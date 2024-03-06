from typing import List, Callable
import transformers
from transformers import TrainerCallback
import dataset.MapDataset as MapDataset
import argparse
import logging
import torchinfo
import torch, torch.utils.data
import tqdm
import os, json
import numpy as np
from copy import deepcopy
from misc.TokenizerBuilder import build_tokenizer
from misc.theoreticalPossibleGuess import guessNoParamMSE
from misc.calcNormalizationParams import calcNormalizationParams
from misc.dataCollator import DataCollatorWithPaddingAndTraceEmbedding
from model.ModelBuilder import build_model
from dataset.DatasetBuilder import build_dataset
from misc.normalization import (
  NormalizerBase,
  Normalizer,
  DummyNormalizer,
  LogPlusNormalizer,
  LogNormalizer
)
from misc.metric import compute_metrics

logger = logging.getLogger(__name__)
rootDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "./")

def summarizeModel(model: 'torch.nn.Module', max_seq_length: int):
    exampleData = {
        'input_ids': torch.tensor([2 for i in range(max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'position_ids': torch.tensor([i for i in range(0, max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'labels': torch.tensor([[100.0]], dtype=torch.float32)
    }
    print(exampleData)
    torchinfo.summary(model, verbose=1, input_data=exampleData)

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        self.prof.step()

class TrainingSetMetricsReportCallback(TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
    
    def on_step_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            return control_copy

# TODO: fix me
class DatasetPostProcessor(MapDataset.MapMixin):
    def __init__(self, tokenizer, targetNormalizer: 'NormalizerBase'=None, regressTime=True):
        self.tokenizer = tokenizer
        self.targetNormalizer = targetNormalizer
        self.regressTime = regressTime

    def doMap(self, elem):
        text = elem["spvText"]
        # TODO: disassemble into text and feed into tokenizer, since they receive text

        encoded_inputs = self.tokenizer(text)

        if self.regressTime:
            encoded_inputs['labels'] = self.targetNormalizer.normalize(1.0 / elem['fpsMean'])
        else:
            encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['fpsMean'])

        # print(len(encoded_inputs['input_ids']))
        return encoded_inputs

class TraceDatasetAsNonTracedPostProcessor(MapDataset.MapMixin):
    """Receives input from FragmentPerformanceWithTraceDataset
    generate the following: {
      # by tokenizer; gives tokenized result of inlined spv binary
      'input_ids': []
      # by dataset - two submode available (fps or time)
      'labels': []
    }
    """
    def __init__(
            self,
            tokenizer,
            targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

    def doMap(self, elem):
        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem["fragSpv"],
            id2TraceIdxMap=elem["bbIdxMap"],
            traceCounters=elem["bbTraceCounters"]
        )

        encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['timeMean'])
        del encoded_inputs['trace_labels']

        return encoded_inputs
    
class TraceDatasetAsConstTracedPostProcessor(MapDataset.MapMixin):
    """Receives input from FragmentPerformanceWithTraceDataset
    generate the following: {
      # by tokenizer; gives tokenized result of inlined spv binary
      'input_ids': []
      # by dataset - two submode available (fps or time)
      'labels': []
      # erased to 1
      'trace_labels': []
    }
    """
    def __init__(
            self,
            tokenizer,
            targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

    def doMap(self, elem):
        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem["fragSpv"],
            id2TraceIdxMap=elem["bbIdxMap"],
            traceCounters=elem["bbTraceCounters"]
        )

        encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['timeMean'])
        # del encoded_inputs['trace_labels']

        encoded_inputs['trace_labels'] = self.traceNormalizer.normalize(
            [1 for i in range(0, len(encoded_inputs['input_ids']))]
        )
        return encoded_inputs

class TracedDatasetPostProcessor(MapDataset.MapMixin):
    """Receives input from FragmentPerformanceWithTraceDataset
    generate the following: {
      # by tokenizer; gives tokenized result of inlined spv binary
      'input_ids': []
      # by dataset - two submode available (fps or time)
      'labels': []
      # by trace - will give traced total cnt on each OpXXX label
      'trace_labels': []
    }
    """
    def __init__(
            self,
            tokenizer,
            targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

    def doMap(self, elem):
        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem["fragSpv"],
            id2TraceIdxMap=elem["bbIdxMap"],
            traceCounters=elem["bbTraceCounters"]
        )

        encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['timeMean'])

        # we convert to float first to avoid the silly overflow caused
        # by things like torch.tensor([UINT64_CAPABLE_BUT_UINT32_NOT])
        encoded_inputs['trace_labels'] = self.traceNormalizer.normalize(
            [float(i) for i in encoded_inputs['trace_labels']]
        )
        return encoded_inputs

class TracedDatasetMALPostProcessor(MapDataset.MapMixin):
    """Receives input from FragmentPerformanceWithTraceDataset
    generate the following: {
      # by tokenizer; gives tokenized result of inlined spv binary
      'input_ids': []
      # by dataset - two submode available (fps or time)
      'labels': []
      # by trace - will give traced total cnt on each OpXXX label
      'trace_labels': []
    }
    """
    def __init__(
            self,
            tokenizer,
            targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

    def doMap(self, elem):
        assert("datasetIdx" in elem)

        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem["fragSpv"],
            id2TraceIdxMap=elem["bbIdxMap"],
            traceCounters=elem["bbTraceCounters"]
        )

        encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['timeMean'])
        encoded_inputs['label_dest_heads'] = elem["datasetIdx"]

        # we convert to float first to avoid the silly overflow caused
        # by things like torch.tensor([UINT64_CAPABLE_BUT_UINT32_NOT])
        encoded_inputs['trace_labels'] = self.traceNormalizer.normalize(
            [float(i) for i in encoded_inputs['trace_labels']]
        )
        return encoded_inputs

class TracedDatasetMTLPostProcessor(MapDataset.MapMixin):
    """Receives input from FragmentPerformanceWithTraceDataset
    generate the following: {
      # by tokenizer; gives tokenized result of inlined spv binary
      'input_ids': []
      # by dataset - two submode available (fps or time)
      'labels': []
      # by trace - will give traced total cnt on each OpXXX label
      'trace_labels': []
    }
    """
    def __init__(
            self,
            tokenizer,
            mlm_prob,
            maskTokenId,
            vocabSize,
            targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

        self.mlm_prob = mlm_prob
        assert(0 <= self.mlm_prob <= 1)

        self.maskTokenId = maskTokenId
        self.vocabSize = vocabSize

    def doMask(
        self,
        elemTokens: 'torch.LongTensor'
    ):

        # print(elemTokens)
        inputs = np.asarray(elemTokens, dtype=np.int64).copy()
        labels = np.asarray(elemTokens, dtype=np.int64).copy()

        # TODO: mask out the prob matrix if necessary
        probability_matrix = np.full(labels.shape, self.mlm_prob)

        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.binomial.html
        # Returns shape (numTokens,)
        masked_indices = np.random.binomial(
            1, probability_matrix, size=probability_matrix.shape
        ).astype(bool)

        # -100 is used in CrossEntropyLoss as ignore_index in PyTorch code
        # Ignore loss for non masked tokens
        labels[~masked_indices] = -100

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        # 0.8 * mlm_prob
        indices_replaced = np.random.binomial(1, 0.8, size=labels.shape).astype(bool) & masked_indices
        inputs[indices_replaced] = self.maskTokenId

        # 10% of the time, we replace masked input tokens with random word
        # 0.5 * mlmprob * (1-0.8) = 0.1 * mlmprob
        indices_random = (
            np.random.binomial(1, 0.5, size=labels.shape).astype(bool) & masked_indices & ~indices_replaced
        )
        random_words = np.random.randint(
            low=0, high=self.vocabSize, size=np.count_nonzero(indices_random), dtype=np.int64
        )
        inputs[indices_random] = random_words

        # and for 10% of the time, the words are unchanged
        return {
            "input_ids": inputs.tolist(),
            "mlm_labels": labels.tolist()
        }

    def doMap(self, elem):
        encoded_inputs = self.tokenizer(
            spvBinaryRepr=elem["fragSpv"],
            id2TraceIdxMap=elem["bbIdxMap"],
            traceCounters=elem["bbTraceCounters"]
        )

        encoded_inputs['labels'] = self.targetNormalizer.normalize(elem['timeMean'])

        # we convert to float first to avoid the silly overflow caused
        # by things like torch.tensor([UINT64_CAPABLE_BUT_UINT32_NOT])
        encoded_inputs['trace_labels'] = self.traceNormalizer.normalize(
            [float(i) for i in encoded_inputs['trace_labels']]
        )

        masked_inputs = self.doMask(encoded_inputs['input_ids'])

        encoded_inputs['input_ids'] = masked_inputs['input_ids']
        encoded_inputs['mlm_labels'] = masked_inputs['mlm_labels']

        return encoded_inputs

def build_normalizer(
    normalizerName: str,
    trainDataset: 'torch.utils.data.Dataset',
    writeCache=True
):
    """Returns normalizer"""
    if normalizerName == "dummy-time":
        return DummyNormalizer()
    elif normalizerName == "collect-from-sample-time":
        setLen = len(trainDataset)

        def timeIter():
            for idx in range(0, setLen):
                yield trainDataset[idx]["timeMean"]

        normalizer = Normalizer.buildFromSamples(timeIter())

        logger.info(
            f"Trained {normalizerName} on {trainDataset.__class__.__name__}, "
            f"mean={normalizer.mean}, stdev={normalizer.stdev}"
        )
        
        if writeCache:
            with open(os.path.join(rootDir, "./intermediates/collect-from-sample-time-cached.json"), "w") as f:
                json.dump({
                    "type": normalizerName,
                    "mean": normalizer.mean,
                    "stdev": normalizer.stdev
                }, f)

        return normalizer

    elif normalizerName == "collect-from-sample-time-cached":
        with open(rootDir, "./collect-from-sample-time-cached.json") as f:
            res = json.load(f)
            assert(res["type"] == "collect-from-sample-time")
            return Normalizer(res["mean"], res["stdev"])

    elif normalizerName == "log-plus-normalizer-time":
        return LogPlusNormalizer()
    elif normalizerName == "log-normalizer-time":
        return LogNormalizer()
    else:
        raise NotImplementedError(f"Unknown normalizer {normalizerName}, implement me")

def run_command(args, outputDir):
    if args.command == 'train':
        trainer.train(args.resume_from_checkpoint)
        trainer.save_model(outputDir)
    elif args.command == 'test':
        metrics = trainer.evaluate()
        print(metrics)
    elif args.command == 'val':
        metrics = trainer.evaluate()
        print(metrics)
    elif args.command == 'baseline':
        trainDatasetMapped = MapDataset.MapDataset(
            trainDataset, DatasetPostProcessor, [lambda x: {}]
        )
        testDatasetMapped = MapDataset.MapDataset(
            testDataset, DatasetPostProcessor, [lambda x: {}]
        )

        def testOnce(dataset: 'MapDataset.MapDataset', testFn: Callable[[List[float]], None]):
            labels = []
            for elem in tqdm.tqdm(dataset):
                labels.append(elem["labels"])

            # print(labels)
            testFn(labels)

        print("train Dataset:")
        testOnce(trainDatasetMapped, guessNoParamMSE)
        
        print("test Dataset:")
        testOnce(testDatasetMapped, guessNoParamMSE)

        print("train Dataset:")
        testOnce(trainDatasetMapped, calcNormalizationParams)

        print("test Dataset:")
        testOnce(testDatasetMapped, calcNormalizationParams)


def loadPretrainedMocoModel(model, mocoModel, tokenizer, loadDir):
    mocoModel = build_model(
        mocoModel,
        "mse",
        4096,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        loadDir=loadDir
    )

    # copy from mocoModel to model
    patched = {f"perfformer.{k}": v for k, v in mocoModel.encoder_q.state_dict().items()}
    model.load_state_dict(patched, strict=False)
    
    # print missing and unexpected keys
    missing_keys = []
    unexpected_keys = []
    for k, v in model.state_dict().items():
        if k not in patched:
            missing_keys.append(k)
    
    for k, v in patched.items():
        if k not in model.state_dict():
            unexpected_keys.append(k)

    print(f"Loaded from MoCo model; unexpected: {unexpected_keys}, missing: {missing_keys}")

    return model
    

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--database-file', default="")
    parser.add_argument('--tokenizer', default="HfSpvTokenizer")
    parser.add_argument('--dataset', default="FragmentPerformanceSnapshotDataset")
    parser.add_argument(
        '--model', default="roberta-base-layer9-regression-vocab40000")
    parser.add_argument('--output-dir-prefix', default="model_output",
                        help="optional prefix for output directory")
    parser.add_argument('--torch-compile', action='store_true')
    parser.add_argument('--save-steps', type=int, default=1500,
                        help="after # steps model is saved")
    parser.add_argument('--save-total-limit', type=int, default=5,
                        help="limit the total amount of checkpoints. Deletes the older checkpoints.")
    parser.add_argument('--num-epochs', type=int, default=3,
                        help="number of training epochs")
    parser.add_argument('--label-normalizer', type=str, default="")
    parser.add_argument('--trace-normalizer', type=str, default="")
    parser.add_argument('--loss-function', type=str, default="mse")
    parser.add_argument('--post-processor', type=str, default="")
    parser.add_argument('--dataset-root-dir-override', type=str, default="")

    # convert from trace_labels -> trace_embeds inside collator
    # Candidate: ['none', 'onehot-base2-shifted', 'onehot-base2']
    parser.add_argument('--collator-trace-embedding', type=str, default="none")

    # scheduler
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help="percent of number of training steps to be in warmup")
    parser.add_argument('--lr-scheduler-type', type=str, default='polynomial', help="SchedulerType")
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help="learning rate")
    parser.add_argument('--per-device-batch-size', type=int, default=16,
                        help="batch size for training")
    parser.add_argument('--max-seq-length', type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument('--no-load', action='store_true',
                        help="Do not load the previous model output")
    parser.add_argument('--load-dir-override', default="")
    parser.add_argument('--load-from-moco-dir', default='')
    parser.add_argument('--no-cuda', action='store_true',
                        help="Do not use cuda")
    parser.add_argument('--use-fp16', action='store_true',
                        help="Use fp16")
    parser.add_argument("--resume-from-checkpoint", action='store_true',
                        help="Boolean, whether to load checkpoint from previous instance of output_dir")
    parser.add_argument("--mtl-mlm-prob", type=float, help="MLM subtask prob for MTL")
    parser.add_argument("--enable-torch-profiler", action='store_true',
                        help="Enable PyTorch Profiler")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    # huggingface trainer options
    parser.add_argument("--load-best-model-at-end", action='store_true')
    parser.add_argument("--metric-for-best-model", type=str, default="eval_mape")
    parser.add_argument("--greater-is-better", type=bool, default=False)
    parser.add_argument("--moco-model", default="perfformer-rope-layer9-moco-m-0.999-T-0.07-trace-input-emmbedding-xformers-memeff")
    parser.add_argument('--pad-to', type=int, default=1,
                        help="Pad to multiple of sth")
    # parser.add_argument('--output-dir-root', type=str, default=rootDir)
    # parser.add_argument('--load-dir-root', type=str, default=rootDir)
    parser.add_argument('command', choices=['train', 'test', 'val', 'baseline'])

    args = parser.parse_args()

    # select and build tokenizer; This tokenizer is used by the main thread
    tokenizer = build_tokenizer(args.tokenizer)

    # assemble output dir according to model and tokenzier pair selected
    outputDir = (args.output_dir_prefix +
                 "_") if args.output_dir_prefix != "" else ""
    outputDir += args.tokenizer + "_" + args.model

    # select and build model
    if not args.no_load:
        model = build_model(
            args.model,
            args.loss_function,
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=outputDir if args.load_dir_override == "" else args.load_dir_override
        )
    else:
        model = build_model(
            args.model,
            args.loss_function,
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=None
        )

    if args.load_from_moco_dir != '':
        logger.warning("Loading from MoCo model will override the default loading behaviour, casuing a force load")
        model = loadPretrainedMocoModel(model, args.moco_model, tokenizer, args.load_from_moco_dir)

    if args.summary:
        summarizeModel(model, args.max_seq_length)

    # initialize dataset
    trainDataset = build_dataset(
        args.dataset,
        "train",
        rootDirOverride=args.dataset_root_dir_override if args.dataset_root_dir_override != "" else None
    )
    testDataset = build_dataset(
        args.dataset,
        "test" if args.command != "val" else "val",
        rootDirOverride=args.dataset_root_dir_override if args.dataset_root_dir_override != "" else None
    )
    if args.command == 'val':
        print(f"Validation set is used instead of test dataset. Please be aware.")

    num_gpus = torch.cuda.device_count()
    effective_bsz = args.per_device_batch_size * num_gpus * args.gradient_accumulation_steps

    logger.info(f"num_gpus={num_gpus} effective_bsz={effective_bsz}")
    logger.info(f"num_epochs={args.num_epochs} train_dataset_len={len(trainDataset)}")
    # define collator and start training
    training_args = transformers.TrainingArguments(
        output_dir=outputDir,  # output directory
        overwrite_output_dir=True,  # overwrite the content of the output directory
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        logging_first_step=True,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        dataloader_num_workers=16,
        optim='adamw_hf',
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        torch_compile=args.torch_compile,
        no_cuda=args.no_cuda,
        fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        # If not specified, hf Trainer will examine the forward method signature
        # for model class provided during instantiation, and 'trace_labels' will be there
        # affecting eval_prediction works
        label_names=["labels"],
        log_level='info'
        # eval_steps=2 # debug use
    )

    trace_embed_dim = model.config.hidden_size
    if args.collator_trace_embedding != 'none':
        # must be in sync with perfformer
        assert(model.config.trace_label_embedding_type == 'input')

    if args.torch_compile:
        # Pad to the possible maximum length, or the torch compiler
        # will retrace for each new input shape
        # GREATLY reducing the speed
        data_collator = DataCollatorWithPaddingAndTraceEmbedding(
            tokenizer=tokenizer,
            padding='max_length',
            max_length=args.max_seq_length,
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )
    elif args.pad_to != 1:
        data_collator = DataCollatorWithPaddingAndTraceEmbedding(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=args.pad_to,
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )
    else:
        data_collator = DataCollatorWithPaddingAndTraceEmbedding(
            tokenizer=tokenizer,
            padding='longest',
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )

    # build combined dataset
    label_normalizer = build_normalizer(args.label_normalizer, trainDataset)
    trace_normalizer = build_normalizer(args.trace_normalizer, trainDataset)

    train_ppset = None
    eval_ppset = None
    if args.post_processor == 'DatasetPostProcessor':
        # TODO: THIS IS TO BE FIXED
        assert(False)
        assert(args.tokenizer not in ('HfTracedSpvTokenizer-single-entrypoint'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, DatasetPostProcessor, [tokenizer, normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, DatasetPostProcessor, [tokenizer, normalizer]
        )
    elif args.post_processor == 'TracedDatasetPostProcessor':
        assert(args.tokenizer not in ('HfSpvTokenizer', 'HfBpeTokenizer'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, TracedDatasetPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TracedDatasetPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
    elif args.post_processor == 'TraceDatasetAsConstTracedPostProcessor':
        assert(args.tokenizer not in ('HfSpvTokenizer', 'HfBpeTokenizer'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, TraceDatasetAsConstTracedPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TraceDatasetAsConstTracedPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
    elif args.post_processor == 'TraceDatasetAsNonTracedPostProcessor':
        assert(args.tokenizer not in ('HfSpvTokenizer', 'HfBpeTokenizer'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, TraceDatasetAsNonTracedPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TraceDatasetAsNonTracedPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
    elif args.post_processor == 'TracedDatasetMTLPostProcessor':
        assert(args.tokenizer not in ('HfSpvTokenizer', 'HfBpeTokenizer'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, TracedDatasetMTLPostProcessor, [
                tokenizer, args.mtl_mlm_prob, tokenizer.mask_token_id, tokenizer.vocab_size, label_normalizer, trace_normalizer
            ]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TracedDatasetMTLPostProcessor, [
                tokenizer, args.mtl_mlm_prob, tokenizer.mask_token_id, tokenizer.vocab_size, label_normalizer, trace_normalizer
            ]
        )
    elif args.post_processor == 'TracedDatasetMALPostProcessor':
        assert(args.tokenizer not in ('HfSpvTokenizer', 'HfBpeTokenizer'))
        train_ppset = MapDataset.MapDataset(
            trainDataset, TracedDatasetMALPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TracedDatasetMALPostProcessor, [tokenizer, label_normalizer, trace_normalizer]
        )
    else:
        raise NotImplementedError(f"Unknown post processor {args.post_processor}")

    assert(train_ppset is not None)
    assert(eval_ppset is not None)

    def compute_fragment_performance_metrics(pred: 'transformers.trainer_utils.EvalPrediction'):
        """pred: inputs, label_ids and predictions, each of them is np.ndarray"""
        metrics = {}

        num_samples = pred.label_ids.shape[0]
        assert(pred.label_ids.shape == (num_samples,))
        assert(pred.predictions.shape == (num_samples, 1) or pred.predictions.shape == (num_samples, ))

        labels = pred.label_ids
        preds = pred.predictions.squeeze()

        # MSE
        metrics["mse_sqrt_raw"] = np.sqrt(((labels - preds)**2).mean(axis=0))

        # MAE
        metrics["mae_raw"] = np.mean(np.abs((labels - preds)))

        # MAPE
        metrics["mape_raw"] = np.mean(np.abs((labels - preds)/labels))

        # original (in time sense)
        orig_labels = label_normalizer.invNormalize(labels)
        orig_predictions = label_normalizer.invNormalize(pred.predictions.reshape((num_samples,)))

        metrics_merged = {
            **compute_metrics(orig_predictions, orig_labels),
            **metrics
        }

        return metrics_merged

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ppset,
        eval_dataset=eval_ppset,
        compute_metrics=compute_fragment_performance_metrics
    )
    # We also do metrics computing on training set
    trainer.add_callback(TrainingSetMetricsReportCallback(trainer))

    if args.enable_torch_profiler:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                        torch.profiler.ProfilerActivity.CUDA], 
                            schedule=torch.profiler.schedule(skip_first=3, wait=1, warmup=1, active=2, repeat=10),
                            on_trace_ready=torch.profiler.tensorboard_trace_handler('hf-training-trainer'),
                            profile_memory=True,
                            with_stack=True,
                            record_shapes=True) as prof:
    
            trainer.add_callback(ProfCallback(prof=prof))
            run_command(args, outputDir)
    else:
        run_command(args, outputDir)

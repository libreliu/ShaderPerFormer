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
from misc.dataCollator import DataCollatorWithPaddingAndTraceEmbedding, DataPairCollatorWithPaddingAndTraceEmbedding
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
from sklearn.neighbors import NearestNeighbors

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

class TracedPairDatasetPostProcessor(MapDataset.MapMixin):
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
            # targetNormalizer: 'NormalizerBase'=None,
            traceNormalizer: 'NormalizerBase'=None
        ):
        # Tokenizer should support trace things
        self.tokenizer = tokenizer
        # self.targetNormalizer = DummyNormalizer() if targetNormalizer is None else targetNormalizer
        self.traceNormalizer = DummyNormalizer() if traceNormalizer is None else traceNormalizer

    def doMap(self, elem):
        elem1, elem2 = elem

        encoded_inputs1 = self.tokenizer(
            spvBinaryRepr=elem1["fragSpv"],
            id2TraceIdxMap=elem1["bbIdxMap"],
            traceCounters=elem1["bbTraceCounters"]
        )
        encoded_inputs1['trace_labels'] = self.traceNormalizer.normalize(
            [float(i) for i in encoded_inputs1['trace_labels']]
        )
        if elem2 is None:
            label = 0
            for i in elem1['shaderId']:
                label = label * 128 + ord(i)
            encoded_inputs1['labels'] = label

        if elem2 is not None:
            encoded_inputs2 = self.tokenizer(
                spvBinaryRepr=elem2["fragSpv"],
                id2TraceIdxMap=elem2["bbIdxMap"],
                traceCounters=elem2["bbTraceCounters"]
            )
            encoded_inputs2['trace_labels'] = self.traceNormalizer.normalize(
                [float(i) for i in encoded_inputs2['trace_labels']]
            )
        else:
            encoded_inputs2 = None
    
        return encoded_inputs1, encoded_inputs2
    
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
    parser.add_argument("--load-best-model-at-end", type=bool, default=True)
    parser.add_argument("--metric-for-best-model", type=str, default="eval_nearest_neighbor_precision")
    parser.add_argument("--greater-is-better", type=bool, default=True)
    parser.add_argument('--pad-to', type=int, default=1,
                        help="Pad to multiple of sth")
    # parser.add_argument('--output-dir-root', type=str, default=rootDir)
    # parser.add_argument('--load-dir-root', type=str, default=rootDir)
    parser.add_argument('command', choices=['train', 'test', 'baseline'])

    args = parser.parse_args()

    # select and build tokenizer; This tokenizer is used by the main thread
    tokenizer = build_tokenizer(args.tokenizer)

    # assemble output dir according to model and tokenzier pair selected
    outputDir = (args.output_dir_prefix +
                 "_") if args.output_dir_prefix != "" else ""
    outputDir += args.tokenizer + "_" + args.model

    # select and build model
    if args.no_load:
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
        "test",
        rootDirOverride=args.dataset_root_dir_override if args.dataset_root_dir_override != "" else None
    )

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
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        fp16=args.use_fp16,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        # If not specified, hf Trainer will examine the forward method signature
        # for model class provided during instantiation, and 'trace_labels' will be there
        # affecting eval_prediction works
        label_names=["labels"],
        # eval_steps=1 # debug use
    )

    trace_embed_dim = model.config.hidden_size
    if args.collator_trace_embedding != 'none':
        # must be in sync with perfformer
        assert(model.config.trace_label_embedding_type == 'input')

    if 'moco' in args.model:
        data_collator_type = DataPairCollatorWithPaddingAndTraceEmbedding
    else:
        data_collator_type = DataCollatorWithPaddingAndTraceEmbedding

    if args.torch_compile:
        # Pad to the possible maximum length, or the torch compiler
        # will retrace for each new input shape
        # GREATLY reducing the speed
        data_collator = data_collator_type(
            tokenizer=tokenizer,
            padding='max_length',
            max_length=args.max_seq_length,
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )
    elif args.pad_to != 1:
        data_collator = data_collator_type(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=args.pad_to,
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )
    else:
        data_collator = data_collator_type(
            tokenizer=tokenizer,
            padding='longest',
            trace_embedding_dim=trace_embed_dim,
            trace_embedding_method=args.collator_trace_embedding
        )

    # build combined dataset
    label_normalizer = build_normalizer('dummy-time', trainDataset)
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
    elif args.post_processor == 'TracedPairDatasetPostProcessor':
        train_ppset = MapDataset.MapDataset(
            trainDataset, TracedPairDatasetPostProcessor, [tokenizer, trace_normalizer]
        )
        eval_ppset = MapDataset.MapDataset(
            testDataset, TracedPairDatasetPostProcessor, [tokenizer, trace_normalizer]
        )
    else:
        raise NotImplementedError(f"Unknown post processor {args.post_processor}")

    assert(train_ppset is not None)
    assert(eval_ppset is not None)

    def compute_moco_performance(pred: 'transformers.trainer_utils.EvalPrediction'):
        """pred: inputs, label_ids and predictions, each of them is np.ndarray"""
        label_ids = pred.label_ids
        predictions = pred.predictions
        k = 4

        # Initialize NearestNeighbors instance
        neigh = NearestNeighbors(n_neighbors=k+1)

        # Fit the model with the data
        neigh.fit(predictions)

        # Find the nearest neighbor for each prediction
        _, indices = neigh.kneighbors(predictions)

        # Get the corresponding labels
        nearest_labels = label_ids[indices]

        nearest_neighbor_precision = np.sum(nearest_labels[:, 0] == nearest_labels[:, 1]) / nearest_labels.shape[0]
        top4_neighbor_precision = np.sum(nearest_labels[:, 0, None] == nearest_labels[:, 1:k+1]) / nearest_labels.shape[0] / k

        # print(f"Nearest neighbor precision: {nearest_neighbor_precision}")
        # print(f"Top 4 neighbor precision: {top4_neighbor_precision}")

        return {
            "nearest_neighbor_precision": nearest_neighbor_precision,
            "top4_neighbor_precision": top4_neighbor_precision
        }

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_ppset,
        eval_dataset=eval_ppset,
        compute_metrics=compute_moco_performance
    )
    # We also do metrics computing on training set
    # trainer.add_callback(TrainingSetMetricsReportCallback(trainer))

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

from dataclasses import dataclass
from typing import List, Callable
import transformers
from transformers import DataCollatorWithPadding
import dataset.MapDataset as MapDataset
import argparse
import logging
import torchinfo
import torch, torch.utils.data
import tqdm
import numpy as np
from misc.TokenizerBuilder import build_tokenizer
from misc.theoreticalPossibleGuess import guessNoParamMSE
from misc.calcNormalizationParams import calcNormalizationParams
from model.ModelBuilder import build_model
from dataset.DatasetBuilder import build_dataset

logger = logging.getLogger(__name__)

def summarizeModel(model: 'torch.nn.Module', max_seq_length: int):
    exampleData = {
        'input_ids': torch.tensor([2 for i in range(max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'position_ids': torch.tensor([i for i in range(0, max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'labels': torch.tensor([[100.0]], dtype=torch.float32)
    }
    print(exampleData)
    torchinfo.summary(model, verbose=1, input_data=exampleData)

# Noop for now
class DatasetPostProcessor(MapDataset.MapMixin):
    def __init__(self):
        pass
    
    def doMap(self, elem):
        return {
            "input_ids": elem["input_ids"],
            "labels": elem["labels"]
        }

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)40s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(prog='train')
    parser.add_argument('--summary', action='store_true')
    parser.add_argument('--database-file', default="")
    parser.add_argument('--tokenizer', default="HfSpvTokenizer")
    parser.add_argument('--dataset', default="FragmentMaskedLMDataset-seqlen1024-epoch15000-mlm0.15-mask1003-pad1000-vocab40000")
    parser.add_argument(
        '--model', default="roberta-base-layer9-maskedlm-vocab40000")
    parser.add_argument('--output-dir-prefix', default="model_output",
                        help="optional prefix for output directory")
    parser.add_argument('--torch-compile', action='store_true')
    parser.add_argument('--save-steps', type=int, default=1500,
                        help="after # steps model is saved")
    parser.add_argument('--save-total-limit', type=int, default=5,
                        help="limit the total amount of checkpoints. Deletes the older checkpoints.")
    parser.add_argument('--num-epochs', type=int, default=3,
                        help="number of training epochs")
    
    # scheduler
    parser.add_argument('--warmup-ratio', type=float, default=0.1,
                        help="percent of number of training steps to be in warmup")
    parser.add_argument('--lr-scheduler-type', type=str, default='polynomial', help="SchedulerType")
    parser.add_argument('--learning-rate', type=float, default=5e-5,
                        help="learning rate")
    parser.add_argument('--per-device-batch-size', type=int, default=16,
                        help="batch size for training")
    parser.add_argument('--max-seq-length', type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument('--no-load', action='store_true',
                        help="Do not load the previous model output")
    parser.add_argument('--load-dir-override', default="")
    parser.add_argument('--no-cuda', action='store_true',
                        help="Do not use cuda")
    parser.add_argument('--use-fp16', action='store_true',
                        help="Use fp16")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument('--pad-to', type=int, default=1,
                        help="Pad to multiple of sth")
    parser.add_argument("--resume-from-checkpoint", action='store_true',
                        help="Boolean, whether to load checkpoint from previous instance of output_dir")
    parser.add_argument('command', choices=['train', 'test'])

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
            None,
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=outputDir if args.load_dir_override == "" else args.load_dir_override
        )
    else:
        model = build_model(
            args.model,
            None,
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=None
        )

    if args.summary:
        summarizeModel(model, args.max_seq_length)

    # initialize dataset
    trainDataset = build_dataset(args.dataset, "train", tokenizer=tokenizer)
    testDataset = build_dataset(args.dataset, "test", tokenizer=tokenizer)

    num_gpus = torch.cuda.device_count()
    effective_bsz = args.per_device_batch_size * num_gpus

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
        dataloader_num_workers=2,
        optim='adamw_hf',
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        torch_compile=args.torch_compile,
        no_cuda=args.no_cuda,
        fp16=args.use_fp16,
        evaluation_strategy="steps",
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    if args.torch_compile:
        # Pad to the possible maximum length, or the torch compiler
        # will retrace for each new input shape
        # GREATLY reducing the speed
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer, padding='max_length', max_length=args.max_seq_length
        )
    elif args.pad_to != 1:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='longest',
            pad_to_multiple_of=args.pad_to
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding='longest'
        )

    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=MapDataset.MapDataset(
            trainDataset, DatasetPostProcessor, []
        ),
        eval_dataset=MapDataset.MapDataset(
            testDataset, DatasetPostProcessor, []
        )
    )

    if args.command == 'train':
        trainer.train(args.resume_from_checkpoint)
        trainer.save_model(outputDir)
    elif args.command == 'test':
        metrics = trainer.evaluate()
        print(metrics)
    else:
        assert(False)

from dataclasses import dataclass
from typing import List, Dict, Optional, Union, Any
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
from model.ModelBuilder import build_model
from dataset.DatasetBuilder import build_dataset

def summarizeModel(model: 'torch.nn.Module', max_seq_length: int):
    exampleData = {
        'input_ids': torch.tensor([2 for i in range(max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'position_ids': torch.tensor([i for i in range(0, max_seq_length)], dtype=torch.long).reshape((1, max_seq_length)),
        'labels': torch.tensor([[100.0]], dtype=torch.float32)
    }
    print(exampleData)
    torchinfo.summary(model, verbose=1, input_data=exampleData)

class DatasetPostProcessor(MapDataset.MapMixin):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def doMap(self, elem):
        text = elem["spvText"]
        encoded_input = self.tokenizer(text)
        encoded_input['labels'] = elem["fpsMean"]
        # print(len(encoded_input['input_ids']))
        return encoded_input

def cliCustom(args, tokenizer, model, trainDataset, testDataset, outputDir):
    proc = DatasetPostProcessor(tokenizer)

    features = proc.doMap(trainDataset[10])
    print(f"len: {len(features['input_ids'])}")
    batch = tokenizer.pad([features], padding='longest', return_tensors='pt')

    model.eval()
    with torch.no_grad():
        output = model(**batch)

    print(output)
    print(features["labels"])

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
    parser.add_argument('--max-seq-length', type=int, default=4096,
                        help="Maximum sequence length")
    parser.add_argument('--no-load', action='store_true',
                        help="Do not load the previous model output")
    parser.add_argument('--load-dir-override', default="")
    parser.add_argument('command', choices=['custom', 'train', 'test'])
    

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
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=outputDir if args.load_dir_override == "" else args.load_dir_override
        )
    else:
        model = build_model(
            args.model,
            args.max_seq_length,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            loadDir=None
        )

    if args.summary:
        summarizeModel(model, args.max_seq_length)

    # initialize dataset
    trainDataset = build_dataset(args.dataset, "train")
    testDataset = build_dataset(args.dataset, "test")

    # load model
    model = model.from_pretrained(outputDir if args.load_dir_override == "" else args.load_dir_override)

    if args.command == "custom":
        cliCustom(args, tokenizer, model, trainDataset, testDataset, outputDir)
    elif args.command == "train":
        pass
    elif args.command == "test":
        pass
    else:
        assert(False)
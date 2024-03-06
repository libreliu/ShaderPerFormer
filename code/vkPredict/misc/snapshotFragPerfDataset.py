from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset

import json
import math
import logging
import numpy as np
import typing
import tqdm

logger = logging.getLogger(__name__)

def snapshot(trainRatio: float, outputFile, maxTokenizedLength=None, tokenizerUsed=None):
    """E.g. trainRatio = 0.6 means 60% of total sample are put into training set"""
    assert (0.0 < trainRatio <= 1.0)

    if tokenizerUsed is None:
        tokenizer = None
        assert(maxTokenizedLength is None)
    elif tokenizerUsed == 'bpe':
        from misc.HfBpeTokenizer import HfBpeTokenizer

        tokenizerHF = HfBpeTokenizer("SpvBpeTokenizer.json")
        def tokenizeBpeWrapped(s) -> typing.List[str]:
            tokenized = tokenizerHF(s)
        
            return tokenized

        tokenizer = tokenizeBpeWrapped
    elif tokenizerUsed == 'spvMultiple':
        from misc.HfSpvTokenizer import HfSpvTokenizer

        tokenizerHF = HfSpvTokenizer(single_entrypoint=False)
        def tokenizeSpvMultipleWrapped(s) -> typing.List[int]:
            tokenized = tokenizerHF(s)["input_ids"]
        
            return tokenized

        tokenizer = tokenizeSpvMultipleWrapped
    elif tokenizerUsed == 'spvSingle':
        raise NotImplementedError("TODO")
    else:
        assert(False)
    
    dset = FragmentPerformanceDataset(tokenizer, maxTokenizedLength=maxTokenizedLength)

    train_size = math.floor(trainRatio * len(dset))

    logger.info(
        f"Total: {len(dset)}, train: {train_size}, test: {len(dset) - train_size}")

    indices = np.arange(len(dset))
    np.random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_samples = []
    test_samples = []

    train_indices_iterator = tqdm.tqdm(train_indices)
    train_indices_iterator.set_description_str("Training sample serialization")
    for idx in train_indices_iterator:
        train_samples.append(dset[idx])
    
    test_indices_iterator = tqdm.tqdm(test_indices)
    test_indices_iterator.set_description_str("Testing sample serialization")
    for idx in test_indices_iterator:
        test_samples.append(dset[idx])

    with open(outputFile, "w") as fp:
        json.dump({
            "train": train_samples,
            "test": test_samples
        }, fp, indent=2)
    
    logger.info(f"Snapshot written to {outputFile}")
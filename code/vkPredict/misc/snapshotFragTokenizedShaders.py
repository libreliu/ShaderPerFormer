from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset

import json
import math
import logging
import numpy as np
import typing
import tqdm

logger = logging.getLogger(__name__)

def snapshot(outputFile, tokenizerUsed=None):
    """Returns a large array, for each element:
    elem = {
      "input_ids": tokenized_array
    }
    """

    if tokenizerUsed is None:
        tokenizer = None
        
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
    
    dset = FragmentPerformanceDataset(tokenizer)
   
    logger.info(
        f"Dataset size: {len(dset)}, tokenizer {tokenizerUsed}, output file {outputFile}")

    samples = []

    indices_iterator = tqdm.tqdm(range(0, len(dset)))
    indices_iterator.set_description_str("Sample serialization")
    for idx in indices_iterator:
        samples.append({
            "input_ids": tokenizer(dset[idx]["spvText"])
        })

    with open(outputFile, "w") as fp:
        json.dump(samples, fp, indent=2)
    
    logger.info(f"Snapshot written to {outputFile}")
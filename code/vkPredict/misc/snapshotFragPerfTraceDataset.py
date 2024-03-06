from dataset.FragmentPerformanceWithTraceDataset import FragmentPerformanceWithTraceDataset
from misc.TokenizerBuilder import build_tokenizer

import pickle
import math
import logging
import numpy as np
import typing
import tqdm

logger = logging.getLogger(__name__)

def snapshot(
        trainRatio: float,
        outputFile: str,
        maxTokenizedLength=None,
        tokenizer=None,
        dsetBuildArgs=None
    ):
    """E.g. trainRatio = 0.6 means 60% of total sample are put into training set"""
    assert (0.0 < trainRatio <= 1.0)
    logger.info(f"maxTokenizedLength == {maxTokenizedLength}")

    dset = FragmentPerformanceWithTraceDataset(
        **({} if dsetBuildArgs is None else dsetBuildArgs)
    )
    dsetLen = len(dset)
    # dsetLen = 50

    if maxTokenizedLength is not None:
        assert(tokenizer is not None)
        failed_samples = []
        all_length = [None for i in range(0, dsetLen)]
        for idx in tqdm.tqdm(range(0, dsetLen)):
            error = False
            try:
                encoded = tokenizer(
                    spvBinaryRepr=dset[idx]["fragSpv"],
                    id2TraceIdxMap=dset[idx]["bbIdxMap"],
                    traceCounters=dset[idx]["bbTraceCounters"]
                )
            except RuntimeError as e:
                logger.warning(f"dataset idx={idx} failed to tokenize, reason: {e}")
                # a big number
                all_length[idx] = maxTokenizedLength + 100
                failed_samples.append(idx)
                error = True

            if error:
                # so it'll be filtered out
                all_length[idx] = maxTokenizedLength + 100
            else:
                all_length[idx] = len(encoded["input_ids"])
        
        print(f"Failed samples (total {len(failed_samples)}): {failed_samples}")
    else:
        all_length = [0 for i in range(0, dsetLen)]

    # print(all_length)

    filtered_length = 0
    filtered_indices = []
    if maxTokenizedLength is not None:
        for idx in range(0, dsetLen):
            if all_length[idx] <= maxTokenizedLength:
                filtered_length += 1
                filtered_indices.append(idx)
    else:
        for idx in range(0, dsetLen):
            if idx not in failed_samples:
                filtered_length += 1
                filtered_indices.append(idx)

    assert(filtered_length == len(filtered_indices))
    # print(filtered_indices)

    train_size = math.floor(trainRatio * filtered_length)

    logger.info(
        f"Total: {dsetLen}, filtered: {filtered_length}, "
        f"({filtered_length / dsetLen * 100.0:.5}% of the original) "
        f"train: {train_size}, test: {filtered_length - train_size}"
    )

    indices = np.asarray(filtered_indices, dtype=np.int32)
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

    with open(outputFile, "wb") as fp:
        pickle.dump({
            "train": train_samples,
            "test": test_samples
        }, fp)

    logger.info(f"Snapshot written to {outputFile}")

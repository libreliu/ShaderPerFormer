import os
rootDirReal = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")

from dataset.FragmentPerformanceSnapshotDataset import FragmentPerformanceSnapshotDataset
from dataset.FragmentMaskedLMDataset import FragmentMaskedLMDataset
from dataset.FragmentPerformanceAugWithTrace import FragmentPerformanceAugWithTracePair
from dataset.FragmentPerformanceTracedSnapshotDataset import (
    FragmentPerformanceTracedSnapshotDataset,
    FragmentPerformanceTraceSnapshotCombinedDataset,
    FragmentPerformanceTraceSnapshotTupleCombinedDataset
)
from misc.HfSpvTokenizer import HfSpvTokenizer

def build_dataset(datasetName, split, tokenizer=None, rootDirOverride=None):
    dataset = None
    if rootDirOverride is None:
        rootDir = rootDirReal
    else:
        rootDir = rootDirOverride

    if datasetName.startswith('FragPerfSnapshotTracedFinalDataset'):
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, f"./intermediates/{datasetName}.dat"),
            split
        )

    assert(dataset is not None)
    return dataset

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

    if datasetName == 'FragmentPerformanceSnapshotDataset':
        dataset = FragmentPerformanceSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotDataset.json"),
            split
        )
    elif datasetName == 'FragmentPerformanceSnapshotDataset1024':
        dataset = FragmentPerformanceSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotDataset1024.json"),
            split
        )
    elif datasetName == 'FragmentPerformanceSnapshotDatasetNormalized':
        dataset = FragmentPerformanceSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotDataset.json"),
            split
        )
    elif datasetName == 'FragmentPerformanceSnapshotDatasetNormalized-6420-2100':
        dataset = FragmentPerformanceSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotDataset.json"),
            split,
            True, "fpsMean", 6420, 2100
        )
    elif datasetName == 'FragmentPerformanceSnapshotDataset1024Normalized-6420-2100':
        dataset = FragmentPerformanceSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotDataset1024.json"),
            split,
            True, "fpsMean", 6420, 2100
        )
    elif datasetName == 'FragmentMaskedLMDataset-seqlen1024-epoch15000-mlm0.15-mask1003-pad1000-vocab40000':
        # Currently only HfSpvTokenizer is supported
        assert(isinstance(tokenizer, HfSpvTokenizer))
        dataset = FragmentMaskedLMDataset(
            os.path.join(rootDir, "./FragTokenizedDataset.json"),
            1024,
            15000,
            0.15,
            1000+3,
            1000+0,
            40000
        )
    elif datasetName == 'FragmentMaskedLMDataset-seqlen1024-epoch15000-mlm0.5-mask1003-pad1000-vocab40000':
        # Currently only HfSpvTokenizer is supported
        assert(isinstance(tokenizer, HfSpvTokenizer))
        dataset = FragmentMaskedLMDataset(
            os.path.join(rootDir, "./FragTokenizedDataset.json"),
            1024,
            15000,
            0.5,
            1000+3,
            1000+0,
            40000
        )
    elif datasetName == 'FragmentMaskedLMDataset-seqlen1024-epoch15000-mlm0.7-mask1003-pad1000-vocab40000':
        # Currently only HfSpvTokenizer is supported
        assert(isinstance(tokenizer, HfSpvTokenizer))
        dataset = FragmentMaskedLMDataset(
            os.path.join(rootDir, "./FragTokenizedDataset.json"),
            1024,
            15000,
            0.7,
            1000+3,
            1000+0,
            40000
        )
    elif datasetName == 'FragmentPerformanceTracedSnapshotDataset':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotTracedDataset.dat"),
            split
        )
    elif datasetName == 'FragmentPerformanceTracedSnapshotDataset4096':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./FragPerfSnapshotTracedDataset4096-Correct.dat"),
            split
        )
    elif datasetName == 'FragmentPerformanceTracedSnapshotDataset4096-3060':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-add3resgroup-train-augmented':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-add3resgroup-train-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-add2resolution-train-augmented':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-add2resolution-train-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-add2resolution-add3resgroup-train-augmented':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-add2resolution-add3resgroup-train-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-add2resolution-add3resgroup-dedup-train-augmented':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-add2resolution-add3resgroup-dedup-train-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-add18resgroup-train-dedup-augmented':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-add18resgroup-train-dedup-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-6600xt-train':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-6600xt-train.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-Optim20000':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-Optim20000.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-optimized-train-augmented':
        dataset = FragmentPerformanceAugWithTracePair(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-optimized-train-augmented.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-uhd630':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-4060':
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-4060.dat"),
            split
        )
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-combined-test3060':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotCombinedDataset([
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat")
                ],
                split
            )
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotCombinedDataset([
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"),
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat")
                ],
                split
            )
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-combined-testuhd630':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotCombinedDataset([
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat")
                ],
                split
            )
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotCombinedDataset([
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"),
                    os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat")
                ],
                split
            )
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-combined-testuhd630-fixed':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "test")
            ])
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "train"),
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "train")
            ])
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-combined-with-testset-testuhd630':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "test")
            ])
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "train"),
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "test"),
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "train")
            ])
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-4060-combined-test3060':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "test")
            ])
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "train"),
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "train"),
                (2, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-4060.dat"), "train")
            ])
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-4060-6600xt-combined-test3060':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "test")
            ])
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "train"),
                (1, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat"), "train"),
                (2, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-4060.dat"), "train"),
                (3, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-6600xt-train.dat"), "train"),
            ])
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-optim-combined':
        if split == "test":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "test")
            ])
        elif split == "train":
            dataset = FragmentPerformanceTraceSnapshotTupleCombinedDataset([
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"), "train"),
                (0, os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060-Optim20000.dat"), "train")
            ])
        else:
            assert(False)
    elif datasetName == 'FragPerfSnapshotTracedDataset4096-3060-uhd630-combined-testAll':
        dataset = FragmentPerformanceTraceSnapshotCombinedDataset([
                os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-3060.dat"),
                os.path.join(rootDir, "./intermediates/FragPerfSnapshotTracedDataset4096-uhd630.dat")
            ],
            split
        )
    elif datasetName.startswith('FragPerfSnapshotTracedFinalDataset'):
        dataset = FragmentPerformanceTracedSnapshotDataset(
            os.path.join(rootDir, f"./intermediates/{datasetName}.dat"),
            split
        )
    assert(dataset is not None)
    return dataset

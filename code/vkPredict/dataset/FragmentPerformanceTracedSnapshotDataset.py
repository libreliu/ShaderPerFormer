import torch.utils.data
import typing
import json
import copy
import pickle
import logging

logger = logging.getLogger(__name__)


class FragmentPerformanceTracedSnapshotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        snapshotFilePath,
        split
    ):
        self.data = None
        with open(snapshotFilePath, "rb") as fp:
            self.data = pickle.load(fp)
        
        self.split = split
        assert(split in ('train', 'test', 'val'))

    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        expr = self.data[self.split][idx]
        expr = copy.deepcopy(expr)

        return expr

class FragmentPerformanceTraceSnapshotCombinedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        snapshotFilePaths: typing.List[str],
        split
    ):
        assert(split in ('train', 'test'))

        datas = []
        for filePath in snapshotFilePaths:
            with open(filePath, "rb") as fp:
                datas.append(pickle.load(fp))
        
        self.data = []
        for datasetIdx, data in enumerate(datas):
            for elem in data[split]:
                self.data.append((datasetIdx, elem))

        logger.info(f"Loaded {len(self.data)} data from {len(snapshotFilePaths)} datasets")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """An extra datasetIdx is appended on the dict returned"""
        expr = copy.deepcopy(self.data[index])
        
        returnDict = {k: v for k, v in expr[1].items()}
        returnDict["datasetIdx"] = expr[0]

        return returnDict
    
class FragmentPerformanceTraceSnapshotTupleCombinedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # (datasetIdx, path, split)
        snapshotFilePathWithSplits: typing.List[typing.Tuple[int, str, str]]
    ):

        self.data = []
        for datasetIdx, filePath, split in snapshotFilePathWithSplits:
            assert(split in ('train', 'test'))

            with open(filePath, "rb") as fp:
                datas = pickle.load(fp)
                for elem in datas[split]:
                    self.data.append((datasetIdx, split, elem))

        logger.info(f"Loaded {len(self.data)} data from {len(snapshotFilePathWithSplits)} dataset shards")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """An extra datasetIdx is appended on the dict returned"""
        expr = copy.deepcopy(self.data[index])
        
        returnDict = {k: v for k, v in expr[2].items()}
        returnDict["datasetIdx"] = expr[0]
        returnDict["datasetSplit"] = expr[1]

        return returnDict
import torch.utils.data
import typing
import json
import copy

class FragmentPerformanceSnapshotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        snapshotFilePath,
        split,
        targetNorm=False,
        targetNormAttribute="fpsMean",
        targetNormMean=0,
        targetNormStdev=1
    ):
        self.data = None
        with open(snapshotFilePath, "r") as fp:
            self.data = json.load(fp)
        
        self.split = split
        assert(split in ('train', 'test'))

        self.targetNorm = targetNorm
        self.targetNormAttribute = targetNormAttribute
        self.targetNormMean = targetNormMean
        self.targetNormStdev = targetNormStdev

    def __len__(self):
        return len(self.data[self.split])

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        expr = self.data[self.split][idx]
        expr = copy.deepcopy(expr)

        if self.targetNorm:
            expr[self.targetNormAttribute] -= self.targetNormMean
            expr[self.targetNormAttribute] /= self.targetNormStdev

        return expr

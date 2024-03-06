import torch.utils.data

class MapMixin:
    def doMap(self, data):
        raise NotImplementedError("Implement me")

class MapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, map_cls, mapClsArgs=[], mapClsKwArgs={}):
        self.dataset = dataset
        self.map = map_cls(*mapClsArgs, **mapClsKwArgs)
        if not isinstance(self.map, MapMixin):
            raise Exception("map class should be derived from MapMixin")

    def __getitem__(self, index):
        return self.map.doMap(self.dataset[index])

    def __len__(self):
        return len(self.dataset)


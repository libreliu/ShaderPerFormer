import torch.utils.data
import peewee
import logging
import json
import os
from toyDb.databases.ExperimentDb import ImageOnlyExperiment
from typing import Union, List
import pickle
import copy
import random

logger = logging.getLogger(__name__)
curDir = os.path.dirname(os.path.abspath(__file__))

class FragmentPerformanceAugWithTraceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # the expr query w.r.t. ImageOnlyExperiment, or the list of expr ids
        modelQuery: Union['peewee.ModelSelect', 'list[int]'],
        filteredIds: List[int] = None,
        maxNumChildren: int = None
    ):
        if isinstance(modelQuery, peewee.ModelSelect):
            self.candidateExprs = modelQuery
        elif isinstance(modelQuery, list):
            self.candidateExprs = [
                ImageOnlyExperiment.get_by_id(idx) for idx in modelQuery
            ]

        self.filteredIds = filteredIds
        self.maxNumChildren = maxNumChildren

        # print(self.candidateExprs)

    def __len__(self):
        return len(self.candidateExprs)
    
    def getChildren(self, expr: ImageOnlyExperiment):
        children = [{
            'environmentId': expr.environment_id,
            'shaderId': expr.shader.shader_id,
            'fragSpv': expr.shader.fragment_spv,
            'traceFragSpv': expr.trace.traced_fragment_spv,
            'bbIdxMap': {int(k): v for k, v in json.loads(expr.trace.bb_idx_map).items()},
            'bbTraceCounters': json.loads(expr.trace.bb_trace_counters)
        }]
        exprs = [expr]
        while len(exprs) > 0:
            expr = exprs.pop(0)
            for child in expr.children:
                exprs.append(child)

                if child.id in self.filteredIds:
                    children.append({
                        'environmentId': child.environment_id,
                        'shaderId': child.shader.shader_id,
                        'fragSpv': child.shader.fragment_spv,
                        'traceFragSpv': child.trace.traced_fragment_spv,
                        'bbIdxMap': {int(k): v for k, v in json.loads(child.trace.bb_idx_map).items()},
                        'bbTraceCounters': json.loads(child.trace.bb_trace_counters)
                    })
                    if len(children) >= self.maxNumChildren:
                        return children

        return children

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        expr = self.candidateExprs[idx]

        results = json.loads(expr.results)
        result_mean = sum(results) / len(results)

        return {
            # "environmentId": expr.environment_id,
            "shaderId": expr.shader.shader_id,
            # # SPIR-V bytes
            # "fragSpv": expr.shader.fragment_spv,
            # # SPIR-V bytes
            # "traceFragSpv": expr.trace.traced_fragment_spv,
            # # float
            "timeMean": result_mean,
            # # dict[int, int]
            # "bbIdxMap": {int(k): v for k, v in json.loads(expr.trace.bb_idx_map).items()},
            # # List[int]
            # "bbTraceCounters": json.loads(expr.trace.bb_trace_counters),
            # List[dict]
            "children": self.getChildren(expr)
        }


class FragmentPerformanceAugWithTracePair(torch.utils.data.Dataset):
    def __init__(
        self,
        snapshotFilePath,
        split,
    ):
        self.data = None
        with open(snapshotFilePath, "rb") as fp:
            self.data = pickle.load(fp)
        
        self.split = split
        assert(split in ('train', 'test'))

        self.allTestChilds = []
        if self.split == 'test':
            for expr in self.data["test"]:
                self.allTestChilds += expr['children']
    
    def __len__(self):
        if self.split == "train":
            return len(self.data[self.split])
        elif self.split == "test":
            return len(self.allTestChilds)
        else:
            assert(False)
    
    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        if self.split == "train":
            expr = self.data[self.split][idx]
            children = random.sample(expr['children'], 2)

            assert(len(children) == 2)
            expr1 = copy.deepcopy(children[0])
            expr2 = copy.deepcopy(children[1])

            return expr1, expr2
        
        elif self.split == "test":
            return self.allTestChilds[idx], None

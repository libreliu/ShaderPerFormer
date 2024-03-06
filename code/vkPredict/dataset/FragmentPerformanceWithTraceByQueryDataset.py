import torch.utils.data
import peewee
import logging
import json
import os
from toyDb.databases.ExperimentDb import ImageOnlyExperiment
from typing import Union

logger = logging.getLogger(__name__)
curDir = os.path.dirname(os.path.abspath(__file__))

class FragmentPerformanceWithTraceByQueryDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        # the expr query w.r.t. ImageOnlyExperiment, or the list of expr ids
        modelQuery: Union['peewee.ModelSelect', 'list[int]']
    ):
        if isinstance(modelQuery, peewee.ModelSelect):
            self.candidateExprs = modelQuery
        elif isinstance(modelQuery, list):
            self.candidateExprs = [
                ImageOnlyExperiment.get_by_id(idx) for idx in modelQuery
            ]

        # print(self.candidateExprs)

    def __len__(self):
        return len(self.candidateExprs)

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        expr = self.candidateExprs[idx]

        results = json.loads(expr.results)
        result_mean = sum(results) / len(results)

        return {
            "environmentId": expr.environment_id,
            "shaderId": expr.shader.shader_id,
            # SPIR-V bytes
            "fragSpv": expr.shader.fragment_spv,
            # SPIR-V bytes
            "traceFragSpv": expr.trace.traced_fragment_spv,
            # float
            "timeMean": result_mean,
            # dict[int, int]
            "bbIdxMap": {int(k): v for k, v in json.loads(expr.trace.bb_idx_map).items()},
            # List[int]
            "bbTraceCounters": json.loads(expr.trace.bb_trace_counters)
        }


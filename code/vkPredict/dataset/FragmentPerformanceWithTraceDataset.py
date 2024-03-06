from toyDb.databases.ExperimentDb import (
    Environment,
    ImageOnlyShader,
    ImageOnlyResource,
    ImageOnlyExperiment,
    CANONICAL_NUM_CYCLES,
    CANONICAL_NUM_TRIALS
)
import torch.utils.data
import peewee
import logging
import json
import os

logger = logging.getLogger(__name__)
curDir = os.path.dirname(os.path.abspath(__file__))

class FragmentPerformanceWithTraceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        environmentId=0,
        filteredNumCycles=None,
        filteredNumTrials=None,
        useBlackList=True
    ):
        self.blacklistJSON = None
        if useBlackList:
            with open(os.path.join(curDir, "shaderBlacklist.json"), "r") as fp:
                self.blacklistJSON = json.load(fp)

        if filteredNumTrials is None:
            filteredNumTrials = CANONICAL_NUM_TRIALS
        
        if filteredNumCycles is None:
            filteredNumCycles = CANONICAL_NUM_CYCLES

        filterArgs = [
            ImageOnlyExperiment.num_cycles == filteredNumCycles,
            ImageOnlyExperiment.num_trials == filteredNumTrials,
            ImageOnlyExperiment.environment == environmentId,
            ImageOnlyExperiment.trace.is_null(False)
        ]

        if useBlackList:
            filterArgs.append(
                ImageOnlyShader.shader_id.not_in(self.blacklistJSON["blacklistedShaderIDs"])
            )

        self.candidateExprs = ImageOnlyExperiment.select().where(*filterArgs).join(
                Environment, peewee.JOIN.LEFT_OUTER
            ).switch(ImageOnlyExperiment) \
            .join(ImageOnlyShader, peewee.JOIN.LEFT_OUTER) \
            .order_by(ImageOnlyExperiment.id)

        self.environmentId = environmentId

        # print(self.candidateExprs)

    def __len__(self):
        return self.candidateExprs.count()

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        expr = self.candidateExprs[idx]

        results = json.loads(expr.results)
        result_mean = sum(results) / len(results)

        return {
            "environmentId": self.environmentId,
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


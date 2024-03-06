from toyDb.ExperimentDB import Environment, ImageOnlyShader, ImageOnlyResource, ImageOnlyExperiment
import torch.utils.data
import peewee
import typing
import logging
import json
import os
import tqdm

logger = logging.getLogger(__name__)
curDir = os.path.dirname(os.path.abspath(__file__))

class FragmentPerformanceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer: typing.Union[typing.Callable[[str], typing.List[int]], None],
        nodeName='libreliu-GCL-Arch',
        filteredNumCycles=50,
        filteredNumTrials=10,
        maxTokenizedLength=None,
        outputRawText=True,
        useBlackList=True
    ):
        self.tokenizer = tokenizer
        self.outputRawText = outputRawText

        self.blacklistJSON = None
        if useBlackList:
            with open(os.path.join(curDir, "shaderBlacklist.json"), "r") as fp:
                self.blacklistJSON = json.load(fp)

        if useBlackList:
            self.candidateExprs = ImageOnlyExperiment.select().where(
                ImageOnlyExperiment.num_cycles == filteredNumCycles,
                ImageOnlyExperiment.num_trials == filteredNumTrials,
                Environment.node == nodeName,
                ImageOnlyShader.shader_id.not_in(self.blacklistJSON["blacklistedShaderIDs"])
            ).join(
                Environment, peewee.JOIN.LEFT_OUTER
            ).switch(ImageOnlyExperiment).join(ImageOnlyShader, peewee.JOIN.LEFT_OUTER) \
            .order_by(ImageOnlyExperiment.id)
        else:
            self.candidateExprs = ImageOnlyExperiment.select().where(
                ImageOnlyExperiment.num_cycles == filteredNumCycles,
                ImageOnlyExperiment.num_trials == filteredNumTrials,
                Environment.node == nodeName
            ).join(
                Environment, peewee.JOIN.LEFT_OUTER
            ).switch(ImageOnlyExperiment).join(ImageOnlyShader, peewee.JOIN.LEFT_OUTER) \
            .order_by(ImageOnlyExperiment.id)

        # print(self.candidateExprs)

        self.filteredIndices = None
        if maxTokenizedLength is not None:
            self.filteredIndices = []
            graphicalIterator = tqdm.tqdm(enumerate(self.candidateExprs))
            graphicalIterator.set_description_str("Filter Processing")
            for exprIdx, expr in graphicalIterator:
                if len(self.tokenizer(expr.shader.fragment_spv)) <= maxTokenizedLength:
                    self.filteredIndices.append(exprIdx)
            
            logger.info(
                f"Raw: {self.candidateExprs.count()}; "
                f"Filtered: {len(self.filteredIndices)} "
                f"({len(self.filteredIndices) / self.candidateExprs.count() * 100.0:.5}% of the original;"
                f"using maxLen={maxTokenizedLength})"
            )
        else:
            logger.info("Tokenized length truncation not applied")

    def __len__(self):
        if self.filteredIndices is not None:
            return len(self.filteredIndices)
        else:
            return self.candidateExprs.count()

    def __getitem__(self, idx):
        """Doesn't support slicing at present"""
        if self.filteredIndices is None:
            expr = self.candidateExprs[idx]
        else:
            expr = self.candidateExprs[self.filteredIndices[idx]]
        
        spvText = expr.shader.fragment_spv
        if self.tokenizer is None or self.outputRawText:
            return {"spvText": spvText, "fpsMean": expr.result_mean}
        else:
            tokens = self.tokenizer(spvText)
            return {"tokens": tokens, "fpsMean": expr.result_mean}

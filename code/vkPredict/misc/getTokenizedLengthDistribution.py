from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset

import json
import math
import logging
import numpy as np
import typing
import matplotlib.pyplot as plt
import tqdm

logger = logging.getLogger(__name__)

def getLength(tokenizer, dsetItem):
    return len(tokenizer(dsetItem["spvText"]))

def getTokenizedLengthDistribution(tokenizerUsed=None, parallel=True, parallelJobs=8):
    if tokenizerUsed is None:
        raise RuntimeError("Requires specifying a tokenizer")
    elif tokenizerUsed == 'bpe':
        from misc.HfBpeTokenizer import HfBpeTokenizer

        tokenizerHF = HfBpeTokenizer("SpvBpeTokenizer.json")
        def tokenizeBpeWrapped(s) -> typing.List[int]:
            tokenized = tokenizerHF(s)["input_ids"]
        
            return tokenized

        tokenizer = tokenizeBpeWrapped
    elif tokenizerUsed == 'spvMultiple':
        from misc.HfSpvTokenizer import HfSpvTokenizer

        tokenizerHF = HfSpvTokenizer(single_entrypoint=False)
        def tokenizeSpvMultipleWrapped(s) -> typing.List[int]:
            tokenized = tokenizerHF(s)["input_ids"]
        
            return tokenized

        tokenizer = tokenizeSpvMultipleWrapped
        
        # Not feasible; since we can't pickle out Cpp extension
        parallel = False
    elif tokenizerUsed == 'spvMultipleRaw':
        import vkExecute, vkExecute.spv
        tokenizerRaw = vkExecute.spv.Tokenizer(True, False, True)
        spvProc = vkExecute.SpvProcessor()

        def tokenizeSpvMultipleRaw(s) -> typing.List[int]:
            success, errMsg = spvProc.assemble(s)
            assert(success)

            tokenizerRaw.loadSpv(spvProc.exportSpv())
            tokenizedSpv, errMsgs = tokenizerRaw.tokenize()
            assert(len(tokenizedSpv) > 0)

            return tokenizedSpv
        
        tokenizer = tokenizeSpvMultipleRaw
        
        # Not feasible; since we can't pickle out Cpp extension
        parallel = False
    else:
        assert(False)
    
    dset = FragmentPerformanceDataset(None, maxTokenizedLength=None)

    if parallel:
        from joblib import Parallel, delayed

        datas = [dset[idx] for idx in range(len(dset))]
        tokLengths = Parallel(n_jobs=parallelJobs, verbose=10)(delayed(getLength)(tokenizer, datas[i]) for i in range(len(datas)))
    else:
        tokLengths = np.ndarray((len(dset),), dtype=np.int32)

        for idx in tqdm.tqdm(range(0, len(dset))):
            data = dset[idx]
            tokLengths[idx] = getLength(tokenizer, data)

    # print(tokLengths)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(f"Token length using {tokenizerUsed}, Total & 95 pencentile")
    
    max_95 = np.percentile(tokLengths, 95)
    ax1.hist(tokLengths, bins='auto', range=(0, max_95))
    # ax1.set_xlabel('# SPIR-V instructions')
    ax1.set_ylabel('# Shaders')
    
    ax2.hist(tokLengths, bins='auto')
    ax2.set_xlabel("# Tokens")
    ax2.set_ylabel('# Shaders')

    plt.show()
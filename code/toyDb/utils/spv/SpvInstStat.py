from .SpvContext import BinaryContext, BasicBlock, getDefaultGrammar
from typing import Dict, List, Union
import dataclasses
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class StatResult:
    numInsts: int
    numInstsByClass: Dict[str, int]

    def mergeResult(self, other: 'StatResult'):
        self.numInsts += other.numInsts
        for k in self.numInstsByClass.keys():
            self.numInstsByClass[k] += other.numInstsByClass[k]
        
        diff = set(other.numInstsByClass.keys()) - set(self.numInstsByClass.keys())
        assert(len(diff) == 0)

def statBasicBlock(bb: BasicBlock):
    grammar = bb.parent.parent.grammar

    numInsts = 0
    numInstsByClass = {k: 0 for k in grammar.instrClasses}

    for inst in bb.insts:
        instClass = grammar.instFormatsByOpcode[inst.opcode].instrClass
        numInstsByClass[instClass] += 1
        
        numInsts += 1

    result = StatResult(numInsts, numInstsByClass)
    # print(result)

    return result

def statWithTrace(
    binCtx: BinaryContext,
    bbIdx2TraceIdx: Dict[int, int],
    traceData: List[int],
    blockIdNotExistCallback=None
) -> StatResult:
    
    bbList: List['BasicBlock'] = []
    for func in binCtx.functions:
        bbList += func.basicBlocks

    grammar = getDefaultGrammar()
    
    resTotal = StatResult(0, {k: 0 for k in grammar.instrClasses})
    for bb in bbList:
        if bb.blockId not in bbIdx2TraceIdx:
            # This is probably not reachable from entrypoint, therefore not labeled.
            logger.warning(f"Basic block #{bb.blockId} not in traced block list. ")
            traceCnt = 0

            if blockIdNotExistCallback is not None:
                blockIdNotExistCallback(bb.blockId)

        else:
            traceId = bbIdx2TraceIdx[bb.blockId]
            traceCnt = traceData[traceId]
    
        res = statBasicBlock(bb)
        res.numInsts *= traceCnt
        for clsName in res.numInstsByClass.keys():
            res.numInstsByClass[clsName] *= traceCnt
        
        resTotal.mergeResult(res)
    return resTotal

def statWithoutTrace(
    binCtx: BinaryContext
) -> StatResult:
    
    bbList: List['BasicBlock'] = []
    for func in binCtx.functions:
        bbList += func.basicBlocks

    grammar = getDefaultGrammar()
    
    resTotal = StatResult(0, {k: 0 for k in grammar.instrClasses})
    for bb in bbList:
    
        res = statBasicBlock(bb)
        resTotal.mergeResult(res)
    return resTotal

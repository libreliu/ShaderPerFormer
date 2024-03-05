import logging
import sys
sys.path.insert(0, sys.path[0]+"/../")
import vkExecute
from databases import ExperimentDb
import dataclasses
import random
import json
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from typing import List, Any, Dict

from utils.spv import SpvContext
from databases.ExperimentDb import (
    ImageOnlyExperiment,
    AugmentationType,
    packSpvToBytes,
    packBytesToSpv,
    packSpvToBytesStream
)
from databases.AugmentationDb import Augmentation

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class AugManagerConfig:
    env: int = None
    width: int = ExperimentDb.CANONICAL_WIDTH
    height: int = ExperimentDb.CANONICAL_HEIGHT
    uniformData: bytes = None
    numCycles: int = ExperimentDb.CANONICAL_NUM_CYCLES
    numTrials: int = ExperimentDb.CANONICAL_NUM_TRIALS
    # augTypes: list[AugmentationType] = None
    # augParams: dict = None


class AugManager:
    GLOBAL_EXCLUDE_PASSES = [AugmentationType.NONE,
                             AugmentationType.O,
                             AugmentationType.OS,
                             AugmentationType.INST_BINDLESS_CHECK,
                             AugmentationType.INST_DESC_IDX_CHECK,
                             AugmentationType.INST_BUFF_OOB_CHECK,
                             AugmentationType.INST_BUFF_ADDR_CHECK,
                             AugmentationType.CONVERT_TO_SAMPLED_IMAGE,
                            ]
    
    PASSES_NEED_PARAMS = [AugmentationType.SET_SPEC_CONST_DEFAULT_VALUE,
                          AugmentationType.SCALAR_REPLACEMENT,
                          AugmentationType.REDUCE_LOAD_SIZE,
                          AugmentationType.LOOP_FISSION,
                          AugmentationType.LOOP_FUSION,
                          AugmentationType.LOOP_UNROLL_PARTIAL,
                          AugmentationType.LOOP_PEELING_THRESHOLD,
                          AugmentationType.CONVERT_TO_SAMPLED_IMAGE
                          ]

    @dataclasses.dataclass
    class AugShaderNode:
        augType: AugmentationType = AugmentationType.NONE
        augParams: str = None           # reserved for future use
        fragmentSpv: List[str] = None   # fragment shader spv
        disFromLast: int = 0            # edit distance from last shader
        disFromOrigin: int = 0          # edit distance from origin shader
        disFromLastRatio: float = 0     # edit distance ratio from last shader


    def __init__(self, config: AugManagerConfig):
        self.augAncestors = []  # list of ancestor shader nodes
        self.augNodeList: List[self.AugShaderNode] = []   # list of shader nodes to be augmented
        self.oriFrgSpv: List[str] = None
        self.config = config


    def initAugList(self, types: List[AugmentationType], params: dict = None):
        for i in range(len(types)):
            shader = self.AugShaderNode()
            shader.augType = types[i]
            shader.augParams = params[i] if params else None
            self.augNodeList.append(shader)


    def initAugListFromFlags(self, flags: List[str], params: dict = None):
        for i in range(len(flags)):
            shader = self.AugShaderNode()
            shader.augType = self.augFlagToType(flags[i])
            shader.augParams = params[i] if params else None
            self.augNodeList.append(shader)


    def initAugListWithRandomPass(self, numPass: int, excludedPasses: List[AugmentationType] = []):
        excludedPasses += AugManager.GLOBAL_EXCLUDE_PASSES
        augTypes = []
        augParams = {}
        for i in range(numPass):
            augType = random.choice([i for i in AugmentationType if i not in excludedPasses])
            
            if augType == AugmentationType.SET_SPEC_CONST_DEFAULT_VALUE:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.SCALAR_REPLACEMENT:
                pass
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.REDUCE_LOAD_SIZE:
                pass
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.LOOP_FISSION:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.LOOP_FUSION:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.LOOP_UNROLL_PARTIAL:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.LOOP_PEELING_THRESHOLD:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            elif augType == AugmentationType.CONVERT_TO_SAMPLED_IMAGE:
                augType = AugmentationType.NONE
                # augParams[i] = random.randint(0, 100)
            if augType == AugmentationType.NONE:
                logger.warning("Some passes with parameters are not supported yet!")
                continue

            augTypes.append(augType)
        self.initAugList(augTypes)


    def initShaderAncestor(self, id):
        expr = ImageOnlyExperiment.get(id)
        self.config.height, self.config.width = expr.height, expr.width
        self.config.numCycles = expr.num_cycles
        self.config.numTrials = expr.num_trials
        if expr.environment.id != self.config.env:
            raise Exception("Shader environment mismatch!")
        if ExperimentDb.ErrorType(expr.errors) is not ExperimentDb.ErrorType.NONE:
            raise Exception(f"Experiment {expr} has errors!")
        
        while expr.shader is None:
            shader = self.AugShaderNode()
            shader.augType = expr.augmentation
            shader.augParams = expr.augmentation_annotation
            self.augAncestors.append(shader)
            expr = expr.parent
            if expr is None:
                raise Exception("No parent experiment with shader found!")
        shader = self.AugShaderNode()
        shader.augType = expr.augmentation
        shader.augParams = expr.augmentation_annotation
        shader.fragmentSpv = packBytesToSpv(expr.shader.fragment_spv)
        self.augAncestors.append(shader)
        self.augAncestors.reverse()


    def initShaderFromSpv(self, fragSpv):
        shader = self.AugShaderNode()
        shader.fragmentSpv = packBytesToSpv(fragSpv)
        self.augAncestors.append(shader)


    def augmentShader(self, shaderNode: AugShaderNode):
        if(shaderNode.fragmentSpv is None):
            raise Exception("FragmentSpv is empty while trying to augment!")
        if(shaderNode.augType == AugmentationType.NONE):
            return
        
        spvProc = vkExecute.SpvProcessor()
        spvProc.loadSpv(shaderNode.fragmentSpv)

        flag = self.augTypeToFlag(shaderNode.augType)
        if flag == "o":
            flag = ["-O"]
        elif flag == "os":
            flag = ["-Os"]
        else:
            flag = ["--" + flag]

        if shaderNode.augParams is not None and shaderNode.augParams != "":
            flag[0] += "=" + shaderNode.augParams

        success, errMsgs = spvProc.runPassSequence(flag)
        if not success:
            logger.error("Augmented spv generation failed: " + errMsgs)
            return
        success, errMsgs = spvProc.validate()
        if not success:
            logger.error("Augmented spv validation failed: " + errMsgs)
            return
        shaderNode.fragmentSpv = spvProc.exportSpv()
        

    def augmentAncestor(self):
        if self.augAncestors[0].fragmentSpv is None:
            raise Exception("Original ancestor fragment spv not initialized!")
        
        for i in range(len(self.augAncestors)):
            self.augmentShader(self.augAncestors[i])
            if i < len(self.augAncestors) - 1:
                self.augAncestors[i+1].fragmentSpv = self.augAncestors[i].fragmentSpv

        self.oriFrgSpv = self.augAncestors[-1].fragmentSpv
        if self.augNodeList is not None:
            self.augNodeList[0].fragmentSpv = self.augAncestors[-1].fragmentSpv
    

    def augment(self):
        if self.augNodeList is None:
            raise Exception("Augmentation list not initialized!")
        
        self.augmentAncestor()
        for i in range(len(self.augNodeList)):
            self.augmentShader(self.augNodeList[i])
            if i < len(self.augNodeList) - 1:
                self.augNodeList[i+1].fragmentSpv = self.augNodeList[i].fragmentSpv
    
    
    def evalDistance(self):
        if self.oriFrgSpv is None:
            raise Exception("Original fragment spv not initialized!")
        for i in range(len(self.augNodeList)):
            oriInstStream = AugManager.SpvtoInstStream(self.oriFrgSpv)
            resultInstStream = AugManager.SpvtoInstStream(self.augNodeList[i].fragmentSpv)
            self.augNodeList[i].disFromOrigin = AugManager.editDistance(oriInstStream, resultInstStream)
            if i > 0:
                lastInstStream = AugManager.SpvtoInstStream(self.augNodeList[i-1].fragmentSpv)
                self.augNodeList[i].disFromLast = AugManager.editDistance(lastInstStream, resultInstStream)
                self.augNodeList[i].disFromLastRatio = self.augNodeList[i].disFromLast / len(lastInstStream)
            else:
                self.augNodeList[i].disFromLast = self.augNodeList[i].disFromOrigin
                self.augNodeList[i].disFromLastRatio = self.augNodeList[i].disFromLast / len(oriInstStream)

    
    def evalSimpleDistance(self):
        if self.oriFrgSpv is None:
            raise Exception("Original fragment spv not initialized!")
        if self.oriFrgSpv == self.augNodeList[-1].fragmentSpv:
            self.augNodeList[-1].disFromOrigin = 0
        else:
            self.augNodeList[-1].disFromOrigin = 1

    
    def evalAncestorDistance(self):
        if self.oriFrgSpv is None:
            raise Exception("Original fragment spv not initialized!")
        oriInstStream = AugManager.SpvtoInstStream(self.oriFrgSpv)
        for i in range(len(self.augAncestors)):
            resultInstStream = AugManager.SpvtoInstStream(self.augAncestors[i].fragmentSpv)
            self.augAncestors[i].disFromOrigin = AugManager.editDistance(oriInstStream, resultInstStream)


    def exportRunAugSpv(self):
        spvList = []
        for augNode in self.augNodeList:
            if augNode.disFromLast > 0:
                spvList.append(augNode.fragmentSpv)
        return spvList
    

    @staticmethod
    def SpvtoInstStream(fragmentSpv: List[str]) -> List[int]:
        parser = SpvContext.BinaryParser()
        binCtx = parser.parse(
        SpvContext.getDefaultGrammar(),
        packSpvToBytesStream(fragmentSpv)
        )

        result = []
        for func in binCtx.functions:
            result.append(func.func.opcode)
            for bb in func.basicBlocks:
                result.append(248)
                for inst in bb.insts:
                    result.append(inst.opcode)
        
        return result
    

    @staticmethod
    def editDistance(arr1, arr2):
        distance = vkExecute.editDistance(arr1, arr2)
        # distanceSlow = AugManager.editDistanceSlow(arr1, arr2)
        # print(distance - distanceSlow)
        return AugManager.editDistanceSlow(arr1, arr2)
    

    @staticmethod
    def editDistanceWithRatio(arr1, arr2):
        distance = vkExecute.editDistance(arr1, arr2)
        disRatio = distance / len(arr2)
        return distance, disRatio
    

    @staticmethod
    def editDistanceSlow(arr1, arr2):
        m, n = len(arr1), len(arr2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            for j in range(n + 1):
                if i == 0:
                    dp[i][j] = j
                elif j == 0:
                    dp[i][j] = i
                elif arr1[i - 1] == arr2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
        return dp[m][n]


    @staticmethod
    def augFlagToType(flag: str):
        if len(flag) > 0:
            flag = flag.replace('-', '_').upper()
            type = getattr(AugmentationType, flag, AugmentationType.NONE)
            if type == AugmentationType.NONE:
                logger.warning(f"Augmentation type {flag} not found!")
            return type
        else:
            logger.warning("Empty augmentation flag!")
            return AugmentationType.NONE


    @staticmethod
    def augTypeToFlag(type: AugmentationType):
        return AugmentationType(type).name.replace('_', '-').lower()


class AugAnalyzer:
    @dataclasses.dataclass
    class AugNodeWithAncestor:
        shaderId: str = None
        fragmentSpv: List[str] = None
        ancestorFragmentSpv: List[str] = None
        distance: int = 0
        renderTime: List[float] = None
        ancestorRenderTime: List[float] = None
        disRatio: float = 0
    

    def __init__(self):
        self.augNodeList: List[AugAnalyzer.AugNodeWithAncestor] = []


    def addNodeWithParent(self, expr: ImageOnlyExperiment):
        if expr.shader is None:
            raise Exception("Shader not found in experiment!")
        if expr.parent is None:
            raise Exception("Parent experiment not found!")
        
        node = AugAnalyzer.AugNodeWithAncestor()
        node.shaderId = expr.shader_shadertoy_id
        node.fragmentSpv = packBytesToSpv(expr.shader.fragment_spv)
        node.ancestorFragmentSpv = packBytesToSpv(expr.parent.shader.fragment_spv)
        node.renderTime = json.loads(expr.results)
        node.ancestorRenderTime = json.loads(expr.parent.results)
        self.augNodeList.append(node)


    def addNodeWithAncestor(self, expr: ImageOnlyExperiment):
        if expr.shader is None:
            raise Exception("Shader not found in experiment!")
        if expr.parent is None:
            raise Exception("Parent experiment not found!")
        parent = expr.parent
        while parent.parent is not None:
            parent = parent.parent
        
        node = AugAnalyzer.AugNodeWithAncestor()
        node.shaderId = expr.shader_shadertoy_id
        node.fragmentSpv = packBytesToSpv(expr.shader.fragment_spv)
        node.ancestorFragmentSpv = packBytesToSpv(parent.shader.fragment_spv)
        node.renderTime = json.loads(expr.results)
        node.ancestorRenderTime = json.loads(parent.results)
        self.augNodeList.append(node)


    @staticmethod
    def calulateDistanceSingleWorkerFn(fragSpv, ancestorFragSpv):
        return AugManager.editDistanceWithRatio(
            AugManager.SpvtoInstStream(fragSpv),
            AugManager.SpvtoInstStream(ancestorFragSpv)
        )


    def calulateDistance(self):
        out = Parallel(n_jobs=-1)\
            (delayed(AugAnalyzer.calulateDistanceSingleWorkerFn)(node.fragmentSpv, node.ancestorFragmentSpv)\
              for node in tqdm(self.augNodeList))
        print("Distance calculated!")
        for node, [distance, disRatio] in zip(self.augNodeList, out):
            node.distance = distance
            node.disRatio = disRatio

    
    def exportTimeResult(self):
        return np.array([node.renderTime for node in self.augNodeList], dtype=np.float64)
    

    def exportAncestorTimeResult(self):
        return np.array([node.ancestorRenderTime for node in self.augNodeList], dtype=np.float64)


    def exportDistance(self):
        return np.array([node.distance for node in self.augNodeList], dtype=np.float64)


    def exportDisRatio(self):
        return np.array([node.disRatio for node in self.augNodeList], dtype=np.float64)


    def exportFragmentSpv(self, index):
        spvProc = vkExecute.SpvProcessor()
        spvProc.loadSpv(self.augNodeList[index].fragmentSpv)
        fragmentSpv = spvProc.disassemble()
        spvProc.loadSpv(self.augNodeList[index].ancestorFragmentSpv)
        ancestorFragmentSpv = spvProc.disassemble()
        return fragmentSpv, ancestorFragmentSpv


def augment(fragment_spv, type, params):
    augManager = AugManager(AugManagerConfig())
    augManager.initShaderFromSpv(fragment_spv)
    if params is not None:
        augManager.initAugList(type, params)
    else:
        augManager.initAugList(type)
    augManager.augment()
    # augManager.evalSimpleDistance()
    augManager.evalDistance()
    return augManager


def augmentAllNoPall(type, params, query):
    for aug in tqdm(query):

        augManager = augment(aug.fragment_spv, type, params)

        if augManager.augNodeList[-1].disFromOrigin > 0:
            augmentation = Augmentation.create(
            augmentation = augManager.augNodeList[-1].augType,
            augmentation_annotation = augManager.augNodeList[-1].augParams,
            fragment_spv = packSpvToBytes(augManager.augNodeList[-1].fragmentSpv),
            shader_id = aug.shader_id,
            depth = aug.depth + 1,
            parent = aug,
            dis_from_last = augManager.augNodeList[-1].disFromLast,
            dis_ratio_from_last = augManager.augNodeList[-1].disFromLastRatio,
            dis_from_origin = 0,
            dis_ratio_from_origin = 1
            )
            augmentation.save()


def augmentAll(type, params, query):
    batch_size = 400
    total_augs = query.count()
    num_batches = (total_augs // batch_size) + 1

    for batch_num in range(num_batches):
        offset = batch_num * batch_size
        batch_query = query.offset(offset).limit(batch_size)
        fragment_spvs = []
        augManagerList = []

        for aug in batch_query:
            fragment_spvs.append(aug.fragment_spv)
        
        augManagerList = Parallel(n_jobs=-1)(delayed(augment)(fragment_spv, type, params) for fragment_spv in tqdm(fragment_spvs))
        fragment_spvs.clear()

        for augManager, i in zip(augManagerList, range(len(augManagerList))):
            if augManager.augNodeList[-1].disFromOrigin > 0:
                augmentation = Augmentation.create(
                augmentation = augManager.augNodeList[-1].augType,
                augmentation_annotation = augManager.augNodeList[-1].augParams,
                fragment_spv = packSpvToBytes(augManager.augNodeList[-1].fragmentSpv),
                shader_id = batch_query[i].shader_id,
                depth = batch_query[i].depth + 1,
                parent = batch_query[i],
                dis_from_last = augManager.augNodeList[-1].disFromLast,
                dis_ratio_from_last = augManager.augNodeList[-1].disFromLastRatio,
                dis_from_origin = 0,
                dis_ratio_from_origin = 1
                )
                augmentation.save()
        augManagerList.clear()

    
if __name__ == '__main__':
    ExperimentDb.init_from_default_db()
    augConfig = AugManagerConfig(env=1)
    augManager = AugManager(augConfig)
    augManager.initShaderAncestor(2)
    augManager.initAugListFromFlags(["loop-peeling-threshold"], {0: "1000"})
    augManager.augment()

    oriInstStream = AugManager.SpvtoInstStream(augManager.oriFrgSpv)
    resultInstStream = AugManager.SpvtoInstStream(augManager.augNodeList[-1].fragmentSpv)
    print(AugManager.editDistance(oriInstStream, resultInstStream))
    print(1)
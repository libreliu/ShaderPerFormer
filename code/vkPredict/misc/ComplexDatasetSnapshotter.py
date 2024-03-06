from typing import List, Dict, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from toyDb.databases import ExperimentDb
from toyDb.databases.ExperimentDb import (
    ImageOnlyExperiment,
    ImageOnlyResource,
    ImageOnlyShader,
    ImageOnlyTrace,
    Environment,
    packBytesToSpv
)
from dataset.FragmentPerformanceWithTraceByQueryDataset import FragmentPerformanceWithTraceByQueryDataset
from dataset.FragmentPerformanceAugWithTrace import FragmentPerformanceAugWithTraceDataset
from toyDb.experiments.ImageOnlyRunner import ImageOnlyRunner

import peewee as pw
import itertools
import joblib
import vkExecute
import json
import tqdm.auto
import functools
import pickle
import random
import math
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class GroupDescription:
    elemExprIds: set
    comment: str

# === Filters ===

class AbstractFilter:
    def getName(self):
        return getattr(self, 'filterName', None)

    def setName(self, name):
        """Set a customized name"""
        self.filterName = name

    def __call__(self, *args, **kwargs) -> List[GroupDescription]:
        raise NotImplementedError("Implement me")

class EnvironmentFilter(AbstractFilter):
    def __init__(self, useAbbreviatedComment=True):
        self.useAbbreviatedComment = useAbbreviatedComment

    def getName(self):
        return 'EnvironmentFilter'

    def _describeEnvironment(self, env, abbreviated):
        if abbreviated:
            return f"EnvId{env.id}"
        else:
            return f"{env.node}-{env.os}-{env.cpu}-{env.gpu}-{env.gpu_driver}-{env.comment}"

    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        envGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.environment,
            pw.fn.COUNT('*').alias('count')
        ).group_by(
            ImageOnlyExperiment.environment
        ).order_by(
            ImageOnlyExperiment.id
        )

        associatedEnvId = []
        for env in envGroups:
            resultSet.append(
                GroupDescription(set(), self._describeEnvironment(env.environment, self.useAbbreviatedComment))
            )
            associatedEnvId.append(env.environment_id)

        # 2. do query and filter out the needed
        for idx, envId in enumerate(associatedEnvId):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.environment_id == envId
            ).order_by(
                ImageOnlyExperiment.id
            ))

        # 3. san check
        for idx, env in enumerate(envGroups):
            assert(env.count == len(resultSet[idx].elemExprIds))

        return resultSet

class CycleTrialsFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        ctGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.num_cycles,
            ImageOnlyExperiment.num_trials
        ).group_by(
            ImageOnlyExperiment.num_cycles,
            ImageOnlyExperiment.num_trials
        ).order_by(
            ImageOnlyExperiment.id
        )

        for ctExpr in ctGroups:
            resultSet.append(
                GroupDescription(set(), f"{ctExpr.num_cycles}cycles-{ctExpr.num_trials}trials")
            )

        # 2. do query and filter out the needed
        for idx, ctExpr in enumerate(ctGroups):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.num_cycles == ctExpr.num_cycles,
                ImageOnlyExperiment.num_trials == ctExpr.num_trials
            ).order_by(
                ImageOnlyExperiment.id
            ))

        return resultSet

class TraceAvailabilityFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        resultSets = [
            GroupDescription(None, "noTrace"),
            GroupDescription(None, "haveTrace")
        ]

        resultSets[0].elemExprIds = set(expr.id for expr in ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.trace.is_null(True)
        ))
        resultSets[1].elemExprIds = set(expr.id for expr in ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.trace.is_null(False)
        ))

        return resultSets

class ResourceFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        ctGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.resource
        ).group_by(
            ImageOnlyExperiment.resource
        ).order_by(
            ImageOnlyExperiment.id
        )

        for ctExpr in ctGroups:
            resultSet.append(
                GroupDescription(set(), f"resource{ctExpr.resource_id}")
            )

        # 2. do query and filter out the needed
        for idx, ctExpr in enumerate(ctGroups):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.resource == ctExpr.resource
            ).order_by(
                ImageOnlyExperiment.id
            ))

        return resultSet

class ImageHashFilter(AbstractFilter):
    """Handles pure white, pure black for all resolutions"""
    def computePureHash(self, width, height):
        # (0,0,0,0) => fully transparent under RGBA mode
        # however Shadertoy will just discard the A component
        transparent_black = ImageOnlyRunner.getImageHash(
            np.zeros((height, width, 4), dtype=np.uint8)
        )
        white = ImageOnlyRunner.getImageHash(
            255 * np.ones((height, width, 4), dtype=np.uint8)
        )

        return white, transparent_black

    def __call__(self) -> List[GroupDescription]:
        resultSets = [
            GroupDescription(None, "normalHash"),
            GroupDescription(None, "victimHash")
        ]
        victimHash = []

        whGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.width,
            ImageOnlyExperiment.height
        ).group_by(
            ImageOnlyExperiment.width,
            ImageOnlyExperiment.height
        ).order_by(
            ImageOnlyExperiment.id
        )

        for whExpr in whGroups:
            white, black = self.computePureHash(whExpr.width, whExpr.height)
            logger.info(f"White hash for ({whExpr.width}, {whExpr.height}): {white}")
            logger.info(f"Transparent black hash for ({whExpr.width}, {whExpr.height}): {black}")

            victimHash.append(white)
            victimHash.append(black)

        resultSets[0].elemExprIds = set(expr.id for expr in ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.image_hash.not_in(victimHash)
        ))
        resultSets[1].elemExprIds = set(expr.id for expr in ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.image_hash.in_(victimHash)
        ))

        return resultSets


class WidthHeightFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        ctGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.width,
            ImageOnlyExperiment.height
        ).group_by(
            ImageOnlyExperiment.width,
            ImageOnlyExperiment.height
        ).order_by(
            ImageOnlyExperiment.id
        )

        for ctExpr in ctGroups:
            resultSet.append(
                GroupDescription(set(), f"{ctExpr.width}-{ctExpr.height}")
            )

        # 2. do query and filter out the needed
        for idx, ctExpr in enumerate(ctGroups):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.width == ctExpr.width,
                ImageOnlyExperiment.height == ctExpr.height
            ).order_by(
                ImageOnlyExperiment.id
            ))

        return resultSet

class ErrorFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        errGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.errors
        ).group_by(
            ImageOnlyExperiment.errors
        ).order_by(
            ImageOnlyExperiment.id
        )

        for errExpr in errGroups:
            resultSet.append(
                GroupDescription(set(), f"error{errExpr.errors}")
            )

        # 2. do query and filter out the needed
        for idx, errExpr in enumerate(errGroups):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.errors == errExpr.errors,
            ).order_by(
                ImageOnlyExperiment.id
            ))

        return resultSet

class ShadertoyIdFilter(AbstractFilter):
    def __init__(self):
        self.groups = []

    def registerGroup(self, groupName: str, groupShdrtoyIds: List[str]):
        selectedIds = set()
        groupShdrtoyIdSets = set(groupShdrtoyIds)
        if len(groupShdrtoyIdSets) != len(groupShdrtoyIds):
            raise Exception("Duplicate items detected in the input")

        for shdrId in groupShdrtoyIds:
            exprs = ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.shader_shadertoy_id == shdrId
            )
            for expr in exprs:
                assert(expr.id not in selectedIds)
                selectedIds.add(expr.id)

        self.groups.append({
            "comment": groupName,
            "set": selectedIds
        })

    def validate(self):
        """Validate that we have disjoint sets"""
        for i in range(0, len(self.groups)):
            for j in range(i, len(self.groups)):
                common = self.groups[i]["set"] & self.groups[j]["set"]
                if len(common) > 0:
                    raise RuntimeError(f"Set {self.groups[i]['comment']} and set {self.groups[j]['comment']} have {len(common)} common elements")

    def __call__(self) -> List[GroupDescription]:
        # or you might have missed to register
        assert(len(self.groups) > 0)

        descs = []
        for grp in self.groups:
            descs.append(GroupDescription(
                elemExprIds=grp['set'],
                comment=grp['comment']
            ))

        return descs

class SpvTokenizedLengthFilter(AbstractFilter):
    def __init__(self):
        self.processed = False

        # Maps ImageOnlyShader.id => tokenized length
        self.idToLength = {}
        self.threshold = None

    def setThreshold(self, threshold: int):
        self.threshold = threshold

    @staticmethod
    def _processFn(id: int, shader_id: str, fragment_spv: bytes):
        tokenizer = vkExecute.spv.Tokenizer(
            False, # single_entrypoint
            False, # compact_types
            False, # convert_ext_insts
            False  # relative inst id pos
        )

        tokenizer.loadSpv(packBytesToSpv(fragment_spv))
        error = False
        try:
            tokenizedSpv, errMsgs = tokenizer.tokenize()
        except RuntimeError as e:
            if "Id exceed max available length" in e.__str__():
                tokenizedSpv = None
            else:
                raise
        
        return id, shader_id, len(tokenizedSpv) + 1 if tokenizedSpv is not None else None

    def process(self, parallel=True):
        # serialize it to make things be able to be passed around by loky.
        # Prior to Python 3.11, sqlite3 connections are not able to be shared among threads
        shdrs = [i for i in ImageOnlyShader.select().dicts().order_by(ImageOnlyShader.id)]
        totalLen = len(shdrs)

        def jobProducer():
            for shdr in shdrs:
                yield (shdr['id'], shdr['shader_id'], shdr['fragment_spv'])

        if parallel:
            for id, shdrId, tokenizedLen in tqdm.auto.tqdm(
                joblib.Parallel(
                    n_jobs=-1, backend='loky', return_as="generator"
                )(joblib.delayed(SpvTokenizedLengthFilter._processFn)(*args) for args in jobProducer()),
                total=totalLen
            ):
                # None if error occurs
                self.idToLength[id] = tokenizedLen
        else:
            for id, shdrId, tokenizedLen in tqdm.tqdm((SpvTokenizedLengthFilter._processFn(*args) for args in jobProducer()), total=totalLen):
                # None if error occurs
                self.idToLength[id] = tokenizedLen

        self.processed = True

    def writeToCache(self, cacheFile):
        assert(self.processed)

        with open(cacheFile, "w") as fp:
            json.dump({
                # a basic fingerprint
                "numExprs": ImageOnlyExperiment.select().count(),
                "idToLength": self.idToLength
            }, fp)

    def readFromCache(self, cacheFile):
        assert(not self.processed)
        with open(cacheFile, "r") as fp:
            cache = json.load(fp)
            curCount = ImageOnlyExperiment.select().count()
            if curCount != cache["numExprs"]:
                raise RuntimeError(f"Cached expr count {cache['numExprs']}, got expr count {curCount}, cache possibly mismatch")

            self.idToLength = cache["idToLength"]
            self.processed = True

    def __call__(self) -> List[GroupDescription]:
        assert(self.threshold is not None)
        assert(self.processed)

        returnSets = [
            GroupDescription(set(), f"belowOrEqualThreshold{self.threshold}"),
            GroupDescription(set(), f"aboveThreshold{self.threshold}"),
            GroupDescription(set(), "failedTokenize")
        ]

        for id, length in tqdm.auto.tqdm(self.idToLength.items(), total=len(self.idToLength)):
            exprs = ImageOnlyExperiment.select(
                ImageOnlyExperiment.id,
            ).where(
                ImageOnlyExperiment.shader_id == id
            ).order_by(
                ImageOnlyExperiment.id
            )

            setIdx = None
            assert(id in self.idToLength)
            if length is None:
                setIdx = 2
            elif length <= self.threshold:
                setIdx = 0
            else:
                setIdx = 1
            
            for expr in exprs:
                returnSets[setIdx].elemExprIds.add(expr.id)

        return returnSets


class TimeThresholdFilter(AbstractFilter):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def getName(self):
        return 'TimeThresholdFilter'

    def __call__(self) -> List[GroupDescription]:
        resultSets = [
            GroupDescription(set(), f"meanBelowOrEqualThreshold{self.threshold}"),
            GroupDescription(set(), f"meanAboveThreshold{self.threshold}"),
            GroupDescription(set(), f"NaN")
        ]

        exprs = ImageOnlyExperiment.select(
            ImageOnlyExperiment.id,
            ImageOnlyExperiment.results
        ).order_by(
            ImageOnlyExperiment.id
        )

        for expr in exprs:
            if expr.results == "":
                resultSets[2].elemExprIds.add(expr.id)
            else:
                results = json.loads(expr.results)
                assert(len(results) > 0)
                timeMean = sum(results) / len(results)
                if timeMean <= self.threshold:
                    resultSets[0].elemExprIds.add(expr.id)
                else:
                    resultSets[1].elemExprIds.add(expr.id)

        return resultSets

class AugmentationFilter(AbstractFilter):
    def getName(self):
        return 'AugmentationFilter'

    def __call__(self) -> List[GroupDescription]:
        resultSet = []

        # 1. decide number of groups
        augGroups = ImageOnlyExperiment.select(
            ImageOnlyExperiment.augmentation
        ).group_by(
            ImageOnlyExperiment.augmentation
        ).order_by(
            ImageOnlyExperiment.id
        )

        for augExpr in augGroups:
            resultSet.append(
                GroupDescription(set(), f"aug{augExpr.augmentation}")
            )

        # 2. do query and filter out the needed
        for idx, augExpr in enumerate(augGroups):
            resultSet[idx].elemExprIds = set(i.id for i in ImageOnlyExperiment.select(
                ImageOnlyExperiment.id
            ).where(
                ImageOnlyExperiment.augmentation == augExpr.augmentation,
            ).order_by(
                ImageOnlyExperiment.id
            ))

        return resultSet


# These are useful for debugging
class DebugFilter(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        return [
            GroupDescription({1,2,3}, "123"),
            GroupDescription({4,5,6}, "456")
        ]

class DebugFilter2(AbstractFilter):
    def __call__(self) -> List[GroupDescription]:
        return [
            GroupDescription({1,2}, "12"),
            GroupDescription({3,4}, "34"),
            GroupDescription({5,6}, "56")
        ]

# ===============

class TraceDuplicationPostFilter:
    def __call__(self, exprIds: List[int]) -> List[int]:
        # shader.id (int) -> set(current_encountered_trace_count_sums)
        shaderIdToTraceCountSums = {}
        acceptedExprIds = []
        rejectedExprIds = []

        for exprIdx in exprIds:
            expr = ImageOnlyExperiment.get_by_id(exprIdx)
            traceCountSum = sum(json.loads(expr.trace.bb_trace_counters))
            if expr.shader_id not in shaderIdToTraceCountSums:
                shaderIdToTraceCountSums[expr.shader_id] = set()
                shaderIdToTraceCountSums[expr.shader_id].add(traceCountSum)
                acceptedExprIds.append(exprIdx)
            elif traceCountSum not in shaderIdToTraceCountSums[expr.shader_id]:
                shaderIdToTraceCountSums[expr.shader_id].add(traceCountSum)
                acceptedExprIds.append(exprIdx)
            else:
                rejectedExprIds.append(exprIdx)
        
        print(f"TraceDuplicationPostFilter: Total {len(acceptedExprIds) + len(rejectedExprIds)}, "
              f"Accepted {len(acceptedExprIds)}, Rejected {len(rejectedExprIds)}")

        return acceptedExprIds
        

class ComplexDatasetSnapshotter:
    def __init__(self):
        self.filters: Dict[str, 'AbstractFilter'] = {}
        self.filterResults: Dict[str, List['GroupDescription']] = {}
        self.postProcInsts: List[Callable[[List[int]], List[int]]] = []

    def registerFilter(self, filterInst, *filterArgs, **filterKwargs):
        """NOTE: Will use class name if not named"""
        name = filterInst.__class__.__name__ if filterInst.getName() is None else filterInst.getName()
        assert(name not in self.filters)

        self.filters[name] = filterInst
        self.filterResults[name] = self.filters[name](*filterArgs, **filterKwargs)

    def examineGroups(self, maxSetCut=None):
        """Return statistics by order 1-set, 2-set, ..., maxSetCut-set (n max) and also do validation"""
        # List[
            # subgroup
        #   List[Dict]
        # ]
        availSubgroupSets = []
        for name, res in self.filterResults.items():
            availSubgroupSets.append([])
            for desc in res:
                availSubgroupSets[-1].append({
                    "set": desc.elemExprIds,
                    "name": name,
                    "comment": desc.comment
                })

                # other set in the same subgroup
                for otherSet in availSubgroupSets[-1][:-1]:
                    # Subgroup in a filter are expected to 
                    assert(len(desc.elemExprIds & otherSet["set"]) == 0)
                    # comments are not expected to be the same
                    assert(otherSet["comment"] != desc.comment)

        if maxSetCut is None:
            maxSetCut = len(availSubgroupSets)
        else:
            assert(maxSetCut <= len(availSubgroupSets))

        for setCnt in range(1, maxSetCut + 1):
            chosenSubgroupCombs = itertools.combinations(range(len(availSubgroupSets)), setCnt)
            for chosenSubgroupComb in chosenSubgroupCombs:
                # chosenSubgroupComb is the indices into availSubgroupSets array
                # chosenSubgroup: List[List[Dict]]
                # print(f"chosenSubgroupComb: {chosenSubgroupComb}")
                chosenSubgroups = [availSubgroupSets[i] for i in chosenSubgroupComb]
                
                # an iterator, returning Tuple[Dicts] on invocation
                prodSubSets = itertools.product(*chosenSubgroups)

                for chosenSubsets in prodSubSets:
                    # do join on those chosen subsets

                    desc = ""
                    begin = True
                    resultSet = None

                    for subset in chosenSubsets:
                        if begin == True:
                            desc = f"{subset['name']}_{subset['comment']}"
                            resultSet = subset['set']
                            begin = False
                        else:
                            desc += f" & {subset['name']}_{subset['comment']}"
                            resultSet = resultSet & subset['set']

                    desc += f" = {len(resultSet)}"
                    print(desc)

    def _findSubfilterDescByComment(self, name, comment) -> GroupDescription:
        for idx, desc in enumerate(self.filterResults[name]):
            if desc.comment == comment:
                return desc
        
        raise RuntimeError(f"Set with name {name} doesn't have subset with comment {comment}")

    def _snapshotExprById(self, exprId):
        exprs = ImageOnlyExperiment.select().where(
            ImageOnlyExperiment.id == exprId
        )
        assert(len(exprs) == 1)

    def registerPostProcessor(self, postProcInst: Callable[[List[int]], List[int]]):
        self.postProcInsts.append(postProcInst)


    def doSnapshotWithOptAug(
            self,
            snapshotDestfile: str,
            filters: List[List[Tuple[str]]],
            maxNumChildren: int = 100,
            applyPostProcessor: bool = False
        ):
        """tuple is of (name, comment) pair
        NOTE: it's caller's responsibility to give a measured and traced dataset

        Outer list describes & (AND)
        Inner list describes | (OR)
        So when combined, they form a Conjunctive normal form.
        """

        candSetsOuter = []

        for innerCands in filters:
            candSetsInner = []
            for filterName, filterComment in innerCands:
                candSetsInner.append(
                    self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                )

            candSetsOuter.append(functools.reduce(
                lambda x, y: x | y,
                candSetsInner,
                candSetsInner[0] if len(candSetsInner) > 0 else set()
            ))

        exprIds = functools.reduce(
            lambda x, y: x & y,
            candSetsOuter,
            candSetsOuter[0] if len(candSetsOuter) > 0 else set()
        )

        filterdIds = exprIds

        filterName = "AugmentationFilter"
        filterComment = "aug0"
        exprIds = list(exprIds & self._findSubfilterDescByComment(filterName, filterComment).elemExprIds)

        if applyPostProcessor:
            for postProc in self.postProcInsts:
                exprIds = postProc(exprIds)
        
        trainExprIds = random.sample(exprIds, math.ceil(len(exprIds) * 0.8))
        testExprIds = list(set(exprIds) - set(trainExprIds))

        trainSet = FragmentPerformanceAugWithTraceDataset(trainExprIds, filteredIds=filterdIds, maxNumChildren=maxNumChildren)
        trainSamples = [trainSet[i] for i in range(0, len(trainSet)) if len(trainSet[i]['children']) >= 2]
        # assert(len(trainSamples) == len(trainExprIds))
        print(f"Train samples: {len(trainSamples)}")

        testSet = FragmentPerformanceAugWithTraceDataset(testExprIds, filteredIds=filterdIds, maxNumChildren=maxNumChildren)
        testSamples = [testSet[i] for i in range(0, len(testSet)) if len(testSet[i]['children']) >= 2]
        # assert(len(testSamples) == len(testExprIds))
        print(f"Test samples: {len(testSamples)}")
            
        with open(snapshotDestfile, "wb") as fp:
            pickle.dump({
                "train": trainSamples,
                "test": testSamples
            }, fp)


    def evalFilters(
        self,
        groupFilters: List[List[Tuple[str]]]
    ):
        # copied from doSnapshot
        trainCandSetsOuter = []

        for trainInnerCands in groupFilters:
            trainCandSetsInner = []
            for filterName, filterComment in trainInnerCands:
                trainCandSetsInner.append(
                    self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                )

            trainCandSetsOuter.append(functools.reduce(
                lambda x, y: x | y,
                trainCandSetsInner,
                trainCandSetsInner[0]
            ))

        trainExprIds = list(functools.reduce(
            lambda x, y: x & y,
            trainCandSetsOuter,
            trainCandSetsOuter[0]
        ))

        return trainExprIds

    def interpretFilters(
            self,
            origExprIds: 'set[str]',
            groupFilters: List[List[Tuple[str]]]
        ):
        """
        the elems in inner list is to be joined
        the elems in the outer list is to be intersected
        """
        candSetsOuter = []
        exprIds = origExprIds

        print(f"  Base expr id: {len(exprIds)} items")
        for innerCands in groupFilters:
            candSetsInner = []
            for filterName, filterComment in innerCands:
                filterSet = self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                res = exprIds & filterSet
                candSetsInner.append(res)

                print(f"  Intersect with {filterName}_{filterComment}: {len(res)} items")

            # union for inner sets
            exprIds = functools.reduce(
                lambda x, y: x | y,
                candSetsInner,
                candSetsInner[0]
            )
            print(f"  After {innerCands[0][0]}: {len(exprIds)} items")

        print(f"  Returning expr id: {len(exprIds)} items")
        return exprIds

    def doSnapshot(
            self,
            snapshotDestfile: str,
            trainGroupFilters: List[List[Tuple[str]]],
            testGroupFilters: Optional[List[List[Tuple[str]]]] = None,
            valGroupFilters: Optional[List[List[Tuple[str]]]] = None,
            testGroupCopySrc: Optional[Any] = None,
            applyPostProcessorToTrain: bool = False
        ):
        """tuple is of (name, comment) pair
        NOTE: it's caller's responsibility to give a measured and traced dataset

        Outer list describes & (AND)
        Inner list describes | (OR)
        So when combined, they form a Conjunctive normal form.
        """

        trainCandSetsOuter = []

        for trainInnerCands in trainGroupFilters:
            trainCandSetsInner = []
            for filterName, filterComment in trainInnerCands:
                trainCandSetsInner.append(
                    self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                )

            trainCandSetsOuter.append(functools.reduce(
                lambda x, y: x | y,
                trainCandSetsInner,
                trainCandSetsInner[0] if len(trainCandSetsInner) > 0 else set()
            ))

        trainExprIds = list(functools.reduce(
            lambda x, y: x & y,
            trainCandSetsOuter,
            trainCandSetsOuter[0] if len(trainCandSetsOuter) > 0 else set()
        ))

        if applyPostProcessorToTrain:
            for postProc in self.postProcInsts:
                trainExprIds = postProc(trainExprIds)

        trainSet = FragmentPerformanceWithTraceByQueryDataset(trainExprIds)
        trainSamples = [trainSet[i] for i in range(0, len(trainSet))]
        assert(len(trainSamples) == len(trainExprIds))
        print(f"Train samples: {len(trainExprIds)}")

        if testGroupFilters is not None:
            assert(testGroupCopySrc is None)
            testCandSetsOuter = []
            for testInnerCands in testGroupFilters:
                testCandSetsInner = []
                for filterName, filterComment in testInnerCands:
                    testCandSetsInner.append(
                        self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                    )

                testCandSetsOuter.append(functools.reduce(
                    lambda x, y: x | y,
                    testCandSetsInner,
                    testCandSetsInner[0] if len(testCandSetsInner) > 0 else set()
                ))

            testExprIds = list(functools.reduce(
                lambda x, y: x & y,
                testCandSetsOuter,
                testCandSetsOuter[0] if len(testCandSetsOuter) > 0 else set()
            ))

            testSet = FragmentPerformanceWithTraceByQueryDataset(testExprIds)
            testSamples = [testSet[i] for i in range(0, len(testSet))]
            
            assert(len(testSamples) == len(testExprIds))
            print(f"Test samples: {len(testExprIds)}")
        else:
            assert(testGroupFilters is None and testGroupCopySrc is not None)
            testSamples = [testGroupCopySrc[i] for i in range(0, len(testGroupCopySrc))]

            print(f"Test samples: {len(testSamples)}")

        if valGroupFilters is not None:
            valCandSetsOuter = []
            for valInnerCands in valGroupFilters:
                valCandSetsInner = []
                for filterName, filterComment in valInnerCands:
                    valCandSetsInner.append(
                        self._findSubfilterDescByComment(filterName, filterComment).elemExprIds
                    )

                valCandSetsOuter.append(functools.reduce(
                    lambda x, y: x | y,
                    valCandSetsInner,
                    valCandSetsInner[0] if len(valCandSetsInner) > 0 else set()
                ))

            valExprIds = list(functools.reduce(
                lambda x, y: x & y,
                valCandSetsOuter,
                valCandSetsOuter[0] if len(valCandSetsOuter) > 0 else set()
            ))

            valSet = FragmentPerformanceWithTraceByQueryDataset(valExprIds)
            valSamples = [valSet[i] for i in range(0, len(valSet))]
            
            assert(len(valSamples) == len(valExprIds))
            print(f"Val samples: {len(valExprIds)}")

        with open(snapshotDestfile, "wb") as fp:
            pickle.dump({
                "train": trainSamples,
                "test": testSamples,
                "val": valSamples if valGroupFilters is not None else None
            }, fp)
        
        return len(trainSamples), len(testSamples), len(valSamples) if valGroupFilters is not None else None
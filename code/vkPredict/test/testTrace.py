import unittest
from toyDb.ExperimentDB import SpvContext
from dataset.DatasetBuilder import build_dataset
from compete.TracedLinearRegression import TracedLinearRegression, SpvInstrClassesOrdered
from compete.TracedPerInstLinearRegression import TracedPerInstLinearRegression
import tqdm
import os
import sklearn.linear_model
import numpy as np

class testTrace(unittest.TestCase):
    def setUp(self):
        pass

    def test_per_inst_trace_equivalence(self):
        """Test the equivalence between per inst trace and per class trace"""
        grammar = SpvContext.getDefaultGrammar()
        trainDataset = build_dataset("FragmentPerformanceTracedSnapshotDataset", "train")

        perInstReg = TracedPerInstLinearRegression(grammar, False)
        reg = TracedLinearRegression(27, False, True, False)

        numSamples = len(trainDataset)
        for idx in tqdm.tqdm(range(0, numSamples)):
            sample = trainDataset[idx]
            x, y = reg.extract_pairs(sample)
            x_inst, y_inst = perInstReg.extract_pairs(sample)

            assert(y == y_inst)

            # check equivalance
            # 1. check if sub-categories adds to the master one
            subSum = sum(x[1:])
            assert(subSum == x[0])

            # 2. check if re-organized insts match sub category result
            numInstByClass = {k: 0 for k in SpvInstrClassesOrdered}
            for i in range(len(x_inst)):
                opcode = grammar.opcode[perInstReg.instNames[i]]
                className = grammar.instFormatsByOpcode[opcode].instrClass
                numInstByClass[className] += x_inst[i]
            
            for i, className in enumerate(SpvInstrClassesOrdered):
                assert(x[i + 1] == numInstByClass[className])

    def test_shift_params(self):
        """Shift the result from 26 class into all insts - mse will improve?"""
        rootDir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "../")
        grammar = SpvContext.getDefaultGrammar()
        trainDataset = build_dataset("FragmentPerformanceTracedSnapshotDataset", "train")

        perInstReg = TracedPerInstLinearRegression(grammar, False)
        perInstReg.model = sklearn.linear_model.LinearRegression()
        perInstReg.model.coef_ = np.ndarray((perInstReg.numInsts,), dtype=np.float32)
        perInstReg.model.intercept_ = np.ndarray((1,), dtype=np.float32)
        reg = TracedLinearRegression(26, False, True, True)

        reg.load(os.path.join(rootDir, "tracedLinearRegression-26feature-exclude-first.json"))

        for idx, instName in enumerate(perInstReg.instNames):
            instClass = grammar.instFormatsByOpcode[grammar.opcode[instName]].instrClass
            clsIdx = SpvInstrClassesOrdered.index(instClass)
            perInstReg.model.coef_[idx] = np.copy(reg.model.coef_[clsIdx])
        
        perInstReg.model.intercept_ = np.copy(reg.model.intercept_)

        print("Val on train set for copied weights")
        perInstReg.validate(trainDataset)

    def test_block_trace_validaty(self):
        """TODO: implement checking for block trace overflow conditions"""
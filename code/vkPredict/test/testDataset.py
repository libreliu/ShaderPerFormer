import unittest, os
from toyDb.ExperimentDB import db, Environment, ImageOnlyShader, ImageOnlyResource, ImageOnlyExperiment
import toyDb.ExperimentDB
import exampleImageOnlyShader
import datetime
from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset
import vkExecute, vkExecute.spv
import typing

# single entrypoint tokenization function
def tokenizeSingle(spvText: str) -> typing.List[int]:
    tokenizer = vkExecute.spv.Tokenizer(True, True, True)
    spvProc = vkExecute.SpvProcessor()
    success, errMsg = spvProc.assemble(spvText)
    assert(success)

    tokenizer.loadSpv(spvProc.exportSpv())
    tokenizedSpv, errMsgs = tokenizer.tokenize()
    assert(len(tokenizedSpv) > 0)
    
    return tokenizedSpv

# multiple entrypoint tokenization function
def tokenizeMultiple(spvText: str) -> typing.List[int]:
    tokenizer = vkExecute.spv.Tokenizer(True, False, True)
    spvProc = vkExecute.SpvProcessor()
    success, errMsg = spvProc.assemble(spvText)
    assert(success)

    tokenizer.loadSpv(spvProc.exportSpv())
    tokenizedSpv, errMsgs = tokenizer.tokenize()
    assert(len(tokenizedSpv) > 0)
    
    return tokenizedSpv

class FragPerfDatasetTest(unittest.TestCase):
    def generateTestData(self):
        self.testDbPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test.db")
        if os.path.exists(self.testDbPath):
            os.remove(self.testDbPath)
        

        db.init(self.testDbPath)
        db.create_tables([Environment, ImageOnlyShader, ImageOnlyResource, ImageOnlyExperiment])

        shdrInst = ImageOnlyShader.create(
            shader_id=exampleImageOnlyShader.shaderId, fragment_spv=exampleImageOnlyShader.fragmentSpv
        )
        shdrInst.save()
        self.shaderRealText = exampleImageOnlyShader.fragmentSpv
        self.shaderRealTokenLength = len(self.tokenizer(self.shaderRealText))

        this_env = toyDb.ExperimentDB.get_or_add_environment("testcase environment")
        self.nodeName = this_env.node
        expr = ImageOnlyExperiment.create(
            time=datetime.datetime.now(),
            environment=this_env,
            augmentation=0,
            width=1024,
            height=768,
            shader=shdrInst,
            resource=None,
            measurement=0,
            image_hash="c81d4b3b4592c931692e4cf44e2a8ec00faa23e097ee6b54bea1b4e56f2fc310",
            num_cycles=50,
            num_trials=10,
            result_mean=111.332069237646,
            result_stdev=0.96453101335189
        )
        expr.save()
        db.close()
        

    def setUp(self):
        self.tokenizer = tokenizeMultiple
        self.generateTestData()
        
    def test_fragment_performance_dataset(self):
        dataset = FragmentPerformanceDataset(tokenizeMultiple, self.nodeName, filteredNumCycles=50, filteredNumTrials=10)

        assert(len(dataset) == 1)
        loaded = dataset[0]
        print(f"=> Tokenized: \n{loaded['tokens']}")

        print(f"=> Target: \n{loaded['fpsMean']}")
        
        tokenizer = vkExecute.spv.Tokenizer(True, False, True)
        detok = tokenizer.deTokenize(loaded['tokens'])
        print(f"=> detokenized: \n{detok}")
        print(f"=> Total {len(loaded['tokens'])} tokens")
        db.close()
    
    def test_fragment_performance_dataset_filtered(self):
        dataset = FragmentPerformanceDataset(tokenizeMultiple, self.nodeName, filteredNumCycles=50, filteredNumTrials=10, maxTokenizedLength=self.shaderRealTokenLength - 1)
        assert(len(dataset) == 0)

        dataset = FragmentPerformanceDataset(tokenizeMultiple, self.nodeName, filteredNumCycles=50, filteredNumTrials=10, maxTokenizedLength=self.shaderRealTokenLength)
        assert(len(dataset) == 1)
        db.close()


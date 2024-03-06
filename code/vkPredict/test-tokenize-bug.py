from dataset.FragmentPerformanceWithTraceDataset import FragmentPerformanceWithTraceDataset
from vkExecute import SpvProcessor
import toyDb.ExperimentDB

toyDb.ExperimentDB.init_from_default_db()

dataset = FragmentPerformanceWithTraceDataset()

bugIdx = 6546

buggy = dataset[bugIdx]
print(f"Id: {dataset.getShaderId(bugIdx)}")

# spvListRepr = [i for i in map(lambda x: chr(x), buggy['spvBlob'])]

# spvProc = SpvProcessor()
# spvProc.loadSpv(spvListRepr)
# fragDisText, fragErrMsgs = spvProc.disassemble()
# print(fragDisText)
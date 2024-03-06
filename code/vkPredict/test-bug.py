from misc.HfSpvTokenizer import HfSpvTokenizer
from misc.HfBpeTokenizer import HfBpeTokenizer
from dataset.FragmentPerformanceDataset import FragmentPerformanceDataset
import tqdm
import toyDb.ExperimentDB


dataset = FragmentPerformanceDataset(None, maxTokenizedLength=None)
toyDb.ExperimentDB.init_from_default_db()

tokenizerHF = HfSpvTokenizer(single_entrypoint=False)

spvText = dataset[166]['spvText']
tokenizerHF(spvText)
from typing import List, Union
import numpy as np

def calcNormalizationParams(targets: Union[List[float], 'np.ndarray']):
    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.float32)

    targets_mean = np.mean(targets)
    targets_var = np.var(targets)
    targets_stdev = np.std(targets)
    print(f"Dataset target mean={targets_mean} var={targets_var} stdev={targets_stdev}")
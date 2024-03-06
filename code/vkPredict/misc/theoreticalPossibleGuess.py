from typing import List, Union
import numpy as np

def guessNoParamMSE(targets: Union[List[float], 'np.ndarray']):
    """
    loss(x) = \sum [(x- targets_i)**2]
    dloss(x) = \sum [2(x-targets_i)] = 0
    => x = mean
    => loss(x) = var(x)

    mape = mean(np.abs(mean(targets_i) - targets_i))
    """

    if not isinstance(targets, np.ndarray):
        targets = np.asarray(targets, dtype=np.float32)

    guess = np.var(targets)
    print(f"Dataset best guess without input (MSE): {guess} (sqrt: {np.sqrt(guess)})")
    
    # mape for this guess
    targets_mean = np.mean(targets)
    guessMape = np.mean(np.abs(targets_mean - targets) / targets)
    print(f"Dataset MAPE w.r.t. best guess for MSE: {guessMape}")
    print(f"Dataset target average: {np.mean(targets)}")

def guessNoParamMape(targets):
    """
    loss(x) = \sum [|x-targets_i| / targets_i]
    dloss(x) = \sum [(x > targets_i ? 1 : -1 ) * (1/targets_i)] = 0
    """
    # TODO: implement me

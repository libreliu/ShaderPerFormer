import numpy as np

def compute_metrics(pred: 'np.ndarray', real: 'np.ndarray') -> 'dict[str, float]':
    numSamples = pred.shape[0]
    assert(pred.shape == (numSamples,) and real.shape == (numSamples,))

    metrics = {}
    
    # an overall report
    metrics["mse_sqrt"] = np.sqrt(((real - pred)**2).mean(axis=0))
    metrics["mae"] = np.mean(np.abs((real - pred)))
    metrics["mape"] = np.mean(np.abs((real - pred)/real))

    # report by range
    ranges = [
        ((0, 1e-4), "_ge_10000fps"),
        ((1e-4, 1e-3), "_ge_1000_le_10000fps"),
        ((1e-3, 1e-2), "_ge_100_le_1000fps"),
        ((1e-2, 1e-1), "_ge_10_le_100fps"),
        ((1e-1, 1e10), "_le_10fps")
    ]

    indices = np.argsort(real)
    realAsc = real[indices]
    predAsc = pred[indices]

    for (rangeMin, rangeMax), descSuffix in ranges:
        candRealTime = []
        candPredTime = []
        for idx, val in enumerate(realAsc):
            if val >= rangeMin and val < rangeMax:
                candRealTime.append(val)
                candPredTime.append(predAsc[idx])
        
        candRealTime = np.array(candRealTime)
        candPredTime = np.array(candPredTime)

        metrics[f"mse_sqrt{descSuffix}"] = np.sqrt(
            ((candRealTime - candPredTime)**2).mean(axis=0)
        )
        metrics[f"mae{descSuffix}"] = np.mean(
            np.abs((candRealTime - candPredTime))
        )
        metrics[f"mape{descSuffix}"] = np.mean(
            np.abs((candRealTime - candPredTime)/candRealTime)
        )

    return metrics

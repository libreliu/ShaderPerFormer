import dataclasses
from typing import Union, Iterable
import numpy as np
import math

class NormalizerBase:
    def normalize(self, sample: Union[float, np.ndarray, list]):
        raise NotImplementedError("Implement me")

    def invNormalize(self, sample: Union[float, np.ndarray, list]):
        raise NotImplementedError("Implement me")
    
# TODO: write unit test on this to make sure things work as expected
class Normalizer(NormalizerBase):
    def __init__(self, mean, stdev):
        self.mean = mean
        self.stdev = stdev
    
    def normalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, list):
            return [(elem - self.mean) / self.stdev for elem in sample]
        else:
            return (sample - self.mean) / self.stdev

    def invNormalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, list):
            return [(elem * self.stdev) + self.mean for elem in sample]
        else:
            return (sample * self.stdev) + self.mean

    @staticmethod
    # https://www.johndcook.com/blog/standard_deviation/
    def buildFromSamples(sampleIterator: Iterable[Union[float, np.ndarray]]):
        count = 0
        mean = None
        S = None

        for sample in sampleIterator:
            count += 1
            if count == 1:
                mean = sample
                S = 0
            else:
                newMean = mean + (sample - mean) / count
                newS = S + (sample - mean) * (sample - newMean)

                mean = newMean
                S = newS
        
        return Normalizer(
            mean if mean is not None else 0.0,
            math.sqrt(S / (count - 1)) if S is not None and count > 1 else 0.0
        )

class DummyNormalizer(NormalizerBase):
    def __init__(self):
        pass

    def normalize(self, sample: Union[float, np.ndarray, list]):
        return sample

    def invNormalize(self, sample: Union[float, np.ndarray, list]):
        return sample

class LogPlusNormalizer(NormalizerBase):
    def __init__(self):
        pass

    def normalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, np.ndarray):
            return np.log(sample + 1)
        elif isinstance(sample, float):
            return math.log(sample + 1)
        elif isinstance(sample, list):
            return [math.log(elem + 1) for elem in sample]
        else:
            raise NotImplementedError(f"Unknown type for {sample}")

    def invNormalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, np.ndarray):
            return np.exp(sample) - 1
        elif isinstance(sample, float):
            return math.exp(sample) - 1
        elif isinstance(sample, list):
            return [math.exp(elem) - 1 for elem in sample]
        else:
            raise NotImplementedError(f"Unknown type for {sample}")

class LogNormalizer(NormalizerBase):
    def __init__(self):
        pass

    def normalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, np.ndarray):
            return np.log(sample)
        elif isinstance(sample, float):
            return math.log(sample)
        elif isinstance(sample, list):
            return [math.log(elem) for elem in sample]
        else:
            raise NotImplementedError(f"Unknown type for {sample}")

    def invNormalize(self, sample: Union[float, np.ndarray, list]):
        if isinstance(sample, np.ndarray):
            return np.exp(sample)
        elif isinstance(sample, float):
            return math.exp(sample)
        elif isinstance(sample, list):
            return [math.exp(elem) for elem in sample]
        else:
            raise NotImplementedError(f"Unknown type for {sample}")


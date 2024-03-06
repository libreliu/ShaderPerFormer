from toyDb.utils.spv import SpvContext, SpvInstStat
import torch
import io, json
from typing import List, Tuple
import numpy as np
import sklearn.linear_model, sklearn.metrics
import tqdm

import logging

logger = logging.getLogger(__name__)

# https://johaupt.github.io/blog/neural_regression.html
# https://scikit-learn.org/stable/getting_started.html

class regression(torch.nn.Module):
    def __init__(self, input_dim):
        # Applies the init method from the parent class nn.Module
        # Lots of backend and definitions that all neural networks modules use
        # Check it out here:
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html
        super().__init__()

        # Safe the input in case we want to use/see it
        self.input_dim = input_dim

        # One layer
        self.linear = torch.nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

"""
sample: {
    "spvBlob": spvBlob,
    "bbIdxMap": {int(k): v for k, v in json.loads(expr.trace.bb_idx_map).items()},
    "bbTraceCounters": json.loads(expr.trace.bb_trace_counters),
    "fpsMean": expr.result_mean
}
"""

SpvInstrClassesOrdered = [
  'Arithmetic',
  'Control-Flow',
  'Miscellaneous',
  'Debug',
  'Annotation',
  'Extension',
  'Mode-Setting',
  'Type-Declaration',
  'Constant-Creation',
  'Memory',
  'Function',
  'Image',
  'Conversion',
  'Composite',
  'Bit',
  'Relational_and_Logical',
  'Derivative',
  'Atomic',
  'Primitive',
  'Barrier',
  'Group',
  'Device-Side_Enqueue',
  'Pipe',
  'Non-Uniform',
  'Reserved',
  '@exclude'
]

def pack_bytes_to_bytes_stream(dataBytes):
    return io.BytesIO(dataBytes)

class TracedLinearRegression:
    def __init__(self, numFeatures, approxMapeLoss=False, withTrace=True, excludeFirst=False):
        self.grammar = SpvContext.Grammar(SpvContext.SpvGrammarPath)
        self.withTrace = withTrace
        self.numFeatures = numFeatures
        self.excludeFirst = excludeFirst
        self.approxMapeLoss = approxMapeLoss
        self.model = None

    def extract_pairs(self, sample) -> Tuple[List[int], float]:
        parser = SpvContext.BinaryParser()

        binCtx = parser.parse(
            self.grammar,
            pack_bytes_to_bytes_stream(sample["fragSpv"])
        )

        if self.withTrace:
            def errorHandler(blockIdx: int):
                logger.info(
                    f"Occured in shaderID = {sample['shaderId']}, environmentId = {sample['environmentId']}"
                )

            statResult = SpvInstStat.statWithTrace(
                binCtx, sample["bbIdxMap"], sample["bbTraceCounters"], errorHandler
            )
        else:
            statResult = SpvInstStat.statWithoutTrace(binCtx)
        
        if self.excludeFirst:
            features = []
        else:
            features = [statResult.numInsts]
        
        assert(len(statResult.numInstsByClass.keys()) == len(SpvInstrClassesOrdered))
        for className in SpvInstrClassesOrdered:
            features.append(statResult.numInstsByClass[className])
        
        features = features[:self.numFeatures]

        # target is frame time used
        target = sample["timeMean"]
        return (features, target)

    def train(self, trainSet: 'torch.utils.data.Dataset'):
        numSamples = len(trainSet)
        # numSamples = 100
        X = np.ndarray((numSamples, self.numFeatures), dtype=np.float32)
        Y = np.ndarray((numSamples,), dtype=np.float32)

        pbar = tqdm.tqdm(range(numSamples), "Training")
        for idx in pbar:
            sample = trainSet[idx]
            pbar.set_postfix({
                "id": sample["shaderId"] 
            })
            x, y = self.extract_pairs(sample)
            X[idx, :] = np.asarray(x, dtype=np.float32)
            Y[idx] = np.asarray(y, dtype=np.float32)
        
        self.model = sklearn.linear_model.LinearRegression()
        if self.approxMapeLoss:
            # Y is inverse of fps, which is around 1e0 ~ 1e-5
            self.model.fit(X, Y, 1.0 / (Y + 1e-6))
        else:
            self.model.fit(X, Y)
        print(f"model coef_={self.model.coef_}, intercept_={self.model.intercept_}")

    def evaluate(self, valSet: 'torch.utils.data.Dataset'):
        numValSamples = len(valSet)
        X = np.ndarray((numValSamples, self.numFeatures), dtype=np.float32)
        Y_real = np.ndarray((numValSamples,), dtype=np.float32)

        for idx in tqdm.tqdm(range(numValSamples), "Evaluation"):
            valSample = valSet[idx]
            x, y = self.extract_pairs(valSample)
            X[idx, :] = np.asarray(x, dtype=np.float32)
            Y_real[idx] = np.asarray(y, dtype=np.float32)

        Y_pred = self.model.predict(X)
        return Y_real, Y_pred

    def validate(self, valSet: 'torch.utils.data.Dataset'):
        Y_real, Y_pred = self.evaluate(valSet)

        fps_real = 1.0 / Y_real
        fps_pred = 1.0 / Y_pred

        mse = sklearn.metrics.mean_squared_error(fps_real, fps_pred)
        mae = sklearn.metrics.mean_absolute_error(fps_real, fps_pred)
        mape = sklearn.metrics.mean_absolute_percentage_error(fps_real, fps_pred)
        print(f"mse={mse} mae={mae} mape={mape}")

        mseOrig = sklearn.metrics.mean_squared_error(Y_real, Y_pred)
        maeOrig = sklearn.metrics.mean_absolute_error(Y_real, Y_pred)
        mapeOrig = sklearn.metrics.mean_absolute_percentage_error(Y_real, Y_pred)
        print(f"mseOrig={mseOrig} maeOrig={maeOrig} mapeOrig={mapeOrig}")

    def save(self, savePath):
        result = {
            "type": "traced-linear-regression",
            "coef_": self.model.coef_.tolist(),
            "intercept_": self.model.intercept_.tolist()
        }
        with open(savePath, "w") as fp:
            json.dump(result, fp)
        
    def load(self, loadPath):
        with open(loadPath, "r") as fp:
            res = json.load(fp)
        
        assert(res["type"] == "traced-linear-regression")
        self.model = sklearn.linear_model.LinearRegression()
        self.model.coef_ = np.asarray(res["coef_"])
        self.model.intercept_ = np.asarray(res["intercept_"])

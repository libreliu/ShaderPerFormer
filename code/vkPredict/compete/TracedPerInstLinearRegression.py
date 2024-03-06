from toyDb.utils.spv import SpvContext
import torch, torch.utils.data.dataset
import logging, io
import dataclasses
import numpy as np
from typing import List, Dict, Tuple
import sklearn.linear_model, sklearn.metrics
import tqdm
import json

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class InstStatResult:
    numInstsByInstName: Dict[str, int]

    def mergeResult(self, other: 'InstStatResult'):
        diff = set(other.numInstsByInstName.keys()) - \
            set(self.numInstsByInstName.keys())
        assert(len(diff) == 0)

        for k in self.numInstsByInstName.keys():
            self.numInstsByInstName[k] += other.numInstsByInstName[k]

def statBasicBlock(instNames: List[str], bb: SpvContext.BasicBlock):
    grammar = bb.parent.parent.grammar

    numInstsByInstName = {k: 0 for k in instNames}

    for inst in bb.insts:
        instName = grammar.instFormatsByOpcode[inst.opcode].opname
        numInstsByInstName[instName] += 1

    result = InstStatResult(numInstsByInstName)
    # print(result)

    return result

def statOpcodeWithTrace(
    instNames: List[str],
    binCtx: SpvContext.BinaryContext,
    bbIdx2TraceIdx: Dict[int, int],
    traceData: List[int],
    blockIdNotExistCallback=None,
    suppressWarning=False
) -> InstStatResult:
    bbList: List['SpvContext.BasicBlock'] = []
    for func in binCtx.functions:
        bbList += func.basicBlocks
    
    resTotal = InstStatResult({k: 0 for k in instNames})
    for bb in bbList:
        if bb.blockId not in bbIdx2TraceIdx:
            # This is probably not reachable from entrypoint, therefore not labeled.
            if not suppressWarning:
                logger.warning(f"Basic block #{bb.blockId} not in traced block list. ")
            traceCnt = 0

            if blockIdNotExistCallback is not None:
                blockIdNotExistCallback(bb.blockId)

        else:
            traceId = bbIdx2TraceIdx[bb.blockId]
            traceCnt = traceData[traceId]
    
        res = statBasicBlock(instNames, bb)
        for instName in res.numInstsByInstName.keys():
            res.numInstsByInstName[instName] *= traceCnt
        
        resTotal.mergeResult(res)
    return resTotal

def statOpcodeWithoutTrace(
    instNames: List[str],
    binCtx: SpvContext.BinaryContext
) -> InstStatResult:
    bbList: List['SpvContext.BasicBlock'] = []
    for func in binCtx.functions:
        bbList += func.basicBlocks
    
    resTotal = InstStatResult({k: 0 for k in instNames})
    for bb in bbList:
        res = statBasicBlock(instNames, bb)
        resTotal.mergeResult(res)
    return resTotal

def pack_bytes_to_bytes_stream(dataBytes):
    return io.BytesIO(dataBytes)

class LinRegNet(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Safe the input in case we want to use/see it
        self.input_dim = input_dim

        # One layer
        self.linear = torch.nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

class PerInstCountDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            grammar: 'SpvContext.Grammar',
            dataset: 'torch.utils.data.Dataset'
        ):
        self.dataset = dataset
        self.grammar = grammar
        self.instNames = sorted(map(
            lambda x: x.opname,
            self.grammar.instFormatsByOpcode.values()
        ))
        self.numInsts = len(self.instNames)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        parser = SpvContext.BinaryParser()
        binCtx = parser.parse(
            self.grammar,
            pack_bytes_to_bytes_stream(sample["spvBlob"])
        )

        statResult = statOpcodeWithTrace(
            self.instNames, binCtx, sample["bbIdxMap"], sample["bbTraceCounters"]
        )

        features = [
            statResult.numInstsByInstName[self.instNames[idx]] \
                for idx in range(0, self.numInsts)
        ]

        target = 1.0 / sample["fpsMean"]

        return (features, target)

def collate_fn(batch):
    features = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return (
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32)
    )

class TracedPerInstLinearRegressionTorch:
    def __init__(self, grammar: 'SpvContext.Grammar', approxMapeLoss=False):
        # build inst order table
        self.grammar = grammar
        self.instNames = sorted(map(
            lambda x: x.opname,
            self.grammar.instFormatsByOpcode.values()
        ))
        self.numInsts = len(self.instNames)
        # self.numInsts = 10

        logger.info(f"Collected {self.numInsts} inst names from grammar")

        self.model = LinRegNet(self.numInsts)
        self.epochs = 20
        self.device = 'cuda:0'
        self.bsz = 512
    
    def train(self, trainSet: 'torch.utils.data.Dataset'):
        loader = torch.utils.data.DataLoader(
            PerInstCountDataset(self.grammar, trainSet),
            batch_size=self.bsz,
            shuffle=False,
            num_workers=60,
            collate_fn=collate_fn
        )

        torch.nn.init.uniform_(self.model.linear.weight, 0, 0)
        self.model.train()
        self.model.to(self.device)

        optim = torch.optim.Adam(self.model.parameters(), lr=1e-6)

        for epoch in range(self.epochs):
            for idx, (X, Y) in enumerate(loader):
        #        print(self.model.linear.weight)

                X = X.to(self.device)
                Y = Y.to(self.device)

                pred = self.model(X)
                loss = (Y-pred).pow(2).mean() / self.bsz

                optim.zero_grad()
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1e25)
                optim.step()

                print(f"epoch={epoch} batch_idx={idx} Loss={loss}")
        
        self.model.to('cpu')
        print(f"model params={self.model.parameters()}")
        print(self.model.linear.weight)

    def validate(self, valSet: 'torch.utils.data.Dataset'):
        numValSamples = len(valSet)
        X = torch.zeros((numValSamples, self.numInsts), dtype=torch.float32)
        Y_real = torch.zeros((numValSamples,), dtype=torch.float32)

        wrappedDataset = PerInstCountDataset(self.grammar, valSet)
        self.model.eval()

        for idx, (x, y) in tqdm.tqdm(enumerate(wrappedDataset), "Validation"):
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)

            X[idx, :] = x
            Y_real[idx] = y
        
        with torch.no_grad():
            Y_pred = self.model(X)

            fps_real = 1.0 / Y_real
            fps_pred = 1.0 / Y_pred

            fps_real = fps_real.numpy()
            fps_pred = fps_pred.numpy()

        mse = sklearn.metrics.mean_squared_error(fps_real, fps_pred)
        mae = sklearn.metrics.mean_absolute_error(fps_real, fps_pred)
        mape = sklearn.metrics.mean_absolute_percentage_error(fps_real, fps_pred)
        print(f"mse={mse} mae={mae} mape={mape}")


class TracedPerInstLinearRegression:
    def __init__(self, grammar: 'SpvContext.Grammar', approxMapeLoss=False, enableTrace=True):
        # build inst order table
        self.grammar = grammar
        self.instNames = sorted(map(
            lambda x: x.opname,
            self.grammar.instFormatsByOpcode.values()
        ))
        self.numInsts = len(self.instNames)

        logger.info(f"Collected {self.numInsts} inst names from grammar")

        self.model = None
        self.approxMapeLoss = approxMapeLoss
        self.enableTrace = enableTrace

    def extract_pairs(self, sample):
        parser = SpvContext.BinaryParser()

        binCtx = parser.parse(
            self.grammar,
            pack_bytes_to_bytes_stream(sample["fragSpv"])
        )
        
        if self.enableTrace:
            def errorHandler(blockIdx: int):
                logger.info(
                    f"Occured in shaderID = {sample['shaderId']}, environmentId = {sample['environmentId']}"
                )

            statResult = statOpcodeWithTrace(
                self.instNames, binCtx, sample["bbIdxMap"], sample["bbTraceCounters"], errorHandler
            )
        else:
            statResult = statOpcodeWithoutTrace(
                self.instNames, binCtx
            )

        features = [
            statResult.numInstsByInstName[self.instNames[idx]] \
                for idx in range(0, self.numInsts)
        ]

        # target is frame time used
        target = sample["timeMean"]
        return (features, target)

    def train(self, trainSet: 'torch.utils.data.Dataset'):
        numSamples = len(trainSet)
        X = np.ndarray((numSamples, self.numInsts), dtype=np.float32)
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

        # np.save("X.npy", X)
        # np.save("Y.npy", Y)

        self.model = sklearn.linear_model.LinearRegression()
        if self.approxMapeLoss:
            # Y is inverse of fps, which is around 1e0 ~ 1e-5
            self.model.fit(X, Y, 1.0 / (Y + 1e-6))
        else:
            self.model.fit(X, Y)
        
        print(f"model coef_={self.model.coef_}, intercept_={self.model.intercept_}")

    def evaluate(self, valSet: 'torch.utils.data.Dataset'):
        numValSamples = len(valSet)
        X = np.ndarray((numValSamples, self.numInsts), dtype=np.float32)
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
            "type": "traced-per-inst-linear-regression",
            "coef_": self.model.coef_.tolist(),
            "intercept_": self.model.intercept_.tolist()
        }
        with open(savePath, "w") as fp:
            json.dump(result, fp)
        
    def load(self, loadPath):
        with open(loadPath, "r") as fp:
            res = json.load(fp)
        
        assert(res["type"] == "traced-per-inst-linear-regression")
        self.model = sklearn.linear_model.LinearRegression()
        self.model.coef_ = np.asarray(res["coef_"])
        self.model.intercept_ = np.asarray(res["intercept_"])

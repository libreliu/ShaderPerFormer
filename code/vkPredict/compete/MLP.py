import torch
import logging
import tqdm
from toyDb.utils.spv import SpvContext
from toyDb.databases.ExperimentDb import (
    packBytesToBytesStream
)
from compete.TracedPerInstLinearRegression import statOpcodeWithTrace

from typing import List, Dict

logger = logging.getLogger(__name__)

class MLPNet(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_dim, self.hidden_dim),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, x, labels=None):
        y_pred = self.linear(x)
        y_pred = torch.squeeze(y_pred, -1)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(y_pred, labels)

        return {
            "loss": loss,
            "y_pred": y_pred
        }

class PerInstCountDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            grammar: 'SpvContext.Grammar',
            dataset: 'torch.utils.data.Dataset',
            suppressWarning: bool = False
        ):
        self.dataset = dataset
        self.grammar = grammar
        self.instNames = sorted(map(
            lambda x: x.opname,
            self.grammar.instFormatsByOpcode.values()
        ))
        self.numInsts = len(self.instNames)
        self.suppressWarning = suppressWarning

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        parser = SpvContext.BinaryParser()
        binCtx = parser.parse(
            self.grammar,
            packBytesToBytesStream(sample["fragSpv"])
        )

        statResult = statOpcodeWithTrace(
            self.instNames, binCtx, sample["bbIdxMap"], sample["bbTraceCounters"],
            suppressWarning=self.suppressWarning
        )

        features = [
            statResult.numInstsByInstName[self.instNames[idx]] \
                for idx in range(0, self.numInsts)
        ]

        target = sample["timeMean"]

        return [features, target]

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

        self.model = MLPNet(self.numInsts, 64)
        self.epochs = 50
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

        for layer in self.linear:
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)

        self.model.train()
        self.model.to(self.device)

        optim = torch.optim.Adam(self.model.parameters(), lr=1e-5)

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

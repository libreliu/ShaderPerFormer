# %% [markdown]
# # Linear Regression Baseline

# %% [markdown]
# We've setup different groups for Linear regression:
# - num_insts only
# - 26 groups (and they sums to num_inst)
# - Per-Inst
# 
# Every group provide options for non-traced and traced. Also, (approx) MAPE vs non-mape loss group is given.
# 
# The metric is calculated using the function inside misc.
# The mape, mae and mse are reported for different time groups.

# %%
import os, sys
sys.path.append(os.path.join(os.path.abspath(''), '../'))

# %%
import logging, sys
from compete.TracedLinearRegression import TracedLinearRegression
from compete.TracedPerInstLinearRegression import (
    TracedPerInstLinearRegression,
    TracedPerInstLinearRegressionTorch
)
from dataset.DatasetBuilder import build_dataset
from misc import metric
from toyDb.utils.spv.SpvContext import getDefaultGrammar
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# %% [markdown]
# TODO: add a constrastive on different optimized shaders having different trace count but same runtime
# 
# use https://pypi.org/project/tabulate/ to show answer
# 
# https://stackoverflow.com/questions/35160256/how-do-i-output-lists-as-a-table-in-jupyter-notebook
# 

# %% [markdown]
# ## 3060 environment

# %%
metrics = {}

def do_compute(
  groupName,
  enableTrace,
  useApproxMape,
  trainDataset,
  testDataset
):
  Y_real = None
  Y_pred = None

  if groupName == 'NumInsts':
    regressor = TracedLinearRegression(1, useApproxMape, enableTrace, False)
    regressor.train(trainDataset)

    print(f"- Model coef: {regressor.model.coef_}")
    print(f"- Model intercept: {regressor.model.intercept_}")

    Y_real, Y_pred = regressor.evaluate(testDataset)

  elif groupName == 'AllSpvCategories':
    regressor = TracedLinearRegression(26, useApproxMape, enableTrace, True)
    regressor.train(trainDataset)

    print(f"- Model coef: {regressor.model.coef_}")
    print(f"- Model intercept: {regressor.model.intercept_}")

    Y_real, Y_pred = regressor.evaluate(testDataset)

  elif groupName == 'PerInst':
    grammar = getDefaultGrammar()
    regressor = TracedPerInstLinearRegression(grammar, useApproxMape, enableTrace)
    regressor.train(trainDataset)

    print(f"- Model coef: {regressor.model.coef_}")
    print(f"- Model intercept: {regressor.model.intercept_}")

    Y_real, Y_pred = regressor.evaluate(testDataset)

  return metric.compute_metrics(Y_pred, Y_real)


# %%
trainDataset = build_dataset('FragmentPerformanceTracedSnapshotDataset4096-3060', 'train')
testDataset = build_dataset('FragmentPerformanceTracedSnapshotDataset4096-3060', 'test')

for groupName in ['NumInsts', 'AllSpvCategories', 'PerInst']:
  for enableTrace in [True, False]:
    for useApproxMape in [True, False]:
      fullName = f"{groupName}-{'withTrace' if enableTrace else 'withoutTrace'}-{'useApproxMape' if useApproxMape else 'useMSE'}"
      logger.info(f"Running test for {fullName}")

      with logging_redirect_tqdm():
        metrics[fullName] = do_compute(**{
          'groupName': groupName,
          'enableTrace': enableTrace,
          'useApproxMape': useApproxMape,
          'trainDataset': trainDataset,
          'testDataset': testDataset
        })





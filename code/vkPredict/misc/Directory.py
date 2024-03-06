# Aid in using Jupyter notebook
import os, sys

def getVkPredictRootDir():
  return os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../"
  )

def getIntermediateDir():
  return os.path.join(
    getVkPredictRootDir(), "./intermediates"
  )
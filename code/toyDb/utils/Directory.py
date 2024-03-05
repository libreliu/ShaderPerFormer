# Aid in using Jupyter notebook
import os, sys

def getToyDbRootDir():
  return os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../"
  )

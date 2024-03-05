from experiments.testStability import testStability
from experiments.testDisassembler import testDisassembler

import os
import argparse
import ShaderDB

if __name__ == '__main__':
  parser = argparse.ArgumentParser("runTests")
  parser.add_argument("testName", type=str)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument("--height", type=int, default=768)

  args = parser.parse_args()

  runDir = os.path.realpath(os.path.dirname(__file__))
  db = ShaderDB.ShaderDB(os.path.join(runDir, "./shaders"))

  db.scan_local()
  shdrProxy = db.get("lllBR7")
  shdrProxy.load()

  globals()[args.testName](shdrProxy, args)
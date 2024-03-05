from .testStability import prepareImageOnlyShaders, buildExecutor, runOnce
import numpy as np
import tqdm
import vkExecute

def testDisassembler(shdrProxy, args):
  vertShdrSpv, fragShdrSpv = prepareImageOnlyShaders(shdrProxy)
  sp = vkExecute.ShaderProcessor()
  sp.loadSpv(fragShdrSpv)
  disasm = sp.disassemble()

  print(disasm)

import vkExecute
from .. import ShaderDB
import os
import numpy as np
import struct
from tqdm import tqdm

vertShdrSrc = """#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;
layout(location = 0) in vec3 inPosition;
void main() {gl_Position = vec4(inPosition, 1.0);}
"""
commonFragShdrPremable = """#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;

layout(location = 0) out vec4 outColor;

layout (binding=0) uniform PrimaryUBO {
  uniform vec3 iResolution;
  uniform float iTime;
  uniform vec4 iChannelTime;
  uniform vec4 iMouse;
  uniform vec4 iDate;
  uniform vec3 iChannelResolution[4];
  uniform float iSampleRate;
  uniform int iFrame;
  uniform float iTimeDelta;
  uniform float iFrameRate;
};

void mainImage(out vec4 c, in vec2 f);
void main() {mainImage(outColor, gl_FragCoord.xy);}

"""

class ShaderCompilationException(Exception):
  def __init__(self, error, glslangError: str):
    super().__init__(error)
    self.glslangError = glslangError
    self.add_note(self.glslangError)

class ShaderResultInconsistencyException(Exception):
  def __init__(self):
    super().__init__(f"Inconsistent result in running the given shader")

def prepareImageOnlyShaders(shdrProxy: ShaderDB.ShaderProxy):
  fragShdrSrc = commonFragShdrPremable
  commonPass = shdrProxy.get_renderpass("common")
  imagePass = shdrProxy.get_renderpass("image")

  if commonPass is not None:
    fragShdrSrc += commonPass["code"]
  
  fragShdrSrc += imagePass["code"]

  vertShaderSpv, vertErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
    vkExecute.ShaderProcessor.ShaderStages.VERTEX,
    vertShdrSrc,
    "VertShader"
  )

  fragShaderSpv, fragErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
    vkExecute.ShaderProcessor.ShaderStages.FRAGMENT,
    fragShdrSrc,
    "FragShader"
  )

  if len(vertShaderSpv) == 0:
    raise ShaderCompilationException("Error while compiling vertex shader", vertErrMsg)

  if len(fragShaderSpv) == 0:
    raise ShaderCompilationException("Error while compiling fragment shader", fragErrMsg)

  return vertShaderSpv, fragShaderSpv

def buildExecutor(
    width, height, vertShaderSpv, fragShaderSpv, traceRun=False, 
    traceBufferSize=None, traceBufferDescSet=None, traceBufferBinding=None):
  pCfg = vkExecute.ImageOnlyExecutor.PipelineConfig()
  pCfg.targetWidth = width
  pCfg.targetHeight = height
  pCfg.vertexShader = vertShaderSpv
  pCfg.fragmentShader = fragShaderSpv
  if traceRun:
    pCfg.traceRun = True
    pCfg.traceBufferSize = traceBufferSize
    pCfg.traceBufferDescSet = traceBufferDescSet
    pCfg.traceBufferBinding = traceBufferBinding

  executor = vkExecute.ImageOnlyExecutor()
  executor.init(True)
  executor.initPipeline(pCfg)

  return executor

def runOnce(executor, cycles, uniform, nsecPerIncrement, width, height):

  # TODO: why iResolution[0] = 1024 things doesn't work?
  imageUniform = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
  if uniform is not None:
    imageUniform = uniform
  else:
    imageUniform.iResolution = [width, height, 0]
    imageUniform.iTime = 1
    imageUniform.iFrame = 1

  executor.setUniform(imageUniform)
  executor.preRender()
  executor.render(cycles)
  imgData, tickUsed = executor.getResults()

  fps = 1e9 / (nsecPerIncrement * tickUsed)

  # imgDataArray = np.asarray([i for i in map(lambda x: ord(x), imgData.data)], dtype=np.int8)
  # imgDataArray = imgDataArray.reshape((height, width, 4))

  # TODO: test if it works under copy=False; this may requires lifetime management with buffer protocol
  imgDataArray = np.array(imgData, copy=True)
  # print(imgDataArray)

  return fps, imgDataArray

def parseTrace(traceDataRaw, clearSeed):
  traceDataBytes = bytes(map(lambda x: ord(x), traceDataRaw))
  assert(len(traceDataBytes) == len(traceDataRaw))

  fmt = "<i"
  fmtSize = struct.calcsize(fmt)
  assert(fmtSize == 4)

  parsedTrace = []
  for i in range(0, len(traceDataBytes), fmtSize):
    parsedTrace.append(struct.unpack(fmt, traceDataBytes[i:i+fmtSize])[0] - clearSeed)
  
  # list[int]
  for elem in parsedTrace:
    assert(elem >= 0)

  return parsedTrace


def traceOnce(executor, uniform, nsecPerIncrement, width, height, clearSeed=0, clearSize=None):
  assert(clearSeed == 0)

  # TODO: store uniform into database, and read from
  imageUniform = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
  if uniform is not None:
    imageUniform = uniform
  else:
    imageUniform.iResolution = [width, height, 0]
    imageUniform.iTime = 1
    imageUniform.iFrame = 1
  
  executor.setUniform(imageUniform)

  # a simple smoke test mechanism
  if clearSeed == 0:
    executor.clearTraceBuffer()
  else:
    raise NotImplementedError("Implement me")

  executor.preRender()
  executor.render(1)
  imgData, tickUsed = executor.getResults()

  # TODO: test if it works under copy=False; this may requires lifetime management with buffer protocol
  imgDataArray = np.array(imgData, copy=True)
  
  fps = 1e9 / (nsecPerIncrement * tickUsed)

  traceDataRaw = executor.getTraceBuffer()
  traceData = parseTrace(traceDataRaw, clearSeed)

  return fps, imgDataArray, traceData

def testShader(shdrProxy, width, height, numCycles=100, numTrials=30, verbose=True):
  vertShdrSpv, fragShdrSpv = prepareImageOnlyShaders(shdrProxy)
  executor = buildExecutor(width, height, vertShdrSpv, fragShdrSpv)

  nsecPerIncrement, nsecValidRange = executor.getTimingParameters()

  # should be greater than 10s
  assert(nsecValidRange >= 10 * 1e9)

  trialFps = np.ndarray((numTrials,), dtype=np.float64)
  imgDataPrev = None

  pbar = tqdm(range(0, numTrials)) if verbose else range(0, numTrials)
  for trialIdx in pbar:
    fps, imgData = runOnce(executor, numCycles, None, nsecPerIncrement, width, height)
    trialFps[trialIdx] = fps * numCycles

    if imgDataPrev is None:
      imgDataPrev = imgData
    else:
      if not np.alltrue(np.equal(imgDataPrev, imgData)):
        raise ShaderResultInconsistencyException()

#   print(trialFps)
#   mean = np.mean(trialFps)
#   std = np.std(trialFps)
#   print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

#   trialFps = np.delete(trialFps, np.argmax(trialFps))
#   trialFps = np.delete(trialFps, np.argmin(trialFps))
#   print(trialFps)
#   print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

  return trialFps, imgDataPrev, fragShdrSpv
  # testStability(shdrProxy)

def traceShader(shdrProxy, width, height, validation=True):
  vertShdrSpv, fragShdrSpv = prepareImageOnlyShaders(shdrProxy)
  
  # inline & instrument trace
  spvProc = vkExecute.SpvProcessor()
  spvProc.loadSpv(fragShdrSpv)
  success, errMsg = spvProc.exhaustiveInlining()
  assert(success)
  fragInlinedSpv = spvProc.exportSpv()

  id2TraceIdxMap = spvProc.instrumentBasicBlockTrace()
  numBasicBlocks = len(id2TraceIdxMap)
  assert(len(id2TraceIdxMap) > 0)
  
  traceFragShdrSpv = spvProc.exportSpv()

  executor = buildExecutor(
    width, height, vertShdrSpv, traceFragShdrSpv,
    traceRun=True, traceBufferSize=numBasicBlocks * 4,
    traceBufferDescSet=5, traceBufferBinding=1
  )

  nsecPerIncrement, nsecValidRange = executor.getTimingParameters()

  # should be greater than 10s
  assert(nsecValidRange >= 10 * 1e9)

  fps, imgData, traceData = traceOnce(
    executor, None, nsecPerIncrement, width, height, clearSeed=0, clearSize=numBasicBlocks * 4
  )

  return {
    "fps": fps, "imgData": imgData, 
    "traceData": traceData,
    "id2TraceIdxMap": id2TraceIdxMap,
    "fragInlinedSpv": fragInlinedSpv,
    "traceFragShdrSpv": traceFragShdrSpv
  }
  
def disassemble(fragShdrSpv):
  sp = vkExecute.ShaderProcessor()
  sp.loadSpv(fragShdrSpv)
  disasm = sp.disassemble()

  return disasm
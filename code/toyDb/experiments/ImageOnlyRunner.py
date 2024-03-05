import databases.ShaderDb
import vkExecute
import dataclasses
import numpy as np
from typing import Union, Dict, List
import tqdm
import struct
import hashlib
import logging
import PIL.Image

import databases.ExperimentDb as ExperimentDb

logger = logging.getLogger(__name__)

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
  def __init__(self, error, auxError: str):
    super().__init__(error)
    self.auxError = auxError
    self.add_note(self.auxError)

class ShaderResultInconsistencyException(Exception):
  def __init__(self):
    super().__init__(f"Inconsistent result in running the given shader")

@dataclasses.dataclass
class ImageOnlyRunnerConfig:
  width: int
  height: int
  numCycles: int
  numTrails: int
  traceRun: bool
  traceBufferDescSet: Union[int, None]
  traceBufferBinding: Union[int, None]
  inlineBeforeRun: bool
  traceWithU64: bool

  @staticmethod
  def defaultNonTracedConfig(
    width=ExperimentDb.CANONICAL_WIDTH, height=ExperimentDb.CANONICAL_HEIGHT,
    numCycles=ExperimentDb.CANONICAL_NUM_CYCLES, numTrails=ExperimentDb.CANONICAL_NUM_TRIALS
    ):
    return ImageOnlyRunnerConfig(
      width, height, numCycles, numTrails, False, None, None, False, False
    )

  @staticmethod
  def defaultTracedConfig(
    width=ExperimentDb.CANONICAL_WIDTH, height=ExperimentDb.CANONICAL_HEIGHT, numCycles=1, numTrails=1,
    traceRun=True, traceBufferDescSet=5, traceBufferBinding=1,
    inlineBeforeRun=False, traceWithU64=False
    ):

    return ImageOnlyRunnerConfig(
      width, height, numCycles, numTrails,
      traceRun, traceBufferDescSet, traceBufferBinding, inlineBeforeRun, traceWithU64
    )
  

@dataclasses.dataclass
class ImageOnlyRunnerResult:
  trialTime: np.ndarray
  imgData: np.ndarray
  traceData: List[int]
  id2TraceIdxMap: Dict[int, int]
  fragShdrSpv: List[str]
  traceFragShdrSpv: List[str]

vkExecuteImageOnlyExecutorInitialized = False
vkExecuteImageOnlyExecutorInitializeOptions = None

def initImageOnlyExecutor(
    executor: 'vkExecute.ImageOnlyExecutor',
    forceEnableValidations: bool,
    u64Supported: bool):
  """ALL calls to vkExecute are supposed to use this to ensure no decay situations"""
  global vkExecuteImageOnlyExecutorInitialized
  global vkExecuteImageOnlyExecutorInitializeOptions
  if vkExecuteImageOnlyExecutorInitialized:
    # Check compatibility
    assert(u64Supported == False or vkExecuteImageOnlyExecutorInitializeOptions[1] == True)
    freshInst = executor.init(forceEnableValidations, u64Supported)

    assert(not freshInst)
  else:
    vkExecuteImageOnlyExecutorInitialized = True
    vkExecuteImageOnlyExecutorInitializeOptions = (forceEnableValidations, u64Supported)
    freshInst = executor.init(forceEnableValidations, u64Supported)

    # Or somewhere else might have called this and we're not tracking this, bad!
    assert(freshInst)

class ImageOnlyRunner:
  def __init__(self, config: ImageOnlyRunnerConfig):
    self.config = config
    self.vertShaderSpv = None
    self.fragShaderSpv = None
    self.uniformBlock = None

    # Useful for trace run
    self.traceFragShaderSpv = None
    self.id2TraceIdxMap = None
    self.numBasicBlocks = None
    self.traceBufferSize = None

  def loadShader(self, shdrProxy: databases.ShaderDb.ShaderProxy):
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

    self.vertShaderSpv = vertShaderSpv

    if self.config.inlineBeforeRun:
      spvProc = vkExecute.SpvProcessor()
      spvProc.loadSpv(fragShaderSpv)
      success, errMsg = spvProc.exhaustiveInlining()
      if not success or len(fragInlinedSpv) == 0:
        raise ShaderCompilationException("Error while inlining fragment shader", errMsg)
      fragInlinedSpv = spvProc.exportSpv()

      self.fragShaderSpv = fragInlinedSpv
    else:
      self.fragShaderSpv = fragShaderSpv

    if self.config.traceRun:
      spvProc = vkExecute.SpvProcessor()
      spvProc.loadSpv(self.fragShaderSpv)

      self.id2TraceIdxMap = spvProc.instrumentBasicBlockTrace(self.config.traceWithU64)
      self.numBasicBlocks = len(self.id2TraceIdxMap)

      if self.config.traceWithU64:
        self.traceBufferSize = self.numBasicBlocks * 8
      else:
        self.traceBufferSize = self.numBasicBlocks * 4

      if not len(self.id2TraceIdxMap) > 0:
        raise ShaderCompilationException("Error while instrumenting fragment shader", "Zero length detected")

      self.traceFragShaderSpv = spvProc.exportSpv()

  def exportShader(self):
    return self.fragShaderSpv

  def loadShaderFromDb(self, fragShaderSpv: List[str]):
    vertShaderSpv, vertErrMsg = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
      vkExecute.ShaderProcessor.ShaderStages.VERTEX,
      vertShdrSrc,
      "VertShader"
    )

    if len(vertShaderSpv) == 0:
      raise ShaderCompilationException("Error while compiling vertex shader", vertErrMsg)

    self.vertShaderSpv = vertShaderSpv

    if self.config.inlineBeforeRun:
      spvProc = vkExecute.SpvProcessor()
      spvProc.loadSpv(fragShaderSpv)
      success, errMsg = spvProc.exhaustiveInlining()
      if not success or len(fragInlinedSpv) == 0:
        raise ShaderCompilationException("Error while inlining fragment shader", errMsg)
      fragInlinedSpv = spvProc.exportSpv()

      self.fragShaderSpv = fragInlinedSpv
    else:
      self.fragShaderSpv = fragShaderSpv

    if self.config.traceRun:
      spvProc = vkExecute.SpvProcessor()
      spvProc.loadSpv(self.fragShaderSpv)

      self.id2TraceIdxMap = spvProc.instrumentBasicBlockTrace(self.config.traceWithU64)
      self.numBasicBlocks = len(self.id2TraceIdxMap)

      if self.config.traceWithU64:
        self.traceBufferSize = self.numBasicBlocks * 8
      else:
        self.traceBufferSize = self.numBasicBlocks * 4

      if not len(self.id2TraceIdxMap) > 0:
        raise ShaderCompilationException("Error while instrumenting fragment shader", "Zero length detected")

      self.traceFragShaderSpv = spvProc.exportSpv()

  def _buildExecutor(self) -> 'vkExecute.ImageOnlyExecutor':
    pCfg = vkExecute.ImageOnlyExecutor.PipelineConfig()
    pCfg.targetWidth = self.config.width
    pCfg.targetHeight = self.config.height
    pCfg.vertexShader = self.vertShaderSpv

    if self.config.traceRun:
      pCfg.fragmentShader = self.traceFragShaderSpv
    else:
      pCfg.fragmentShader = self.fragShaderSpv
    
    if self.config.traceRun:
      assert(self.traceBufferSize is not None)
      pCfg.traceRun = True
      pCfg.traceBufferSize = self.traceBufferSize
      pCfg.traceBufferDescSet = self.config.traceBufferDescSet
      pCfg.traceBufferBinding = self.config.traceBufferBinding

    executor = vkExecute.ImageOnlyExecutor()
    initImageOnlyExecutor(executor, True, self.config.traceRun and self.config.traceWithU64)
    executor.initPipeline(pCfg)

    return executor

  def fillUniform(self, iTime=1, iFrame=1) -> 'bytes':
    self.uniformBlock = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
    self.uniformBlock.iResolution = [self.config.width, self.config.height, 0]
    self.uniformBlock.iTime = iTime
    self.uniformBlock.iFrame = iFrame

    return self.uniformBlock.exportAsBytes()

  # TODO: support for clearSeed
  def _parseTrace(self, traceDataRaw):
    traceDataBytes = bytes(map(lambda x: ord(x), traceDataRaw))
    assert(len(traceDataBytes) == len(traceDataRaw))

    if self.config.traceWithU64:
      fmt = "<Q"
    else:
      fmt = "<I"

    fmtSize = struct.calcsize(fmt)
    if self.config.traceWithU64:
      assert(fmtSize == 8)
    else:
      assert(fmtSize == 4)

    parsedTrace = []
    for i in range(0, len(traceDataBytes), fmtSize):
      parsedTrace.append(struct.unpack(fmt, traceDataBytes[i:i+fmtSize])[0])
    
    # list[int]
    for elem in parsedTrace:
      assert(elem >= 0)

    return parsedTrace

  def _runOnce(self, executor: 'vkExecute.ImageOnlyExecutor', nsecPerIncrement: float):
    executor.setUniform(self.uniformBlock)

    if self.config.traceRun:
      executor.clearTraceBuffer()

    executor.preRender()
    executor.render(self.config.numCycles)
    imgData, tickUsed = executor.getResults()

    # unit: sec.
    totalTimeUsed = (nsecPerIncrement * tickUsed) / 1e9

    # TODO: test if it works under copy=False; this may requires lifetime management with buffer protocol
    imgDataArray = np.array(imgData, copy=True)

    traceData = None
    if self.config.traceRun:
      traceDataRaw = executor.getTraceBuffer()
      traceData = self._parseTrace(traceDataRaw)

    return totalTimeUsed, imgDataArray, traceData

  @staticmethod
  def getImageHash(imgData: np.ndarray):
    """SHA256"""
    bytes = imgData.tobytes()
    return hashlib.sha256(bytes).hexdigest()
  
  @staticmethod
  def saveImage(fileName, imgData):
    """imgData: shape (w, h, 4) of int32, 0-255 each"""
    im = PIL.Image.fromarray(imgData, mode='RGBA')
    im.save(fileName)

  def run(self, checkImageConsistency=True, verbose=False):
    assert(self.vertShaderSpv is not None and self.fragShaderSpv is not None)
    executor = self._buildExecutor()

    nsecPerIncrement, nsecValidRange = executor.getTimingParameters()

    # should be greater than 10s
    assert(nsecValidRange >= 10 * 1e9)

    if self.config.traceRun:
      assert(self.config.numTrails == 1)
      assert(self.config.numCycles == 1)

    trialTime = np.ndarray((self.config.numTrails,), dtype=np.float64)
    imgDataPrev = None

    if verbose:
      pbar = tqdm.tqdm(range(0, self.config.numTrails))
    else:
      pbar = range(0, self.config.numTrails)

    for trialIdx in pbar:
      totalTimeUsed, imgData, traceData = self._runOnce(executor, nsecPerIncrement)
      trialTime[trialIdx] = totalTimeUsed / self.config.numCycles

      if checkImageConsistency:
        if imgDataPrev is None:
          imgDataPrev = imgData
        else:
          if not np.alltrue(np.equal(imgDataPrev, imgData)):
            raise ShaderResultInconsistencyException()

    return ImageOnlyRunnerResult(
      trialTime=trialTime,
      imgData=imgData,
      traceData=traceData,
      id2TraceIdxMap=self.id2TraceIdxMap,
      fragShdrSpv=self.fragShaderSpv,
      traceFragShdrSpv=self.traceFragShaderSpv
    )


#   print(trialFps)
#   mean = np.mean(trialFps)
#   std = np.std(trialFps)
#   print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

#   trialFps = np.delete(trialFps, np.argmax(trialFps))
#   trialFps = np.delete(trialFps, np.argmin(trialFps))
#   print(trialFps)
#   print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

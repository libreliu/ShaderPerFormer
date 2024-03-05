import vkExecute
from .. import ShaderDB
import os
import numpy as np
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

def prepareImageOnlyShaders(shdrProxy: ShaderDB.ShaderProxy):
  fragShdrSrc = commonFragShdrPremable
  commonPass = shdrProxy.get_renderpass("common")
  imagePass = shdrProxy.get_renderpass("image")

  if commonPass is not None:
    fragShdrSrc += commonPass["code"]
  
  fragShdrSrc += imagePass["code"]

  vertShaderSpv = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
    vkExecute.ShaderProcessor.ShaderStages.VERTEX,
    vertShdrSrc,
    "VertShader"
  )

  fragShaderSpv = vkExecute.ShaderProcessor.compileShaderToSPIRV_Vulkan(
    vkExecute.ShaderProcessor.ShaderStages.FRAGMENT,
    fragShdrSrc,
    "FragShader"
  )

  return vertShaderSpv, fragShaderSpv

def buildExecutor(width, height, vertShaderSpv, fragShaderSpv):
  pCfg = vkExecute.ImageOnlyExecutor.PipelineConfig()
  pCfg.targetWidth = width
  pCfg.targetHeight = height
  pCfg.vertexShader = vertShaderSpv
  pCfg.fragmentShader = fragShaderSpv

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
    imageUniform.iResolution = [1024, 768, 0]
    imageUniform.iTime = 1
    imageUniform.iFrame = 1

  executor.setUniform(imageUniform)
  executor.preRender()
  executor.render(cycles)
  imgData, tickUsed = executor.getResults()

  fps = 1e9 / (nsecPerIncrement * tickUsed)

  imgDataArray = np.asarray([i for i in map(lambda x: ord(x), imgData.data)], dtype=np.int32)
  imgDataArray = imgDataArray.reshape((height, width, 4))

  return fps, imgDataArray

def testStability(shdrProxy, args):
  vertShdrSpv, fragShdrSpv = prepareImageOnlyShaders(shdrProxy)
  executor = buildExecutor(args.width, args.height, vertShdrSpv, fragShdrSpv)

  nsecPerIncrement, nsecValidRange = executor.getTimingParameters()

  numCycles = 50
  numTrials = 30
  trialFps = np.ndarray((numTrials,), dtype=np.float64)
  for trialIdx in tqdm(range(0, numTrials)):
    fps, imgData = runOnce(executor, numCycles, None, nsecPerIncrement, args.width, args.height)
    trialFps[trialIdx] = fps * numCycles

  print(trialFps)
  print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

  trialFps = np.delete(trialFps, np.argmax(trialFps))
  trialFps = np.delete(trialFps, np.argmin(trialFps))
  print(trialFps)
  print(f"mean = {np.mean(trialFps)} ± {np.std(trialFps)}")

  # testStability(shdrProxy)
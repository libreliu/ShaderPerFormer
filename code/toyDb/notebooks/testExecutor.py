import vkExecute

vertShdrSrc = """#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;
layout(location = 0) in vec3 inPosition;
void main() {gl_Position = vec4(inPosition, 1.0);}
"""

fragShdrSrc = """#version 310 es
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

float Circle( vec2 uv, vec2 p, float r, float blur )
{
	float d = length(uv - p);
  float c = smoothstep(r, r-blur, d);
  return c;
}

float Hash( float h )
{
  return h = fract(cos(h) * 5422.2465);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
  float c = 0.0;
    
  uv -= .5;
  uv.x *= iResolution.x / iResolution.y;
  float sizer = 1.0;
  float steper = .1;
  for(float i = -sizer; i<sizer; i+=steper)
    for(float j = -sizer; j<sizer; j+=steper)
    {	
      float timer = .5;
      float resetTimer = 7.0;
      if(c<=1.0){
        c += Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));
      }
      else if(c>=1.0)
      {
        c -= Circle(uv, vec2(i, j),sin(Hash(i))*cos(Hash(j))*(mod(iTime*timer, resetTimer)), sin(Hash(j)));     
      }
    }
  fragColor = vec4(vec3(c),1.0);
}
"""

# compile shader
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

print(f"Compiled: Vert={len(vertShaderSpv)} bytes, Frag={len(fragShaderSpv)} bytes")

pCfg = vkExecute.ImageOnlyExecutor.PipelineConfig()
pCfg.targetWidth = 1024
pCfg.targetHeight = 768
pCfg.vertexShader = vertShaderSpv
pCfg.fragmentShader = fragShaderSpv

executor = vkExecute.ImageOnlyExecutor()
executor.init(True)
executor.initPipeline(pCfg)

nsecPerIncrement, nsecValidRange = executor.getTimingParameters()
print(f"nsecPerIncr: {nsecPerIncrement}, validRange: {nsecValidRange / 1e9} sec")

imageUniform = vkExecute.ImageOnlyExecutor.ImageUniformBlock()
# TODO: why iResolution[0] = 1024 things doesn't work?
imageUniform.iResolution = [1024, 768, 0]
print(imageUniform.iResolution)
imageUniform.iTime = 1
imageUniform.iFrame = 1

executor.setUniform(imageUniform)
executor.preRender()
executor.render(100)
imgData, tickUsed = executor.getResults()

print(f"tickUsed: {tickUsed} ({1e9 / (nsecPerIncrement * tickUsed)} fps)")
# print(imgData.data)
# print(ord(imgData.data[0]))

import numpy as np
imgDataArray = np.asarray([i for i in map(lambda x: ord(x), imgData.data)], dtype=np.int32)
imgDataArray = imgDataArray.reshape((768, 1024, 4))

# print(imgDataArray)

from matplotlib import pyplot as plt
plt.imshow(imgDataArray, interpolation='nearest')
plt.show()


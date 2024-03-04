#include <cstdio>
#include <iostream>
#include <ostream>
#include <tuple>

#include "Common.hpp"
#include "ImageOnlyExecutor.hpp"
#include "ShaderProcessor.hpp"
#include "SpvProcessor.hpp"
#include "image_data.hpp"

static const char *vertexShaderSrc = R"shdrSrc(#version 310 es
precision highp float;
precision highp int;
precision mediump sampler3D;
layout(location = 0) in vec3 inPosition;
void main() {gl_Position = vec4(inPosition, 1.0);}
)shdrSrc";

static const char *fragmentShaderSrc = R"shdrSrc(#version 310 es
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
)shdrSrc";

using namespace vkExecute;

std::tuple<BinaryBlob, RGBAUIntImageBlob> testExecutor() {
  auto vertShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::VERTEX, vertexShaderSrc, "VertShader");

  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderSrc, "FragShader");

  auto pCfg = ImageOnlyExecutor::PipelineConfig();
  pCfg.targetWidth = 1024;
  pCfg.targetHeight = 768;
  pCfg.vertexShader = std::move(std::get<BinaryBlob>(vertShaderSpvTuple));
  pCfg.fragmentShader = std::move(std::get<BinaryBlob>(fragShaderSpvTuple));

  auto executor = ImageOnlyExecutor();
  executor.init(true, false);

  // This won't move shader blob away according to current implementation
  executor.initPipeline(pCfg);

  auto imageUniform = ImageOnlyExecutor::ImageUniformBlock();
  imageUniform.iResolution = {1024, 768, 0};
  imageUniform.iTime = 1;
  imageUniform.iFrame = 1;

  executor.setUniform(imageUniform);
  executor.preRender();
  executor.render(10);

  auto imgData = executor.getResults();

  return std::make_tuple<BinaryBlob, RGBAUIntImageBlob>(
    std::move(pCfg.fragmentShader),
    std::move(std::get<0>(imgData))
  );
}

void testSpvProcessor(BinaryBlob fragShaderSpv) {
  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(fragShaderSpv);
  
  std::cerr << "Original:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  printErrors(spvProc.exhaustiveInlining());

  std::cerr << "Inlined:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;
  
  auto bbs = spvProc.separateBasicBlocks();

  for (auto &bb: bbs) {
    bb.dump();
    std::cerr << std::endl;
  }
}

int main(int argc, char *argv[]) {

  auto result = testExecutor();

  testSpvProcessor(
    std::get<0>(result)
  );

  return 0;
}
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <ostream>
#include <tuple>

#include "Common.hpp"
#include "ImageOnlyExecutor.hpp"
#include "ShaderProcessor.hpp"
#include "SpvProcessor.hpp"
#include "image_data.hpp"
#include "ShadertoyCircle.inc.h"

using namespace vkExecute;

TEST_CASE("vkExecute executor test") {
  auto vertShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::VERTEX, vertexShaderSrc, "VertShader");

  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderSrc, "FragShader");

  REQUIRE(std::get<BinaryBlob>(vertShaderSpvTuple).size() > 0);
  REQUIRE(std::get<BinaryBlob>(fragShaderSpvTuple).size() > 0);

  auto pCfg = ImageOnlyExecutor::PipelineConfig();
  pCfg.targetWidth = 1024;
  pCfg.targetHeight = 768;
  pCfg.vertexShader = std::move(std::get<BinaryBlob>(vertShaderSpvTuple));
  pCfg.fragmentShader = std::move(std::get<BinaryBlob>(fragShaderSpvTuple));

  auto executor = ImageOnlyExecutor();
  executor.init(true, true);

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

  auto res = std::make_tuple<BinaryBlob, RGBAUIntImageBlob>(
    std::move(pCfg.fragmentShader),
    std::move(std::get<0>(imgData))
  );
}

TEST_CASE("vkExecute compile and run with trace test") {
  auto vertShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::VERTEX, vertexShaderSrc, "VertShader");

  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderTestLargeU64Src, "FragShader");

  REQUIRE(std::get<BinaryBlob>(vertShaderSpvTuple).size() > 0);
  REQUIRE(std::get<BinaryBlob>(fragShaderSpvTuple).size() > 0);

  BinaryBlob fragShaderSpv = std::move(std::get<0>(fragShaderSpvTuple));

  // prepare trace fragment
  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(fragShaderSpv);

  REQUIRE(printErrors(spvProc.exhaustiveInlining()) == true);
  REQUIRE(printErrors(spvProc.validate()) == true);

  auto id2TraceIdxU32 = spvProc.instrumentBasicBlockTrace(false);

  REQUIRE(printErrors(spvProc.validate()) == true);
  auto processedFragShaderSpv = spvProc.exportSpv();

  auto pCfg = ImageOnlyExecutor::PipelineConfig();
  pCfg.targetWidth = 1024;
  pCfg.targetHeight = 768;
  pCfg.vertexShader = std::move(std::get<BinaryBlob>(vertShaderSpvTuple));
  pCfg.fragmentShader = std::move(processedFragShaderSpv);
  pCfg.traceRun = true;
  pCfg.traceBufferDescSet = 5;
  pCfg.traceBufferBinding = 1;
  pCfg.traceBufferSize = id2TraceIdxU32.size() * sizeof(int);

  auto executor = ImageOnlyExecutor();
  executor.init(true, true);

  // This won't move shader blob away according to current implementation
  executor.initPipeline(pCfg);

  auto imageUniform = ImageOnlyExecutor::ImageUniformBlock();
  imageUniform.iResolution = {1024, 768, 0};
  imageUniform.iTime = 1;
  imageUniform.iFrame = 1;

  executor.setUniform(imageUniform);
  executor.clearTraceBuffer();
  executor.preRender();
  executor.render(1);

  auto imgData = executor.getResults();

  auto res = std::make_tuple<BinaryBlob, RGBAUIntImageBlob>(
    std::move(pCfg.fragmentShader),
    std::move(std::get<0>(imgData))
  );

  auto traceData = executor.getTraceBuffer();
  std::vector<uint32_t> traceDecoded(id2TraceIdxU32.size(), 0);
  for (int i = 0; i < traceDecoded.size(); i++) {
    traceDecoded[i] = *reinterpret_cast<uint32_t *>(traceData.data() + i * sizeof(uint32_t));
  }

  std::cerr << ToString(traceDecoded) << std::endl;
}

TEST_CASE("vkExecute compile and run with trace test U64") {
  auto vertShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::VERTEX, vertexShaderSrc, "VertShader");

  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderTestLargeU64Src, "FragShader");

  REQUIRE(std::get<BinaryBlob>(vertShaderSpvTuple).size() > 0);
  REQUIRE(std::get<BinaryBlob>(fragShaderSpvTuple).size() > 0);

  BinaryBlob fragShaderSpv = std::move(std::get<0>(fragShaderSpvTuple));

  // prepare trace fragment
  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(fragShaderSpv);

  REQUIRE(printErrors(spvProc.exhaustiveInlining()) == true);
  REQUIRE(printErrors(spvProc.validate()) == true);

  auto id2TraceIdxU64 = spvProc.instrumentBasicBlockTrace(true);

  REQUIRE(printErrors(spvProc.validate()) == true);
  auto processedFragShaderSpv = spvProc.exportSpv();

  auto pCfg = ImageOnlyExecutor::PipelineConfig();
  pCfg.targetWidth = 1024;
  pCfg.targetHeight = 768;
  pCfg.vertexShader = std::move(std::get<BinaryBlob>(vertShaderSpvTuple));
  pCfg.fragmentShader = std::move(processedFragShaderSpv);
  pCfg.traceRun = true;
  pCfg.traceBufferDescSet = 5;
  pCfg.traceBufferBinding = 1;
  pCfg.traceBufferSize = id2TraceIdxU64.size() * sizeof(uint64_t);

  auto executor = ImageOnlyExecutor();
  executor.init(true, true);

  // This won't move shader blob away according to current implementation
  executor.initPipeline(pCfg);

  auto imageUniform = ImageOnlyExecutor::ImageUniformBlock();
  imageUniform.iResolution = {1024, 768, 0};
  imageUniform.iTime = 1;
  imageUniform.iFrame = 1;

  executor.setUniform(imageUniform);
  executor.clearTraceBuffer();
  executor.preRender();
  executor.render(1);

  auto imgData = executor.getResults();

  auto res = std::make_tuple<BinaryBlob, RGBAUIntImageBlob>(
    std::move(pCfg.fragmentShader),
    std::move(std::get<0>(imgData))
  );

  auto traceData = executor.getTraceBuffer();
  std::vector<uint64_t> traceDecoded(id2TraceIdxU64.size(), 0);
  for (int i = 0; i < traceDecoded.size(); i++) {
    traceDecoded[i] = *reinterpret_cast<uint64_t *>(traceData.data() + i * sizeof(uint64_t));
  }

  std::cerr << ToString(traceDecoded) << std::endl;
}


TEST_CASE("vkExecute driver version report test") {
  auto executor = ImageOnlyExecutor();
  executor.init(true, true);

  std::cerr << "Device name: " << executor.getDeviceName() << std::endl;
  std::cerr << "Driver description: " << executor.getDriverDescription() << std::endl;
}

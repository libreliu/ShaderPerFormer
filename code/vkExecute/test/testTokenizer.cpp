#include <catch2/catch_test_macros.hpp>

#include <cstdio>
#include <iostream>
#include <ostream>
#include <tuple>

#include "Common.hpp"
#include "ImageOnlyExecutor.hpp"
#include "ShaderProcessor.hpp"
#include "SpvProcessor.hpp"
#include "image_data.hpp"
#include "spv/Tokenizer.hpp"
#include "ShadertoyCircle.inc.h"


using namespace vkExecute;
using namespace vkExecute::spv;

TEST_CASE("vkExecute Tokenizer test") {
  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderSrc, "FragShader");

  BinaryBlob fragShaderSpv = std::move(std::get<0>(fragShaderSpvTuple));

  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(fragShaderSpv);

  std::cerr << "Original:" << std::endl;
  auto disassembled = printErrors(spvProc.disassemble());
  REQUIRE(disassembled.size() > 0);

  std::cerr << disassembled;
  std::cerr << std::endl;

  bool res = printErrors(spvProc.exhaustiveInlining());
  std::cerr << "Status: " << (res ? "True" : "False") << std::endl;
  REQUIRE(res == true);

  auto inlinedSpv = spvProc.exportSpv();
  REQUIRE(inlinedSpv.size() > 0);

  std::cerr << "Inlined:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  REQUIRE(printErrors(spvProc.validate()) == true);

  // test for relative id pos
  Tokenizer tokenizer(true, true, true, true);
  tokenizer.loadSpv(inlinedSpv);

  auto tokenVector = printErrors(tokenizer.tokenize());
  REQUIRE(tokenVector.size() > 9);

  std::cerr << "Got " << tokenVector.size() << " tokens" << std::endl;
  auto detokenizeResult = tokenizer.deTokenize(tokenVector);
  REQUIRE(detokenizeResult.size() > 0);

  std::cerr << detokenizeResult << std::endl; 
}

TEST_CASE("vkExecute TokenizerWithTrace test") {
  auto fragShaderSpvTuple = ShaderProcessor::compileShaderToSPIRV_Vulkan(
      ShaderProcessor::ShaderStages::FRAGMENT, fragmentShaderSrc, "FragShader");

  BinaryBlob fragShaderSpv = std::move(std::get<0>(fragShaderSpvTuple));

  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(fragShaderSpv);

  std::cerr << "Original:" << std::endl;
  auto disassembled = printErrors(spvProc.disassemble());
  REQUIRE(disassembled.size() > 0);

  std::cerr << disassembled;
  std::cerr << std::endl;

  bool res = printErrors(spvProc.exhaustiveInlining());
  std::cerr << "Status: " << (res ? "True" : "False") << std::endl;
  REQUIRE(res == true);

  auto inlinedSpv = spvProc.exportSpv();
  REQUIRE(inlinedSpv.size() > 0);

  std::cerr << "Inlined:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  REQUIRE(printErrors(spvProc.validate()) == true);

  // instrument to get a valid id2TraceIdx map & traceData array
  auto id2TraceIdxU32 = spvProc.instrumentBasicBlockTrace(false);
  std::cerr << "id2TraceIdx: " << ToString(id2TraceIdxU32) << std::endl;

  size_t traceCount = id2TraceIdxU32.size();
  std::vector<vkExecute::TraceCounter_t> traceData;
  for (size_t i = 0; i < traceCount; i++) {
    traceData.push_back(i + 1);
  }

  Tokenizer tokenizer(true, true, true, true);
  tokenizer.loadSpv(inlinedSpv);

  auto tokenized = printErrorsTuple(tokenizer.tokenizeWithTrace(id2TraceIdxU32, traceData));
  auto tokenVector = std::move(std::get<0>(tokenized));
  auto tokenTraceVector = std::move(std::get<1>(tokenized));

  REQUIRE(tokenVector.size() > 9);
  REQUIRE(tokenTraceVector.size() == tokenVector.size());

  std::cerr << "Got " << tokenVector.size() << " tokens" << std::endl;
  auto detokenizeResult = tokenizer.deTokenize(tokenVector);
  REQUIRE(detokenizeResult.size() > 0);

  std::cerr << detokenizeResult << std::endl; 

  std::cerr << "Got " << tokenTraceVector.size() << " trace tokens" << std::endl;
  std::cerr << ToString(tokenTraceVector) << std::endl;
}
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
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"
#include "spv/ModuleBuilder.hpp"
#include "spv/Type.hpp"

#include "ShadertoyCircle.inc.h"

using namespace vkExecute;
using namespace vkExecute::spv;

TEST_CASE("vkExecute SpvProcessor test") {
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

  std::cerr << "Inlined:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  REQUIRE(printErrors(spvProc.validate()) == true);

  auto bbs = spvProc.separateBasicBlocks();

  for (auto &bb : bbs) {
    bb.dump();
    std::cerr << std::endl;
  }
}

// %328 = OpLoad %6 %212
// %329 = OpLoad %6 %213
// %330 = OpFAdd %6 %329 %328
// OpStore %213 %330

TEST_CASE("vkExecute basic block emit test") {
  ModuleBuilder mBuilder;

  uint32_t extInstId = mBuilder.getOrAddExtInstImport(u8"GLSL.std.450");

  // prepare main func
  Type::Void typeVoid;
  Type::Function mainFuncType{typeVoid};

  // prepare PrimaryUBO type
  Type::Float typeFloat(32);
  Type::Integer typeUInt(32, 0);
  Type::Integer typeInt(32, 1);
  Type::Vector typeVec3(3, typeFloat);
  Type::Vector typeVec4(4, typeFloat);
  Type::Array typeFloat4(
      4, typeFloat, 16); // DON'T KNOW WHY, but glslang generates ArrayStride 16
  Type::Array typeVec34(4, typeVec3, 16);
  Type::Struct typePrimaryUBO;
  typePrimaryUBO.addMember(typeVec3, 0);    // vec3 iResolution
  typePrimaryUBO.addMember(typeFloat, 12);  // float iTime
  typePrimaryUBO.addMember(typeFloat4, 16); // float iChannelTime[4]
  typePrimaryUBO.addMember(typeVec4, 80);   // vec4 iMouse
  typePrimaryUBO.addMember(typeVec4, 96);   // vec4 iDate
  typePrimaryUBO.addMember(typeFloat, 112); // float iSampleRate
  typePrimaryUBO.addMember(typeVec34, 128); // vec3 iChannelResolution[4]
  typePrimaryUBO.addMember(typeInt, 192);   // int iFrame
  typePrimaryUBO.addMember(typeFloat, 196); // float iTimeDelta
  typePrimaryUBO.addMember(typeFloat, 200); // float iFrameRate
  Type::Pointer ptrTypePrimaryUBO((uint32_t)::spv::StorageClass::Uniform,
                                  typePrimaryUBO);

  uint32_t pTypeUBOId = mBuilder.getOrAddType(&ptrTypePrimaryUBO);
  // prepare primary uniform variable
  uint32_t pUBOId = mBuilder.addGlobalVariable(
      &ptrTypePrimaryUBO, (uint32_t)::spv::StorageClass::Uniform);

  // prepare shader input and output variable
  Type::Pointer ptrOutTypeVec4((uint32_t)::spv::StorageClass::Output, typeVec4);
  Type::Pointer ptrInTypeVec4((uint32_t)::spv::StorageClass::Input, typeVec4);
  uint32_t pOutColorId = mBuilder.addGlobalVariable(
      &ptrOutTypeVec4, (uint32_t)::spv::StorageClass::Output);
  uint32_t pFragCoordId = mBuilder.addGlobalVariable(
      &ptrInTypeVec4, (uint32_t)::spv::StorageClass::Input);

  mBuilder.addVariableDecoration(pFragCoordId, {
    Operand::create(SPV_OPERAND_TYPE_DECORATION, {(uint32_t)::spv::Decoration::BuiltIn}),
    Operand::create(SPV_OPERAND_TYPE_BUILT_IN, {(uint32_t)::spv::BuiltIn::FragCoord})
  });
  mBuilder.addVariableDecoration(pOutColorId, {
    Operand::create(SPV_OPERAND_TYPE_DECORATION, {(uint32_t)::spv::Decoration::Location}),
    Operand::createLiteralInteger(0)
  });
  mBuilder.addVariableDecoration(pTypeUBOId, {
    Operand::create(SPV_OPERAND_TYPE_DECORATION, {(uint32_t)::spv::Decoration::Block})
  });
  mBuilder.addVariableDecoration(pUBOId, {
    Operand::create(SPV_OPERAND_TYPE_DECORATION, {(uint32_t)::spv::Decoration::DescriptorSet}),
    Operand::createLiteralInteger(0)
  });
  mBuilder.addVariableDecoration(pUBOId, {
    Operand::create(SPV_OPERAND_TYPE_DECORATION, {(uint32_t)::spv::Decoration::Binding}),
    Operand::createLiteralInteger(0)
  });

  // emit main func content
  uint32_t mainFuncId = mBuilder.emitFunctionBegin(&mainFuncType);
  mBuilder.emitFunctionEnd();

  mBuilder.finalize(mainFuncId, u8"main");

  std::vector<uint32_t> spirvOut = mBuilder.getSpv();

  auto spvProc = vkExecute::SpvProcessor();
  spvProc.loadSpv(toBinaryBlob(spirvOut));
  std::cerr << "Emitted:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;
}

TEST_CASE("vkExecute instrument trace test") {
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

  std::cerr << "Inlined:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  REQUIRE(printErrors(spvProc.validate()) == true);

  auto id2TraceIdxU32 = spvProc.instrumentBasicBlockTrace(false);
  REQUIRE(id2TraceIdxU32.size() > 0);

  auto id2TraceIdxU64 = spvProc.instrumentBasicBlockTrace(true);
  REQUIRE(id2TraceIdxU64.size() > 0);

  std::cerr << "Basic block instrumented:" << std::endl;
  std::cerr << printErrors(spvProc.disassemble());
  std::cerr << std::endl;

  std::cerr << "id2TraceIdxU32 array:" << std::endl;
  std::cerr << ToString(id2TraceIdxU32);
  std::cerr << std::endl;

  std::cerr << "id2TraceIdxU64 array:" << std::endl;
  std::cerr << ToString(id2TraceIdxU64);
  std::cerr << std::endl;
}
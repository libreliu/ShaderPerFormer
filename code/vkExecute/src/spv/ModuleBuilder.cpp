#include "spv/ModuleBuilder.hpp"

#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp"
#include "spirv/unified1/spirv.hpp11"
#include "spv/BasicBlock.hpp"
#include "spv/Type.hpp"
#include "spv/TypeBuilder.hpp"
#include <assimp/scene.h>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>

using namespace vkExecute::spv;

// always assume little endian for now
// TODO: check SPIRV-Tools on endianness

// Ref:
// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_physical_layout_of_a_spir_v_module_and_instruction
ModuleBuilder::SpvBlob ModuleBuilder::emitSpirvPremable() {
  SpvBlob spv;

  // 3.1 magic number
  spv.push_back(0x07230203);
  // version number: 1.6
  spv.push_back(0x00010600);
  // generator number: 0
  spv.push_back(0x0);

  // 0 < id < bound
  spv.push_back(cur_id + 1);
  // instruction schema
  spv.push_back(0);

  return spv;
}

// NOTE: check conditions on resultid and typeid
// see build/vkExecute/thirdParty/SPIRV-Tools/core.insts-unified1.inc for more
// info

// Ref:
// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_logical_layout_of_a_module
void ModuleBuilder::emitModulePremable(uint32_t entrypointId,
                                       std::u8string entrypointName) {

  // clang-format off
  // 1. OpCapability
  capabilities.push_back(Instr::create(::spv::Op::OpCapability, false, false, {
    Operand::create(SPV_OPERAND_TYPE_CAPABILITY, {(uint32_t)::spv::Capability::Shader}),
  }));

  // 2. OpExtension
  // 3. OpExtInstImport
  emitExtInstImports();

  // 4. OpMemoryModel
  memoryModel.push_back(Instr::create(::spv::Op::OpMemoryModel, false, false, {
    Operand::create(SPV_OPERAND_TYPE_ADDRESSING_MODEL, {(uint32_t)::spv::AddressingModel::Logical}),
    Operand::create(SPV_OPERAND_TYPE_MEMORY_MODEL, {(uint32_t)::spv::MemoryModel::GLSL450})
  }));

  // 5. OpEntryPoint
  entryPoints.push_back(Instr::create(::spv::Op::OpEntryPoint, false, false, {
    Operand::create(SPV_OPERAND_TYPE_EXECUTION_MODEL, {(uint32_t)::spv::ExecutionModel::Fragment}),
    Operand::createId(entrypointId),
    Operand::createLiteralString(entrypointName)
  }));

  // 6. OpExecutionMode / OpExecutionModeId
  executionModes.push_back(Instr::create(::spv::Op::OpExecutionMode, false, false, {
    Operand::createId(entrypointId),
    Operand::create(SPV_OPERAND_TYPE_EXECUTION_MODE, {(uint32_t)::spv::ExecutionMode::OriginUpperLeft})
  }));

  // 7. debug instructions
  // 8. annotation instruction (all decoration instructions)
  // 9. All type declarations
  //    All constant instructions
  //    All global variable declarations
  // 10. All function declarations
  // 11. All function definitions

  // clang-format on
}

void ModuleBuilder::emitExtInstImports() {
  for (auto &extPair : extInstImportIds) {
    extInstImports.push_back(
        Instr::create(::spv::Op::OpExtInstImport, true, false,
                      {Operand::createResultId(extPair.first),
                       Operand::createLiteralString(extPair.second)}));
  }
}

uint32_t ModuleBuilder::getOrAddType(const Type::Type *type) {
  // search for type
  for (size_t i = 0; i < typesInUse.size(); i++) {
    if (*std::get<0>(typesInUse[i]) == *type) {
      return std::get<1>(typesInUse[i]);
    }
  }

  // create new instance to type
  TypeBuilder *tBuilder = getTypeBuilder();
  uint32_t resTypeId = tBuilder->build(type);

  return resTypeId;
}

bool ModuleBuilder::isTypeBuilt(const Type::Type *type) {
  // search for type
  for (size_t i = 0; i < typesInUse.size(); i++) {
    if (*std::get<0>(typesInUse[i]) == *type) {
      return true;
    }
  }

  return false;
}

uint32_t ModuleBuilder::getOrAddOpConstant(const Type::Type *type,
                                           Operand value) {

  // the OpConstant instruction only support these
  assert(type->getKind() == Type::kInteger || type->getKind() == Type::kFloat);

  // search for existing
  for (size_t i = 0; i < constsInUse.size(); i++) {
    if (*std::get<0>(constsInUse[i]) == *type &&
        std::get<1>(constsInUse[i]) == value) {
      return std::get<2>(constsInUse[i]);
    }
  }

  uint32_t constResultId = getNextId();
  typesValues.push_back(
      Instr::create(::spv::Op::OpConstant, true, true,
                    {Operand::createTypeId(getOrAddType(type)),
                     Operand::createResultId(constResultId), value}));

  constsInUse.push_back(std::make_tuple(type->clone(), value, constResultId));

  return constResultId;
}

uint32_t ModuleBuilder::getNextId() { return cur_id++; }

TypeBuilder *ModuleBuilder::getTypeBuilder() {
  if (typeBuilder.get() == nullptr) {
    typeBuilder = std::make_unique<TypeBuilder>(this);
  }

  return typeBuilder.get();
}

uint32_t ModuleBuilder::emitFunctionBegin(const Type::Function *functionType) {
  assert(std::get<1>(currentScope.back()) == Scope::Global);

  uint32_t resultId = getNextId();

  functionDefinitions.push_back(InstructionList{});
  auto &activeDefn = functionDefinitions.back();

  uint32_t funcResId = getNextId();
  activeDefn.push_back(Instr::create(
      ::spv::Op::OpFunction, true, true,
      {Operand::createTypeId(getOrAddType(functionType->returnType.get())),
       Operand::createResultId(funcResId),
       Operand::create(SPV_OPERAND_TYPE_FUNCTION_CONTROL,
                       {(uint32_t)::spv::FunctionControlMask::MaskNone}),
       Operand::createId(
           getOrAddType(dynamic_cast<const Type::Type *>(functionType)))}));

  currentScope.push_back(std::make_tuple(funcResId, Scope::Function));

  return funcResId;
}

std::vector<uint32_t>
ModuleBuilder::emitFunctionParameters(const Type::Function *functionType) {
  assert(std::get<1>(currentScope.back()) == Scope::Function);

  auto &activeDefn = functionDefinitions.back();
  std::vector<uint32_t> resultIds;

  for (size_t i = 0; i < functionType->parameterTypes.size(); i++) {
    uint32_t resultId = getNextId();
    resultIds.push_back(resultId);

    activeDefn.push_back(
        Instr::create(::spv::Op::OpFunctionParameter, true, true,
                      {Operand::createTypeId(
                           getOrAddType(functionType->parameterTypes[i].get())),
                       Operand::createResultId(resultId)}));
  }

  return resultIds;
}

void ModuleBuilder::emitFunctionEnd() {
  assert(std::get<1>(currentScope.back()) == Scope::Function);

  auto &activeDefn = functionDefinitions.back();
  activeDefn.push_back(
      Instr::create(::spv::Op::OpFunctionEnd, false, false, {}));
}

uint32_t ModuleBuilder::addGlobalVariable(const Type::Type *pointerType,
                                          uint32_t storageClass) {
  uint32_t pointerTypeId = getOrAddType(pointerType);
  uint32_t resultId = getNextId();
  typesValues.push_back(Instr::create(
      ::spv::Op::OpVariable, true, true,
      {Operand::createTypeId(pointerTypeId), Operand::createResultId(resultId),
       Operand::create(SPV_OPERAND_TYPE_STORAGE_CLASS, {storageClass})}));

  return resultId;
}

uint32_t ModuleBuilder::getOrAddExtInstImport(std::u8string extInstSetName) {
  for (auto &extInstImport : extInstImportIds) {
    if (extInstImport.second == extInstSetName) {
      return extInstImport.first;
    }
  }

  uint32_t instSetId = getNextId();
  extInstImportIds[instSetId] = extInstSetName;

  return instSetId;
}

void ModuleBuilder::finalize(uint32_t entrypointId,
                             std::u8string entrypointName) {
  emitModulePremable(entrypointId, entrypointName);

}

std::vector<uint32_t> ModuleBuilder::getSpv() {
  std::vector<uint32_t> spv;
  auto spvPremable = emitSpirvPremable();
  spv.insert(spv.end(), spvPremable.begin(), spvPremable.end());

  const InstructionList *spvInstLists[] = {&capabilities,
                                           &extensions,
                                           &extInstImports,
                                           &memoryModel,
                                           &sampledImageAddressMode,
                                           &entryPoints,
                                           &executionModes,
                                           &annotations,
                                           &typesValues};

  for (auto &instListPtr : spvInstLists) {
    for (auto &inst : *instListPtr) {
      auto binary = inst.toSpvBinary();
      spv.insert(spv.end(), binary.begin(), binary.end());
    }
  }

  for (auto &funcDefn : functionDefinitions) {
    for (auto &inst : funcDefn) {
      auto binary = inst.toSpvBinary();
      spv.insert(spv.end(), binary.begin(), binary.end());
    }
  }

  return spv;
}

void ModuleBuilder::addVariableDecoration(uint32_t variableId,
                                          std::vector<Operand> decoration) {
  std::vector<Operand> operands;
  operands.push_back(Operand::createId(variableId));
  operands.insert(operands.end(), decoration.begin(), decoration.end());

  annotations.push_back(
      Instr::create(::spv::Op::OpDecorate, false, false, operands));
}

ModuleBuilder::ModuleBuilder() {
  currentScope.push_back(std::make_tuple(-1, Scope::Global));
}

// Must do this; destructor requires info on TypeBuilder instance
// So incomplete type can actually prevent compilation
// when a user includes just <ModuleBuilder.hpp> but not <TypeBuilder.hpp>
ModuleBuilder::~ModuleBuilder() = default;

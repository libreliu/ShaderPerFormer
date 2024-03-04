#pragma once

#include "Common.hpp"
#include "spirv/unified1/spirv.hpp11"
#include "spv/BasicBlockTestEmitter.hpp"
#include "spv/Type.hpp"
#include "spv/BasicBlock.hpp"
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace vkExecute::spv {

class TypeBuilder;
class BasicBlockTestEmitter;

// A dummy builder; requires calculating everything in advance
class ModuleBuilder {
  friend class TypeBuilder;
  friend class Type::Struct;
  friend class Type::Array;
  friend class BasicBlockTestEmitter;

public:
  using SpvBlob = std::vector<uint32_t>;
  using InstructionList = std::vector<Instr>;

  uint32_t getOrAddType(const Type::Type *type);
  bool isTypeBuilt(const Type::Type *type);
  
  // this is dumb, but works for now
  uint32_t getOrAddOpConstant(const Type::Type *type, Operand value);

  uint32_t addGlobalVariable(const Type::Type *pointerType, uint32_t storageClass);
  void addVariableDecoration(uint32_t variableId, std::vector<Operand> decoration);

  uint32_t getOrAddExtInstImport(std::u8string extInstSetName);

  uint32_t emitFunctionBegin(const Type::Function *functionType);
  std::vector<uint32_t> emitFunctionParameters(const Type::Function *functionType);

  void emitFunctionEnd();

  // TODO: implement the following
  void emitIfBegin();
  void emitIfElse();
  void emitIfEnd();
  
  void emitLoopHeader();
  void emitLoopTest();
  void emitLoopCond();
  void emitLoopBody();
  void emitLoopCont();
  void emitLoopMerge();

  void finalize(uint32_t entrypointId, std::u8string entrypointName);
  std::vector<uint32_t> getSpv();

  ModuleBuilder();
  ~ModuleBuilder();

  uint32_t getNextId();
  TypeBuilder *getTypeBuilder();

private:
  // called by finalize
  SpvBlob emitSpirvPremable();
  void emitModulePremable(uint32_t entrypointId, std::u8string entrypointName);
  void emitExtInstImports();

  // Arrange according to Logical Layout of a Module
  InstructionList capabilities;
  InstructionList extensions;
  InstructionList extInstImports;
  // NOTE: A module only has one memory model instruction.
  InstructionList memoryModel;
  // NOTE: A module can only have one optional sampled image addressing mode
  InstructionList sampledImageAddressMode;
  InstructionList entryPoints;
  InstructionList executionModes;
  // InstructionList debugs1;
  // InstructionList debugs2;
  // InstructionList debugs3;
  // InstructionList extInstDebuginfo;
  InstructionList annotations;

  // Type declarations, constants, and global variable declarations.
  InstructionList typesValues;

  // this is optional - useful when linking with other (imported) functions
  // std::vector<InstructionList> functionDeclarartions;
  std::vector<InstructionList> functionDefinitions;

  // keep track of the cur id, used in finalize for patching id bound
  uint32_t cur_id = 1;

  // associated type storage - (typePtr, type_id)
  std::vector<std::tuple<std::unique_ptr<Type::Type>, uint32_t>> typesInUse;

  // associated constant storage - (typePtr, constant operand, result_id)
  std::vector<std::tuple<std::unique_ptr<Type::Type>, Operand, uint32_t>> constsInUse;

  std::map<uint32_t, std::u8string> extInstImportIds;

  std::unique_ptr<TypeBuilder> typeBuilder;

  // current scope
  enum class Scope {
    Global,
    Function,
    BasicBlock
  };

  // the scope stack
  std::vector<std::tuple<uint32_t, Scope>> currentScope;
};

}
#pragma once

#include "spv/BasicBlock.hpp"
#include "spv/ModuleBuilder.hpp"
#include "spv/Type.hpp"
#include "spv/SubVariableInfo.hpp"
#include <cstdint>
#include <memory>
#include <stdint.h>
#include <vector>

namespace vkExecute::spv {

// BasicBlockTestEmitter
// - input global input and output variable, their type (variable / register)
//   - use a functional type definition; every variable have emitting options,
//   so on
// - add candidate basicblock
// - make connection

// 1. unpack all composite structure
//    - record into vector??
// 2. for all basic type, do connection according to connection strategy

class ModuleBuilder;

// TODO: implement register input
// struct RegisterDescriptor {
//   uint32_t ssaRegisterId;
//   std::unique_ptr<Type::Type> type;
// };

// void registerInputRegister(RegisterDescriptor inputRegDesc);

class BasicBlockTestEmitter {
public:
  struct ConnectionStrategy {
    // Only consider leaf node during matching
    bool leafOnly;

    bool sampleConst;
    // 0 to 1
    float sampleConstProb;
  };

  using NextIdGetter = uint32_t (*)(void);

  // everything needed to emit load / store code
  struct VariableDescriptor {
    uint32_t ssaVariableId;
    std::string prettyName;
    bool readable;
    bool writable;
    std::unique_ptr<Type::Type> type;

    VariableDescriptor() = default;
    ~VariableDescriptor() = default;
    VariableDescriptor(VariableDescriptor &&) = default;
    VariableDescriptor &operator=(VariableDescriptor &&) = default;
    inline VariableDescriptor(const VariableDescriptor &rhs) {
      this->ssaVariableId = rhs.ssaVariableId;
      this->readable = rhs.readable;
      this->writable = rhs.writable;
      this->type = rhs.type.get() != nullptr ? rhs.type->clone() : nullptr;
    }
    inline VariableDescriptor &operator=(const VariableDescriptor &rhs) {
      this->ssaVariableId = rhs.ssaVariableId;
      this->readable = rhs.readable;
      this->writable = rhs.writable;
      this->type = rhs.type.get() != nullptr ? rhs.type->clone() : nullptr;

      return *this;
    }

    inline VariableDescriptor clone() const {
      VariableDescriptor newDesc(*this);
      return newDesc;
    }
  };

  BasicBlockTestEmitter(ModuleBuilder *mBuilder);

  void registerInputVariable(VariableDescriptor inputVarDesc);
  void registerOutputVariable(VariableDescriptor outputVarDesc);

  // TODO: set input-output connection strategy
  void putBasicBlock(BasicBlock &bb, ConnectionStrategy inputConnStrategy);

  void finalize(ConnectionStrategy outputConnStrategy);

private:
  static bool isScalarType(const Type::Type *type);
  static bool isCompositeType(const Type::Type *type);
  static bool isPointerType(const Type::Type *type);
  static bool isDenseType(const Type::Type *type);

  // check if the type is meaningful for us to connect
  static bool isLegalType(const Type::Type *type);

  static std::function<void(uint32_t, std::vector<Instr> &, ModuleBuilder *)>
  generateCompositeStoreEmitFn(const std::unique_ptr<Type::Type> &valType,
                               uint32_t rootVarId,
                               const std::vector<uint32_t> &extractPath,
                               uint32_t storageClass);

  static std::function<uint32_t(std::vector<Instr> &, ModuleBuilder *)>
  generateCompositeLoadEmitFn(const std::unique_ptr<Type::Type> &valType,
                              uint32_t rootVarId,
                              const std::vector<uint32_t> &extractPath,
                              uint32_t storageClass);

  static void generateCompositeEmitFn(SubVariableInfo &subInfo);

  std::vector<SubVariableInfo> expandVariable(const VariableDescriptor &desc,
                                              bool appendIntermediate);

\
  void breakdownBBInputAndAssign();

  // bbIdRemap should have handled all input variable related remapping
  // For resource and output remapping we need to handle separately
  void emitBasicBlock(std::vector<Instr> &insts, BasicBlock &bb, std::map<uint32_t, uint32_t> &bbIdRemap);

  // =0 means no connectivity at all
  // >0 means connectable; the larger the more recommended
  int32_t getConnectivity(const Type::Type &srcType, const Type::Type &dstType);

  ModuleBuilder *mBuilder;

  // active variable
  std::map<uint32_t, VariableDescriptor> activeInputVar;
  std::map<uint32_t, VariableDescriptor> activeOutputVar;
};
} // namespace vkExecute::spv
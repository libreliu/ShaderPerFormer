#pragma once

#include "spv/Type.hpp"
#include "spv/BasicBlock.hpp"

namespace vkExecute::spv {
class ModuleBuilder;

// this gives tree traverse info
struct SubVariableInfo {
  // e.g. glFragCoord_0
  std::string prettyName;

  // emits load to given (sub)variable
  // Returns uint32_t - varable id as the result of the load
  std::function<uint32_t(std::vector<Instr> &, ModuleBuilder *)> loadEmitFn;

  // emits store to given (sub)variable
  // Args: uint32_t - variable of given type to be stored
  std::function<void(uint32_t, std::vector<Instr> &, ModuleBuilder *)>
      storeEmitFn;

  // the value type
  std::unique_ptr<Type::Type> valueType;

  uint32_t rootVariableId;
  // for composite variable use
  std::vector<uint32_t> extractPath;

  // this will always be inherited; useful for creating the pointer
  uint32_t storageClass;

  SubVariableInfo() = default;
  ~SubVariableInfo() = default;

  inline SubVariableInfo(const SubVariableInfo &rhs) {
    this->prettyName = rhs.prettyName;
    this->loadEmitFn = rhs.loadEmitFn;
    this->storeEmitFn = rhs.storeEmitFn;
    this->rootVariableId = rhs.rootVariableId;
    this->extractPath = rhs.extractPath;
    this->valueType =
        rhs.valueType.get() != nullptr ? rhs.valueType->clone() : nullptr;
  }
  inline SubVariableInfo &operator=(const SubVariableInfo &rhs) {
    this->prettyName = rhs.prettyName;
    this->loadEmitFn = rhs.loadEmitFn;
    this->storeEmitFn = rhs.storeEmitFn;
    this->rootVariableId = rhs.rootVariableId;
    this->extractPath = rhs.extractPath;
    this->valueType =
        rhs.valueType.get() != nullptr ? rhs.valueType->clone() : nullptr;

    return *this;
  }
  inline SubVariableInfo clone() const {
    SubVariableInfo subVarInfo(*this);
    return subVarInfo;
  }
};

}
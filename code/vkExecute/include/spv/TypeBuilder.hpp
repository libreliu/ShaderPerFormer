#pragma once

#include "spv/BasicBlock.hpp"
#include "spv/Type.hpp"
#include <cstdint>
#include <functional>

namespace vkExecute::spv {

class ModuleBuilder;

class TypeBuilder {
public:
  inline TypeBuilder(ModuleBuilder *mb) : mBuilder(mb) {}

  uint32_t build(const Type::Type *type);

private:
  uint32_t build(const Type::Integer *type);
  uint32_t build(const Type::Float *type);
  uint32_t build(const Type::Void *type);
  uint32_t build(const Type::Bool *type);
  uint32_t build(const Type::Vector *type);
  uint32_t build(const Type::Matrix *type);
  uint32_t build(const Type::Struct *type);
  uint32_t build(const Type::Array *type);
  uint32_t build(const Type::Pointer *type);
  uint32_t build(const Type::Function *type);

  ModuleBuilder *mBuilder;
};

} // namespace vkExecute::spv
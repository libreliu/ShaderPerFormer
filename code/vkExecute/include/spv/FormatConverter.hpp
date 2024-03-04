#pragma once

#include "spv/BasicBlock.hpp"
#include "spv/Type.hpp"

namespace vkExecute::spv {
class FormatConverter {
public:
  using FormatConversionFn = uint32_t (*)(uint32_t, std::vector<Instr> &,
                                          ModuleBuilder *);
  static const std::map<std::pair<Type::Kind, Type::Kind>, FormatConversionFn>
      fmtConvTable;

  static std::optional<FormatConversionFn>
  getFormatConvFn(const Type::Type &srcType, const Type::Type &dstType);
};

} // namespace vkExecute::spv
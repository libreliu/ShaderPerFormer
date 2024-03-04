#pragma once

#include "BasicBlock.hpp"
#include "SubVariableInfo.hpp"
#include "Type.hpp"
#include <memory>
#include <unordered_map>

namespace vkExecute::spv {

class ModuleBuilder;

// 1. Breakdown composite bb input variable type into subtype
// 2. for assignable subtype, emit assignment code
// 3. propagate up the tree; run construction code from values given to
// subtype

class AssignmentBuilder {
public:
  inline AssignmentBuilder(ModuleBuilder *mB) : mBuilder(mB) {}

  void registerInputSubVariableInfo(SubVariableInfo info);

  // emit assignment code from registered inputs; will call the loadEmitFn if
  // the sub variable is chosen.
  std::optional<uint32_t> emitAssignment(std::vector<Instr> &insts,
                                         const Type::Type *targetType);

private:
  std::optional<uint32_t> tryAssignComposite(std::vector<Instr> &insts,
                                             const Type::Type *target);
  std::optional<uint32_t> tryAssignScalar(std::vector<Instr> &insts,
                                          const Type::Type *target);

  // TODO: merge with BasicBlockTestEmitter or just delete the other
  int32_t getConnectivity(const Type::Type &srcType, const Type::Type &dstType);

  std::vector<SubVariableInfo> inputSubVars;
  ModuleBuilder *mBuilder;
};

} // namespace vkExecute::spv
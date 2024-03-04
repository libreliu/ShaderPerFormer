#include "spv/AssignmentBuilder.hpp"
#include "imgui.h"
#include "spv/BasicBlock.hpp"
#include "spv/FormatConverter.hpp"
#include "spv/ModuleBuilder.hpp"
#include "spv/SubVariableInfo.hpp"
#include "spv/Type.hpp"
#include <memory>
#include <stdexcept>

using namespace vkExecute::spv;

int32_t AssignmentBuilder::getConnectivity(const Type::Type &srcType,
                                           const Type::Type &dstType) {
  if (srcType == dstType) {
    return 100;
  } else if (FormatConverter::getFormatConvFn(srcType, dstType).has_value()) {
    return 10;
  } else {
    return 0;
  }
}

void AssignmentBuilder::registerInputSubVariableInfo(SubVariableInfo info) {
  inputSubVars.push_back(info);
}

std::optional<uint32_t>
AssignmentBuilder::emitAssignment(std::vector<Instr> &insts,
                                  const Type::Type *targetType) {
  // todo: implement me
  return {};
}

std::optional<uint32_t>
AssignmentBuilder::tryAssignScalar(std::vector<Instr> &insts,
                                   const Type::Type *target) {

  assert(target->getKind() == Type::kBool ||
         target->getKind() == Type::kInteger ||
         target->getKind() == Type::kFloat);

  // check for all possible assignments
  bool feasible = false;
  std::vector<float> candScores(inputSubVars.size());
  for (size_t i = 0; i < candScores.size(); i++) {
    candScores[i] = getConnectivity(*inputSubVars[i].valueType, *target);
    if (candScores[i] > 0) {
      feasible = true;
    }
  }

  if (!feasible) {
    return {};
  }

  // TODO: do sampling
  auto candIdx = argmax(candScores);
  uint32_t convertedResultId =
      inputSubVars[candIdx].loadEmitFn(insts, mBuilder);

  if (*inputSubVars[candIdx].valueType != *target) {
    auto convFn = FormatConverter::getFormatConvFn(
        *inputSubVars[candIdx].valueType, *target);
    assert(convFn.has_value());

    convertedResultId = convFn.value()(convertedResultId, insts, mBuilder);
  }

  return convertedResultId;
}

std::optional<uint32_t>
AssignmentBuilder::tryAssignComposite(std::vector<Instr> &insts,
                                      const Type::Type *target) {

  // TODO: do sampling / argmax according to ConnectionStrategy
  auto res = std::find_if(inputSubVars.begin(), inputSubVars.end(),
                          [&target](SubVariableInfo &elem) -> bool {
                            return *elem.valueType == *target;
                          });

  if (res != std::end(inputSubVars)) {
    uint32_t loadedResultId = res->loadEmitFn(insts, mBuilder);
    return loadedResultId;
  }

  std::vector<uint32_t> subAssignResults;
  std::vector<std::unique_ptr<Type::Type>> subTypes =
      Type::getFiniteCompositeSubTypes(target);
  for (size_t i = 0; i < subTypes.size(); i++) {
    auto assignRes = emitAssignment(insts, subTypes[i].get());
    if (!assignRes.has_value() && subAssignResults.size() > 0) {
      throw std::runtime_error("Partially assigned; may cause inconsistencies");
    } else if (!assignRes.has_value()) {
      return {};
    }

    subAssignResults.push_back(assignRes.value());
  }

  // build composite structure
  uint32_t compositeResult = mBuilder->getNextId();
  std::vector<Operand> operands{
      Operand::createTypeId(mBuilder->getOrAddType(target)),
      Operand::createResultId(compositeResult)};

  for (auto &subAssignResId : subAssignResults) {
    operands.push_back(Operand::createId(subAssignResId));
  }

  insts.push_back(
      Instr::create(::spv::Op::OpCompositeConstruct, true, true, operands));

  return compositeResult;
}
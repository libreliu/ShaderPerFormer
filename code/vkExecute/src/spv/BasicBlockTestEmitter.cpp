#include "spv/BasicBlockTestEmitter.hpp"
#include "Common.hpp"
#include "imgui.h"
#include "source/operand.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"
#include "spv/BasicBlock.hpp"
#include "spv/ModuleBuilder.hpp"
#include "spv/FormatConverter.hpp"
#include "spv/Type.hpp"
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using namespace vkExecute::spv;

void BasicBlockTestEmitter::registerInputVariable(
    VariableDescriptor inputVarDesc) {
  activeInputVar[inputVarDesc.ssaVariableId] = inputVarDesc;
}

void BasicBlockTestEmitter::registerOutputVariable(
    VariableDescriptor outputVarDesc) {
  activeOutputVar[outputVarDesc.ssaVariableId] = outputVarDesc;
}

void BasicBlockTestEmitter::putBasicBlock(
    BasicBlock &bb, ConnectionStrategy inputConnStrategy) {
  std::vector<SubVariableInfo> expandedInputVars;
  for (auto &inputVar : activeInputVar) {
    auto expandedVar =
        expandVariable(inputVar.second, !inputConnStrategy.leafOnly);
    expandedInputVars.insert(expandedInputVars.end(), expandedVar.begin(),
                             expandedVar.end());
  }

  VKEXECUTE_LOG("Got %zd expanded input vars", expandedInputVars.size());

  // match (expanded) input vars with input variables defined by basic block
  // Q: how about input *registers*?
  //    or to say, how do we treat basic block pieces?
  // For basic block pieces, actually *variables* are not necessary since it
  // merely describes transformations
  // And it's the cost of specific transformations that we eventually learnt in
  // the network
  // So for bb's striping all the variable informations away seems a good idea
  // since we have infinite registers available, and we determine which to
  // connect here.
  // However, when emitting the bridge code, the bb itself have to be renamed
  //
  // Q: how do we extract backend knowledge from learnt model?
  // A: align behavior with a simulator?? - nn it self is a good simulator so no
  //    we can just discard the idea

  assert(expandedInputVars.size() > 0);
  std::vector<Instr> premableInstrs;

  // bb_id -> remapped_id
  std::map<uint32_t, uint32_t> bbIdRemap;
  // remapped bb instrs; not including OpLabel & OpBranch (or any other control
  // flow instructions)
  std::vector<Instr> bbInstrs;
  for (auto &inputReg : bb.inputDescription) {
    auto &inputType = bb.associatedTypes[inputReg.type];

    // TODO: switch between assigning a const and assigning previous value
    if (inputConnStrategy.sampleConst && false) {
      assert(inputConnStrategy.sampleConstProb > 0 and
             inputConnStrategy.sampleConstProb <= 1);
      std::array<float, 2> probs = {inputConnStrategy.sampleConstProb,
                                    1 - inputConnStrategy.sampleConstProb};
      int res = sampleOnce(probs);

      // assign a const here
      if (res == 0) {
        // TODO: implement me; this may involve constructing a feasible constant
        // for arbitrary complex type; leave as future work
      }
    }

    bool feasible = false;
    std::vector<float> candScores(expandedInputVars.size());
    for (size_t i = 0; i < expandedInputVars.size(); i++) {
      candScores[i] =
          getConnectivity(*expandedInputVars[i].valueType, *inputType);
      if (candScores[i] > 0) {
        feasible = true;
      }
    }

    if (feasible) {
      auto candIdx = argmax(candScores);

      uint32_t convertedResultId =
          expandedInputVars[candIdx].loadEmitFn(premableInstrs, mBuilder);
      if (*expandedInputVars[candIdx].valueType != *inputType) {
        auto &srcType = *expandedInputVars[candIdx].valueType;
        auto &dstType = *inputType;
        auto convFn = FormatConverter::getFormatConvFn(srcType, dstType).value();
        convertedResultId = convFn(convertedResultId, premableInstrs, mBuilder);
      }

      // rename all use of inputReg.ssaId to convertedResultId
      bbIdRemap[inputReg.ssaId] = convertedResultId;
    } else {
      // need to break down into basic types, and then find corresponding
      // input variable
      breakdownBBInputAndAssign();
    }
  }

  // process output block
  std::vector<Instr> bbInsts;
  emitBasicBlock(bbInsts, bb, bbIdRemap);
}

void BasicBlockTestEmitter::finalize(ConnectionStrategy outputConnStrategy) {}

bool BasicBlockTestEmitter::isLegalType(const Type::Type *type) {
  switch (type->getKind()) {
  case Type::Kind::kBool:
  case Type::Kind::kInteger:
  case Type::Kind::kFloat:
  case Type::Kind::kArray:
  case Type::Kind::kVector:
  case Type::Kind::kStruct:
  case Type::Kind::kImage:
  case Type::Kind::kMatrix:
  case Type::Kind::kPointer:
    return true;
  default:
    return false;
  }
}

// TODO: add runtime array as dense type
bool BasicBlockTestEmitter::isDenseType(const Type::Type *type) {
  switch (type->getKind()) {
  case Type::Kind::kImage:
    return true;
  default:
    return false;
  }
}

bool BasicBlockTestEmitter::isScalarType(const Type::Type *type) {
  switch (type->getKind()) {
  case Type::Kind::kBool:
  case Type::Kind::kInteger:
  case Type::Kind::kFloat:
    return true;
  default:
    return false;
  }
}

bool BasicBlockTestEmitter::isCompositeType(const Type::Type *type) {
  switch (type->getKind()) {
  case Type::Kind::kArray:
  case Type::Kind::kStruct:
  case Type::Kind::kVector:
  case Type::Kind::kMatrix:
    return true;
  default:
    return false;
  }
}

bool BasicBlockTestEmitter::isPointerType(const Type::Type *type) {
  return type->getKind() == Type::Kind::kPointer;
}

std::vector<SubVariableInfo>
BasicBlockTestEmitter::expandVariable(const VariableDescriptor &desc,
                                      bool appendIntermediate) {
  std::vector<SubVariableInfo> result, workQueue;

  auto isFinalType = [](const Type::Type *type) -> bool {
    assert(isLegalType(type));
    return isScalarType(type) || isDenseType(type);
  };

  SubVariableInfo rootInfo;
  rootInfo.prettyName = desc.prettyName;
  rootInfo.valueType = desc.type->asPointer()->pointeeType->clone();
  rootInfo.rootVariableId = desc.ssaVariableId;
  rootInfo.storageClass = desc.type->asPointer()->storageClass;
  if (isFinalType(rootInfo.valueType.get()) || appendIntermediate) {
    rootInfo.loadEmitFn =
        [ssaId = desc.ssaVariableId,
         varType = std::shared_ptr<Type::Type>(rootInfo.valueType->clone())](
            std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      uint32_t loadResultId = mBuilder->getNextId();
      insts.push_back(Instr::create(
          ::spv::Op::OpLoad, true, true,
          {Operand::createTypeId(mBuilder->getOrAddType(varType.get())),
           Operand::createResultId(loadResultId), Operand::createId(ssaId)}));

      return loadResultId;
    };
    rootInfo.storeEmitFn =
        [ssaId = desc.ssaVariableId,
         varType = std::shared_ptr<Type::Type>(desc.type->clone())](
            uint32_t resultToBeStored, std::vector<Instr> &insts,
            ModuleBuilder *mBuilder) -> void {
      insts.push_back(Instr::create(
          ::spv::Op::OpStore, false, false,
          {Operand::createId(ssaId), Operand::createId(resultToBeStored)}));
    };
  }

  workQueue.push_back(rootInfo);

  while (!workQueue.empty()) {
    auto workInfo = workQueue.back();
    workQueue.pop_back();

    assert(isLegalType(workInfo.valueType.get()) &&
           !isPointerType(workInfo.valueType.get()));

    if (isFinalType(workInfo.valueType.get())) {
      result.push_back(workInfo);
      continue;
    }

    if (appendIntermediate) {
      result.push_back(workInfo);
    }

    switch (workInfo.valueType->getKind()) {
    case Type::Kind::kArray: {
      auto typed = workInfo.valueType->asArray();
      for (size_t i = 0; i < typed->count; i++) {
        SubVariableInfo subInfo;
        subInfo.prettyName = workInfo.prettyName + "_arr" + std::to_string(i);
        subInfo.valueType = typed->elementType->clone();
        subInfo.rootVariableId = workInfo.rootVariableId;
        subInfo.extractPath = workInfo.extractPath;
        subInfo.extractPath.push_back(i);
        subInfo.storageClass = workInfo.storageClass;

        // only emit when we need to do so
        if (isFinalType(subInfo.valueType.get()) || appendIntermediate) {
          generateCompositeEmitFn(subInfo);
        }

        workQueue.push_back(std::move(subInfo));
      }
    }
    case Type::Kind::kStruct: {
      auto typed = workInfo.valueType->asStruct();
      for (size_t i = 0; i < typed->members.size(); i++) {
        SubVariableInfo subInfo;
        subInfo.prettyName = workInfo.prettyName + "_str" + std::to_string(i);
        subInfo.valueType = typed->members[i]->clone();
        subInfo.rootVariableId = workInfo.rootVariableId;
        subInfo.extractPath = workInfo.extractPath;
        subInfo.extractPath.push_back(i);
        subInfo.storageClass = workInfo.storageClass;

        // only emit when we need to do so
        if (isFinalType(subInfo.valueType.get()) || appendIntermediate) {
          generateCompositeEmitFn(subInfo);
        }

        workQueue.push_back(std::move(subInfo));
      }
    }
    case Type::Kind::kVector: {
      auto typed = workInfo.valueType->asVector();
      for (size_t i = 0; i < typed->count; i++) {
        SubVariableInfo subInfo;

        subInfo.prettyName = workInfo.prettyName + "_vec" + std::to_string(i);
        subInfo.valueType = typed->componentType->clone();
        subInfo.rootVariableId = workInfo.rootVariableId;
        subInfo.extractPath = workInfo.extractPath;
        subInfo.extractPath.push_back(i);
        subInfo.storageClass = workInfo.storageClass;

        // only emit when we need to do so
        if (isFinalType(subInfo.valueType.get()) || appendIntermediate) {
          generateCompositeEmitFn(subInfo);
        }

        workQueue.push_back(std::move(subInfo));
      }
    }
    case Type::Kind::kMatrix: {
      auto typed = workInfo.valueType->asMatrix();
      for (size_t i = 0; i < typed->cols; i++) {
        SubVariableInfo subInfo;

        subInfo.prettyName = workInfo.prettyName + "_vec" + std::to_string(i);
        subInfo.valueType = typed->colType->clone();
        subInfo.rootVariableId = workInfo.rootVariableId;
        subInfo.extractPath = workInfo.extractPath;
        subInfo.extractPath.push_back(i);
        subInfo.storageClass = workInfo.storageClass;

        // only emit when we need to do so
        if (isFinalType(subInfo.valueType.get()) || appendIntermediate) {
          generateCompositeEmitFn(subInfo);
        }

        workQueue.push_back(std::move(subInfo));
      }
    }
    default:
      throw std::runtime_error("Unexpected kind");
    }
  }

  return result;
}

std::function<void(uint32_t, std::vector<Instr> &, ModuleBuilder *)>
BasicBlockTestEmitter::generateCompositeStoreEmitFn(
    const std::unique_ptr<Type::Type> &valType, uint32_t rootVarId,
    const std::vector<uint32_t> &extractPath, uint32_t storageClass) {

  return [valType = std::shared_ptr<Type::Type>(valType->clone()),
          rootVarId = rootVarId, extractPath = extractPath,
          storageClass = storageClass](uint32_t resultToBeStored,
                                       std::vector<Instr> &insts,
                                       ModuleBuilder *mBuilder) -> void {
    std::vector<Operand> operands;
    auto ptrResultId = mBuilder->getNextId();

    auto valPointer = Type::Pointer(storageClass, *valType);
    operands.push_back(
        Operand::createTypeId(mBuilder->getOrAddType(&valPointer)));
    operands.push_back(Operand::createResultId(ptrResultId));
    operands.push_back(Operand::createId(rootVarId));
    for (auto &id : extractPath) {
      operands.push_back(Operand::createId(id));
    }

    insts.push_back(
        Instr::create(::spv::Op::OpAccessChain, true, true, operands));

    insts.push_back(Instr::create(
        ::spv::Op::OpStore, false, false,
        {Operand::createTypeId(mBuilder->getOrAddType(valType.get())),
         Operand::createResultId(resultToBeStored)}));
  };
}

std::function<uint32_t(std::vector<Instr> &, ModuleBuilder *)>
BasicBlockTestEmitter::generateCompositeLoadEmitFn(
    const std::unique_ptr<Type::Type> &valType, uint32_t rootVarId,
    const std::vector<uint32_t> &extractPath, uint32_t storageClass) {

  return [valType = std::shared_ptr<Type::Type>(valType->clone()),
          rootVarId = rootVarId, extractPath = extractPath,
          storageClass = storageClass](std::vector<Instr> &insts,
                                       ModuleBuilder *mBuilder) -> uint32_t {
    std::vector<Operand> operands;
    auto ptrResultId = mBuilder->getNextId();

    auto valPointer = Type::Pointer(storageClass, *valType);
    operands.push_back(
        Operand::createTypeId(mBuilder->getOrAddType(&valPointer)));
    operands.push_back(Operand::createResultId(ptrResultId));
    operands.push_back(Operand::createId(rootVarId));
    for (auto &id : extractPath) {
      operands.push_back(Operand::createId(id));
    }

    auto loadResultId = mBuilder->getNextId();
    insts.push_back(
        Instr::create(::spv::Op::OpAccessChain, true, true, operands));
    insts.push_back(Instr::create(
        ::spv::Op::OpLoad, true, true,
        {Operand::createTypeId(mBuilder->getOrAddType(valType.get())),
         Operand::createResultId(loadResultId),
         Operand::createId(ptrResultId)}));

    return loadResultId;
  };
}

void BasicBlockTestEmitter::generateCompositeEmitFn(SubVariableInfo &subInfo) {
  subInfo.loadEmitFn =
      generateCompositeLoadEmitFn(subInfo.valueType, subInfo.rootVariableId,
                                  subInfo.extractPath, subInfo.storageClass);

  subInfo.storeEmitFn =
      generateCompositeStoreEmitFn(subInfo.valueType, subInfo.rootVariableId,
                                   subInfo.extractPath, subInfo.storageClass);
}

int32_t BasicBlockTestEmitter::getConnectivity(const Type::Type &srcType,
                                               const Type::Type &dstType) {
  if (srcType == dstType) {
    return 100;
  } else if (FormatConverter::getFormatConvFn(srcType, dstType).has_value()) {
    return 10;
  } else {
    return 0;
  }
}

void BasicBlockTestEmitter::emitBasicBlock(
    std::vector<Instr> &insts, BasicBlock &bb,
    std::map<uint32_t, uint32_t> &bbIdRemap) {

  // copy by value to avoid accidentally changing original bb
  for (auto bbInst : bb.bbInstructions) {
    for (auto &opnd : bbInst.operands) {
      if (spvIsIdType(opnd.type)) {
        assert(opnd.data.size() == 1);
        uint32_t oldId = opnd.data[0];
        if (bbIdRemap.count(oldId) > 0) {
          opnd.data[0] = bbIdRemap[oldId];
        } else if (bb.associatedTypes.count(oldId) > 0) {
          // this is a type associated value
          auto typePointed = bb.associatedTypes[oldId]->clone();
          uint32_t newId = mBuilder->getOrAddType(typePointed.get());
          bbIdRemap[oldId] = newId;
          opnd.data[0] = newId;
        } else {
          auto resIt = bb.findResourceValue(oldId);
          if (resIt != bb.resourceDescription.end()) {
            // transition the resource into correct segment
            // and update id accordingly
            switch (resIt->type) {
            case Resource::ExtInstImport: {
              auto &concretePayload =
                  std::get<ExtInstImportResource>(resIt->payload);
              uint32_t newId = mBuilder->getOrAddExtInstImport(
                  concretePayload.extInstSetName);

              bbIdRemap[oldId] = newId;
              opnd.data[0] = newId;
              break;
            }
            default: {
              throw std::runtime_error("Unexpected resource type");
            }
            }

            // separate input and output item; input item should have been
            // mapped
          } else if (bb.findOutputValue(oldId) != bb.outputDescription.end()) {
            // remap output value to new item
            uint32_t newId = mBuilder->getNextId();
            bbIdRemap[oldId] = newId;
            opnd.data[0] = newId;
          } else if (bb.findInputValue(oldId) != bb.inputDescription.end()) {
            throw std::runtime_error(
                "Input value shall already be remapped earlier");
          } else {
            // error
            throw std::runtime_error(
                "Unexpected old Id while processing basic block instructions");
          }
        }
      }
    }

    insts.push_back(bbInst);
  }
}
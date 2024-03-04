#include "spv/Type.hpp"
#include "Common.hpp"
#include "spv/BasicBlock.hpp"
#include "spv/TypeBuilder.hpp"

#include "source/opt/ir_context.h"
#include "source/opt/types.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"
#include "spv/ModuleBuilder.hpp"
#include <algorithm>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <optional>
#include <stdexcept>
#include <stdint.h>
#include <string>

using namespace vkExecute::spv;

#define SpvToolsKind(kindName) spvtools::opt::analysis::Type::Kind::kindName

std::unique_ptr<Type::Type>
Type::Type::create(const spvtools::opt::analysis::Type *type,
                   spvtools::opt::IRContext *irCtx) {
  switch (type->kind()) {

  case SpvToolsKind(kVoid):
    return std::make_unique<Void>();
  case SpvToolsKind(kBool):
    return std::make_unique<Bool>();
  case SpvToolsKind(kFloat):
    return std::make_unique<Float>(type->AsFloat()->width());
  case SpvToolsKind(kInteger):
    return std::make_unique<Integer>(type->AsInteger()->width(),
                                     type->AsInteger()->IsSigned() ? 1 : 0);
  case SpvToolsKind(kVector): {
    auto vectorSubType = create(type->AsVector()->element_type(), irCtx);

    return std::make_unique<Vector>(type->AsVector()->element_count(),
                                    *vectorSubType);
  }
  case SpvToolsKind(kMatrix): {
    auto matrixSubType = create(type->AsMatrix()->element_type(), irCtx);
    return std::make_unique<Matrix>(type->AsMatrix()->element_count(),
                                    *matrixSubType);
  }
  case SpvToolsKind(kPointer): {
    auto pointeeType = create(type->AsPointer()->pointee_type(), irCtx);
    return std::make_unique<Pointer>(
        static_cast<unsigned>(type->AsPointer()->storage_class()),
        *pointeeType);
  }
  case SpvToolsKind(kStruct): {
    auto newStruct = std::make_unique<Struct>();

    std::vector<uint32_t> memberOffsets;
    // check for potential decorations
    auto decorations = type->AsStruct()->element_decorations();
    for (size_t i = 0; i < decorations.size(); i++) {
      bool found = false;
      for (size_t j = 0; j < decorations.at(i).size(); j++) {
        if (decorations.at(i).at(j).at(0) ==
            (uint32_t)::spv::Decoration::Offset) {
          memberOffsets.push_back(decorations[i][j][1]);
          found = true;
          break;
        }
      }

      if (!found) {
        VKEXECUTE_LOG(
            "Didn't found member offset decorations at index %zu, skip", i);
        memberOffsets.clear();
        break;
      }
    }

    if (type->AsStruct()->element_types().size() != memberOffsets.size() &&
        memberOffsets.size() > 0) {
      VKEXECUTE_LOG("Member offset mismatch with element size, skip");
      memberOffsets.clear();
    }

    for (size_t i = 0; i < type->AsStruct()->element_types().size(); i++) {
      auto &spvToolsMemType = type->AsStruct()->element_types()[i];
      auto memberType = create(spvToolsMemType, irCtx);

      if (memberOffsets.size() == 0) {
        newStruct->addMember(*memberType);
      } else {
        newStruct->addMember(*memberType, memberOffsets[i]);
      }
    }

    return newStruct;
  }
  case SpvToolsKind(kArray): {
    auto elemType = create(type->AsArray()->element_type(), irCtx);

    int length = -1;
    // parse length info
    {
      auto &lengthInfo = type->AsArray()->length_info();
      if (lengthInfo.words[0] !=
          spvtools::opt::analysis::Array::LengthInfo::Case::kConstant) {
        // The rest needs specialization; we don't support it for now
        VKEXECUTE_WARN("Unsupported array length kind: %d",
                       static_cast<int>(lengthInfo.words[0]));
        throw std::out_of_range("Unsupported array length kind");
      }

      uint32_t definingId = lengthInfo.id;
      auto constant =
          irCtx->get_constant_mgr()->FindDeclaredConstant(definingId);
      if (!constant) {
        VKEXECUTE_WARN("Defining Id %u doesn't form a constant", definingId);
        throw std::out_of_range("Defining Id doesn't form a constant");
      }

      if (constant->type()->kind() != SpvToolsKind(kInteger)) {
        VKEXECUTE_WARN("Defining Id %u is not an integer constant", definingId);
        throw std::out_of_range("Defining Id is not an integer constant");
      }

      length = constant->AsIntConstant()->GetU32();
    }

    // parse array stride info
    std::optional<uint32_t> arrayStride;
    {
      auto decorations = type->AsArray()->decorations();

      if (decorations.size() > 0) {
        for (auto &decoration : decorations) {
          if (decoration[0] == (uint32_t)::spv::Decoration::ArrayStride) {
            arrayStride = decoration[1];
            break;
          }
        }
      }
    }

    return std::make_unique<Array>(length, *elemType, arrayStride);
  }

  default:
    VKEXECUTE_WARN("Unexpected spvtools kind: %d",
                   static_cast<int>(type->kind()));
    throw std::out_of_range("Unexpected spvtools kind");
  }
}

#undef SpvToolsKind

std::unique_ptr<Type::Type> Type::Type::createInt32() {
  return std::make_unique<Integer>(32, 1);
}

std::unique_ptr<Type::Type> Type::Type::createUInt32() {
  return std::make_unique<Integer>(32, 0);
}

std::unique_ptr<Type::Type> Type::Type::createF32() {
  return std::make_unique<Float>(32);
}

void Type::Struct::emitOffsetDecorations(ModuleBuilder *mBuilder) const {
  assert(memberOffsets.size() > 0);
  // or we'll have a circular dependency and infinite recursion!
  assert(mBuilder->isTypeBuilt(this));

  for (size_t i = 0; i < memberOffsets.size(); i++) {
    mBuilder->annotations.push_back(
        Instr::create(::spv::Op::OpMemberDecorate, false, false,
                      {Operand::createId(mBuilder->getOrAddType(this)),
                       Operand::createLiteralInteger(i),
                       Operand::create(SPV_OPERAND_TYPE_DECORATION,
                                       {(uint32_t)::spv::Decoration::Offset}),
                       Operand::createLiteralInteger(memberOffsets[i])}));
  }
}

void Type::Array::emitStrideDecorations(ModuleBuilder *mBuilder) const {
  assert(arrayStride.has_value());
  assert(mBuilder->isTypeBuilt(this));

  mBuilder->annotations.push_back(Instr::create(
      ::spv::Op::OpDecorate, false, false,
      {Operand::createId(mBuilder->getOrAddType(this)),
       Operand::create(SPV_OPERAND_TYPE_DECORATION,
                       {(uint32_t)::spv::Decoration::ArrayStride}),
       Operand::createLiteralInteger(arrayStride.value())}));
}

std::vector<std::unique_ptr<Type::Type>>
getFiniteCompositeSubTypes(const Type::Type *type) {
  std::vector<std::unique_ptr<Type::Type>> resVec;

  switch (type->getKind()) {
  case Type::Kind::kArray: {
    for (size_t i = 0; i < type->asArray()->count; i++) {
      resVec.push_back(type->asArray()->elementType->clone());
    }
    return resVec;
  }
  case Type::Kind::kMatrix: {
    for (size_t i = 0; i < type->asMatrix()->cols; i++) {
      resVec.push_back(type->asMatrix()->colType->clone());
    }
    return resVec;
  }
  case Type::Kind::kStruct: {
    for (size_t i = 0; i < type->asStruct()->members.size(); i++) {
      resVec.push_back(type->asStruct()->members[i]->clone());
    }
    return resVec;
  }
  case Type::Kind::kVector: {
    for (size_t i = 0; i < type->asVector()->count; i++) {
      resVec.push_back(type->asVector()->componentType->clone());
    }
    return resVec;
  }
  default:
    throw std::runtime_error("Unsupported type");
  }
}
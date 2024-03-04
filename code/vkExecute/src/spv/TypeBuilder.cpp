#include "spv/TypeBuilder.hpp"
#include "spirv-tools/libspirv.h"
#include "spv/BasicBlock.hpp"
#include "spv/ModuleBuilder.hpp"
#include "spv/Type.hpp"
#include <cstdint>
#include <stdexcept>

using namespace vkExecute::spv;

uint32_t TypeBuilder::build(const Type::Type *type) {
  switch (type->getKind()) {
  case Type::Kind::kVoid:
    return build(type->asVoid());
  case Type::Kind::kBool:
    return build(type->asBool());
  case Type::Kind::kInteger:
    return build(type->asInteger());
  case Type::Kind::kFloat:
    return build(type->asFloat());
  case Type::Kind::kVector:
    return build(type->asVector());
  case Type::Kind::kMatrix:
    return build(type->asMatrix());
  case Type::Kind::kArray:
    return build(type->asArray());
  case Type::Kind::kStruct:
    return build(type->asStruct());
  case Type::Kind::kPointer:
    return build(type->asPointer());
  case Type::Kind::kFunction:
    return build(type->asFunction());

  case Type::Kind::kImage:
  case Type::Kind::kSampler:
  case Type::Kind::kSampledImage:
  case Type::Kind::kRuntimeArray:
  case Type::Kind::kOpaque:
  case Type::Kind::kEvent:
  case Type::Kind::kDeviceEvent:
  case Type::Kind::kReserveId:
  case Type::Kind::kQueue:
  case Type::Kind::kPipe:
  case Type::Kind::kForwardPointer:
  case Type::Kind::kPipeStorage:
  case Type::Kind::kNamedBarrier:
  case Type::Kind::kAccelerationStructureNV:
  case Type::Kind::kCooperativeMatrixNV:
  case Type::Kind::kRayQueryKHR:
  case Type::Kind::kHitObjectNV:
  case Type::Kind::kLast:
  default:
    throw std::runtime_error("Unhandled type kind");
  }
}

uint32_t TypeBuilder::build(const Type::Integer *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeInt, true, false,
                    {Operand::createResultId(typeResultId),
                     Operand::createLiteralInteger(type->width),
                     Operand::createLiteralInteger(type->signedness)}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Float *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeFloat, true, false,
                    {Operand::createResultId(typeResultId),
                     Operand::createLiteralInteger(type->width)}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Void *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeVoid, true, false,
                    {Operand::createResultId(typeResultId)}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Bool *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeBool, true, false,
                    {Operand::createResultId(typeResultId)}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Vector *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(Instr::create(
      ::spv::Op::OpTypeVector, true, false,
      {Operand::createResultId(typeResultId),
       Operand::createId(mBuilder->getOrAddType(type->componentType.get())),
       Operand::createLiteralInteger(type->count)}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Matrix *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  mBuilder->typesValues.push_back(Instr::create(
      ::spv::Op::OpTypeMatrix, true, false,
      {Operand::createResultId(typeResultId),
       Operand::createId(mBuilder->getOrAddType(type->colType.get())),
       Operand::createLiteralInteger(type->cols)}));
  
  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Struct *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  std::vector<Operand> operands;

  operands.push_back(Operand::createResultId(typeResultId));
  for (auto &member : type->members) {
    operands.push_back(
        Operand::createId(mBuilder->getOrAddType(member.get())));
  }

  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeStruct, true, false, std::move(operands)));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));

  if (type->memberOffsets.size() > 0) {
    type->emitOffsetDecorations(mBuilder);
  }

  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Array *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  auto uInt32UPtr = Type::Type::createUInt32();

  mBuilder->typesValues.push_back(Instr::create(
      ::spv::Op::OpTypeArray, true, false,
      {Operand::createResultId(typeResultId),
       Operand::createId(mBuilder->getOrAddType(type->elementType.get())),
       Operand::createId(mBuilder->getOrAddOpConstant(
           uInt32UPtr.get(), Operand::createLiteralInteger(type->count)))}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));

  if (type->arrayStride.has_value()) {
    type->emitStrideDecorations(mBuilder);
  }

  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Pointer *type) {
  uint32_t typeResultId = mBuilder->getNextId();

  mBuilder->typesValues.push_back(Instr::create(
      ::spv::Op::OpTypePointer, true, false,
      {Operand::createResultId(typeResultId),
       Operand::create(SPV_OPERAND_TYPE_STORAGE_CLASS, {type->storageClass}),
       Operand::createId(
           mBuilder->getOrAddType(type->pointeeType.get()))}));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

uint32_t TypeBuilder::build(const Type::Function *type) {
  uint32_t typeResultId = mBuilder->getNextId();
  std::vector<Operand> operands;

  operands.push_back(Operand::createResultId(typeResultId));
  operands.push_back(
      Operand::createId(mBuilder->getOrAddType(type->returnType.get())));
  for (auto &param : type->parameterTypes) {
    operands.push_back(
        Operand::createId(mBuilder->getOrAddType(param.get())));
  }

  mBuilder->typesValues.push_back(
      Instr::create(::spv::Op::OpTypeFunction, true, false, operands));

  mBuilder->typesInUse.push_back(std::make_tuple(type->clone(), typeResultId));
  return typeResultId;
}

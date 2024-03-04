#include "spv/FormatConverter.hpp"
#include "spv/ModuleBuilder.hpp"
#include "spv/Type.hpp"
#include <map>
#include <vector>

using namespace vkExecute::spv;

// clang-format off
const std::map<std::pair<Type::Kind, Type::Kind>, FormatConverter::FormatConversionFn>
FormatConverter::fmtConvTable = {
  {
    {Type::Kind::kBool, Type::Kind::kFloat},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto f32Type = Type::Type::createF32();
      auto resultId = mBuilder->getNextId();
      insts.push_back(Instr::create(::spv::Op::OpSelect, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(f32Type.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx),
        Operand::createId(mBuilder->getOrAddOpConstant(f32Type.get(), Operand::createLiteralF32(1.0f))),
        Operand::createId(mBuilder->getOrAddOpConstant(f32Type.get(), Operand::createLiteralF32(0.0f)))
      }));
      return resultId;
    }
  },
  {
    {Type::Kind::kFloat, Type::Kind::kBool},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto boolType = std::make_unique<Type::Bool>();
      auto f32Type = Type::Type::createF32();
      auto resultId = mBuilder->getNextId();
      insts.push_back(Instr::create(::spv::Op::OpFUnordNotEqual, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(boolType.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx),
        Operand::createId(mBuilder->getOrAddOpConstant(f32Type.get(), Operand::createLiteralF32(0.0f)))
      }));
      return resultId;
    }
  },
  {
    {Type::Kind::kBool, Type::Kind::kInteger},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto i32Type = Type::Type::createInt32();
      auto resultId = mBuilder->getNextId();
      insts.push_back(Instr::create(::spv::Op::OpSelect, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(i32Type.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx),
        Operand::createId(mBuilder->getOrAddOpConstant(i32Type.get(), Operand::createLiteralInteger(1.0f))),
        Operand::createId(mBuilder->getOrAddOpConstant(i32Type.get(), Operand::createLiteralInteger(0.0f)))
      }));
      return resultId;
    }
  },
  {
    {Type::Kind::kInteger, Type::Kind::kBool},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto boolType = std::make_unique<Type::Bool>();
      auto i32Type = Type::Type::createInt32();
      auto resultId = mBuilder->getNextId();
      insts.push_back(Instr::create(::spv::Op::OpINotEqual, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(boolType.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx),
        Operand::createId(mBuilder->getOrAddOpConstant(i32Type.get(), Operand::createLiteralInteger(0)))
      }));
      return resultId;
    }
  },
  {
    {Type::Kind::kInteger, Type::Kind::kFloat},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto resultId = mBuilder->getNextId();
      auto f32Type = Type::Type::createF32();
      insts.push_back(Instr::create(::spv::Op::OpConvertSToF, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(f32Type.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx)
      }));
      return resultId;
    }
  },
  {
    {Type::Kind::kFloat, Type::Kind::kInteger},
    [](uint32_t srcIdx, std::vector<Instr> &insts, ModuleBuilder *mBuilder) -> uint32_t {
      auto resultId = mBuilder->getNextId();
      auto i32Type = Type::Type::createInt32();
      insts.push_back(Instr::create(::spv::Op::OpConvertFToS, true, true, {
        Operand::createTypeId(mBuilder->getOrAddType(i32Type.get())),
        Operand::createResultId(resultId),
        Operand::createId(srcIdx)
      }));
      return resultId;
    }
  }
};
// clang-format on

std::optional<FormatConverter::FormatConversionFn>
FormatConverter::getFormatConvFn(const Type::Type &srcType,
                                       const Type::Type &dstType) {
  auto typePair = std::make_pair(srcType.getKind(), dstType.getKind());
  assert(fmtConvTable.count(typePair) > 0);

  // https://stackoverflow.com/questions/13902742/why-does-stdmap-not-have-a-const-accessor
  return fmtConvTable.at(typePair);
}
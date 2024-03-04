#include "spv/Tokenizer.hpp"
#include "Common.hpp"
#include "source/opt/build_module.h"
#include "source/opt/instruction.h"
#include "source/opt/ir_context.h"
#include "source/opt/types.h"
#include "spirv-tools/libspirv.h"
#include "spv/Type.hpp"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace vkExecute::spv;

constexpr auto kDefaultEnvironment = SPV_ENV_VULKAN_1_3;

void Tokenizer::initCommonTypeLUT() {
  // NOTE: ORDER MATTERS! So be sure not to modify too often
  enum CommonTypes {
    kVoid = 0,
    kUInt32,
    kInt32,
    kFloat32,
    kFloat64,
    kBool,
    kVec2U,
    kVec2I,
    kVec2F,
    kVec2D,
    kVec3U,
    kVec3I,
    kVec3F,
    kVec3D,
    kVec4U,
    kVec4I,
    kVec4F,
    kVec4D,
    kMat33F,
    kMat33D,
    kMat34F,
    kMat34D,
    kMat43F,
    kMat43D,
    kMat44F,
    kMat44D
  };

  commonTypes.clear();

  commonTypes.push_back(std::make_unique<Type::Void>());

  commonTypes.push_back(std::make_unique<Type::Integer>(32, 0));
  commonTypes.push_back(std::make_unique<Type::Integer>(32, 1));

  commonTypes.push_back(std::make_unique<Type::Float>(32));
  commonTypes.push_back(std::make_unique<Type::Float>(64));

  commonTypes.push_back(std::make_unique<Type::Bool>());

  // kVec2U...2D, 3U...3D, 4U...4D
  for (int i = 2; i < 5; i++) {
    commonTypes.push_back(
        std::make_unique<Type::Vector>(i, *commonTypes[kUInt32]));
    commonTypes.push_back(
        std::make_unique<Type::Vector>(i, *commonTypes[kInt32]));
    commonTypes.push_back(
        std::make_unique<Type::Vector>(i, *commonTypes[kFloat32]));
    commonTypes.push_back(
        std::make_unique<Type::Vector>(i, *commonTypes[kFloat64]));
  }

  // kMat33F, 33D
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(3, *commonTypes[kVec3F]));
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(3, *commonTypes[kVec3D]));

  // kMat34F, 34D
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(3, *commonTypes[kVec4F]));
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(3, *commonTypes[kVec4D]));

  // kMat43F, 43D
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(4, *commonTypes[kVec3F]));
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(4, *commonTypes[kVec3D]));

  // kMat44F, 44D
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(4, *commonTypes[kVec4F]));
  commonTypes.push_back(
      std::make_unique<Type::Matrix>(4, *commonTypes[kVec4D]));
}

std::tuple<std::vector<int>, std::string> Tokenizer::tokenize() {
  auto res = doTokenize(false, nullptr, nullptr);
  return std::make_tuple(
    std::move(std::get<0>(res)),
    std::move(std::get<2>(res))
  );
}

// unified interface
std::tuple<std::vector<int>,
           std::optional<std::vector<vkExecute::TraceCounter_t>>, std::string>
Tokenizer::doTokenize(bool withTrace,
                      // Params for trace
                      std::map<int, int> *bbIdxMap,
                      std::vector<vkExecute::TraceCounter_t> *bbTraceCounters) {

  if (spvBlob.size() == 0) {
    throw std::runtime_error("Load your SPIR-V first!");
  }

  spv_context context = spvContextCreate(spv_target_env::SPV_ENV_VULKAN_1_3);
  std::stringstream errMsgs;
  std::vector<int> tokenized;
  std::vector<TraceCounter_t> traced;

  auto irModule = spvtools::BuildModule(
      kDefaultEnvironment,
      [&errMsgs](spv_message_level_t level, const char *source,
                 const spv_position_t &position, const char *message) {
        const char *errorLevels[] = {"Fatal",   "Internal error", "Error",
                                     "Warning", "Info",           "Debug"};
        errMsgs << "[" << errorLevels[level] << "] " << source << " (L"
                << position.line << ":" << position.column << "): " << message
                << std::endl;
      },
      spvBlob.data(), spvBlob.size());

  // check id bound
  // Removed since this is infeasible - the spec requires 4000000+ id at least
  // if (irModule->max_id_bound() >= IdEnd - IdBegin) {
  //   throw std::runtime_error("Id exceed the tokenizer bound");
  // }

  spvtools::opt::Function *entryPointFn;
  uint32_t rootFunction;
  {
    int numEntrypoints = 0;
    for (auto &ep : irModule->module()->entry_points()) {
      if (numEntrypoints >= 1) {
        throw std::runtime_error("Not supported for now");
      } else {
        numEntrypoints++;
      }

      auto funcInstId = ep.GetSingleWordInOperand(kEntryPointFunctionIdInIdx);
      entryPointFn = irModule->GetFunction(funcInstId);
      rootFunction = entryPointFn->result_id();
    }
  }

  std::unordered_set<uint32_t> functionsToBeConverted;
  if (entrypointOnly) {
    irModule->CollectCallTreeFromRoots(rootFunction, &functionsToBeConverted);
    if (functionsToBeConverted.size() > 1) {
      throw std::runtime_error("Got >1 reachable");
    }
  } else {
    for (auto &f : *irModule->module()) {
      functionsToBeConverted.insert(f.result_id());
    }
  }

  assert(functionsToBeConverted.count(rootFunction) > 0);

  std::vector<uint32_t> functionsToBeProcessed;

  if (entrypointOnly) {
    functionsToBeProcessed.push_back(rootFunction);
  } else {
    // alway serialize entrypoint first
    // this gives implicit information on entrypoint position
    functionsToBeProcessed.push_back(rootFunction);

    // the rest are not guaranteed with a consistent order
    // Hence this mode is not recommended since it may harm prediction perf
    // TODO: eliminate this mode
    for (auto funcId : functionsToBeConverted) {
      if (funcId == rootFunction) {
        continue;
      }

      functionsToBeProcessed.push_back(funcId);
    }
  }

  if (relativeInstIdPos) {
    instSeqByResultId.clear();
    typeInstTokByResultId.clear();
    int curInstIdx = 0;

    for (auto &inst : irModule->module()->types_values()) {
      if (inst.HasResultId()) {
        instSeqByResultId[inst.result_id()] = curInstIdx;
        typeInstTokByResultId[inst.result_id()] = SymbolOffsets::IdBegin + inst.result_id();
      }

      curInstIdx++;
    }

    for (auto funcId: functionsToBeProcessed) {
      for (auto &bb: *irModule->GetFunction(funcId)) {
        bb.ForEachInst(
            [this, &tokenized,
            irCtx = irModule.get(), &curInstIdx](spvtools::opt::Instruction *inst) -> void {
              if (inst->HasResultId()) {
                instSeqByResultId[inst->result_id()] = curInstIdx;
              }

              curInstIdx++;
            },
            false
        );
      }
    }
  }

  // begin actual tokenization
  int curInstIdx = 0;

  // begin actual tokenization
  for (auto &inst : irModule->module()->types_values()) {
    // these things have no trace available so set to 0
    if (withTrace) {
      tokenizeInst(irModule.get(), tokenized, &traced, 0, &inst, curInstIdx);
    } else {
      tokenizeInst(irModule.get(), tokenized, nullptr, 0, &inst, curInstIdx);
    }

    curInstIdx++;
  }

  for (auto funcId: functionsToBeProcessed) {
    for (auto &bb: *irModule->GetFunction(funcId)) {
      if (withTrace) {
        uint32_t bbIdx = bb.id();
        int traceIdx = (*bbIdxMap)[bbIdx];
        int traceCnt = (*bbTraceCounters)[traceIdx];

        bb.ForEachInst(
          [this, &tokenized, &traced, traceCnt, &curInstIdx,
          irCtx = irModule.get()](spvtools::opt::Instruction *inst) -> void {
            tokenizeInst(irCtx, tokenized, &traced, traceCnt, inst, curInstIdx);
            curInstIdx++;
          },
          false
        );
      } else {
        bb.ForEachInst(
          [this, &tokenized, &curInstIdx,
          irCtx = irModule.get()](spvtools::opt::Instruction *inst) -> void {
            tokenizeInst(irCtx, tokenized, nullptr, 0, inst, curInstIdx);
            curInstIdx++;
          },
          false
        );
      }
    }
  }

  if (withTrace) {
    return std::make_tuple(
      tokenized,
      traced,
      errMsgs.str()
    );
  } else {
    return std::make_tuple(
      tokenized,
      std::nullopt,
      errMsgs.str()
    );
  }
  
}

// Reference: SPIRV-Tools/source/binary.cpp Parser::parseOperand
// 
// TODO: implement type reduction
void Tokenizer::tokenizeInst(spvtools::opt::IRContext *ctx,
                             std::vector<int> &tokVec,
                             std::vector<TraceCounter_t> *tokTraceVec,
                             TraceCounter_t traceCount,
                             spvtools::opt::Instruction *inst,
                             int curInstIdx) {

  if (tokTraceVec != nullptr) {
    tokTraceVec->push_back(traceCount);
  }
  tokVec.push_back(SymbolOffsets::OpCodeBegin + (uint32_t)inst->opcode());

  size_t padBegin = tokVec.size();
  for (auto &op : *inst) {
    if (spvIsIdType(op.type)) {
      // ID type operand

      if (relativeInstIdPos) {
        if (inst->HasResultId()
            && inst->result_id() == op.AsId()
            && typeInstTokByResultId.count(op.AsId()) == 0) {
          // this is the result id and should be processed in a relative id manner, so skip
          continue;
        }

        int idSym = 0;
        if (typeInstTokByResultId.count(op.AsId()) > 0) {
          idSym = typeInstTokByResultId[op.AsId()];
        } else if (instSeqByResultId.count(op.AsId()) == 0) {
          // use IdBegin as exception
          idSym = SymbolOffsets::IdBegin;
        } else {
          int offset = instSeqByResultId[op.AsId()] - curInstIdx;
          idSym = SymbolOffsets::IdRelativeZero + offset;

          if (idSym >= SymbolOffsets::IdEnd) {
            printf("Got distant relative offset: %d, clip to end\n", offset);
            idSym = SymbolOffsets::IdEnd;
          } else if (idSym < SymbolOffsets::IdBegin) {
            printf("Got distant relative offset: %d, clip to start\n", offset);
            idSym = SymbolOffsets::IdBegin;
          }
        }

        tokVec.push_back(idSym);
      } else {
        if (SymbolOffsets::IdBegin + op.AsId() >= SymbolOffsets::IdEnd) {
          throw std::runtime_error(
            "Id exceed max available length: Got Id="+ std::to_string(op.AsId())
          );
        }
        tokVec.push_back(SymbolOffsets::IdBegin + op.AsId());
      }
      
    } else if (op.type == SPV_OPERAND_TYPE_LITERAL_INTEGER ||
               op.type == SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER) {
      // literal number type operand
      // 1. determine number kind
      uint32_t numberKind;
      uint32_t numberBitWidth;

      if (op.type == SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER) {
        spvtools::opt::analysis::Type *refType = nullptr;
        if (inst->opcode() == ::spv::Op::OpSwitch) {
          uint32_t selectorVarId = inst->GetOperand(0).AsId();
          auto selectorVar = ctx->get_def_use_mgr()->GetDef(selectorVarId);
          if (!selectorVar->HasResultType()) {
            throw std::runtime_error(
              "Instruction defining %" + std::to_string(selectorVarId) + 
              " does not have a result id; Opcode = " +
              std::to_string((uint32_t)inst->opcode())
            );
          }

          refType = ctx->get_type_mgr()->GetType(selectorVar->type_id());
        } else {
          if (!inst->HasResultType()) {
            throw std::runtime_error(
              "Expected result id while parsing operand of type "
              "SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER; Opcode = " +
              std::to_string((uint32_t)inst->opcode())
            );
          }
          refType = ctx->get_type_mgr()->GetType(inst->type_id());
        }

        if (refType == nullptr) {
          throw std::runtime_error(
            "Expected non-null refType while parsing operand of type "
            "SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER; Opcode = " +
            std::to_string((uint32_t)inst->opcode())
          );
        }

        switch (refType->kind()) {
        case spvtools::opt::analysis::Type::kFloat:
          numberKind = SPV_NUMBER_FLOATING;
          numberBitWidth = refType->AsFloat()->width();
          break;
        case spvtools::opt::analysis::Type::kInteger:
          numberKind = refType->AsInteger()->IsSigned() ? SPV_NUMBER_SIGNED_INT
                                                        : SPV_NUMBER_UNSIGNED_INT;
          numberBitWidth = refType->AsInteger()->width();
          break;
        default:
          throw std::runtime_error(
              "Unexpected type to instruction which uses literal number");
        }
      } else {
        numberKind = SPV_NUMBER_UNSIGNED_INT;
        numberBitWidth = 32;
      }

      // 2. emit
      if (op.words.size() == 1) {
        switch (numberKind) {
        case SPV_NUMBER_SIGNED_INT:
          assert(numberBitWidth == 32);
          tokenizeInt64Literal(tokVec, int32_t(op.words[0]));
          break;
        case SPV_NUMBER_UNSIGNED_INT:
          assert(numberBitWidth == 32);
          tokenizeUInt64Literal(tokVec, op.words[0]);
          break;
        case SPV_NUMBER_FLOATING:
          if (numberBitWidth == 16) {
            throw std::runtime_error("TODO: support float16");
          } else {
            assert(numberBitWidth == 32);
            float reinterpret = *(float *)(&op.words[0]);
            tokenizeDoubleLiteral(tokVec, reinterpret);
          }
          break;
        default:
          assert(false);
          break;
        }
      } else if (op.words.size() == 2) {
        // Multi-word numbers are presented with lower order words first.
        uint64_t bits = uint64_t(op.words[0]) | (uint64_t(op.words[1]) << 32);
        assert(numberBitWidth == 64);
        switch (numberKind) {
        case SPV_NUMBER_SIGNED_INT:
          tokenizeInt64Literal(tokVec, int64_t(bits));
          break;
        case SPV_NUMBER_UNSIGNED_INT:
          tokenizeUInt64Literal(tokVec, bits);
          break;
        case SPV_NUMBER_FLOATING: {
          float reinterpret = *(double *)(&bits);
          tokenizeDoubleLiteral(tokVec, reinterpret);
        } break;
        default:
          assert(false);
          break;
        }
      }

    } else if (op.type == SPV_OPERAND_TYPE_EXTENSION_INSTRUCTION_NUMBER ||
               op.type == SPV_OPERAND_TYPE_SPEC_CONSTANT_OP_NUMBER) {
      // maybe float / int;
      // checked from Parser::parseOperand and
      // spvtools::disassemble::InstructionDisassembler::EmitOperand
      assert(op.words.size() == 1);
      tokenizeInt64Literal(tokVec, op.words[0]);
    } else if (op.type == SPV_OPERAND_TYPE_LITERAL_STRING) {
      // string type operand
      auto str = op.AsString();
      tokenizeStringLiteral(tokVec, str);
    } else {
      // treat as single uint32_t literal; if not possible, abort
      if (op.words.size() != 1) {
        throw std::runtime_error("Unhandled operand type");
      } else {
        tokenizeUInt64Literal(tokVec, op.words[0]);
      }
    }
  }

  size_t padEnd = tokVec.size();
  if (tokTraceVec != nullptr) {
    for (size_t i = padBegin; i < padEnd; i++) {
      tokTraceVec->push_back(0);
    }
  }
}

std::tuple<std::vector<int>, std::vector<vkExecute::TraceCounter_t>,
           std::string>
Tokenizer::tokenizeWithTrace(std::map<int, int> bbIdxMap,
                             std::vector<TraceCounter_t> bbTraceCounters) {
  // pass
  auto res = doTokenize(true, &bbIdxMap, &bbTraceCounters);

  if (!std::get<1>(res).has_value()) {
    throw std::runtime_error("Unexpected tokenize result while calling internal impl");
  }

  return std::make_tuple(
    std::get<0>(res),
    std::get<1>(res).value(),
    std::get<2>(res)
  );
}

void Tokenizer::tokenizeStringLiteral(std::vector<int> &tokVec,
                                      std::string literal) {
  for (auto ch : literal) {
    uint8_t uch = static_cast<uint8_t>(ch);

    // ensure we get positive add vars
    tokVec.push_back(SymbolOffsets::ByteEncodedLiteralBegin + uch);
  }
}

void Tokenizer::tokenizeInt64Literal(std::vector<int> &tokVec,
                                     int64_t integer) {
  return tokenizeStringLiteral(tokVec, std::to_string(integer));
}

void Tokenizer::tokenizeUInt64Literal(std::vector<int> &tokVec,
                                      uint64_t integer) {
  return tokenizeStringLiteral(tokVec, std::to_string(integer));
}

void Tokenizer::tokenizeDoubleLiteral(std::vector<int> &tokVec, double number) {
  return tokenizeStringLiteral(tokVec, std::to_string(number));
}

std::string Tokenizer::deTokenize(const std::vector<int> &tokens) {
  const spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;
  spv_opcode_table opcodeTable;
  spv_result_t res;

  res = spvOpcodeTableGet(&opcodeTable, target_env);
  assert(res == SPV_SUCCESS);

  bool inLiteral = false;
  std::vector<std::string> colElems;
  std::stringstream ss;

  auto emitColumn = [&colElems, &ss]() {
    bool start = true;
    for (auto &colElem : colElems) {
      if (start) {
        start = false;
      } else {
        ss << " ";
      }
      ss << colElem;
    }
    ss << "\n";
    colElems.clear();
  };

  for (auto token : tokens) {
    if (token < SymbolBegin || token >= SymbolEnd) {
      throw std::runtime_error("Unexpected token " + std::to_string(token));
    }

    if (token >= ByteEncodedLiteralBegin && token < ByteEncodedLiteralEnd) {
      if (!inLiteral) {
        colElems.push_back("\"");
        inLiteral = true;
      }
      colElems.back().push_back(token - ByteEncodedLiteralBegin);
      continue;
    }

    // close the literal if we're out
    if (inLiteral) {
      assert(colElems.size() > 1);
      colElems.back().push_back('"');
      inLiteral = false;
    }

    if (token >= IdBegin && token < IdEnd) {
      colElems.push_back("%");
      colElems.back().append(std::to_string(token - IdBegin));
      continue;
    }

    if (token >= OpCodeBegin && token < OpCodeEnd) {
      spv_opcode_desc entry;
      res = spvOpcodeTableValueLookup(target_env, opcodeTable,
                                      (::spv::Op)(token - OpCodeBegin), &entry);

      if (res != SPV_SUCCESS) {
        throw std::runtime_error("Unknown opcode");
      }

      // emit previous instruction if needed
      if (colElems.size() > 0) {
        emitColumn();
      }

      colElems.push_back("Op" + std::string(entry->name));
      continue;
    }

    if (token >= SpecialSymbolBegin && token < SpecialSymbolEnd) {
      // TODO: implement me
      int symOffset = token - SpecialSymbolBegin;
      switch (symOffset) {
        case Pad:
        case BoS:
        case EoS:
        case Mask:
        case Sep:
        case Unk:
        case Cls: {
          // emit previous instruction if needed
          // Might be in an illegal status, but 
          // will be okay for the next command
          if (colElems.size() > 0) {
            emitColumn();
          }

          colElems.push_back(specialSymbolNames[symOffset]);
          emitColumn();
        }

        break;
        default:
          throw std::runtime_error("Undefined special symbol");
      }
    }
  }

  return ss.str();
}
#include "spv/BasicBlock.hpp"
#include "source/opcode.h"
#include "spv/Type.hpp"

#include "Common.hpp"
#include "source/disassemble.h"
#include "source/latest_version_spirv_header.h"
#include "source/opt/build_module.h"
#include "source/opt/ir_context.h"
#include "source/opt/pass.h"
#include "source/opt/types.h"
#include "source/table.h"
#include "spirv-tools/libspirv.h"
#include "spirv-tools/optimizer.hpp"
#include "spirv/unified1/spirv.hpp11"
#include <algorithm>
#include <cstring>
#include <exception>
#include <iterator>
#include <memory>
#include <stdint.h>
#include <string>

using namespace vkExecute::spv;

Instr Instr::create(const spvtools::opt::Instruction *inst) {
  Instr newInst;
  // newInst.operands.resize(inst->NumOperandWords());
  newInst.opcode = static_cast<uint32_t>(inst->opcode());
  newInst.has_result_id = inst->HasResultId();
  newInst.has_type_id = inst->HasResultType();

  for (int i = 0; i < inst->NumOperands(); i++) {
    auto &operand = inst->GetOperand(i);
    auto myOperand = Operand();
    myOperand.data = std::vector(operand.words.begin(), operand.words.end());
    myOperand.type = operand.type;

    newInst.operands.push_back(myOperand);
  }

  return newInst;
}

void BasicBlock::dump() {
  const spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;
  spv_opcode_table opcode_table;
  spv_operand_table operand_table;
  spv_ext_inst_table ext_inst_table;
  spv_result_t result;

  // prepare tables
  {
    result = spvOpcodeTableGet(&opcode_table, target_env);
    if (result != SPV_SUCCESS) {
      throw std::runtime_error("Error while retriving opcode table");
    }

    result = spvOperandTableGet(&operand_table, target_env);
    if (result != SPV_SUCCESS) {
      throw std::runtime_error("Error while retriving operand table");
    }

    result = spvExtInstTableGet(&ext_inst_table, target_env);
    if (result != SPV_SUCCESS) {
      throw std::runtime_error("Error while retriving ext inst table");
    }
  }

  spv_context_t context{
      target_env,
      opcode_table,
      operand_table,
      ext_inst_table,
  };

  const spvtools::AssemblyGrammar grammar(&context);
  if (!grammar.isValid()) {
    throw std::runtime_error("Error while building assembly grammar");
  }

  uint32_t disassembly_options = SPV_BINARY_TO_TEXT_OPTION_PRINT;
  spvtools::NameMapper name_mapper = spvtools::GetTrivialNameMapper();

  spvtools::disassemble::InstructionDisassembler dis(
      grammar, std::cerr, disassembly_options, name_mapper);

  std::cerr << "============================" << std::endl;
  std::cerr << "=> Basic Block Instructions:" << std::endl;
  for (auto &inst : this->bbInstructions) {
    spv_parsed_instruction_t parsed_inst;
    std::vector<spv_parsed_operand_t> parsed_operands;
    std::vector<uint32_t> inst_binary;

    toParsedInstruction(inst, parsed_inst, parsed_operands, inst_binary);

    // do disasm
    dis.EmitInstruction(parsed_inst, 0);
  }

  std::cerr << std::endl;
  std::cerr << "=> Associated Inputs:" << std::endl;
  for (auto &inputVar : inputDescription) {
    std::cerr << "  - %" << inputVar.ssaId << ": "
              << associatedTypes[inputVar.type]->str() << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "=> Associated Outputs:" << std::endl;
  for (auto &outputVar : outputDescription) {
    std::cerr << "  - %" << outputVar.ssaId << ": "
              << associatedTypes[outputVar.type]->str() << std::endl;
  }

  std::cerr << std::endl;
  std::cerr << "=> Associated Resources:" << std::endl;
  for (auto &resInsts: resourceDescription) {
    resInsts.dump();
  }

  std::cerr << "============================" << std::endl;
}

void Instr::dump() {
  const spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;
  spv_opcode_table opcodeTable;
  spv_result_t res;

  res = spvOpcodeTableGet(&opcodeTable, target_env);
  assert(res == SPV_SUCCESS);
  
  spv_opcode_desc entry;
  res = spvOpcodeTableValueLookup(target_env, opcodeTable, (::spv::Op)this->opcode, &entry);
  assert(res == SPV_SUCCESS);

  std::cerr << entry->name << std::endl;
}

void BasicBlock::toParsedInstruction(
    const Instr &inst, spv_parsed_instruction_t &parsed_inst,
    std::vector<spv_parsed_operand_t> &parsed_operands,
    std::vector<uint32_t> &inst_binary) {

  // prepare inst binary
  {
    // +1 is for the (opcode & wordcount) uint32_t word
    uint32_t num_words = inst.numOperandWords() + 1;
    inst_binary.push_back((num_words << 16) |
                          static_cast<uint16_t>(inst.opcode));
    for (const auto &operand : inst.operands) {
      inst_binary.insert(inst_binary.end(), operand.data.begin(),
                         operand.data.end());
    }
  }

  parsed_inst.opcode = inst.opcode;
  parsed_inst.num_words = static_cast<uint16_t>(inst_binary.size());
  parsed_inst.words = inst_binary.data();

  // See Differ::GetExtInstType
  {
    if (inst.opcode == static_cast<uint32_t>(::spv::Op::OpExtInst)) {
      // TODO: fix me; use spvExtInstImportTypeGet & referred type id to
      // retrieve associated info
      parsed_inst.ext_inst_type = SPV_EXT_INST_TYPE_GLSL_STD_450;
    } else {
      parsed_inst.ext_inst_type = SPV_EXT_INST_TYPE_NONE;
    }
  }

  parsed_inst.type_id = inst.hasResultType() ? inst.getSingleWordOperand(0) : 0;
  parsed_inst.result_id = inst.hasResultId() ? inst.result_id() : 0;

  // fill parsed_operands
  {
    parsed_operands.resize(inst.numOperands());
    parsed_inst.operands = parsed_operands.data();
    parsed_inst.num_operands = static_cast<uint16_t>(parsed_operands.size());

    // Word 0 is always op and num_words, so operands start at offset 1.
    uint32_t offset = 1;
    for (uint16_t operand_index = 0; operand_index < parsed_inst.num_operands;
         ++operand_index) {
      auto &operand = inst.getOperand(operand_index);
      spv_parsed_operand_t &parsed_operand = parsed_operands[operand_index];

      parsed_operand.offset = static_cast<uint16_t>(offset);
      parsed_operand.num_words = static_cast<uint16_t>(operand.data.size());

      parsed_operand.type = operand.type;

      // number kind
      {
        switch (parsed_operand.type) {
        case SPV_OPERAND_TYPE_LITERAL_INTEGER:
        case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER:
          // Always unsigned integers.
          parsed_operand.number_bit_width = 32;
          parsed_operand.number_kind = SPV_NUMBER_UNSIGNED_INT;
          break;
        case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
        case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
          switch (::spv::Op(inst.opcode)) {
          case ::spv::Op::OpSwitch:
          case ::spv::Op::OpConstant:
          case ::spv::Op::OpSpecConstant: {
            // need to lookup the type involved
            uint32_t id_to = inst.type_id();
            auto &refType = associatedTypes.at(id_to);
            if (refType->getKind() == Type::Kind::kFloat) {
              parsed_operand.number_bit_width = refType->asFloat()->width;
              parsed_operand.number_kind = SPV_NUMBER_FLOATING;
            } else if (refType->getKind() == Type::Kind::kInteger) {
              parsed_operand.number_bit_width = refType->asInteger()->width;
              parsed_operand.number_kind = refType->asInteger()->signedness
                                               ? SPV_NUMBER_SIGNED_INT
                                               : SPV_NUMBER_UNSIGNED_INT;
            }
          } break;
          default:
            assert(false && "Unreachable");
            break;
          }
          break;
        default:
          break;
        }
      }

      offset += parsed_operand.num_words;
    }
  }
}

void BasicBlock::eliminateOpVariable() {
  for (auto instIt = bbInstructions.begin(); instIt != bbInstructions.end();
       instIt++) {
    if (::spv::Op(instIt->opcode) == ::spv::Op::OpVariable) {
      Register extVariable;
      extVariable.type = instIt->type_id();
      extVariable.ssaId = instIt->result_id();

      inputDescription.push_back(extVariable);
      instIt = bbInstructions.erase(instIt);
    }
  }
}

Instr Instr::create(uint32_t opcode, bool has_result_id, bool has_type_id,
                    std::vector<Operand> operands) {
  Instr ret;
  ret.opcode = opcode;
  ret.has_result_id = has_result_id;
  ret.has_type_id = has_type_id;
  ret.operands = operands;

  return ret;
}

Operand Operand::create(spv_operand_type_t type, OperandData data) {
  Operand operand;
  operand.type = type;
  operand.data = data;
  return operand;
}

Operand Operand::createTypeId(uint32_t type_id) {
  return create(SPV_OPERAND_TYPE_TYPE_ID, {type_id});
}

Operand Operand::createId(uint32_t type_id) {
  return create(SPV_OPERAND_TYPE_ID, {type_id});
}

Operand Operand::createResultId(uint32_t result_id) {
  return create(SPV_OPERAND_TYPE_RESULT_ID, {result_id});
}

// A string is interpreted as a nul-terminated stream of characters.
// All string comparisons are case sensitive. The character set is
// Unicode in the UTF-8 encoding scheme. The UTF-8 octets (8-bit bytes)
// are packed four per word, following the little-endian convention
// (i.e., the first octet is in the lowest-order 8 bits of the word).
Operand Operand::createLiteralString(const std::u8string literal) {
  Operand operand;
  operand.type = SPV_OPERAND_TYPE_LITERAL_STRING;

  size_t literalOffset = 0, packOffset = 0;
  uint32_t packed = 0;
  // the extra +1 is for '\0'
  while (literalOffset < literal.size() + 1) {
    // TODO: fix for big endian - this only works for small endian
    packed |= (literal.c_str()[literalOffset] << packOffset);
    packOffset += 8;

    if (packOffset == 32) {
      operand.data.push_back(packed);
      packOffset = 0;
      packed = 0;
    }
    literalOffset++;
  }

  if (packOffset != 0) {
    operand.data.push_back(packed);
  }

  return operand;
}

Operand Operand::createLiteralInteger(uint32_t integer) {
  return create(SPV_OPERAND_TYPE_LITERAL_INTEGER, {integer});
}

Operand Operand::createLiteralF32(float f) {
  uint32_t casted = 0;
  float *pf = reinterpret_cast<float *>(&casted);
  *pf = f;

  return create(SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER, {casted});
}

// TODO: implement me
BasicBlock BasicBlock::concatenate(const std::vector<const BasicBlock *> bbs) {
  BasicBlock destBB;

  return destBB;
}

void BasicBlock::makeCanonical() {
  if (isCanonical) {
    return;
  }

  eliminateOpVariable();

  // reorder id used
  reorderId();

  isCanonical = true;
}

// See also spvtools::opt::CompactIdsPass
void BasicBlock::reorderId() {
  std::map<uint32_t, uint32_t> externalIdMap;
  std::map<uint32_t, uint32_t> resultIdMap;
  std::map<uint32_t, uint32_t> typeIdMap;

  uint32_t curId = 1;

  // TODO: implement me and implement basic block comparaison
}


std::vector<Register>::iterator BasicBlock::findInputValue(uint32_t ssaId) {
  for (size_t i = 0; i < inputDescription.size(); i++) {
    if (inputDescription[i].ssaId == ssaId) {
      return inputDescription.begin() + i;
    }
  }

  return inputDescription.end();
}

std::vector<Register>::iterator BasicBlock::findOutputValue(uint32_t ssaId) {
  for (size_t i = 0; i < outputDescription.size(); i++) {
    if (outputDescription[i].ssaId == ssaId) {
      return outputDescription.begin() + i;
    }
  }

  return outputDescription.end();
}

std::vector<Resource>::iterator BasicBlock::findResourceValue(uint32_t ssaId) {
  for (size_t i = 0; i < resourceDescription.size(); i++) {
    if (resourceDescription[i].ssaId == ssaId) {
      return resourceDescription.begin() + i;
    }
  }

  return resourceDescription.end();
}
#pragma once

#include "Common.hpp"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"
#include <memory>
#include <variant>

// forward declaration
// to keep things separate
namespace spvtools::opt {
class IRContext;
class Instruction;
namespace analysis {
class Type;
}
} // namespace spvtools::opt

namespace vkExecute::spv {

// forward declaration of Type
namespace Type {
struct Type;
}; // namespace Type

struct Operand {
  using OperandData = std::vector<uint32_t>;

  spv_operand_type_t type; // Type of this logical operand.
  OperandData data;

  inline bool operator==(const Operand &rhs) const {
    if (type != rhs.type || data.size() != rhs.data.size()) {
      return false;
    }
    for (size_t i = 0; i < data.size(); i++) {
      if (data[i] != rhs.data[i]) {
        return false;
      }
    }

    return true;
  }

  // generic create helper
  static Operand create(spv_operand_type_t type, OperandData data);

  static Operand createId(uint32_t type_id);
  static Operand createTypeId(uint32_t type_id);
  static Operand createResultId(uint32_t result_id);
  static Operand createLiteralString(const std::u8string literal);
  static Operand createLiteralInteger(uint32_t integer);
  static Operand createLiteralF32(float f);
};

// in a way that is easy to do conversion from spvtools::opt::Instruction and
// this
struct Instr {
  uint32_t opcode;
  std::vector<Operand> operands;
  bool has_result_id;
  bool has_type_id;

  static Instr create(uint32_t opcode, bool has_result_id, bool has_type_id,
                      std::vector<Operand> operands);

  static inline Instr create(::spv::Op opcode, bool has_result_id,
                             bool has_type_id, std::vector<Operand> operands) {
    return create(static_cast<uint32_t>(opcode), has_result_id, has_type_id,
                  operands);
  }

  inline uint32_t typeResultIdCount() const {
    if (has_type_id && has_result_id)
      return 2;
    if (has_type_id || has_result_id)
      return 1;
    return 0;
  }

  inline uint32_t numInOperandWords() const {
    uint32_t size = 0;
    for (uint32_t i = typeResultIdCount(); i < operands.size(); ++i)
      size += static_cast<uint32_t>(operands[i].data.size());
    return size;
  }

  inline uint32_t numOperandWords() const {
    return numInOperandWords() + typeResultIdCount();
  }

  inline bool hasResultType() const { return has_type_id; }
  inline bool hasResultId() const { return has_result_id; }

  inline const Operand &getOperand(uint32_t index) const {
    assert(index < operands.size() && "operand index out of bound");
    return operands[index];
  }

  inline uint32_t getSingleWordOperand(uint32_t index) const {
    const auto &words = getOperand(index);
    assert(words.data.size() == 1 &&
           "expected the operand only taking one word");
    return words.data.front();
  }

  uint32_t type_id() const { return has_type_id ? getSingleWordOperand(0) : 0; }
  uint32_t result_id() const {
    return has_result_id ? getSingleWordOperand(has_type_id ? 1 : 0) : 0;
  }
  uint32_t numOperands() const {
    return static_cast<uint32_t>(operands.size());
  }

  inline std::vector<uint32_t> toSpvBinary() const {
    std::vector<uint32_t> inst_binary;
    // +1 is for the (opcode & wordcount) uint32_t word
    uint32_t num_words = numOperandWords() + 1;
    inst_binary.push_back((num_words << 16) | static_cast<uint16_t>(opcode));
    for (const auto &operand : operands) {
      inst_binary.insert(inst_binary.end(), operand.data.begin(),
                         operand.data.end());
    }

    return inst_binary;
  }

  static Instr create(const spvtools::opt::Instruction *inst);
  // not so feature rich
  void dump();
};

// Mutable; we could permute this
struct Register {
  uint32_t type;
  uint32_t ssaId;
};

// Unlikely to get changed; but we still need it
// Current considered:
// - OpExtInstImport "GLSL.std.450"

struct ExtInstImportResource {
  std::u8string extInstSetName;
};

struct Resource {
  enum Type {
    Invalid,
    // Put things in (the very) front and your program will be ok
    ExtInstImport
  };

  Type type;
  uint32_t ssaId;
  std::variant<std::monostate, ExtInstImportResource> payload;

  void dump() {
    switch (type) {
      case ExtInstImport: {
        auto &concretePayload = std::get<ExtInstImportResource>(payload);
        
        std::cerr << "ExtInstSetId=" << std::to_string(ssaId)
              << ", name=" << toString(concretePayload.extInstSetName) << std::endl;

        break;
      }
      default:
        assert(false);
    }
  }
};

struct BasicBlock {
  BasicBlock() = default;
  BasicBlock(const BasicBlock &) = delete;
  BasicBlock(BasicBlock &&) = default;
  BasicBlock &operator=(const BasicBlock &) = delete;
  BasicBlock &operator=(BasicBlock &&) = default;

  std::vector<Register> inputDescription;
  std::vector<Register> outputDescription;
  std::vector<Resource> resourceDescription;
  std::vector<Instr> bbInstructions;
  std::map<uint32_t, std::unique_ptr<Type::Type>> associatedTypes;

  bool isCanonical = false;

  void dump();

  void toParsedInstruction(const Instr &inst,
                           spv_parsed_instruction_t &parsed_inst,
                           std::vector<spv_parsed_operand_t> &parsed_operands,
                           std::vector<uint32_t> &inst_binary);

  std::vector<Register>::iterator findInputValue(uint32_t ssaId);
  std::vector<Register>::iterator findOutputValue(uint32_t ssaId);
  std::vector<Resource>::iterator findResourceValue(uint32_t ssaId);

  // Move OpVariable into inputDescription
  void eliminateOpVariable();

  // // eliminate load & store
  // void stripLoadStore();

  void reorderId();

  void makeCanonical();

  static BasicBlock concatenate(const std::vector<const BasicBlock *> bbs);
};

}; // namespace vkExecute::spv
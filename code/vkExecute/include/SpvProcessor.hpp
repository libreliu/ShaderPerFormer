#pragma once

#include "Common.hpp"
#include "spv/BasicBlock.hpp"
#include "spv/Type.hpp"
#include <cstring>
#include <stdexcept>

namespace vkExecute {

// Wrapper for SPIRV-Tools
class SpvProcessor {
public:
  inline void loadSpv(const BinaryBlob &chrBlob) {
    if (chrBlob.size() % sizeof(uint32_t) != 0) {
      throw std::runtime_error("Not valid SPIR-V blob");
    }

    spvBlob.resize(chrBlob.size() / sizeof(uint32_t));
    memcpy(spvBlob.data(), chrBlob.data(), chrBlob.size());
  }

  inline BinaryBlob exportSpv() const { return toBinaryBlob(spvBlob); }

  // direct wrapping of SPIRV-Tools
  std::tuple<std::string, std::string> disassemble();
  std::tuple<bool, std::string> exhaustiveInlining();
  std::vector<spv::BasicBlock> separateBasicBlocks();
  std::tuple<bool, std::string> assemble(std::string asmText);
  std::tuple<bool, std::string> validate();
  std::map<int, int> instrumentBasicBlockTrace(bool traceWithU64);
  std::tuple<bool, std::string>
  runPassSequence(std::vector<std::string> passSequence);

private:
  std::vector<uint32_t> spvBlob;
};

}; // namespace vkExecute

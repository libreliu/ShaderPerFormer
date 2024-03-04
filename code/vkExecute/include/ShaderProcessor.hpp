#pragma once

#include <vector>
#include <cstdint>
#include <string>

#include "Common.hpp"

namespace vkExecute {
class ShaderProcessor {
public:

  enum ShaderStages {
    VERTEX,
    TESSCONTROL,
    TESSEVALUATION,
    GEOMETRY,
    FRAGMENT,
    COMPUTE
  };

  static std::tuple<BinaryBlob, std::string> compileShaderToSPIRV_Vulkan(
    ShaderStages stage, const char *shaderSource, const char *shaderName
  );

  inline void loadSpv(BinaryBlob spvBlob) {
    this->spvBlob = std::move(spvBlob);
  }

  std::string disassemble();

private:

  BinaryBlob spvBlob;
};

const void *get_glslang_builtin_resource();

} // namespace vkExecute

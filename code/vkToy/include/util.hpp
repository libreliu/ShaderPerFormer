#pragma once

// #include "common.hpp"
// #include "glslang/Include/glslang_c_interface.h"
// #include "glslang/Public/resource_limits_c.h"
// #include "glslang/Public/ShaderLang.h"
// #include "StandAlone/DirStackFileIncluder.h"
// #include "glslang/Public/ResourceLimits.h"
// #include "glslang/Include/ShHandle.h"

// #include "glslang/Include/ResourceLimits.h"
// #include "glslang/MachineIndependent/Versions.h"
// #include "glslang/MachineIndependent/localintermediate.h"
// #include <utility>

// std::vector<uint32_t> compileShaderToSPIRV(
//   EShLanguage stage,
//   std::vector<std::string> shaderCodes
// ) {
//   for (auto &shaderSrc: shaderCodes) {
//     // this is a silly but efficient way to make lifetime correct
//     // since glslang requires "const char * const *" as its code input
//     auto shader = std::make_unique<std::pair<glslang::TShader, const char *>>(
//       std::make_pair(stage, nullptr)
//     );
//     shader->second = shaderSrc.c_str();
//     shader->first.setStrings(&shader->second, 1);
//     shader->first.setEnvInput(glslang::EShSourceGlsl, stage,
//                         glslang::EShClientVulkan, 100);
//     shader->first.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_2);
//     shader->first.setEnvTarget(glslang::EShTargetSpv,
//                         glslang::EShTargetSpv_1_5);

//     DirStackFileIncluder Includer;
//     /* TODO: use custom callbacks if they are available in 'i->callbacks' */
//     return shader->first.preprocess(
//         reinterpret_cast<const TBuiltInResource*>(input->resource),
//         100,
//         ENoProfile,
//         input->force_default_version_and_profile != 0,
//         input->forward_compatible != 0,
//         (EShMessages)c_shader_messages(input->messages),
//         &shader->preprocessedGLSL,
//         Includer
//     );
//   }
// }



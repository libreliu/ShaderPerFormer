#include "ShaderProcessor.hpp"
#include <exception>

#include <glslang/Include/glslang_c_shader_types.h>

#include <glslang/Include/glslang_c_interface.h>
#include <glslang/Public/resource_limits_c.h>
#include <sstream>
#include <tuple>

#include "spirv-tools/libspirv.h"

using namespace vkExecute;

static glslang_stage_t ToGlslangStages(ShaderProcessor::ShaderStages stage) {
  switch (stage) {
  case ShaderProcessor::VERTEX:
    return glslang_stage_t::GLSLANG_STAGE_VERTEX;
  case ShaderProcessor::FRAGMENT:
    return glslang_stage_t::GLSLANG_STAGE_FRAGMENT;
  default:
    throw std::out_of_range("Invalid shader stage");
  }
}

std::tuple<BinaryBlob, std::string> ShaderProcessor::compileShaderToSPIRV_Vulkan(
    ShaderStages stage, const char *shaderSource, const char *shaderName) {
  static bool glslangInitialized = false;
  if (!glslangInitialized) {
    glslang_initialize_process();
    glslangInitialized = true;
  }

  // https://registry.khronos.org/OpenGL/specs/es/3.1/GLSL_ES_Specification_3.10.pdf
  // https://www.wikiwand.com/en/OpenGL_Shading_Language
  // ShaderToy uses GLES 300, but 310 have all the features included in 300
  // and glslang complains for el support lower than 310
  // so use 310 and es profile here
  const glslang_input_t input = {
      .language = GLSLANG_SOURCE_GLSL,
      .stage = ToGlslangStages(stage),
      .client = GLSLANG_CLIENT_VULKAN,
      .client_version = GLSLANG_TARGET_VULKAN_1_2,
      .target_language = GLSLANG_TARGET_SPV,
      .target_language_version = GLSLANG_TARGET_SPV_1_5,
      .code = shaderSource,
      .default_version = 310,
      .default_profile = GLSLANG_ES_PROFILE,
      .force_default_version_and_profile = false,
      .forward_compatible = false,
      .messages = GLSLANG_MSG_DEFAULT_BIT,
      .resource = reinterpret_cast<const glslang_resource_t *>(
          vkExecute::get_glslang_builtin_resource())};

  glslang_shader_t *shader = glslang_shader_create(&input);

  if (!glslang_shader_preprocess(shader, &input)) {
    VKEXECUTE_WARN("GLSL preprocessing failed %s\n", shaderName);

    std::stringstream ss;
    ss << "GLSL preprocessing failed.\n";
    ss << "=> glslang_program_get_info_log():\n";
    ss << glslang_shader_get_info_log(shader);
    ss << "\n";
    ss << "=> glslang_shader_get_info_debug_log():\n";
    ss << glslang_shader_get_info_debug_log(shader);
    ss << "\n";
    ss << "=> input.code:\n";
    ss << input.code;

    glslang_shader_delete(shader);
    return std::make_tuple(std::vector<char>(), "");
  }

  if (!glslang_shader_parse(shader, &input)) {
    VKEXECUTE_WARN("GLSL parsing failed %s\n", shaderName);

    std::stringstream ss;
    ss << "GLSL parsing failed.\n";
    ss << "=> glslang_program_get_info_log():\n";
    ss << glslang_shader_get_info_log(shader);
    ss << "\n";
    ss << "=> glslang_shader_get_info_debug_log():\n";
    ss << glslang_shader_get_info_debug_log(shader);
    ss << "\n";
    ss << "=> glslang_shader_get_preprocessed_code():\n";
    ss << glslang_shader_get_preprocessed_code(shader);

    glslang_shader_delete(shader);
    return std::make_tuple(std::vector<char>(), ss.str());
  }

  glslang_program_t *program = glslang_program_create();
  glslang_program_add_shader(program, shader);

  if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT |
                                         GLSLANG_MSG_VULKAN_RULES_BIT)) {
    VKEXECUTE_WARN("GLSL linking failed %s\n", shaderName);

    std::stringstream ss;
    ss << "GLSL linking failed.\n";
    ss << "=> glslang_program_get_info_log():\n";
    ss << glslang_program_get_info_log(program);
    ss << "\n";
    ss << "=> glslang_program_get_info_debug_log():\n";
    ss << glslang_program_get_info_debug_log(program);

    glslang_program_delete(program);
    glslang_shader_delete(shader);
    return std::make_tuple(std::vector<char>(), ss.str());
  }

  std::stringstream ss;

  glslang_program_SPIRV_generate(program, ToGlslangStages(stage));

  std::vector<char> outShaderModule(glslang_program_SPIRV_get_size(program) *
                                    sizeof(unsigned int));
  glslang_program_SPIRV_get(
      program, reinterpret_cast<unsigned int *>(outShaderModule.data()));

  const char *spirv_messages = glslang_program_SPIRV_get_messages(program);
  if (spirv_messages) {
    VKEXECUTE_WARN("(%s) %s\b", shaderName, spirv_messages);
    ss << "spirv_messages:\n";
    ss << spirv_messages;
  }

  glslang_program_delete(program);
  glslang_shader_delete(shader);

  return std::make_tuple(outShaderModule, ss.str());
}

std::string ShaderProcessor::disassemble() {
  std::string output = "";

  // disassemble
  spv_context context = spvContextCreate(spv_target_env::SPV_ENV_VULKAN_1_3);
  spv_text text;
  spv_diagnostic diagnostic = nullptr;
  spvBinaryToText(context, reinterpret_cast<const uint32_t *>(spvBlob.data()),
                  spvBlob.size() / sizeof(unsigned int),
                  SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES |
                      SPV_BINARY_TO_TEXT_OPTION_INDENT,
                  &text, &diagnostic);

  // dump
  if (diagnostic == nullptr)
    output = text->str;
  else
    spvDiagnosticPrint(diagnostic);

  // teardown
  spvDiagnosticDestroy(diagnostic);
  spvContextDestroy(context);

  return output;
}
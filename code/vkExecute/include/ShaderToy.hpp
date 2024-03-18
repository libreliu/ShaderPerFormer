#pragma once

#include <string>
#include <filesystem>
#include <optional>
#include <iostream>
#include <sstream>
#include <exception>
#include <nlohmann/json.hpp>
#include <stb_image.h>
#include <auto_vk_toolkit.hpp>

#include "Common.hpp"

//
//template<class Archive>
//inline void serialize(Archive& archive, glm::vec3 & glm_vec3)
//{
//  archive(cereal::make_nvp("x", glm_vec3.x), cereal::make_nvp("y", glm_vec3.y), cereal::make_nvp("z", glm_vec3.z));
//}
//
//template<class Archive>
//inline void serialize(Archive& archive, glm::vec4& glm_vec4)
//{
//  archive(
//    cereal::make_nvp("x", glm_vec4.x),
//    cereal::make_nvp("y", glm_vec4.y),
//    cereal::make_nvp("z", glm_vec4.z),
//    cereal::make_nvp("w", glm_vec4.w)
//  );
//}

class ShaderToy {
public:
  using path = std::filesystem::path;
  using binary_blob = std::vector<uint8_t>;

  typedef enum {
      SHADERTOY_SHADER_STAGE_VERTEX,
      SHADERTOY_SHADER_STAGE_TESSCONTROL,
      SHADERTOY_SHADER_STAGE_TESSEVALUATION,
      SHADERTOY_SHADER_STAGE_GEOMETRY,
      SHADERTOY_SHADER_STAGE_FRAGMENT,
      SHADERTOY_SHADER_STAGE_COMPUTE
  } ShaderToyStages;

  struct ImageUniformBlock {
    glm::vec3 iResolution;
    float iTime = 0;
    // NOTE: this is float iChannelTime[4] inside shader block
    // doing this is for the ease of serialization
    glm::vec4 iChannelTime;
    glm::vec4 iMouse;
    glm::vec4 iDate;
    float iSampleRate;
    glm::vec3 iChannelResolution;
    int iFrame = 0;
    float iTimeDelta;
    float iFrameRate;

    template<class Archive>
    inline void serialize(Archive& archive) {
      archive(CEREAL_NVP(iResolution), CEREAL_NVP(iTime), CEREAL_NVP(iChannelTime));
      archive(
        CEREAL_NVP(iMouse), CEREAL_NVP(iDate), CEREAL_NVP(iSampleRate),
        CEREAL_NVP(iChannelResolution), CEREAL_NVP(iFrame), CEREAL_NVP(iTimeDelta),
        CEREAL_NVP(iFrameRate)
      );
    }
  };


  static inline std::unique_ptr<ShaderToy> parse(
    path shaderRoot,
    std::string shaderID,
    std::optional<path> jsonName = {},
    bool cacheRead = false
  ) {
    path jPath = jsonName.value_or(findJsonPathByShaderID(shaderRoot, shaderID));

    if (jPath.empty()) {
      throw std::invalid_argument(
        "Parse failed: can't locate for shader (ID=" + shaderID + ")"
      );
    }

    std::unique_ptr<ShaderToy> toy = std::make_unique<ShaderToy>();
    toy->shaderRoot = shaderRoot;

    std::ifstream jF(jPath);
    toy->mainData = nlohmann::json::parse(jF);
    for (auto &renderpass: toy->mainData["Shader"]["renderpass"]) {
      for (auto &inputCh: renderpass["inputs"]) {
        if (inputCh.object().find("src") == inputCh.object().end()) {
          VKEXECUTE_WARN("Input channel doesn't have src field available");
          continue;
        }

        auto srcName = inputCh["src"].get<std::string>();
        VKEXECUTE_LOG("Asset blob related: %s", srcName.c_str());
      }
    }

    // todo: sanitize

    return toy;
  }

  inline std::vector<nlohmann::json> getRenderPassMultiple(std::string type) {
    std::vector<nlohmann::json> rpObjects;
    for (auto &obj: mainData["Shader"]["renderpass"]) {
      if (obj.find("type") != obj.end() && obj["type"] == type) {
        rpObjects.push_back(obj);
      }
    }
    return rpObjects;
  }

  inline std::optional<nlohmann::json> getRenderPassSingle(std::string type) {
    auto rpObjects = getRenderPassMultiple(type);
    if (rpObjects.size() != 1) {
      VKEXECUTE_WARN("Got %zu passes of type %s, expected 1.", rpObjects.size(), type.c_str());
      return {};
    } else {
      return std::optional<nlohmann::json>(rpObjects[0]);
    }
  }

  inline std::tuple<avk::image, avk::command::action_type_command> getMediaImage(std::string mediaURL) {
    // retrive blob and convert to image
    auto mediaPath = getMediaPath(mediaURL);
    auto imgData = avk::image_data(mediaPath.string());

    // Will do
    // - create image with avk::context().create_image()
    //   - create VkImage object, and alloc memory using avk::vma_handle or avk::mem_handle
    // - create staging buffer with avk::context().create_buffer()
    //   - create VkBuffer object, alloc memory with memory type host_coherent
    // - fill with buffer_t::fill() -> returns command::action_type_command
    //   - this will not emit any command, and will do map -> memcpy -> unmap
    // - assemble copyBufferToMemory, mipmap gen, image mem barriers along with image layout transitions
    // - return the command buffer and the image created
    return avk::create_image_from_file(mediaPath.string());
  }

  inline std::string getCommonCode() {
    auto commonPass = getRenderPassSingle("common");
    if (!commonPass.has_value()) {
      return "";
    }

    return commonPass->at("code");
  }

  inline std::tuple<std::vector<char>, std::string> prepareVertexShader() {
    if (bufImgVertexShaderSpv.has_value()) {
      // An implicit copy construction here - but since codes are small so OK
      return std::make_tuple(bufImgVertexShaderSpv.value(), bufImgVertexShaderSpvCode);
    }

    std::stringstream ssCode;
    {
      ssCode << "#version 310 es\n";
      ssCode << "precision highp float;\n";
      ssCode << "precision highp int;\n";
      ssCode << "precision mediump sampler3D;\n";
      ssCode << "layout(location = 0) in vec3 inPosition;\n";
      ssCode << "void main() {gl_Position = vec4(inPosition, 1.0);}\n";
    }

    bufImgVertexShaderSpvCode = ssCode.str();
    auto res = compileShaderToSPIRV_Vulkan(
      SHADERTOY_SHADER_STAGE_VERTEX,
      bufImgVertexShaderSpvCode.c_str(),
      "common-vertex"
    );

    if (res.size() == 0) {
      throw std::runtime_error("Failed to compile vertex shader");
    }

    bufImgVertexShaderSpv = std::move(res);

    return std::make_tuple(bufImgVertexShaderSpv.value(), bufImgVertexShaderSpvCode);
  }

  inline std::tuple<std::vector<char>, std::string> prepareFragmentShader() {
    if (true) {
      std::stringstream ssCode;
      // preamble
      {
        ssCode << "#version 310 es\n";
        ssCode << "precision highp float;\n";
        ssCode << "precision highp int;\n";
        ssCode << "precision mediump sampler3D;\n";
      }

      // shader output declaration
      // note: the location here needs to be consistent with
      //       the values in RenderPass's subpass creation
      {
        ssCode << "layout(location = 0) out vec4 outColor;\n";
      }

      // uniform declaration
      // -> Writing only uniform type name and no struct name will
      //    import all members into shader program
      // https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL)#Uniform_blocks
      {
        ssCode << "layout (binding=0) uniform PrimaryUBO {\n";
        ssCode << "  uniform vec3 iResolution;\n";
        ssCode << "  uniform float iTime;\n";
        ssCode << "  uniform float iChannelTime[4];\n";
        ssCode << "  uniform vec4 iMouse;\n";
        ssCode << "  uniform vec4 iDate;\n";
        ssCode << "  uniform float iSampleRate;\n";
        ssCode << "  uniform vec3 iChannelResolution[4];\n";
        ssCode << "  uniform int iFrame;\n";
        ssCode << "  uniform float iTimeDelta;\n";
        ssCode << "  uniform float iFrameRate;\n";
        ssCode << "};\n";
      }

      // main program declaration
      {
        ssCode << "void mainImage(out vec4 c, in vec2 f);\n";
        ssCode << "void main() {mainImage(outColor, gl_FragCoord.xy);}\n";
      }

      // ssCode << getCommonCode() << "\n";
      // ssCode << renderPass["code"].get<std::string>();
      ssCode << shaderSrc;

      std::string codeStr = ssCode.str();
      auto spvBlob = compileShaderToSPIRV_Vulkan(
        SHADERTOY_SHADER_STAGE_FRAGMENT,
        codeStr.c_str(), "image-or-buffer-fragment"
      );

      if (spvBlob.size() == 0) {
        // throw std::runtime_error("Failed to compile fragment shader");
        setShaderSrc("void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n fragColor = vec4(0.0,1.0,1.0,1.0);\n}");
        return prepareFragmentShader();
      }

      return std::make_tuple(spvBlob, codeStr);
    } 
  }

  void setShaderSrc(std::string shaderSrc) {
    this->shaderSrc = shaderSrc;
  }

protected:
  static inline path findJsonPathByShaderID(
    const path &shaderRoot,
    const std::string &shaderID
  ) {
    path jsonFolder = shaderRoot / path(/*
      "json/" + shaderID.substr(0, 1) +
      "/" + shaderID.substr(1, 1) + "/" +
      shaderID.substr(2, shaderID.size() - 2) + "/"*/
        "E:/NGPP/Shadertoy-Offline-Database/shaders/json"
    );
    path jsonPath;

    // This is safe, since shaderID only contains ASCII
    std::u8string shaderIDInU8(reinterpret_cast<const char8_t*>(shaderID.c_str()));

    for (const auto &entry: std::filesystem::directory_iterator(jsonFolder)) {
      // std::cout << entry.path() << std::endl;
      // NOTE: don't use .string() here - will cause std::system_error on Windows with filename
      // characters that not representable using currrent locale setting
      auto filename = entry.path().filename().u8string();
      if (filename.rfind(shaderIDInU8, 0) == 0) {
        jsonPath = entry.path();
        break;
      }
    }

    return jsonPath;
  }

  inline path getMediaPath(std::string mediaURL) {
    return shaderRoot / path("media/") / path(mediaURL);
  }

  // https://stackoverflow.com/questions/18816126/c-read-the-whole-file-in-buffer
  static inline binary_blob readBlob(const path &blobPath) {
    auto file = std::ifstream(blobPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    binary_blob blob(size);
    file.read(reinterpret_cast<char *>(blob.data()), size);

    return blob;
  }

  std::vector<char> compileShaderToSPIRV_Vulkan(
    ShaderToyStages stage,
    const char *shaderSource,
    const char *shaderName
  );

  nlohmann::json mainData;
  std::string shaderSrc;
  path shaderRoot;

  // common vertex shader spir-v
  std::optional<std::vector<char>> bufImgVertexShaderSpv;
  std::string bufImgVertexShaderSpvCode;
};
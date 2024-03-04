#pragma once

#include "Common.hpp"
#include "avk/command_buffer.hpp"
#include <glm/glm.hpp>
#include <stdexcept>

namespace vkExecute {

// A simple image-pass only shadertoy executor
// renders into a separate texture
// NOTE: this is not thread-safe!
class ImageOnlyExecutor {
public:

  // TODO: give more flexibility to parameter binding
  struct PipelineConfig {
    int targetWidth;
    int targetHeight;
    BinaryBlob vertexShader;
    BinaryBlob fragmentShader;

    bool traceRun;
    int traceBufferSize;
    int traceBufferDescSet;
    int traceBufferBinding;
  };

  using vec3 = std::array<float, 3>;
  using vec4 = std::array<float, 4>;

  // layout (binding=0) uniform PrimaryUBO {
  //   uniform vec3 iResolution;              // Offset 0
  //   uniform float iTime;                   // Offset 12
  //   uniform vec4 iChannelTime;             // Offset 16
  //   uniform vec4 iMouse;                   // Offset 32
  //   uniform vec4 iDate;                    // Offset 48
  //   uniform vec3 iChannelResolution[4];    // Offset 64
  //   uniform float iSampleRate;             // Offset 128
  //   uniform int iFrame;                    // Offset 132
  //   uniform float iTimeDelta;              // Offset 136
  //   uniform float iFrameRate;              // Offset 140
  // };

  struct ImageUniformBlock {
    vec3 iResolution;
    float iTime;
    // NOTE: this is float iChannelTime[4] inside shader block
    // doing this is for the ease of serialization
    vec4 iChannelTime;
    vec4 iMouse;
    vec4 iDate;
    float iSampleRate;
    std::array<vec4, 4> iChannelResolution;    // for padding purpose
    int iFrame;
    float iTimeDelta;
    float iFrameRate;
  };

  static_assert(sizeof(ImageUniformBlock) == 144, "Image Uniform is not consistent with glsl side");

  struct VertexData {
    glm::vec3 pos;
  };

  ImageOnlyExecutor();
  ImageOnlyExecutor(const ImageOnlyExecutor& other) = delete;
  ~ImageOnlyExecutor();

  // auxiliaries
  inline bool isRenderDocAPIEnabled() {
    return renderDocAPIEnabled;
  }

  // NOTE: a serious flaw of this is init arguments are only passed the first time
  // the application is being run, so be sure to enable asap
  // The return value returns if this is a fresh ctx, or a ctx already initialized
  // (hence may not follow the current one)
  bool init(bool forceEnableValidations, bool enableU64Features);
  void initPipeline(PipelineConfig cfg);
  void setUniform(ImageUniformBlock imageUB);
  void preRender();
  void render(int cycles);
  std::tuple<RGBAUIntImageBlob, uint64_t> getResults();
  std::tuple<float, float> getTimingParameters();

  BinaryBlob getTraceBuffer();
  void clearTraceBuffer(const BinaryBlob &content);
  void clearTraceBuffer();

  inline std::string getDeviceName() {
    if (!vkCtxInitialized) {
      throw std::runtime_error("Vulkan context haven't been initialized yet.");
    }
    
    return deviceName;
  }
  inline std::string getDriverDescription() {
    if (!vkCtxInitialized) {
      throw std::runtime_error("Vulkan context haven't been initialized yet.");
    }

    return driverDescription;
  }

private:

  // todo: fix this
  static bool vkCtxInitialized;
  static avk::queue* cmdQueue;

  // This is reused across different runs
  static avk::descriptor_cache descriptorCache;

  bool renderDocAPIEnabled;

  avk::graphics_pipeline imagePipeline;
  avk::framebuffer imageFramebuffer;

  // Buffers for drawing a quad
  avk::buffer quadVertexBuffer;
  avk::buffer imageUniformBuffer;
  ImageUniformBlock imageUniformData;

  // Buffers for tracing
  bool traceRun;
  avk::buffer traceBuffer;
  int traceBufferDescSet;
  int traceBufferBinding;

  avk::image renderColorOutput;
  avk::image renderDepthOutput;
  avk::image_view renderColorView;
  avk::image_view renderDepthView;

  // useful in case of read-back
  avk::buffer stagingColorOutput;

  // for performance counting
  avk::query_pool timestampQueryPool;

  float nsecPerIncrement;
  float nsecValidRange;

  std::string deviceName;
  std::string driverDescription;
};

} // namespace vkExecute
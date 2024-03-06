#pragma once

#include <auto_vk_toolkit.hpp>
#include <bitset>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <nlohmann/json.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "ShaderToy.hpp"
#include "avk/command_buffer.hpp"
#include "avk/commands.hpp"
#include "avk/cpp_utils.hpp"
#include "avk/pipeline_stage.hpp"
#include "util.hpp"
#include "window.hpp"
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>


class vkToy : public avk::invokee
{
private: // v== Member variables ==v

  avk::queue* cmdQueue;
  std::unique_ptr<ShaderToy> toyData;
  bool enableUI;

  using PlayerPass = struct {
    std::string name;
    avk::graphics_pipeline avkPipeline;
  };

  std::vector<PlayerPass> BufferPasses;
  avk::graphics_pipeline imagePipeline;

  avk::descriptor_cache descriptorCache;

  // Buffers for drawing a quad
  avk::buffer quadVertexBuffer;

  avk::buffer imageUniform;
  ShaderToy::ImageUniformBlock imageUniformData;

  // timed control
  bool useTimer;
  bool isPlaying;
  bool isShaderMdf;
  std::chrono::high_resolution_clock::time_point startTime;

#pragma region Profiler
  // Reference:
  // https://github.com/Chaf-Libraries/Ilum/blob/main/Source/Plugin/RHI/Vulkan/Profiler.cpp

  struct Profiler {
    typedef void (profile_logging_t)(avk::window::frame_id_t frameId, float nsecGpuTime);

    inline void initDeviceAttrs(uint32_t queueFamilyIndex) {
      assert(!deviceAttrInited);

      auto pdProps = avk::context().physical_device().getProperties();

      nsecPerIncrement = pdProps.limits.timestampPeriod;
      auto qfProperties = avk::context().physical_device().getQueueFamilyProperties();

      uint32_t validBits = qfProperties[queueFamilyIndex].timestampValidBits;
      nsecValidRange = nsecPerIncrement * std::pow(2, validBits);
      LOG_INFO(fmt::format(
        "Profiler inited for qf={}, {} ns/incr, {} ns valid (validBits={})",
        queueFamilyIndex,
        nsecPerIncrement,
        nsecValidRange,
        validBits
      ));

      deviceAttrInited = true;
    }

    // NOTE: Legacy resources are moved to the respective window
    // TODO: handle swapchain framecount change event
    inline void initPools(
      avk::window* window,
      int newConcurrentFrameCount,
      avk::window::frame_id_t currentFrame
    ) {
      // handle previous resources gracefully
      // When resize, please wait till idle before calling this
      if (timestampQueryPools.size() > 0) {
        timestampQueryPools.clear();
      }

      timestampQueryPools.reserve(newConcurrentFrameCount);
      for (int i = 0; i < newConcurrentFrameCount; i++) {
        timestampQueryPools.push_back(avk::context().create_query_pool_for_timestamp_queries(2));
      }

      concurrentFrameCount = newConcurrentFrameCount;

      assert(frameOffset <= currentFrame);
      frameOffset = currentFrame;
    }

    inline void setLoggingFunc(std::function<profile_logging_t> loggerFunc) {
      this->loggerFunc = loggerFunc;
    }

    inline std::vector<avk::recorded_commands_t> begin(avk::window::frame_id_t currentFrame) {
      assert(concurrentFrameCount != 0 && deviceAttrInited);

      int frameSlotIdx = currentFrame % concurrentFrameCount;
      return avk::command::gather(
        timestampQueryPools[frameSlotIdx]->reset(),
        // Vk Spec: After query pool creation, each query must be reset before it is used.
        //          Queries must also bereset between uses.
        timestampQueryPools[frameSlotIdx]->write_timestamp(0, avk::stage::top_of_pipe)
      );
    }

    inline std::vector<avk::recorded_commands_t> end(avk::window::frame_id_t currentFrame) {
      assert(concurrentFrameCount != 0 && deviceAttrInited);

      int frameSlotIdx = currentFrame % concurrentFrameCount;
      return avk::command::gather(
        timestampQueryPools[frameSlotIdx]->write_timestamp(1, avk::stage::bottom_of_pipe)
      );
    }

    // The user must sure that this is behind the fence
    // and for each frame this callback is only called once
    inline void beforeRender(avk::window::frame_id_t currentFrame) {
      int frameSinceLastInit = currentFrame - frameOffset;

      // onRender must be called for each frame exactly once!
      assert(onRenderLastUpdate == 0 || onRenderLastUpdate == currentFrame - 1);

      // log the result
      if (frameSinceLastInit >= concurrentFrameCount) {
        int finishedFrameSlot = frameSinceLastInit % concurrentFrameCount;
        auto result = timestampQueryPools[finishedFrameSlot]->get_results<uint64_t, 2>(0, vk::QueryResultFlagBits::e64);

        float nsecGpuTime = nsecPerIncrement * (result[1] - result[0]);
        loggerFunc(currentFrame - concurrentFrameCount, nsecGpuTime);
      }
    }


  private:
    int concurrentFrameCount = 0;
    avk::window::frame_id_t frameOffset = 0;
    avk::window::frame_id_t onRenderLastUpdate = 0;
    std::vector<avk::query_pool> timestampQueryPools;

    std::function<profile_logging_t> loggerFunc;
    float nsecPerIncrement;
    float nsecValidRange;

    bool deviceAttrInited = false;
  };

  std::unique_ptr<Profiler> profiler;
#pragma endregion

  public:
#pragma region PlayManager
  /* Export mode:
   * - export the frame [0, wrapBackFrame]'s associated performance info and its associated shader trace.
   *
   * Expected:
   * 1. Warm up loop (e.g. play for 5x)
   * 2. Performance gathering loop (e.g. play for 5x)
   * 3. Export {shaderUniformBlock, perf_data} into a json file
   * 4. Get shader trace
   *    (toyTrace, a separate application that reads in the json file and process):
   *   4.1. mutate shader, prepare new graphics pipeline
   *   4.2. run to get result
   */

  class PlayManager {
  public:
    struct PlayManagerConfig {
      bool constantFps;
      int constantFpsTarget;
      uint64_t wrapBackFrame;
      int maxLoop;
      std::string dumpJsonPath;
    };

    inline void initialize(PlayManagerConfig cfg) {
      if (cfg.wrapBackFrame == 0) {
        throw std::runtime_error("wrapBackFrame should be greater than 0");
      }

      this->constantFps = cfg.constantFps;
      this->constantFpsTarget = constantFpsTarget;
      this->wrapBackFrame = cfg.wrapBackFrame;
      this->maxLoop = cfg.maxLoop;
      this->curLoopTaken = 0;
      this->dumpJsonPath = cfg.dumpJsonPath;

      trialData.frameDatas.resize(maxLoop * wrapBackFrame);
    }

    // To be called end rendering ends
    inline void advanceFrame() {
      currentFrame++;

      if (currentFrame == wrapBackFrame) {
        currentFrame = 0;
        curLoopTaken++;

        if (curLoopTaken == maxLoop) {
          // We can end now - TODO: figure out
          renderEnd = true;
        }
      }
    }

    inline bool shouldClose() {
      return renderEnd && lastLoggedFrame + 1 == trialData.frameDatas.size();
    }

    inline void updateUniformData(ShaderToy::ImageUniformBlock& uniforms) {
      uniforms.iResolution = glm::vec3(
        avk::context().main_window()->resolution().x,
        avk::context().main_window()->resolution().y,
        0
      );
      if (constantFps) {
        uniforms.iTime = currentFrame * (1.0f / constantFpsTarget);
      }
      else {
        if (currentFrame == 0) {
          uniforms.iTime = 0;
          timer = std::make_unique<avk::varying_update_timer>();
        }
        else {
          assert(timer.get() != nullptr);
          timer->tick();
          uniforms.iTime = timer->time_since_start();
        }
      }

      uniforms.iFrame = static_cast<int>(currentFrame);

      if (currentFrame + curLoopTaken * wrapBackFrame >= this->trialData.frameDatas.size()) {
        VKTOY_WARN("Discard frame #%lld uniform because frameData is full", currentFrame + curLoopTaken * wrapBackFrame);
      }
      else {
        this->trialData.frameDatas[currentFrame + curLoopTaken * wrapBackFrame].uniforms = uniforms;
      }
    }

    inline void setUpProfilerLogFunction(Profiler& profiler) {
      // (avk::window::frame_id_t curFrame, float nsecGpuTime)
      profiler.setLoggingFunc([&](avk::window::frame_id_t curFrame, float nsecGpuTime) {
        if (curFrame < trialData.frameDatas.size()) {
          this->trialData.frameDatas[curFrame].gpuTime = nsecGpuTime;
          if (lastLoggedFrame + 1 != curFrame) {
            frameLost += static_cast<int>(curFrame) - lastLoggedFrame - 1;
            VKTOY_WARN(
              "Frame lost detected, previous frame got is frame #%d, got %d frame lost for now.",
              lastLoggedFrame,
              frameLost
            );
          }
          lastLoggedFrame = curFrame;
        }
        else {
          VKTOY_WARN("Discard frame #%lld (%f nsec) because frameData is full", curFrame, nsecGpuTime);
        }
      });
    }

    // curFrame (the real one) => (imageUniformData, time_nsec)
    struct PerFrameData {
      ShaderToy::ImageUniformBlock uniforms;
      double gpuTime = 0.0;

      // https://uscilab.github.io/cereal/serialization_functions.html
      template<class Archive>
      void serialize(Archive& archive) {
        archive(CEREAL_NVP(uniforms), CEREAL_NVP(gpuTime));
      }
    };

    struct TrialData {
      std::vector<PerFrameData> frameDatas;
      std::map<
        std::string,
        std::map<std::string, std::vector<char>>
      > shaderMap;
      std::map<
        std::string,
        std::map<std::string, std::string>
      > shaderSrcMap;
      int windowWidth, windowHeight;
      int wrapBackFrame, maxLoop;

      template<class Archive>
      void serialize(Archive& archive) {
        archive(CEREAL_NVP(frameDatas), CEREAL_NVP(shaderMap), CEREAL_NVP(shaderSrcMap));
        archive(CEREAL_NVP(windowWidth), CEREAL_NVP(windowHeight));
        archive(CEREAL_NVP(wrapBackFrame), CEREAL_NVP(maxLoop));
      }
    };

    inline void dumpTrialData() {
      // TODO: simple statistics generation
      auto curTime = std::time(nullptr);
      std::string destPath = fmt::format(dumpJsonPath, fmt::localtime(curTime));

      std::ofstream outJsonFile(destPath);
      VKTOY_LOG("Trial data exported to %s", destPath.c_str());

      trialData.windowWidth = avk::context().main_window()->resolution().x;
      trialData.windowHeight = avk::context().main_window()->resolution().y;
      trialData.wrapBackFrame = this->wrapBackFrame;
      trialData.maxLoop = this->maxLoop;

      cereal::JSONOutputArchive archive(outJsonFile);
      archive(trialData);
    }

    inline void saveShaders(std::string passName, std::string shaderStage, const std::vector<char> &spvBlob) {
      trialData.shaderMap[passName][shaderStage] = spvBlob;
    }

    inline void saveShaderSrcs(std::string passName, std::string shaderStage, const std::string &shaderSrc) {
      trialData.shaderSrcMap[passName][shaderStage] = shaderSrc;
    }

  private:
    bool constantFps;
    int constantFpsTarget;
    uint64_t wrapBackFrame;
    uint64_t currentFrame;
    std::unique_ptr<avk::varying_update_timer> timer;

    std::string dumpJsonPath;
    int maxLoop;
    int curLoopTaken;

    // (NOTE: please handle -1 with special care)
    int lastLoggedFrame = -1;
    // A counter to keep track of the frame lost, probably due to things like
    // swapchain recreation
    int frameLost = 0;
    // 
    bool renderEnd = false;

    TrialData trialData;
  };

private:
  std::unique_ptr<PlayManager> playMgr;

#pragma endregion

public: // v== avk::invokee overrides which will be invoked by the framework ==v
  vkToy(
    avk::queue& aQueue, // queue for command submission
    std::unique_ptr<ShaderToy> toyData,
    std::optional<PlayManager::PlayManagerConfig> playCfg,
    bool enableUI
  ) : cmdQueue{ &aQueue }, toyData(std::move(toyData)), enableUI(enableUI)
  {
    if (playCfg.has_value()) {
      playMgr = std::make_unique<PlayManager>();
      playMgr->initialize(playCfg.value());
    }
  }

  struct VertexData {
    glm::vec3 pos;
  };

  inline void initQuadBuffers() {
    const std::vector<VertexData> vertexData = {
      {{-1.0f, -1.0f,  0.0f}}, // first triangle
      {{ 1.0f, -1.0f,  0.0f}},
      {{-1.0f,  1.0f,  0.0f}},
      {{-1.0f,  1.0f,  0.0f}}, // second triangle
      {{ 1.0f, -1.0f,  0.0f}},
      {{ 1.0f,  1.0f,  0.0f}}
    };

    quadVertexBuffer = avk::context().create_buffer(
      avk::memory_usage::device, {},
      avk::vertex_buffer_meta::create_from_data(vertexData)
    );

    auto fence = avk::context().record_and_submit_with_fence({
      quadVertexBuffer->fill(vertexData.data(), 0)
    }, *cmdQueue);
    fence->wait_until_signalled();
  }

  inline void initProfiler() {
    profiler = std::make_unique<Profiler>();
    auto mainWnd = avk::context().main_window();

    profiler->initDeviceAttrs(cmdQueue->queue_index());
    profiler->initPools(mainWnd, mainWnd->get_config_number_of_concurrent_frames(), mainWnd->current_frame());
    if (playMgr.get() != nullptr) {
      playMgr->setUpProfilerLogFunction(*profiler);
    }
    else {
      profiler->setLoggingFunc([](avk::window::frame_id_t curFrame, float nsecGpuTime) {
        float throughput = 1e9 / nsecGpuTime;
        // std::cout << "Frame #" << curFrame << ": " << nsecGpuTime << "ns" 
        //           << " (throughput=" << throughput << "fps)" << std::endl;
      });
    }

  }

  inline void initBufferPasses() {

  }

  inline void initImagePass() {
    // prepare shaders
    std::string vertCodeSrc, fragCodeSrc;

    avk::shader_info fragShaderInfo;
    fragShaderInfo.mShaderType = avk::shader_type::fragment;
    fragShaderInfo.mEntryPoint = "main";
    std::tie(fragShaderInfo.mSpVCode, fragCodeSrc) = toyData->prepareFragmentShader();
    
    avk::shader_info vertShaderInfo;
    vertShaderInfo.mShaderType = avk::shader_type::vertex;
    vertShaderInfo.mEntryPoint = "main";
    std::tie(vertShaderInfo.mSpVCode, vertCodeSrc) = toyData->prepareVertexShader();
    
    std::cout << "fragment shader code:\n" << fragCodeSrc << std::endl;

    if (playMgr.get() != nullptr) {
      playMgr->saveShaders("image", "vertex", vertShaderInfo.mSpVCode.value());
      playMgr->saveShaders("image", "fragment", fragShaderInfo.mSpVCode.value());
      playMgr->saveShaderSrcs("image", "vertex", vertCodeSrc);
      playMgr->saveShaderSrcs("image", "fragment", fragCodeSrc);
    }

    auto fragShader = avk::context().create_shader(fragShaderInfo);
    auto vertShader = avk::context().create_shader(vertShaderInfo);

    imageUniform = avk::context().create_buffer(
      avk::memory_usage::host_coherent, {},
      avk::uniform_buffer_meta::create_from_size(sizeof(ShaderToy::ImageUniformBlock))
    );

    // create pipeline
    imagePipeline = avk::context().create_graphics_pipeline_for(
      avk::from_buffer_binding(0) -> stream_per_vertex(0, vk::Format::eR32G32B32Sfloat, sizeof(VertexData::pos)) -> to_location(0),
      avk::descriptor_binding(0, 0, imageUniform),
      [&vertShader, &fragShader](avk::graphics_pipeline_t& result) -> void {
        // insert compiled spir-v as fragment shader
        // no specialization infos, so skip the SpecInfos - the dummies are for indices sync
        result.shaders().push_back(std::move(vertShader));
        result.shader_stage_create_infos().emplace_back()
          .setStage(vk::ShaderStageFlagBits::eVertex)
          .setModule(result.shaders().back().handle())
          .setPName(result.shaders().back().info().mEntryPoint.c_str());
        result.specialization_infos().emplace_back();

        result.shaders().push_back(std::move(fragShader));
        result.shader_stage_create_infos().emplace_back()
          .setStage(vk::ShaderStageFlagBits::eFragment)
          .setModule(result.shaders().back().handle())
          .setPName(result.shaders().back().info().mEntryPoint.c_str());
        result.specialization_infos().emplace_back();
      },
      avk::cfg::front_face::define_front_faces_to_be_clockwise(),
      avk::cfg::viewport_depth_scissors_config::from_framebuffer(avk::context().main_window()->backbuffer_reference_at_index(0)),
      avk::context().main_window()->renderpass()
    );
  }

  inline void updateUniformData() {
    // Use PlayManager when we have one
    if (playMgr.get() != nullptr) {
      playMgr->updateUniformData(imageUniformData);
      return;
    }

    // image uniform
    imageUniformData.iResolution = glm::vec3(
      avk::context().main_window()->resolution().x,
      avk::context().main_window()->resolution().y,
      0
    );
    if(isPlaying)
    {
      imageUniformData.iTime += avk::time().delta_time();
      if (imageUniformData.iTime > 10) {
        imageUniformData.iTime -= 10;
      }
      imageUniformData.iFrame = (imageUniformData.iFrame + 1) % 1000;
    }
  }

  inline void initialize() override
  {
    // Print some information about the available memory on the selected physical device:
    avk::context().print_available_memory_types();
    isPlaying = true;
    isShaderMdf = false;
    // toyData->setShaderSrc("vec3 createRect(vec2 uv, vec2 position, vec2 size, vec3 color) {\n    if(uv.x > position.x && uv.x < position.x + size.x && uv.y > position.y && uv.y < position.y + size.y) {\n        return color;\n    }\n    return vec3(0,0,0);\n}\nvec3 createCircle(vec2 uv, vec2 position, float radius, vec3 color) {\n    float x = uv.x - position.x - pow(uv.x - position.x, 2.);\n    if(x > .0) {\n        if(uv.y < sqrt(x) && uv.y < sqrt(x)) {\n            return color;\n        }\n    }\n    return vec3(0,0,0);\n}   void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n // Normalized pixel coordinates (from 0 to 1)\n    vec2 uv = fragCoord/iResolution.xy;\n\n    // Time varying pixel color\n    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n\n    float W = 1. / 2.;\n    \n    // Output to screen\n    fragColor = vec4(\n                     createRect(uv, vec2(0,0),           vec2(W,W), col)\n                   + createRect(uv, vec2(1. - W,1. - W), vec2(W,W), abs(cos(col)))\n                   + createRect(uv, vec2(1. - W,0),      vec2(W,W), 0.5 + 0.25 * abs(sin(tan(4. * col))))\n                   + createRect(uv, vec2(0,1. - W),      vec2(W,W), abs(fract(1. - col))),1.0);\n}");


    descriptorCache = avk::context().create_descriptor_cache();

    initQuadBuffers();
    initBufferPasses();
    initImagePass();
    initProfiler();
    updateUniformData();
    
    // We want to use an updater => gotta create one:
    mUpdater.emplace();
    mUpdater->on(
      avk::swapchain_resized_event(avk::context().main_window()) // In the case of window resizes
    )
    .update(imagePipeline); // ... it will recreate the pipeline.

    if (enableUI) {
      auto imguiManager = avk::current_composition()->element_by_type<avk::imgui_manager>();
      if (nullptr != imguiManager) {

        imguiManager->add_callback([this]() {
          bool isEnabled = this->is_enabled();
          float frameRate = ImGui::GetIO().Framerate;
          float frameTime = 1000.0f / frameRate;
          if (!isPlaying) {
            frameRate = 0;
            frameTime = 0;
          }
          ImGui::Begin("vkToy");
          ImGui::SetWindowPos(ImVec2(1.0f, 1.0f), ImGuiCond_FirstUseEver);
          ImGui::Text("%.3f ms/frame", frameTime);
          ImGui::Text("%.1f FPS", frameRate);
          ImGui::Checkbox("Enable/Disable invokee", &isEnabled);
          if (isEnabled != this->is_enabled())
          {
            if (!isEnabled) this->disable();
            else this->enable();
          }
          static std::vector<float> values;
          values.push_back(frameTime);
          if (values.size() > 90) {
            values.erase(values.begin());
          }
          ImGui::PlotLines("ms/frame", values.data(), static_cast<int>(values.size()), 0, nullptr, 0.0f, FLT_MAX, ImVec2(0.0f, 100.0f));
          ImGui::End();
        });
      }
    }
  }

  inline void update() override
  {
    // On H pressed,
    if (avk::input().key_pressed(avk::key_code::h)) {
      // log a message:
      LOG_INFO_EM("Hello Auto-Vk-Toolkit!");
    }

    // On C pressed,
    if (avk::input().key_pressed(avk::key_code::c)) {
      // center the cursor:
      auto resolution = avk::context().main_window()->resolution();
      avk::context().main_window()->set_cursor_pos({ resolution[0] / 2.0, resolution[1] / 2.0 });
    }

    // On Esc pressed,
    if (avk::input().key_pressed(avk::key_code::escape)) {
      // stop the current composition:
      avk::current_composition()->stop();
    }

    // On R pressed,recompile shader, only for testing
    if (avk::input().key_pressed(avk::key_code::r)) {
      toyData->setShaderSrc("vec3 createRect(vec2 uv, vec2 position, vec2 size, vec3 color) {\n    if(uv.x > position.x && uv.x < position.x + size.x && uv.y > position.y && uv.y < position.y + size.y) {\n        return color;\n    }\n    return vec3(0,0,0);\n}\nvec3 createCircle(vec2 uv, vec2 position, float radius, vec3 color) {\n    float x = uv.x - position.x - pow(uv.x - position.x, 2.);\n    if(x > .0) {\n        if(uv.y < sqrt(x) && uv.y < sqrt(x)) {\n            return color;\n        }\n    }\n    return vec3(0,0,0);\n}   void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n // Normalized pixel coordinates (from 0 to 1)\n    vec2 uv = fragCoord/iResolution.xy;\n\n    // Time varying pixel color\n    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n\n    float W = 1. / 2.;\n    \n    // Output to screen\n    fragColor = vec4(\n                     createRect(uv, vec2(0,0),           vec2(W,W), col)\n                   + createRect(uv, vec2(1. - W,1. - W), vec2(W,W), abs(cos(col)))\n                   + createRect(uv, vec2(1. - W,0),      vec2(W,W), 0.5 + 0.25 * abs(sin(tan(4. * col))))\n ,1.0);\n}");
      initImagePass();
    }

    // when shader is modified, recompile
    if (isShaderMdf) {
      isShaderMdf = false;
      initImagePass();
    }

    // On Space pressed, pause/unpause
    if (avk::input().key_pressed(avk::key_code::space)) {
      isPlaying = !isPlaying;
    }

    if (playMgr.get() != nullptr && playMgr->shouldClose()) {
      playMgr->dumpTrialData();
      avk::current_composition()->stop();
    }

    if(isPlaying) {
      updateUniformData();
      profiler->beforeRender(avk::context().main_window()->current_frame());
    }
  }

  /**	Render callback which is invoked by the framework every frame after every update() callback has been invoked.
   *
   *	Important: We must establish a dependency to the "swapchain image available" condition, i.e., we must wait for the
   *	           next swap chain image to become available before we may start to render into it.
   *			   This dependency is expressed through a semaphore, and the framework demands us to use it via the function:
   *			   context().main_window()->consume_current_image_available_semaphore() for the main_window (our only window).
   *
   *			   More background information: At one point, we also must tell the presentation engine when we are done
   *			   with rendering by the means of a semaphore. Actually, we would have to use the framework function:
   *			   mainWnd->add_present_dependency_for_current_frame() for that purpose, but we don't have to do it in our case
   *			   since we are rendering a GUI. imgui_manager will add a semaphore as dependency for the presentation engine.
   */
  inline void render() override
  {
    auto mainWnd = avk::context().main_window();

    // The main window's swap chain provides us with an "image available semaphore" for the current frame.
    // Only after the swapchain image has become available, we may start rendering into it.
    auto imageAvailableSemaphore = mainWnd->consume_current_image_available_semaphore();

    // Get a command pool to allocate command buffers from:
    auto& commandPool = avk::context().get_command_pool_for_single_use_command_buffers(*cmdQueue);

    // Create a command buffer and render into the *current* swap chain image:
    auto cmdBfr = commandPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    auto allCmds = avk::command::gather();

    // Assemble render commands
    {
      auto profileStartCmds = profiler->begin(avk::context().main_window()->current_frame());
      auto renderCmds = avk::command::gather(
        // Call fill here, this will insert memory barrier automatically
        imageUniform->fill(&imageUniformData, 0),

        // Begin and end one renderpass:
        avk::command::render_pass(imagePipeline->renderpass_reference(), avk::context().main_window()->current_backbuffer_reference(), {
          // And within, bind a pipeline and draw three vertices:
          avk::command::bind_pipeline(imagePipeline.as_reference()),
          avk::command::bind_descriptors(
            imagePipeline->layout(),
            descriptorCache->get_or_create_descriptor_sets({
              avk::descriptor_binding(0, 0, imageUniform)
            })
          ),
          avk::command::draw_vertices(quadVertexBuffer.get())
        })
      );
      auto profileEndCmds = profiler->end(avk::context().main_window()->current_frame());

      allCmds.insert(allCmds.end(), profileStartCmds.begin(), profileStartCmds.end());
      allCmds.insert(allCmds.end(), renderCmds.begin(), renderCmds.end());
      allCmds.insert(allCmds.end(), profileEndCmds.begin(), profileEndCmds.end());
    }

    avk::recorded_commands(&avk::context(), allCmds)
      .into_command_buffer(cmdBfr)
      .then_submit_to(*cmdQueue)
      .waiting_for(imageAvailableSemaphore >> avk::stage::color_attachment_output)
      .submit();

    // Use a convenience function of avk::window to take care of the command buffer's lifetime:
    // It will get deleted in the future after #concurrent-frames have passed by.
    avk::context().main_window()->handle_lifetime(std::move(cmdBfr));

    if (playMgr.get() != nullptr) {
      playMgr->advanceFrame();
    }
  }

  void setShader(std::string shader)
  {
    toyData->setShaderSrc(shader);
    isShaderMdf = true;
  }

};
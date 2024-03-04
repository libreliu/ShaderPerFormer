#include "ImageOnlyExecutor.hpp"
#include "Common.hpp"
#include "avk/binding_data.hpp"
#include "avk/bindings.hpp"
#include "avk/buffer_meta.hpp"
#include "avk/command_buffer.hpp"
#include "avk/commands.hpp"
#include "avk/descriptor_set.hpp"
#include "avk/layout.hpp"
#include "avk/pipeline_stage.hpp"

#include <auto_vk_toolkit.hpp>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vulkan/vulkan_structs.hpp>

#ifdef VKEXECUTE_USE_RENDERDOC
#include <renderdoc_app.h>
static RENDERDOC_API_1_1_2 *rdoc_api = NULL;

#ifdef __linux__
#include <dlfcn.h>
static void enable_renderdoc() {
  if(void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD))
  {
      pRENDERDOC_GetAPI RENDERDOC_GetAPI = (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
      int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
      assert(ret == 1);
  }

  if(rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);
}
#endif

#ifdef WIN32
static void enable_renderdoc() {
  if(HMODULE mod = GetModuleHandleA("renderdoc.dll"))
  {
      pRENDERDOC_GetAPI RENDERDOC_GetAPI =
          (pRENDERDOC_GetAPI)GetProcAddress(mod, "RENDERDOC_GetAPI");
      int ret = RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
      assert(ret == 1);
  }

  if(rdoc_api) rdoc_api->StartFrameCapture(NULL, NULL);
}
#endif

#endif

using namespace vkExecute;

bool ImageOnlyExecutor::vkCtxInitialized = false;
avk::queue* ImageOnlyExecutor::cmdQueue = nullptr;
avk::descriptor_cache ImageOnlyExecutor::descriptorCache;

ImageOnlyExecutor::ImageOnlyExecutor() {
#ifdef VKEXECUTE_USE_RENDERDOC
  renderDocAPIEnabled = true;
#else
  renderDocAPIEnabled = false;
#endif

}

ImageOnlyExecutor::~ImageOnlyExecutor() {
  if (descriptorCache.has_value()) {
    descriptorCache->cleanup();
  }
  
#ifdef VKEXECUTE_USE_RENDERDOC
  if(rdoc_api) rdoc_api->EndFrameCapture(NULL, NULL);
#endif
}

bool ImageOnlyExecutor::init(bool forceEnableValidations, bool enableU64Features) {

  bool freshInitialization = false;
  if (!vkCtxInitialized) {
    // pending as event handler; will be dispatched and created by
    // work_off_event_handler inside context_vulkan::initialize
    cmdQueue = &avk::context().create_queue(
        {}, avk::queue_selection_preference::versatile_queue, nullptr);

    avk::settings s{};
    s.mValidationLayers.enable_in_release_mode(true);

    vk::PhysicalDeviceFeatures phdf =
        vk::PhysicalDeviceFeatures{}
            .setGeometryShader(VK_TRUE)
            .setTessellationShader(VK_TRUE)
            .setSamplerAnisotropy(VK_TRUE)
            .setVertexPipelineStoresAndAtomics(VK_TRUE)
            .setFragmentStoresAndAtomics(VK_TRUE)
            .setShaderStorageImageExtendedFormats(VK_TRUE)
            .setSampleRateShading(VK_TRUE)
            .setFillModeNonSolid(VK_TRUE);

    if (enableU64Features) {
      phdf.setShaderInt64(VK_TRUE);
    }

    vk::PhysicalDeviceVulkan11Features v11f =
        vk::PhysicalDeviceVulkan11Features{};

    vk::PhysicalDeviceVulkan12Features v12f =
        vk::PhysicalDeviceVulkan12Features{}
            .setDescriptorBindingVariableDescriptorCount(VK_TRUE)
            .setRuntimeDescriptorArray(VK_TRUE)
            .setShaderUniformTexelBufferArrayDynamicIndexing(VK_TRUE)
            .setShaderStorageTexelBufferArrayDynamicIndexing(VK_TRUE)
            .setDescriptorIndexing(VK_TRUE)
            .setBufferDeviceAddress(VK_FALSE);

    if (enableU64Features) {
      v12f.setShaderBufferInt64Atomics(VK_TRUE);
    }

  #if VK_HEADER_VERSION >= 162
      auto acsFtrs = vk::PhysicalDeviceAccelerationStructureFeaturesKHR{}.setAccelerationStructure(VK_FALSE);
      auto rtpFtrs = vk::PhysicalDeviceRayTracingPipelineFeaturesKHR{}.setRayTracingPipeline(VK_FALSE);
      auto rquFtrs = vk::PhysicalDeviceRayQueryFeaturesKHR{}.setRayQuery(VK_FALSE);
  #else
      vk::PhysicalDeviceRayTracingFeaturesKHR rtf = vk::PhysicalDeviceRayTracingFeaturesKHR{}
      .setRayTracing(VK_FALSE);
  #endif

    avk::context().initialize(s, phdf, v11f, v12f,
                              RAY_TRACING_PASS_ON_PARAMETERS);

#ifdef VKEXECUTE_USE_RENDERDOC
    // This is called *after* opening vulkan devices but before any real work,
    // to signal the start of RenderDoc frame capture.
    enable_renderdoc();
#endif

    freshInitialization = true;
    vkCtxInitialized = true;
  }

  auto pdProps = avk::context().physical_device().getProperties();

  nsecPerIncrement = pdProps.limits.timestampPeriod;
  auto qfProperties = avk::context().physical_device().getQueueFamilyProperties();

  uint32_t validBits = qfProperties[cmdQueue->family_index()].timestampValidBits;
  nsecValidRange = nsecPerIncrement * std::pow(2, validBits);

  deviceName = pdProps.deviceName.data();
  
  vk::PhysicalDeviceDriverProperties drvProps;
	vk::PhysicalDeviceProperties2 props2;
	props2.pNext = &drvProps;

  avk::context().physical_device().getProperties2(&props2);
  driverDescription = fmt::format(
    "{} {}", drvProps.driverName, drvProps.driverInfo
  );

  return freshInitialization;
}

void ImageOnlyExecutor::initPipeline(PipelineConfig cfg) {
  if (!descriptorCache.has_value()) {
    descriptorCache = avk::context().create_descriptor_cache();
  }

  // Init vertex quad buffer
  {
    const std::vector<VertexData> vertexData = {
        {{-1.0f, -1.0f, 0.0f}}, // first triangle
        {{1.0f, -1.0f, 0.0f}},  {{-1.0f, 1.0f, 0.0f}},
        {{-1.0f, 1.0f, 0.0f}}, // second triangle
        {{1.0f, -1.0f, 0.0f}},  {{1.0f, 1.0f, 0.0f}}};

    quadVertexBuffer = avk::context().create_buffer(
        avk::memory_usage::device, {},
        avk::vertex_buffer_meta::create_from_data(vertexData));

    auto fence = avk::context().record_and_submit_with_fence(
        {quadVertexBuffer->fill(vertexData.data(), 0)}, *cmdQueue);
    fence->wait_until_signalled();
  }

  // init image uniform buffer
  {
    imageUniformBuffer = avk::context().create_buffer(
      avk::memory_usage::host_coherent, {},
      avk::uniform_buffer_meta::create_from_size(sizeof(ImageUniformBlock))
    );
  }

  // init counter buffer if necessary
  if (cfg.traceRun) {
    traceBuffer = avk::context().create_buffer(
      avk::memory_usage::device_readback, {},
      avk::storage_buffer_meta::create_from_size(cfg.traceBufferSize)
    );

    traceBufferBinding = cfg.traceBufferBinding;
    traceBufferDescSet = cfg.traceBufferDescSet;

    traceRun = true;
  } else {
    // set to monostate - destroy previous buffer if not empty
    if (traceBuffer.has_value()) {
      traceBuffer = avk::buffer();
    }
    
    traceRun = false;
  }

  // prepare shaders
  avk::shader fragShader;
  avk::shader vertShader;
  {
    avk::shader_info fragShaderInfo;
    fragShaderInfo.mShaderType = avk::shader_type::fragment;
    fragShaderInfo.mEntryPoint = "main";
    fragShaderInfo.mSpVCode = cfg.fragmentShader;

    avk::shader_info vertShaderInfo;
    vertShaderInfo.mShaderType = avk::shader_type::vertex;
    vertShaderInfo.mEntryPoint = "main";
    vertShaderInfo.mSpVCode = cfg.vertexShader;

    fragShader = avk::context().create_shader(fragShaderInfo);
    vertShader = avk::context().create_shader(vertShaderInfo);
  }

  // create render texture
  {
    renderColorOutput = avk::context().create_image(
        cfg.targetWidth, cfg.targetHeight, avk::default_srgb_4comp_format(), 1,
        avk::memory_usage::device, avk::image_usage::general_color_attachment);

    renderDepthOutput = avk::context().create_depth_image(
        cfg.targetWidth, cfg.targetHeight, {}, 1, avk::memory_usage::device,
        avk::image_usage::general_depth_stencil_attachment);

    renderColorView = avk::context().create_image_view(renderColorOutput);
    renderDepthView = avk::context().create_depth_image_view(renderDepthOutput);

    renderColorOutput.enable_shared_ownership();
    renderDepthOutput.enable_shared_ownership();
  }

  // create renderpass, pipeline & framebuffer
  {
    std::vector<avk::attachment> attachments;
    attachments.push_back(avk::attachment::declare(
        renderColorOutput->format(), avk::on_load::clear,
        avk::usage::color(0),
        avk::on_store::store.in_layout(avk::layout::color_attachment_optimal)));

    attachments.push_back(avk::attachment::declare(
        renderDepthOutput->format(), avk::on_load::clear,
        avk::usage::depth_stencil,
        avk::on_store::store.in_layout(avk::layout::depth_stencil_attachment_optimal)));

    auto newRenderpass = avk::context().create_renderpass(
        attachments,
        {avk::subpass_dependency(
             avk::subpass::external >> avk::subpass::index(0),
             // ... we have to synchronize all these stages with
             // color+dept_stencil write access:
             avk::stage::color_attachment_output |
                 avk::stage::early_fragment_tests |
                 avk::stage::late_fragment_tests >>
                     avk::stage::color_attachment_output |
                 avk::stage::early_fragment_tests |
                 avk::stage::late_fragment_tests,
             avk::access::color_attachment_write |
                 avk::access::depth_stencil_attachment_write >>
                     avk::access::color_attachment_read |
                 avk::access::depth_stencil_attachment_write),
         avk::subpass_dependency(
             avk::subpass::index(0) >> avk::subpass::external,
             avk::stage::color_attachment_output >>
                 avk::stage::none, // assume semaphore afterwards
             avk::access::color_attachment_write >> avk::access::none)});

    // this is a bit ugly; TODO: improve
    if (traceRun) {
      imagePipeline = avk::context().create_graphics_pipeline_for(
          avk::from_buffer_binding(0)
              ->stream_per_vertex(0, vk::Format::eR32G32B32Sfloat,
                                  sizeof(VertexData::pos))
              ->to_location(0),
          avk::descriptor_binding(0, 0, imageUniformBuffer),
          avk::descriptor_binding(traceBufferDescSet, traceBufferBinding,
                                  traceBuffer),
          [&vertShader, &fragShader](avk::graphics_pipeline_t &result) -> void {
            // insert compiled spir-v as fragment shader
            // no specialization infos, so skip the SpecInfos - the dummies are
            // for indices sync
            result.shaders().push_back(std::move(vertShader));
            result.shader_stage_create_infos()
                .emplace_back()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(result.shaders().back().handle())
                .setPName(result.shaders().back().info().mEntryPoint.c_str());
            result.specialization_infos().emplace_back();

            result.shaders().push_back(std::move(fragShader));
            result.shader_stage_create_infos()
                .emplace_back()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(result.shaders().back().handle())
                .setPName(result.shaders().back().info().mEntryPoint.c_str());
            result.specialization_infos().emplace_back();
          },
          avk::cfg::front_face::define_front_faces_to_be_clockwise(),
          avk::cfg::viewport_depth_scissors_config::from_extent(
              cfg.targetWidth, cfg.targetHeight),
          newRenderpass);
    } else {
      imagePipeline = avk::context().create_graphics_pipeline_for(
          avk::from_buffer_binding(0)
              ->stream_per_vertex(0, vk::Format::eR32G32B32Sfloat,
                                  sizeof(VertexData::pos))
              ->to_location(0),
          avk::descriptor_binding(0, 0, imageUniformBuffer),
          [&vertShader, &fragShader](avk::graphics_pipeline_t &result) -> void {
            // insert compiled spir-v as fragment shader
            // no specialization infos, so skip the SpecInfos - the dummies are
            // for indices sync
            result.shaders().push_back(std::move(vertShader));
            result.shader_stage_create_infos()
                .emplace_back()
                .setStage(vk::ShaderStageFlagBits::eVertex)
                .setModule(result.shaders().back().handle())
                .setPName(result.shaders().back().info().mEntryPoint.c_str());
            result.specialization_infos().emplace_back();

            result.shaders().push_back(std::move(fragShader));
            result.shader_stage_create_infos()
                .emplace_back()
                .setStage(vk::ShaderStageFlagBits::eFragment)
                .setModule(result.shaders().back().handle())
                .setPName(result.shaders().back().info().mEntryPoint.c_str());
            result.specialization_infos().emplace_back();
          },
          avk::cfg::front_face::define_front_faces_to_be_clockwise(),
          avk::cfg::viewport_depth_scissors_config::from_extent(
              cfg.targetWidth, cfg.targetHeight),
          newRenderpass);
    }


    imageFramebuffer = avk::context().create_framebuffer(
        newRenderpass, renderColorView, renderDepthView);
  }

  // create staging images
  {
    stagingColorOutput = avk::context().create_buffer(
        avk::memory_usage::host_coherent, vk::BufferUsageFlagBits::eTransferDst,
        // https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#copies-buffers-images-addressing
        // will be the following for R8G8B8A8_SRGB
        avk::generic_buffer_meta::create_from_size(cfg.targetWidth *
                                                   cfg.targetHeight * 4));

    stagingColorOutput.enable_shared_ownership();
  }

  // create query pool
  {
    timestampQueryPool = avk::context().create_query_pool_for_timestamp_queries(2);
  }
}

void ImageOnlyExecutor::setUniform(ImageUniformBlock imageUB) {
  imageUniformData = imageUB;
  auto fence = avk::context().record_and_submit_with_fence(
      {imageUniformBuffer->fill(&imageUniformData, 0)}, *cmdQueue);
  fence->wait_until_signalled();
}

// transfer layouts
void ImageOnlyExecutor::preRender() {
  auto fence = avk::context().record_and_submit_with_fence(
      {avk::sync::image_memory_barrier(
           renderColorOutput.as_reference(),
           avk::stage_and_access{avk::stage::none, avk::access::none} >>
               avk::stage_and_access{avk::stage::auto_stage,
                                     avk::access::auto_access})
           .with_layout_transition(avk::layout::undefined >>
                                   avk::layout::color_attachment_optimal),
       avk::sync::image_memory_barrier(
           renderDepthOutput.as_reference(),
           avk::stage_and_access{avk::stage::none, avk::access::none} >>
               avk::stage_and_access{avk::stage::auto_stage,
                                     avk::access::auto_access})
           .with_layout_transition(avk::layout::undefined >>
                                   avk::layout::depth_stencil_attachment_optimal),
       timestampQueryPool->reset()},
      *cmdQueue);
  fence->wait_until_signalled();
}

void ImageOnlyExecutor::render(int cycles) {
  auto &commandPool =
      avk::context().get_command_pool_for_single_use_command_buffers(*cmdQueue);
  auto cmdBfr = commandPool->alloc_command_buffer(
      vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

  // Assemble render commands
  std::vector<avk::descriptor_set> descriptorSets;
  
  if (traceRun) {
    descriptorSets = descriptorCache->get_or_create_descriptor_sets(
        {avk::descriptor_binding(0, 0, imageUniformBuffer),
         avk::descriptor_binding(traceBufferDescSet, traceBufferBinding, traceBuffer)});
  } else {
    descriptorSets = descriptorCache->get_or_create_descriptor_sets(
        {avk::descriptor_binding(0, 0, imageUniformBuffer)});
  }

  auto renderCmd = avk::command::gather(avk::command::render_pass(
      imagePipeline->renderpass_reference(), imageFramebuffer.as_reference(),
      {avk::command::bind_pipeline(imagePipeline.as_reference()),
       avk::command::bind_descriptors(imagePipeline->layout(), descriptorSets),
       avk::command::draw_vertices(quadVertexBuffer.get())}));

  auto allCmds = avk::command::gather(
    timestampQueryPool->write_timestamp(0, avk::stage::top_of_pipe)
  );

  for (int i = 0; i < cycles; i++) {
    allCmds.insert(allCmds.end(), renderCmd.begin(), renderCmd.end());
  }

  allCmds.push_back(timestampQueryPool->write_timestamp(1, avk::stage::bottom_of_pipe));

  auto fence = avk::context().create_fence();

  avk::recorded_commands(&avk::context(), allCmds)
      .into_command_buffer(cmdBfr)
      .then_submit_to(*cmdQueue)
      .signaling_upon_completion(fence)
      .submit();

  fence->wait_until_signalled();
}

std::tuple<RGBAUIntImageBlob, uint64_t> ImageOnlyExecutor::getResults() {
  auto imgCopyCmdBuilder = [](avk::image src, avk::buffer dest) {
    auto cmd = avk::command::action_type_command{
        {}, // Define a resource-specific sync hint here and let the general
            // sync hint be inferred afterwards (because it is supposed to be
            // exactly the same)
        {
            std::make_tuple(
                src->handle(),
                avk::sync::sync_hint{
                    avk::stage_and_access{avk::stage::copy,
                                          avk::access::transfer_read},
                    avk::stage_and_access{avk::stage::copy, avk::access::none}})
            // No need for any dependencies for the staging buffer
        },
        [lOutput = src, lStagingOutput = dest](avk::command_buffer_t &cb) {
          // vk::ImageCopy region(vk::ImageSubresourceLayers(
          //                          vk::ImageAspectFlagBits::eColor, 0, 0, 1),
          //                      vk::Offset3D(0, 0, 0),
          //                      vk::ImageSubresourceLayers(
          //                          vk::ImageAspectFlagBits::eColor, 0, 0, 1),
          //                      vk::Offset3D(0, 0, 0));

          // cb.handle().copyImage(
          //     lOutput->handle(), vk::ImageLayout::eTransferSrcOptimal,
          //     lStagingOutput->handle(), vk::ImageLayout::eTransferDstOptimal,
          //     1, &region);

          vk::BufferImageCopy region(
              0, 0, 0,
              vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0,
                                         1),
              vk::Offset3D(0, 0, 0),
              vk::Extent3D(lOutput->width(), lOutput->height(), 1));

          cb.handle().copyImageToBuffer(lOutput->handle(),
                                        vk::ImageLayout::eTransferSrcOptimal,
                                        lStagingOutput->handle(), 1, &region);
        }};
    cmd.infer_sync_hint_from_resource_sync_hints();

    return cmd;
  };

  auto copyColorCmd = imgCopyCmdBuilder(renderColorOutput, stagingColorOutput);

  auto fence = avk::context().record_and_submit_with_fence(
      {avk::sync::image_memory_barrier(
           renderColorOutput.as_reference(),
           avk::stage_and_access{avk::stage::none, avk::access::none} >>
               avk::stage_and_access{avk::stage::auto_stage,
                                     avk::access::auto_access})
           .with_layout_transition(avk::layout::color_attachment_optimal >>
                                   avk::layout::transfer_src),
       avk::sync::buffer_memory_barrier(
           stagingColorOutput.as_reference(),
           avk::stage_and_access{avk::stage::none, avk::access::none} >>
               avk::stage_and_access{avk::stage::auto_stage,
                                     avk::access::auto_access}),
       copyColorCmd},
      *cmdQueue);
  fence->wait_until_signalled();

  // TODO: pass back depth image data

  auto colorBlob = RGBAUIntImageBlob(renderColorOutput->width(), renderColorOutput->height());

  // ImageData colorData;
  // colorData.format = static_cast<int>(renderColorOutput->format());
  // colorData.width = renderColorOutput->width();
  // colorData.height = renderColorOutput->height();

  size_t imgSize =
      stagingColorOutput->meta_at_index<avk::generic_buffer_meta>(0)
          .total_size();
  assert(imgSize ==
         4 * renderColorOutput->width() * renderColorOutput->height());

  {
    auto map = stagingColorOutput->map_memory(avk::mapping_access::read);
    memcpy(colorBlob.data(), map.get(), imgSize);
  }

  // read out time information
  auto timestampResults = timestampQueryPool->get_results<uint64_t, 2>(0, vk::QueryResultFlagBits::e64);
  uint64_t incrCount = (timestampResults[1] - timestampResults[0]);

  return std::make_tuple(std::move(colorBlob), incrCount);
}

std::tuple<float, float> ImageOnlyExecutor::getTimingParameters() {
  return std::make_tuple(nsecPerIncrement, nsecValidRange);
}

void ImageOnlyExecutor::clearTraceBuffer() {
  if (!traceBuffer.has_value()) {
    throw std::runtime_error("Trace buffer hasn't been initialized yet");
  }

  std::vector<char> traceBufferContents(traceBuffer->meta<avk::storage_buffer_meta>().total_size(), 0);
  auto fence = avk::context().record_and_submit_with_fence(
    {traceBuffer->fill(traceBufferContents.data(), 0)}, *cmdQueue);
  fence->wait_until_signalled();
}

void ImageOnlyExecutor::clearTraceBuffer(const BinaryBlob &content) {
  if (!traceBuffer.has_value()) {
    throw std::runtime_error("Trace buffer hasn't been initialized yet");
  }

  if (content.size() != traceBuffer->meta<avk::storage_buffer_meta>().total_size()) {
    throw std::runtime_error("Trace buffer size is different from the meta");
  }

  auto fence = avk::context().record_and_submit_with_fence(
    {traceBuffer->fill(content.data(), 0)}, *cmdQueue);
  fence->wait_until_signalled();
}

BinaryBlob ImageOnlyExecutor::getTraceBuffer() {
  // Initialize with -1 to reveal potential problems
  BinaryBlob content(traceBuffer->meta<avk::storage_buffer_meta>().total_size(), -1);

  auto& cmdPool = avk::context().get_command_pool_for_single_use_command_buffers(*cmdQueue);
  auto cmdBfr = cmdPool->alloc_command_buffer(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
  auto fen = avk::context().create_fence();

  cmdBfr.enable_shared_ownership();

  avk::context().record({ traceBuffer->read_into(content.data(), 0) })
    .into_command_buffer(cmdBfr)
    .then_submit_to(*cmdQueue)
    .signaling_upon_completion(fen)
    .submit();

  //fen->handle_lifetime_of(cmdBfr);
  fen->wait_until_signalled();

  // actual works are done here
  cmdBfr->invoke_post_execution_handler();

  return content;
}


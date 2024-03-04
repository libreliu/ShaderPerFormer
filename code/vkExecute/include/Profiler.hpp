#pragma once

#include <auto_vk_toolkit.hpp>

namespace vkExecute {

// Reference:
// https://github.com/Chaf-Libraries/Ilum/blob/main/Source/Plugin/RHI/Vulkan/Profiler.cpp

struct Profiler {
  typedef void(profile_logging_t)(avk::window::frame_id_t frameId,
                                  float nsecGpuTime);

  inline void initDeviceAttrs(uint32_t queueFamilyIndex) {
    assert(!deviceAttrInited);

    auto pdProps = avk::context().physical_device().getProperties();

    nsecPerIncrement = pdProps.limits.timestampPeriod;
    auto qfProperties =
        avk::context().physical_device().getQueueFamilyProperties();

    uint32_t validBits = qfProperties[queueFamilyIndex].timestampValidBits;
    nsecValidRange = nsecPerIncrement * std::pow(2, validBits);
    LOG_INFO(fmt::format(
        "Profiler inited for qf={}, {} ns/incr, {} ns valid (validBits={})",
        queueFamilyIndex, nsecPerIncrement, nsecValidRange, validBits));

    deviceAttrInited = true;
  }

  // NOTE: Legacy resources are moved to the respective window
  // TODO: handle swapchain framecount change event
  inline void initPools(avk::window *window, int newConcurrentFrameCount,
                        avk::window::frame_id_t currentFrame) {
    // handle previous resources gracefully
    // When resize, please wait till idle before calling this
    if (timestampQueryPools.size() > 0) {
      timestampQueryPools.clear();
    }

    timestampQueryPools.reserve(newConcurrentFrameCount);
    for (int i = 0; i < newConcurrentFrameCount; i++) {
      timestampQueryPools.push_back(
          avk::context().create_query_pool_for_timestamp_queries(2));
    }

    concurrentFrameCount = newConcurrentFrameCount;

    assert(frameOffset <= currentFrame);
    frameOffset = currentFrame;
  }

  inline void setLoggingFunc(std::function<profile_logging_t> loggerFunc) {
    this->loggerFunc = loggerFunc;
  }

  inline std::vector<avk::recorded_commands_t>
  begin(avk::window::frame_id_t currentFrame) {
    assert(concurrentFrameCount != 0 && deviceAttrInited);

    int frameSlotIdx = currentFrame % concurrentFrameCount;
    return avk::command::gather(
        timestampQueryPools[frameSlotIdx]->reset(),
        // Vk Spec: After query pool creation, each query must be reset before
        // it is used.
        //          Queries must also bereset between uses.
        timestampQueryPools[frameSlotIdx]->write_timestamp(
            0, avk::stage::top_of_pipe));
  }

  inline std::vector<avk::recorded_commands_t>
  end(avk::window::frame_id_t currentFrame) {
    assert(concurrentFrameCount != 0 && deviceAttrInited);

    int frameSlotIdx = currentFrame % concurrentFrameCount;
    return avk::command::gather(
        timestampQueryPools[frameSlotIdx]->write_timestamp(
            1, avk::stage::bottom_of_pipe));
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
      auto result =
          timestampQueryPools[finishedFrameSlot]->get_results<uint64_t, 2>(
              0, vk::QueryResultFlagBits::e64);

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

}; // namespace vkExecute

#include <iostream>
#include <popl.hpp>
#include "spirv-tools/libspirv.h"
#include <vkToy.hpp>
#include <vector>

struct TraceToy {
  using TrialData = vkToy::PlayManager::TrialData;
  void loadTrace(std::string dumpJsonInputPath) {
    std::ifstream jsonFs(dumpJsonInputPath);
    cereal::JSONInputArchive iArchive(jsonFs);

    iArchive(trialData);
  }

  void generateSimpleStatistics() {
    printf("Total number of frames: %ulld\n", trialData.frameDatas.size());
    printf("Number of frames in a loop: %d\n", trialData.wrapBackFrame);
    printf("Number of loops in total: %d\n", trialData.maxLoop);

    //{
    //  // average across loop instances
    //  std::vector<double> frameTimeAvg(trialData.wrapBackFrame, 0);
    //  std::vector<double> frameTimeSquared(trialData.wrapBackFrame, 0);

    //  for (size_t i = 0; i < trialData.maxLoop; i++) {
    //    for (size_t j = 0; j < trialData.wrapBackFrame; j++) {
    //      frameTimeAvg[j] += trialData.frameDatas[i * trialData.wrapBackFrame + j].gpuTime;
    //      frameTimeSquared[j] += (trialData.frameDatas[i * trialData.wrapBackFrame + j].gpuTime * trialData.frameDatas[i * trialData.wrapBackFrame + j].gpuTime);
    //    }
    //  }

    //  std::vector<double> frameTimeSampleVar(trialData.wrapBackFrame, 0);
    //  if (trialData.maxLoop > 1) {
    //    for (size_t i = 0; i < trialData.wrapBackFrame; i++) {
    //      frameTimeSampleVar[i] = frameTimeSquared[i] - trialData.maxLoop * (frameTimeAvg[i] * frameTimeAvg[i]);
    //      frameTimeSampleVar[i] /= (trialData.maxLoop - 1);
    //    }
    //  }
    //}
  }

  void disassembleShader(const std::vector<char>& spvBlob) {
    // disassemble
    spv_context context = spvContextCreate(spv_target_env::SPV_ENV_VULKAN_1_3);
    spv_text text;
    spv_diagnostic diagnostic = nullptr;
    spvBinaryToText(context, reinterpret_cast<const uint32_t *>(spvBlob.data()), spvBlob.size() / sizeof(unsigned int),
      SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES | SPV_BINARY_TO_TEXT_OPTION_INDENT,
      &text, &diagnostic);

    // dump
    if (diagnostic == nullptr)
      std::cout << text->str;
    else
      spvDiagnosticPrint(diagnostic);

    // teardown
    spvDiagnosticDestroy(diagnostic);
    spvContextDestroy(context);
  }

  void disassembleShaders() {
    printf("; ==== VERTEX SHADER ====\n");
    disassembleShader(trialData.shaderMap["image"]["vertex"]);
    printf("; ==== FRAGMENT SHADER ====\n");
    disassembleShader(trialData.shaderMap["image"]["fragment"]);
  }

  // This assumes a fragment shader setting for now
  void generateFragmentTraceShaders(const std::vector<char>& spvBlob) {
    spv_context context = spvContextCreate(spv_target_env::SPV_ENV_VULKAN_1_3);

  }

private:
  TrialData trialData;
};

int main(int argc, char *argv[]) {
  popl::OptionParser op("Allowed options");
  auto dumpJsonInputPath = op.add<popl::Value<std::string>>("", "dump-json-input-path", "Destination JSON path for input", "./play.json");

  TraceToy traceToy;
  traceToy.loadTrace(dumpJsonInputPath->value());
  traceToy.generateSimpleStatistics();
  traceToy.disassembleShaders();

  printf("Hello world!\n");
}
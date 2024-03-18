#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include "Common.hpp"
#include "ShaderProcessor.hpp"
#include "ImageOnlyExecutor.hpp"
#include "SpvProcessor.hpp"
#include "spv/Tokenizer.hpp"
#include "utils.hpp"
#include "vkDisplay.hpp"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

namespace py = pybind11;
using namespace vkExecute;

PYBIND11_MODULE(vkExecute, m) {
  m.doc() = R"pbdoc(
        Source code is documentation!
    )pbdoc";

  py::class_<RGBAUIntImageBlob> rgbaUIntImageBlob(m, "RGBAUIntImageBlob", py::buffer_protocol());
  rgbaUIntImageBlob.def_buffer([](RGBAUIntImageBlob &m) -> py::buffer_info {
    return py::buffer_info(
        m.data(),     /* Pointer to buffer */
        m.compSize(), /* Size of one scalar */
        py::format_descriptor<RGBAUIntImageBlob::ComponentType>::
            format(),              /* Python struct-style format descriptor */
        3,                         /* Number of dimensions */
        {m.height(), m.width(), m.numComp()},      /* Buffer dimensions */
        {m.numComp() * m.compSize() * m.width(),   /* Strides (in bytes) for each index */
         m.numComp() * m.compSize(), m.compSize()});
  });

  rgbaUIntImageBlob.def(py::init<size_t, size_t>())
                   .def("width", &vkExecute::RGBAUIntImageBlob::width)
                   .def("height", &vkExecute::RGBAUIntImageBlob::height)
                   .def("setWidth", &vkExecute::RGBAUIntImageBlob::setWidth)
                   .def("setHeight", &vkExecute::RGBAUIntImageBlob::setHeight)
                   .def("compSize", &vkExecute::RGBAUIntImageBlob::compSize)
                   .def("numComp", &vkExecute::RGBAUIntImageBlob::numComp);

  py::class_<ImageData> imageData(m, "ImageData");
  imageData.def(py::init<>())
           .def_readwrite("format", &vkExecute::ImageData::format)
           .def_readwrite("width", &vkExecute::ImageData::width)
           .def_readwrite("height", &vkExecute::ImageData::height)
           .def_readwrite("data", &vkExecute::ImageData::data);

  py::class_<ShaderProcessor> shaderProcessor(m, "ShaderProcessor");

  py::enum_<ShaderProcessor::ShaderStages>(shaderProcessor, "ShaderStages")
    .value("VERTEX", ShaderProcessor::ShaderStages::VERTEX)
    .value("TESSCONTROL", ShaderProcessor::ShaderStages::TESSCONTROL)
    .value("TESSEVALUATION", ShaderProcessor::ShaderStages::TESSEVALUATION)
    .value("GEOMETRY", ShaderProcessor::ShaderStages::GEOMETRY)
    .value("FRAGMENT", ShaderProcessor::ShaderStages::FRAGMENT)
    .value("COMPUTE", ShaderProcessor::ShaderStages::COMPUTE)
    .export_values();
  
  shaderProcessor.def(py::init<>())
                 .def("loadSpv", &ShaderProcessor::loadSpv)
                 .def("disassemble", &ShaderProcessor::disassemble)
                 .def_static("compileShaderToSPIRV_Vulkan", &ShaderProcessor::compileShaderToSPIRV_Vulkan);

  py::class_<ImageOnlyExecutor> imageOnlyExecutor(m, "ImageOnlyExecutor");

  py::class_<ImageOnlyExecutor::ImageUniformBlock> imageUniformBlock(imageOnlyExecutor, "ImageUniformBlock");

  imageUniformBlock.def(py::init<>())
                   .def_readwrite("iResolution", &ImageOnlyExecutor::ImageUniformBlock::iResolution)
                   .def_readwrite("iTime", &ImageOnlyExecutor::ImageUniformBlock::iTime)
                   .def_readwrite("iChannelTime", &ImageOnlyExecutor::ImageUniformBlock::iChannelTime)
                   .def_readwrite("iMouse", &ImageOnlyExecutor::ImageUniformBlock::iMouse)
                   .def_readwrite("iDate", &ImageOnlyExecutor::ImageUniformBlock::iDate)
                   .def_readwrite("iSampleRate", &ImageOnlyExecutor::ImageUniformBlock::iSampleRate)
                   .def_readwrite("iChannelResolution", &ImageOnlyExecutor::ImageUniformBlock::iChannelResolution)
                   .def_readwrite("iFrame", &ImageOnlyExecutor::ImageUniformBlock::iFrame)
                   .def_readwrite("iTimeDelta", &ImageOnlyExecutor::ImageUniformBlock::iTimeDelta)
                   .def_readwrite("iFrameRate", &ImageOnlyExecutor::ImageUniformBlock::iFrameRate)
                   .def("exportAsBytes", [](ImageOnlyExecutor::ImageUniformBlock &self) {
                     // This is used to remind of the possible change on the Python side
                     // while changing this struct
                     static_assert(sizeof(ImageOnlyExecutor::ImageUniformBlock) == 144);

                     const char *begin = reinterpret_cast<const char*>(&self);
                     return py::bytes(begin, sizeof(ImageOnlyExecutor::ImageUniformBlock));
                   })
                   .def("importFromBytes", [](ImageOnlyExecutor::ImageUniformBlock &self, const py::bytes& input) {
                     // This is used to remind of the possible change on the Python side
                     // while changing this struct
                     static_assert(sizeof(ImageOnlyExecutor::ImageUniformBlock) == 144);

                     std::string_view content{input};
                     if (content.size() != sizeof(ImageOnlyExecutor::ImageUniformBlock)) {
                      throw std::runtime_error(
                        "The bytes object to be imported does not have the correct length. Expected: " +
                        std::to_string(sizeof(ImageOnlyExecutor::ImageUniformBlock)) + ", Got: " +
                        std::to_string(content.size())
                      );
                     }
                     std::memcpy(&self, content.data(), content.size());
                   });

  py::class_<ImageOnlyExecutor::PipelineConfig> pipelineConfig(imageOnlyExecutor, "PipelineConfig");

  pipelineConfig.def(py::init<>())
                .def_readwrite("targetWidth", &ImageOnlyExecutor::PipelineConfig::targetWidth)
                .def_readwrite("targetHeight", &ImageOnlyExecutor::PipelineConfig::targetHeight)
                .def_readwrite("vertexShader", &ImageOnlyExecutor::PipelineConfig::vertexShader)
                .def_readwrite("fragmentShader", &ImageOnlyExecutor::PipelineConfig::fragmentShader)
                .def_readwrite("traceRun", &ImageOnlyExecutor::PipelineConfig::traceRun)
                .def_readwrite("traceBufferSize", &ImageOnlyExecutor::PipelineConfig::traceBufferSize)
                .def_readwrite("traceBufferDescSet", &ImageOnlyExecutor::PipelineConfig::traceBufferDescSet)
                .def_readwrite("traceBufferBinding", &ImageOnlyExecutor::PipelineConfig::traceBufferBinding);

  // defer declaration to solve the docstring problem described in
  // https://pybind11.readthedocs.io/en/latest/advanced/misc.html#avoiding-cpp-types-in-docstrings
  imageOnlyExecutor.def(py::init<>())
                   .def("isRenderDocAPIEnabled", &ImageOnlyExecutor::isRenderDocAPIEnabled)
                   .def("init", &ImageOnlyExecutor::init)
                   .def("initPipeline", &ImageOnlyExecutor::initPipeline)
                   .def("setUniform", &ImageOnlyExecutor::setUniform)
                   .def("preRender", &ImageOnlyExecutor::preRender)
                   .def("render", &ImageOnlyExecutor::render)
                   .def("getResults", &ImageOnlyExecutor::getResults, pybind11::return_value_policy::move)
                   .def("getTimingParameters", &ImageOnlyExecutor::getTimingParameters)
                   .def("getTraceBuffer", &ImageOnlyExecutor::getTraceBuffer)
                   .def("clearTraceBuffer", static_cast<void (ImageOnlyExecutor::*)()>(&ImageOnlyExecutor::clearTraceBuffer))
                   .def("clearTraceBuffer", py::overload_cast<const BinaryBlob &>(&ImageOnlyExecutor::clearTraceBuffer))
                   .def("getDeviceName", &ImageOnlyExecutor::getDeviceName)
                   .def("getDriverDescription", &ImageOnlyExecutor::getDriverDescription);

  py::class_<SpvProcessor> spvProcessor(m, "SpvProcessor");

  spvProcessor.def(py::init<>())
              .def("loadSpv", &SpvProcessor::loadSpv)
              .def("exportSpv", &SpvProcessor::exportSpv)
              .def("disassemble", &SpvProcessor::disassemble)
              .def("exhaustiveInlining", &SpvProcessor::exhaustiveInlining)
              // TODO: bind after vkExecute::spv::BasicBlock have correct binding
              //.def("separateBasicBlocks", &SpvProcessor::separateBasicBlocks)
              .def("assemble", &SpvProcessor::assemble)
              .def("validate", &SpvProcessor::validate)
              .def("runPassSequence", &SpvProcessor::runPassSequence)
              .def("instrumentBasicBlockTrace", &SpvProcessor::instrumentBasicBlockTrace)
              .def(py::pickle(
                  [](const vkExecute::SpvProcessor &p) { // __getstate__
                    return py::make_tuple(
                      p.exportSpv()
                    );
                  },
                  [](py::tuple t) { // __setstate__
                    if (t.size() != 1) {
                      throw std::runtime_error("Invalid state!");
                    }

                    vkExecute::SpvProcessor p;
                    p.loadSpv(t[0].cast<BinaryBlob>());
                    return p;
                  }
              ));

  auto m_spv = m.def_submodule("spv", "spv submodule");

  py::class_<vkExecute::spv::Tokenizer> tokenizer(m_spv, "Tokenizer");

  tokenizer.def(py::init<bool, bool, bool, bool>())
           .def("loadSpv", &vkExecute::spv::Tokenizer::loadSpv)
           .def("tokenize", &vkExecute::spv::Tokenizer::tokenize)
           .def("tokenizeWithTrace", &vkExecute::spv::Tokenizer::tokenizeWithTrace)
           .def("deTokenize", &vkExecute::spv::Tokenizer::deTokenize)
           .def(py::pickle(
              [](const vkExecute::spv::Tokenizer &p) { // __getstate__
                return py::make_tuple(
                  p.isCompactTypesEnabled(),
                  p.isEntrypointOnlyEnabled(),
                  p.isConvertExtInstsEnabled(),
                  p.isRelativeInstIdPosEnabled(),
                  p.exportSpv()
                );
              },
              [](py::tuple t) { // __setstate__
                if (t.size() != 5) {
                  throw std::runtime_error("Invalid state!");
                }

                vkExecute::spv::Tokenizer p(
                  t[0].cast<bool>(),
                  t[1].cast<bool>(),
                  t[2].cast<bool>(),
                  t[3].cast<bool>()
                );

                p.loadSpv(t[4].cast<BinaryBlob>());
                return p;
              }
           ));

  py::class_<vkDisplay> vk_display(m, "vkDisplay");

  vk_display.def(py::init<>())
           .def("setResolution", &vkDisplay::setResolution)
           .def("setShader", &vkDisplay::setShader)
           .def("setUiEnabled", &vkDisplay::setUiEnabled)
           .def("initMainWnd", &vkDisplay::initMainWnd)
           .def("initVkToy", &vkDisplay::initVkToy)
           .def("render", &vkDisplay::render)
           .def("renderUnlocked", [&](vkDisplay &self) {
             py::gil_scoped_release release_gil;
             self.render();
           });

  m.def("editDistance", &vkExecute::editDistance);

  m.def("version_info", []() {
#ifdef VKEXECUTE_VERSIONINFO
    const std::string ExtVersion = MACRO_STRINGIFY(VKEXECUTE_VERSIONINFO);
#else
    const std::string ExtVersion = "Unknown";
#endif
    
#ifdef VKEXECUTE_COMPILE_STATUS
    const std::string CompileStatus = MACRO_STRINGIFY(VKEXECUTE_COMPILE_STATUS);
#else
    const std::string CompileStatus = "Unknown";
#endif

    return ExtVersion + "_" + CompileStatus;
  });

#ifdef VKEXECUTE_VERSIONINFO
  m.attr("__version__") = MACRO_STRINGIFY(VKEXECUTE_VERSIONINFO);
#else
  m.attr("__version__") = "dev";
#endif
}
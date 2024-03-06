#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <nlohmann/json.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <thread>
#include <pybind11/pybind11.h>

#include "ShaderToy.hpp"
#include "avk/bindings.hpp"
#include "avk/buffer_meta.hpp"
#include "avk/command_buffer.hpp"
#include "avk/commands.hpp"
#include "avk/memory_usage.hpp"
#include "avk/shader_type.hpp"
#include "util.hpp"
#include "vkToy.hpp"
#include "vkDisplay.hpp"

namespace py = pybind11;

PYBIND11_MODULE(vkDisplay, m) {
  m.doc() = "vkDisplay bindings";
  py::class_<vkDisplay>(m, "vkDisplay")
    .def(py::init<>())
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
}
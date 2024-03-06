cmake_minimum_required(VERSION 3.20)
project(vkToy VERSION 0.1)

find_package(Vulkan REQUIRED) 
find_package(glslang REQUIRED)
find_package(SPIRV-Tools REQUIRED)
find_package(nlohmann_json REQUIRED)

enable_language(C CXX)

# include_directories("include")


set(VKTOY_COMMON_SRCS
    "src/vkToy/ShaderToy.cpp"
    "src/vkToy/glslang_resources.cpp"
    "src/vkToy/vkDisplay.cpp"
)

add_library(libVkToy STATIC ${VKTOY_COMMON_SRCS})

set_target_properties(libVkToy PROPERTIES CXX_STANDARD 20)
target_include_directories(libVkToy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(thirdParty/Auto-Vk-Toolkit)

function (avk_dll_postbuild_fix targetName)
  if (WIN32)
      if (CMAKE_BUILD_TYPE STREQUAL "Release" OR avk_toolkit_ReleaseDLLsOnly)
          set(avk_toolkit_DLLDirectory "${Auto_Vk_Toolkit_SOURCE_DIR}/external/release/bin/x64")
      else()
          set(avk_toolkit_DLLDirectory "${Auto_Vk_Toolkit_SOURCE_DIR}/external/debug/bin/x64")
      endif (CMAKE_BUILD_TYPE STREQUAL "Release" OR avk_toolkit_ReleaseDLLsOnly)
      file(GLOB dllsToCopy "${avk_toolkit_DLLDirectory}/*.dll")
      foreach(dllToCopy ${dllsToCopy})
          get_filename_component(dllFileName ${dllToCopy} NAME)
          add_custom_command(TARGET ${targetName} POST_BUILD
              COMMAND ${CMAKE_COMMAND} -E copy_if_different
                  ${dllToCopy}
                  $<TARGET_FILE_DIR:${targetName}>/${dllFileName}
              COMMENT "Post Build Helper: copying DLL ${dllFileName}")
      endforeach(dllToCopy)
  else()
      message(STATUS "Shared libraries should be in the rpath of vkExecute and are therefore not copied to the target location. Also, the location of built libraries should be configured via CMAKE_RUNTIME_OUTPUT_DIRECTORY, etc.")
  endif (WIN32)
endfunction(avk_dll_postbuild_fix)

target_link_libraries(libVkToy PUBLIC Auto_Vk_Toolkit nlohmann_json::nlohmann_json SPIRV-Tools)
target_link_libraries(libVkToy PUBLIC glslang MachineIndependent GenericCodeGen SPIRV glslang-default-resource-limits)

# pybind
set(PYBIND ${CMAKE_CURRENT_SOURCE_DIR}/thirdParty/pybind)
add_subdirectory(${PYBIND})

pybind11_add_module(vkDisplay src/vkToy/PythonMain.cpp)
set_property(TARGET libVkToy PROPERTY POSITION_INDEPENDENT_CODE ON)

# add_executable(vkDisplay src/vkToy/main.cpp)

set_target_properties(vkDisplay PROPERTIES CXX_STANDARD 20)
target_link_libraries(vkDisplay PUBLIC libVkToy)

avk_dll_postbuild_fix(vkDisplay)


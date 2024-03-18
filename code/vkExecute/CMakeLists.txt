cmake_minimum_required(VERSION 3.4)
project(vkExecute)

include(FetchContent)
find_package(Vulkan REQUIRED)
# Use fixed glslang version for now
#find_package(glslang)
#if (NOT glslang_FOUND)

set(FETCHCONTENT_QUIET FALSE)
FetchContent_Declare(
    glslang
    GIT_REPOSITORY https://github.com/KhronosGroup/glslang.git
    GIT_TAG sdk-1.3.250
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(glslang)
#endif()

option(VKEXECUTE_USE_EXTERNAL_SPIRV_TOOLS "Use external SPIRV-Tools and SPIRV-Headers" OFF)

# Force enable fuzzer build
set(SPIRV_BUILD_FUZZER OFF)
set(SPIRV_BUILD_TESTS OFF)
set(SKIP_SPIRV_TOOLS_INSTALL OFF)

message(STATUS "NOTE: use `python utils/git-sync-deps --treeless` to checkout necessary components if haven't")

if (${VKEXECUTE_USE_EXTERNAL_SPIRV_TOOLS})
  message(FATAL_ERROR "External SPIRV-Tools is not available; we have coupling between internal implementations")

  find_package(SPIRV-Tools)
  if (NOT SPIRV-Tools_FOUND)
  FetchContent_Declare(
    SPIRV-Headers
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Headers
      GIT_TAG sdk-1.3.250
  )

  FetchContent_MakeAvailable(SPIRV-Headers)

  FetchContent_Declare(
    SPIRV-Tools
      GIT_REPOSITORY https://github.com/KhronosGroup/SPIRV-Tools
      GIT_TAG sdk-1.3.250
  )

  FetchContent_MakeAvailable(SPIRV-Tools)
  endif()
else()
  add_subdirectory(thirdParty/SPIRV-Headers)
  add_subdirectory(thirdParty/SPIRV-Tools)
endif()

set(VKEXECUTE_COMMON_SRCS
  "src/ShaderProcessor.cpp"
  "src/SpvProcessor.cpp"
  "src/GlslangResources.cpp"
  "src/ImageOnlyExecutor.cpp"
  "src/utils.hpp"
  "src/spv/BasicBlock.cpp"
  "src/spv/BasicBlockTestEmitter.cpp"
  "src/spv/FormatConverter.cpp"
  "src/spv/AssignmentBuilder.cpp"
  "src/spv/ModuleBuilder.cpp"
  "src/spv/TypeBuilder.cpp"
  "src/spv/Tokenizer.cpp"
  "src/spv/Type.cpp"
  "src/vkToy/ShaderToy.cpp"
  "src/vkToy/glslang_resources.cpp"
  "src/vkToy/RenderWindow.cpp"
)

add_library(libVkExecute STATIC ${VKEXECUTE_COMMON_SRCS})

set_target_properties(libVkExecute PROPERTIES CXX_STANDARD 20)
target_include_directories(libVkExecute PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Uncomment the following if uses renderdoc
target_compile_definitions(libVkExecute PUBLIC VKEXECUTE_USE_RENDERDOC)
add_subdirectory(thirdParty/renderdoc)
target_link_libraries(libVkExecute PUBLIC renderdoc-api)

if (WIN32)
  target_link_libraries(libVkExecute PUBLIC dbghelp)
endif()

set(avk_toolkit_LibraryType "INTERFACE" CACHE STRING
  "The type of library avk_toolkit should be built as. Must be one of ${avk_toolkit_AllowedLibraryTypes}. Default: STATIC"
  FORCE
)
add_subdirectory(thirdParty/Auto-Vk-Toolkit)

# copy shared libraries
function (vkexecute_avk_dll_postbuild_fix targetName)
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
endfunction(vkexecute_avk_dll_postbuild_fix)

# spirv_tools_external_target_options() sets options for external library to use spirv_tools inner function and symbols
function(spirv_tools_external_target_options target)
  # spvtools_default_compile_options(${target})
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${spirv-tools_BINARY_DIR}
    PRIVATE ${SPIRV-Headers_SOURCE_DIR}/include
    PRIVATE ${spirv-tools_SOURCE_DIR}
  )
  # set_property(TARGET ${target} PROPERTY FOLDER "SPIRV-Tools libraries")
  # spvtools_check_symbol_exports(${target})
  add_dependencies(${target} spirv-tools-build-version core_tables enum_string_mapping extinst_tables)
endfunction()

target_link_libraries(libVkExecute PUBLIC Auto_Vk_Toolkit)
target_link_libraries(libVkExecute PUBLIC SPIRV-Tools-static SPIRV-Tools-opt)
spirv_tools_external_target_options(libVkExecute)
target_link_libraries(libVkExecute PUBLIC glslang MachineIndependent GenericCodeGen SPIRV glslang-default-resource-limits)

# == vkExecute ==
if(${BUILD_PYTHON_MODULE})
  add_subdirectory(pybind11)
  pybind11_add_module(vkExecute src/PythonMain.cpp)
  set_property(TARGET libVkExecute PROPERTY POSITION_INDEPENDENT_CODE ON)
else()
  add_executable(vkExecute src/Main.cpp)
  set(BUILD_VKEXECUTE_TESTS ON)
endif()


target_compile_definitions(vkExecute PRIVATE VKEXECUTE_COMPILE_STATUS=$<$<CONFIG:Debug>:Debug>$<$<CONFIG:Release>:Release>)
set_target_properties(vkExecute PROPERTIES CXX_STANDARD 20)
target_link_libraries(vkExecute PUBLIC libVkExecute)

if(DEFINED VKEXECUTE_VERSIONINFO)
  # EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
  # define (VERSION_INFO) here.
  target_compile_definitions(vkExecute PRIVATE VKEXECUTE_VERSIONINFO=${VKEXECUTE_VERSIONINFO})
endif()

vkexecute_avk_dll_postbuild_fix(vkExecute)

# => tests
set(VKEXECUTE_TESTS
  "testDummy"
  "testExecutor"
  "testProcessor"
  "testTokenizer"
)

# todo: refactor vkExecute into a interface target; avoid copying these over and over
function(vkexecute_add_test testName)
  add_executable(${testName} test/${testName}.cpp)
  target_link_libraries(${testName} PRIVATE Catch2::Catch2WithMain libVkExecute)

  set_target_properties(${testName} PROPERTIES CXX_STANDARD 20)
  vkexecute_avk_dll_postbuild_fix(${testName})
endfunction()

if (${BUILD_VKEXECUTE_TESTS})
  # add tests
  add_subdirectory(thirdParty/Catch2)
  foreach(vkExecuteTestName ${VKEXECUTE_TESTS})
    message(STATUS "Adding test ${vkExecuteTestName}")
    vkexecute_add_test(${vkExecuteTestName})
  endforeach(vkExecuteTestName)
endif()

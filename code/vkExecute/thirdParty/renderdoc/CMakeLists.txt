project(renderdoc-api)

add_library(renderdoc-api INTERFACE)
target_include_directories(renderdoc-api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/app)
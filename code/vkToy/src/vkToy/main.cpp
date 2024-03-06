#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <nlohmann/json.hpp>
#include <vulkan/vulkan_enums.hpp>
#include <ctime>
#include <thread>

#include "ShaderToy.hpp"
#include "util.hpp"
#include "vkToy.hpp"
#include "vkDisplay.hpp"

void setSdr(vkDisplay* display)
{
  std::string shader;
  while(true)
  {
    std::cin >> shader;
    display->setShader("vec3 createRect(vec2 uv, vec2 position, vec2 size, vec3 color) {\n    if(uv.x > position.x && uv.x < position.x + size.x && uv.y > position.y && uv.y < position.y + size.y) {\n        return color;\n    }\n    return vec3(0,0,0);\n}\nvec3 createCircle(vec2 uv, vec2 position, float radius, vec3 color) {\n    float x = uv.x - position.x - pow(uv.x - position.x, 2.);\n    if(x > .0) {\n        if(uv.y < sqrt(x) && uv.y < sqrt(x)) {\n            return color;\n        }\n    }\n    return vec3(0,0,0);\n}   void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n // Normalized pixel coordinates (from 0 to 1)\n    vec2 uv = fragCoord/iResolution.xy;\n\n    // Time varying pixel color\n    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n\n    float W = 1. / 2.;\n    \n    // Output to screen\n    fragColor = vec4(\n                     createRect(uv, vec2(0,0),           vec2(W,W), col)\n                   + createRect(uv, vec2(1. - W,1. - W), vec2(W,W), abs(cos(col)))\n                   + createRect(uv, vec2(1. - W,0),      vec2(W,W), 0.5 + 0.25 * abs(sin(tan(4. * col))))\n ,1.0);\n}");
  }
}

int main(int argc, char *argv[])
{
  vkDisplay display;
  display.setResolution(1024, 768);  
  display.setUiEnabled(true);
  display.initMainWnd();
  display.initVkToy();
  display.setShader("vec3 createRect(vec2 uv, vec2 position, vec2 size, vec3 color) {\n    if(uv.x > position.x && uv.x < position.x + size.x && uv.y > position.y && uv.y < position.y + size.y) {\n        return color;\n    }\n    return vec3(0,0,0);\n}\nvec3 createCircle(vec2 uv, vec2 position, float radius, vec3 color) {\n    float x = uv.x - position.x - pow(uv.x - position.x, 2.);\n    if(x > .0) {\n        if(uv.y < sqrt(x) && uv.y < sqrt(x)) {\n            return color;\n        }\n    }\n    return vec3(0,0,0);\n}   void mainImage( out vec4 fragColor, in vec2 fragCoord )\n{\n // Normalized pixel coordinates (from 0 to 1)\n    vec2 uv = fragCoord/iResolution.xy;\n\n    // Time varying pixel color\n    vec3 col = 0.5 + 0.5*cos(iTime+uv.xyx+vec3(0,2,4));\n\n    float W = 1. / 2.;\n    \n    // Output to screen\n    fragColor = vec4(\n                     createRect(uv, vec2(0,0),           vec2(W,W), col)\n                   + createRect(uv, vec2(1. - W,1. - W), vec2(W,W), abs(cos(col)))\n                   + createRect(uv, vec2(1. - W,0),      vec2(W,W), 0.5 + 0.25 * abs(sin(tan(4. * col))))\n                   + createRect(uv, vec2(0,1. - W),      vec2(W,W), abs(fract(1. - col))),1.0);\n}");
  std::thread t1 = std::thread(setSdr, &display);
  
  display.render();

  t1.join();
  
  return EXIT_SUCCESS;
}

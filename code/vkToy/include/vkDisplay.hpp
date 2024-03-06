#include <auto_vk_toolkit.hpp>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include <nlohmann/json.hpp>
#include <vulkan/vulkan_enums.hpp>

#include "ShaderToy.hpp"
#include "avk/bindings.hpp"
#include "avk/buffer_meta.hpp"
#include "avk/command_buffer.hpp"
#include "avk/commands.hpp"
#include "avk/memory_usage.hpp"
#include "avk/shader_type.hpp"
#include "util.hpp"
#include "vkToy.hpp"
#include <thread>

class vkDisplay
{
public:
    vkDisplay();
    ~vkDisplay();
    void setResolution(int width, int height);
    void setShader(std::string shader);
    void setUiEnabled(bool enabled);
    void initMainWnd();
    void initVkToy();
    void render();

private:
    int m_width;
    int m_height;
    bool uiEnabled;
    vkToy* m_vkToy;
    avk::queue* m_queue;
    avk::window* m_window;
};
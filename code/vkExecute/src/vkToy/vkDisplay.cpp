#include "vkDisplay.hpp"

vkDisplay::vkDisplay()
{
  uiEnabled = true;
  m_vkToy = nullptr;
  m_window = nullptr;
  m_queue = nullptr;
}

vkDisplay::~vkDisplay()
{
  delete m_vkToy;
}

void vkDisplay::setResolution(int width, int height)
{
    m_width = width;
    m_height = height;
    // Update window resolution if the window is already created
    if (m_window) {
      m_window->set_resolution({ m_width, m_height });
    }
}

void vkDisplay::setShader(std::string shader)
{
    m_vkToy->setShader(shader);
}

void vkDisplay::setUiEnabled(bool enabled)
{
    uiEnabled = enabled;
}

void vkDisplay::initMainWnd()
{
  // Create a window and open it
  m_window = avk::context().create_window("vkToy");
  m_window->set_resolution({ m_width, m_height });
  m_window->enable_resizing(true);
  m_window->set_presentaton_mode(avk::presentation_mode::mailbox);
  m_window->set_number_of_concurrent_frames(3u);
  m_window->set_number_of_presentable_images(3u);
  m_window->open();

  m_queue = &avk::context().create_queue({}, avk::queue_selection_preference::versatile_queue, m_window);
  m_window->set_queue_family_ownership(m_queue->family_index());
  m_window->set_present_queue(*m_queue);
}

void vkDisplay::initVkToy()
{
  // Create an instance of our main "invokee" which contains all the functionality:
  m_vkToy = new vkToy(
      *m_queue,
      std::make_unique<ShaderToy>(),
      std::nullopt,
      uiEnabled
  );
}

void vkDisplay::render() 
{
  auto ui = avk::imgui_manager(*m_queue);
  ui.set_use_fence_for_font_upload();

  auto composition = configure_and_compose(
    avk::application_name("vkToy"),
    [](avk::validation_layers& config) {
      config.enable_feature(vk::ValidationFeatureEnableEXT::eSynchronizationValidation);
    },
    // Pass windows:
    m_window,
    // Pass invokees:
    *m_vkToy, ui
  );

  // Create an invoker object, which defines the way how invokees/elements are invoked
  // (In this case, just sequentially in their execution order):
  avk::sequential_invoker invoker;

  composition.start_render_loop(
    // Callback in the case of update:
    [&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
      // Call all the update() callbacks:
      invoker.invoke_updates(aToBeInvoked);
    },
    // Callback in the case of render:
      [&invoker](const std::vector<avk::invokee*>& aToBeInvoked) {
      // Sync (wait for fences and so) per window BEFORE executing render callbacks
      avk::context().execute_for_each_window([](avk::window* wnd) {
        wnd->sync_before_render();
        });

      // Call all the render() callbacks:
      invoker.invoke_renders(aToBeInvoked);

      // Render per window:
      avk::context().execute_for_each_window([](avk::window* wnd) {
        wnd->render_frame();
        });
    }
    ); // This is a blocking call, which loops until avk::current_composition()->stop(); has been called (see update())

}
------------
add_requires("vulkansdk", "vulkan-hpp")
add_requires("glfw", "glm", "assimp", "nlohmann_json", "cereal", "fmt", "gli", "glad", "stb")
add_requires("imgui v1.87", {configs = {
    glfw = true,
    vulkan = true
}})

-- add_requires("glslang 1.3.239+0", {configs = {
--     debug = true
-- }})

includes("override/glslang.lua")
includes("override/spirv-tools.lua")

local avk_dir = "thirdParty/Auto-Vk-Toolkit"
local glslang_package_name = "glslang-local"
local spirv_tools_package_name = "spirv-tools"

add_requires(spirv_tools_package_name .. " v2023.1", {configs = {
    debug = true
}})

add_requires(glslang_package_name .. " 1.3.239+0", {configs = {
    debug = true
}})

---------
target("FileWatcher")
    set_kind("static")
    -- on_load(function (target)
    --     print("FileWatcher builddir = $(builddir)")
    --     os.cp(avk_dir .. "/external/universal/include/FileWatcher/*.h", "$(builddir)/include/FileWatcher")
    --     os.cp(avk_dir .. "/external/universal/src/FileWatcher/*.cpp", "$(builddir)/src")
    --     print(target)
    --     target:add_files("$(builddir)/src/*.cpp")
    --     target:add_files("$(builddir)/src/*.h")
    --     target:add_includedirs("$(builddir)/include/")
    -- end)

    add_files("thirdParty/FileWatcher/src/*.cpp")
    add_includedirs("thirdParty/FileWatcher/include/", {public = true})


-- package("FileWatcher-local")
--     on_load(function (package)
--         -- package:set("installdir", path.join(os.scriptdir(), "mustache"))
--         print("avk_dir: " ..  avk_dir)
--         print(debug.traceback())
--         -- os.cp(avk_dir .. "/external/universal/include/FileWatcher/*.h", package:installdir("include/FileWatcher"))
--         -- os.cp(avk_dir .. "/external/universal/src/FileWatcher/*.cpp", package:installdir("src"))
--     end)

--     on_install(function (package)
--         print(package:installdir())
--         os.vrun("cmd /k dir ")
--         os.cp(avk_dir .. "/external/universal/include/FileWatcher/*.h", package:installdir("include/FileWatcher"))
--         os.cp(avk_dir .. "/external/universal/src/FileWatcher/*.cpp", package:installdir("src"))
--         local xmake_lua = [[
--             add_rules("mode.debug", "mode.release")
--             target("FileWatcher-local")
--                 set_kind("static")
--                 add_files("src/*.cpp")
--                 add_headerfiles("include/FileWatcher/*.h")
--         ]]
--         io.writefile("xmake.lua", xmake_lua)
--         print("Invoking xmake for installation ...")
--         import("package.tools.xmake").install(package)
--     end)
--     -- add_includedirs(avk_dir .. "/external/universal/include/")
--     -- add_headerfiles(avk_dir .. "/external/universal/include/FileWatcher/*.h")
-- package_end()

-- add_requires("FileWatcher-local")

target("auto-vk")
    set_kind("static")
    
    add_defines("AVK_USE_VMA")
    add_includedirs(avk_dir .. "/auto_vk/include", {public = true})
    add_files(avk_dir .. "/auto_vk/src/*.cpp")
    
    add_packages("vulkansdk", "vulkan-hpp", {public = true})

    if is_plat("windows") then
        add_cxxflags("/bigobj")
    end

------------

target("auto-vk-toolkit")
    set_kind("static")

    add_defines("AVK_USE_VMA")
    add_deps("auto-vk", "FileWatcher")
    add_includedirs(avk_dir .. "/auto_vk_toolkit/include", {public = true})
    add_files(avk_dir .. "/auto_vk_toolkit/src/*.cpp")
    
    if is_plat("windows") then
        add_cxxflags("/bigobj")
    end

    add_packages("imgui", "glfw", "glm", "assimp", "nlohmann_json", "cereal", "fmt",
        "gli", "glad", "stb", {public = true})

------------

target("vkToy")
    set_kind("binary")
    set_default(true)
    set_rundir("$(projectdir)")
    set_runargs("--enable-play-manager")
    
    add_files("src/vkToy/*.cpp")
    add_includedirs("include", {public = false})

    add_deps("auto-vk-toolkit")
    -- add_defines("AVK_USE_VMA")
    add_packages(glslang_package_name, "nlohmann_json")

    if is_plat("windows") then
        add_cxxflags("/bigobj")
    end

target("traceToy")
    set_kind("binary")
    set_rundir("$(projectdir)")
    
    add_files("src/traceToy/*.cpp")
    add_includedirs("include", {public = false})

    add_deps("auto-vk-toolkit")
    -- add_defines("AVK_USE_VMA")
    add_packages(spirv_tools_package_name, "nlohmann_json")

    if is_plat("windows") then
        add_cxxflags("/bigobj")
    end
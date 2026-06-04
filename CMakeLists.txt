cmake_minimum_required(VERSION 3.21)
project(hand_motion_viewer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

include(FetchContent)

# ──────────────────────────────────────────────
#  Dependencies (all fetched and built by CMake)
# ──────────────────────────────────────────────

# SDL2 — windowing, GL context, input. Building from source keeps the project
# self-contained on every platform; the runtime DLL is copied next to the exe
# below so there is no loose SDL2.dll to hunt down.
set(SDL_TEST OFF CACHE BOOL "" FORCE)
set(SDL2_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    SDL2
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-2.30.11
    GIT_SHALLOW TRUE
)

# glm — vector/matrix math that replaces numpy + the GLU matrix helpers.
FetchContent_Declare(
    glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 1.0.1
    GIT_SHALLOW TRUE
)

# nlohmann/json — the JSON config (config.cpp).
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
    GIT_SHALLOW TRUE
)

# Dear ImGui (docking branch) — the UI. It ships no CMake of its own, so the
# core sources plus the SDL2 + OpenGL3 backends are compiled into a static
# library below. The OpenGL3 backend targets the programmable pipeline, matching
# the renderer's core 3.3 context; it bundles its own GL loader so ImGui needs no
# external one.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG docking
)

# portable-file-dialogs — native open-file / select-folder dialogs. This is the
# C++ original of the Python ``portable_file_dialogs`` the viewer already used;
# a single header, no binary dependency, easy to swap out. SOURCE_SUBDIR points
# at a non-existent directory so MakeAvailable only downloads the header and
# skips the project's own (outdated) CMakeLists; the interface target below
# exposes the include path.
FetchContent_Declare(
    portable_file_dialogs
    GIT_REPOSITORY https://github.com/samhocevar/portable-file-dialogs.git
    GIT_TAG 0.1.0
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR do-not-configure
)

# rapidobj — fast, parallel OBJ parser used for the single-file mesh loader (the
# per-frame sequence path keeps its own hand-tuned streaming parser). Disable its
# tests/examples/tools so only the header library is built.
set(RAPIDOBJ_BuildTests OFF CACHE BOOL "" FORCE)
set(RAPIDOBJ_BuildExamples OFF CACHE BOOL "" FORCE)
set(RAPIDOBJ_BuildTools OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    rapidobj
    GIT_REPOSITORY https://github.com/guybrush77/rapidobj.git
    GIT_TAG v1.1
    GIT_SHALLOW TRUE
)

# stb — single-header image loader (stb_image.h) used to decode the per-frame
# keypoint .jpg shown in the image viewport. The repo ships no usable CMake, so
# SOURCE_SUBDIR points at a non-existent directory to download the headers only;
# the interface target below exposes the include path.
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
    GIT_SHALLOW TRUE
    SOURCE_SUBDIR do-not-configure
)

FetchContent_MakeAvailable(SDL2 glm nlohmann_json imgui portable_file_dialogs rapidobj stb)

add_library(portable_file_dialogs INTERFACE)
target_include_directories(portable_file_dialogs INTERFACE ${portable_file_dialogs_SOURCE_DIR})

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

find_package(OpenGL REQUIRED)

# ── Dear ImGui static library ──────────────────
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl2.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui_lib PUBLIC SDL2::SDL2 OpenGL::GL)

# ── The viewer ─────────────────────────────────
add_executable(hand_motion_viewer
    src/gl_loader.cpp
    src/config.cpp
    src/geometry.cpp
    src/npy.cpp
    src/crossview.cpp
    src/sequence.cpp
    src/rendering.cpp
    src/image.cpp
    src/window.cpp
    src/gui.cpp
    src/main.cpp
)

target_link_libraries(hand_motion_viewer PRIVATE
    SDL2::SDL2
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    OpenGL::GL
    glm::glm
    nlohmann_json::nlohmann_json
    imgui_lib
    portable_file_dialogs
    rapidobj::rapidobj
    stb
)

if (APPLE)
    # Legacy fixed-function GL is deprecated on macOS but still functional;
    # silence the wall of warnings.
    target_compile_definitions(hand_motion_viewer PRIVATE GL_SILENCE_DEPRECATION)
endif ()

# Copy the SDL2 runtime library next to the executable on Windows so it launches
# straight from the build tree with no manual DLL placement.
if (WIN32)
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL2::SDL2>
        $<TARGET_FILE_DIR:hand_motion_viewer>
    )
endif ()

# Make the bundled font available next to the executable (matches the Python
# viewer's fonts/ lookup relative to the binary).
add_custom_command(TARGET hand_motion_viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/fonts
    $<TARGET_FILE_DIR:hand_motion_viewer>/fonts
)

cmake_minimum_required(VERSION 3.21)
project(hand_motion_viewer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

# When ON, link everything statically (SDL2, the C/C++ runtime, libgcc,
# winpthread) so the release exe runs with no bundled DLLs. opengl32.dll is the
# one exception — it is the system GPU driver loader and is always dynamic. Set
# at configure time: cmake -B build -DRELEASE_STATIC=ON
option(RELEASE_STATIC "Statically link the release exe (except opengl32.dll)" OFF)

# When ON, gather only the files that need shipping (exe + fonts/ + mano/, plus
# SDL2.dll for a non-static build) into <build>/dist after each build, so that
# folder can be zipped and distributed as-is. Set at configure time:
# cmake -B build -DPACKAGE_RELEASE=ON
option(PACKAGE_RELEASE "Assemble the shippable files into <build>/dist" OFF)

# Under MSVC a static build means the static CRT (/MT). Set globally and before
# any target is created so SDL, imgui and the exe all use the same runtime; the
# GCC/MinGW equivalent (-static) is applied to the exe link below.
if (RELEASE_STATIC AND MSVC)
    cmake_policy(SET CMP0091 NEW) # honour CMAKE_MSVC_RUNTIME_LIBRARY
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>" CACHE STRING "" FORCE)
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
# Build SDL as a static lib for a static release, shared otherwise.
if (RELEASE_STATIC)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
else ()
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
endif ()
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

FetchContent_MakeAvailable(SDL2 glm nlohmann_json imgui portable_file_dialogs stb)

add_library(portable_file_dialogs INTERFACE)
target_include_directories(portable_file_dialogs INTERFACE ${portable_file_dialogs_SOURCE_DIR})

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

find_package(OpenGL REQUIRED)

# SDL exposes SDL2::SDL2 when built shared and SDL2::SDL2-static when static;
# pick whichever this configuration produced.
if (RELEASE_STATIC)
    set(sdl2_target SDL2::SDL2-static)
else ()
    set(sdl2_target SDL2::SDL2)
endif ()

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
target_link_libraries(imgui_lib PUBLIC ${sdl2_target} OpenGL::GL)

# ── The viewer ─────────────────────────────────
# Glob the sources recursively so new files (and subfolders) are picked up
# without editing this list. CONFIGURE_DEPENDS re-globs when the source tree
# changes, so a build re-runs CMake instead of silently missing a new file.
file(GLOB_RECURSE viewer_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(hand_motion_viewer ${viewer_sources})

target_include_directories(hand_motion_viewer PRIVATE src)

target_link_libraries(hand_motion_viewer PRIVATE
    ${sdl2_target}
    $<TARGET_NAME_IF_EXISTS:SDL2::SDL2main>
    OpenGL::GL
    glm::glm
    nlohmann_json::nlohmann_json
    imgui_lib
    portable_file_dialogs
    stb
)

# Fold the C/C++ runtime into the exe for a static release. opengl32 stays
# dynamic (system import lib, no static form). The mechanism is compiler
# specific: GCC/MinGW links libstdc++, libgcc and winpthread in via -static (and
# -s strips the larger binary); MSVC uses the static CRT (/MT), set on the cache
# variable above so SDL and imgui match.
if (RELEASE_STATIC)
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        target_link_options(hand_motion_viewer PRIVATE
            -static
            -static-libgcc
            -static-libstdc++
            -s
        )
    endif ()
endif ()

if (APPLE)
    # Legacy fixed-function GL is deprecated on macOS but still functional;
    # silence the wall of warnings.
    target_compile_definitions(hand_motion_viewer PRIVATE GL_SILENCE_DEPRECATION)
endif ()

# Build as a GUI-subsystem app on Windows so launching it shows only the app
# window, with no console popping up. SDL2main supplies the WinMain entry point
# that forwards to main(), so no source change is needed.
if (WIN32)
    set_target_properties(hand_motion_viewer PROPERTIES WIN32_EXECUTABLE ON)
endif ()

# Copy the SDL2 runtime library next to the executable on Windows so it launches
# straight from the build tree with no manual DLL placement. Skipped for a static
# release, where SDL is linked into the exe and there is no DLL to copy.
if (WIN32 AND NOT RELEASE_STATIC)
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

# Copy the shared MANO face topology next to the executable so the .hmesh loader
# finds it (main.cpp resolves mano/mano_faces.bin relative to the binary). This
# keeps mano/ a single source of truth and avoids a stale hand-copied build copy.
add_custom_command(TARGET hand_motion_viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/mano
    $<TARGET_FILE_DIR:hand_motion_viewer>/mano
)

# Assemble a clean, distributable folder containing only what ships: the exe and
# the fonts/ and mano/ runtime assets (plus SDL2.dll when not statically linked).
# Built fresh each time (the dir is removed first) so it never holds stale files.
if (PACKAGE_RELEASE)
    set(dist_dir ${CMAKE_BINARY_DIR}/dist)
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${dist_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${dist_dir}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hand_motion_viewer> ${dist_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/fonts ${dist_dir}/fonts
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/mano ${dist_dir}/mano
        COMMENT "Packaging shippable files into ${dist_dir}"
    )
    if (WIN32 AND NOT RELEASE_STATIC)
        add_custom_command(TARGET hand_motion_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:SDL2::SDL2> ${dist_dir}
        )
    endif ()

    # Zip the assembled files into <build>/dist.zip for one-file distribution.
    # Run from inside dist/ and archive its entries (".") so the zip holds the
    # exe and asset folders at its root, not a nested dist/ directory. Uses
    # cmake -E tar (no external zip tool needed). Ordered after the copies above
    # since POST_BUILD commands on a target run in the order added.
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E tar cf ${CMAKE_BINARY_DIR}/dist.zip --format=zip .
        WORKING_DIRECTORY ${dist_dir}
        COMMENT "Zipping contents of ${dist_dir} into ${CMAKE_BINARY_DIR}/dist.zip"
    )
endif ()

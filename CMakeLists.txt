cmake_minimum_required(VERSION 3.21)
project(hand_motion_viewer CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif ()

# When ON, link everything statically (SDL3, the C/C++ runtime, libgcc,
# winpthread) so the release exe runs with no bundled DLLs. opengl32.dll is the
# one exception — it is the system GPU driver loader and is always dynamic. Set
# at configure time: cmake -B build -DRELEASE_STATIC=ON
option(RELEASE_STATIC "Statically link the release exe (except opengl32.dll)" OFF)

# When ON, gather only the files that need shipping (exe + assets/ + mano/, plus
# SDL3.dll for a non-static build) into <build>/dist/hand_motion_viewer after each
# build and zip it into <build>/hand_motion_viewer.zip. Set at configure time:
# cmake -B build -DPACKAGE_RELEASE=ON
option(PACKAGE_RELEASE "Assemble the shippable files into <build>/dist" OFF)

# When ON, build the Windows exe against the console subsystem so it keeps an
# attached console (useful for seeing stdout/stderr while debugging) instead of
# launching as a windowless GUI app. Set at configure time:
# cmake -B build -DWIN32_CONSOLE=ON
option(WIN32_CONSOLE "Build the Windows exe as a console app instead of a GUI app" OFF)

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

# SDL3 — windowing, GL context, input. Building from source keeps the project
# self-contained on every platform; the runtime DLL is copied next to the exe
# below so there is no loose SDL3.dll to hunt down.
set(SDL_TEST_LIBRARY OFF CACHE BOOL "" FORCE)
set(SDL_DISABLE_INSTALL ON CACHE BOOL "" FORCE)
# Build SDL as a static lib for a static release, shared otherwise.
if (RELEASE_STATIC)
    set(SDL_SHARED OFF CACHE BOOL "" FORCE)
    set(SDL_STATIC ON CACHE BOOL "" FORCE)
else ()
    set(SDL_SHARED ON CACHE BOOL "" FORCE)
    set(SDL_STATIC OFF CACHE BOOL "" FORCE)
endif ()
FetchContent_Declare(
    SDL3
    GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
    GIT_TAG release-3.4.8
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
# core sources plus the SDL3 + OpenGL3 backends are compiled into a static
# library below. The OpenGL3 backend targets the programmable pipeline, matching
# the renderer's core 3.3 context; it bundles its own GL loader so ImGui needs no
# external one.
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG docking
)

# FreeType — font rasterizer for Dear ImGui. The default stb_truetype builder is
# unhinted and renders UI text softly; FreeType grid-fits (hints) the glyphs for
# crisp text. Disable the optional helper libraries (zlib, png, harfbuzz, brotli,
# bzip2) so the build is self-contained and never depends on system packages.
set(FT_DISABLE_ZLIB ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BZIP2 ON CACHE BOOL "" FORCE)
set(FT_DISABLE_PNG ON CACHE BOOL "" FORCE)
set(FT_DISABLE_HARFBUZZ ON CACHE BOOL "" FORCE)
set(FT_DISABLE_BROTLI ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    freetype
    GIT_REPOSITORY https://github.com/freetype/freetype.git
    GIT_TAG VER-2-13-3
    GIT_SHALLOW TRUE
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

FetchContent_MakeAvailable(SDL3 glm nlohmann_json imgui freetype portable_file_dialogs stb)

add_library(portable_file_dialogs INTERFACE)
target_include_directories(portable_file_dialogs INTERFACE ${portable_file_dialogs_SOURCE_DIR})

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${stb_SOURCE_DIR})

find_package(OpenGL REQUIRED)

# SDL exposes SDL3::SDL3 when built shared and SDL3::SDL3-static when static;
# pick whichever this configuration produced.
if (RELEASE_STATIC)
    set(sdl3_target SDL3::SDL3-static)
else ()
    set(sdl3_target SDL3::SDL3)
endif ()

# ── Dear ImGui static library ──────────────────
add_library(imgui_lib STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/misc/freetype/imgui_freetype.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui_lib PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
    ${imgui_SOURCE_DIR}/misc/freetype
)
# IMGUI_ENABLE_FREETYPE switches the font atlas builder to FreeType (imgui_freetype.cpp);
# it is PUBLIC so the viewer's translation units that include imgui.h see the same
# config and the freetype builder flags are available where fonts are loaded.
target_compile_definitions(imgui_lib PUBLIC IMGUI_ENABLE_FREETYPE)
target_link_libraries(imgui_lib PUBLIC ${sdl3_target} OpenGL::GL freetype)

# ── The viewer ─────────────────────────────────
# Glob the sources recursively so new files (and subfolders) are picked up
# without editing this list. CONFIGURE_DEPENDS re-globs when the source tree
# changes, so a build re-runs CMake instead of silently missing a new file.
file(GLOB_RECURSE viewer_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
add_executable(hand_motion_viewer ${viewer_sources})

target_include_directories(hand_motion_viewer PRIVATE src)

target_link_libraries(hand_motion_viewer PRIVATE
    ${sdl3_target}
    $<TARGET_NAME_IF_EXISTS:SDL3::SDL3main>
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
# window, with no console popping up. SDL3 supplies the WinMain entry point (via
# SDL_main.h, included in main.cpp) that forwards to main().
if (WIN32 AND NOT WIN32_CONSOLE)
    set_target_properties(hand_motion_viewer PROPERTIES WIN32_EXECUTABLE ON)
endif ()

# Copy the SDL3 runtime library next to the executable on Windows so it launches
# straight from the build tree with no manual DLL placement. Skipped for a static
# release, where SDL is linked into the exe and there is no DLL to copy.
if (WIN32 AND NOT RELEASE_STATIC)
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:SDL3::SDL3>
        $<TARGET_FILE_DIR:hand_motion_viewer>
    )
endif ()

# Make the runtime assets available next to the executable: the explorer's icon
# PNGs and the bundled font (assets/fonts/), both resolved relative to the binary.
add_custom_command(TARGET hand_motion_viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_CURRENT_SOURCE_DIR}/assets
    $<TARGET_FILE_DIR:hand_motion_viewer>/assets
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
# the assets/ and mano/ runtime assets (plus SDL3.dll when not statically linked).
# Built fresh each time (the dir is removed first) so it never holds stale files.
if (PACKAGE_RELEASE)
    set(dist_dir ${CMAKE_BINARY_DIR}/dist)
    set(package_dir ${dist_dir}/hand_motion_viewer)
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E rm -rf ${dist_dir}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${package_dir}
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:hand_motion_viewer> ${package_dir}
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/assets ${package_dir}/assets
        COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/mano ${package_dir}/mano
        COMMENT "Packaging shippable files into ${package_dir}"
    )
    if (WIN32 AND NOT RELEASE_STATIC)
        add_custom_command(TARGET hand_motion_viewer POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE:SDL3::SDL3> ${package_dir}
        )
    endif ()

    # Zip the assembled files into <build>/hand_motion_viewer.zip for one-file
    # distribution. Run from dist/ and archive the hand_motion_viewer/ entry so the
    # zip holds a single top-level hand_motion_viewer folder, keeping everything
    # contained when unzipped. Uses cmake -E tar (no external zip tool needed).
    # Ordered after the copies above since POST_BUILD commands on a target run in
    # the order added.
    add_custom_command(TARGET hand_motion_viewer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E tar cf ${CMAKE_BINARY_DIR}/hand_motion_viewer.zip --format=zip hand_motion_viewer
        WORKING_DIRECTORY ${dist_dir}
        COMMENT "Zipping ${package_dir} into ${CMAKE_BINARY_DIR}/hand_motion_viewer.zip"
    )
endif ()

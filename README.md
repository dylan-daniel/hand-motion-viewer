# Hand Motion Viewer

A C++/OpenGL desktop viewer for inspecting reconstructed hand-motion sequences.
It plays back per-frame MANO hand meshes recovered from video and renders them in
an interactive 3D viewport.

## Features

- **Sequence playback** — loads a folder of per-frame `frame_NNNN_<slot>.hmesh`
  hand meshes (one compact binary mesh per detected hand per video frame) and
  plays them back at 30 FPS. Each `.hmesh` stores only the 778 MANO surface
  vertices and 21 joint positions; the face topology is shared globally (loaded
  once from `mano/mano_faces.bin`) and the colored joint skeleton is regenerated
  procedurally, so only the moving vertices are streamed per frame. Each frame's
  meshes are read and decoded from disk on demand as playback reaches them.
- **3D viewport** — core OpenGL 3.3 renderer with an orbit camera and grid.
- **Keypoint image view** — shows the per-frame `*_all_keypoints` image (`.jpg`
  or `.png`) alongside the 3D scene.
- **Dockable UI** — Dear ImGui (docking branch); dock layout and window geometry
  persist across launches.
- **Native file dialogs** — open a mesh sequence folder via the OS file picker.

## Requirements

- A C++20 compiler (MSVC, Clang, or GCC)
- [CMake](https://cmake.org/) 3.21 or newer
- A GPU/driver supporting OpenGL 3.3 core
- [Ninja](https://ninja-build.org/) (optional, recommended)
- Git (CMake fetches all dependencies from source)

All third-party libraries — SDL3, glm, nlohmann/json, Dear ImGui,
portable-file-dialogs, and stb — are downloaded and built
automatically by CMake via `FetchContent`. There is nothing to install by hand.
On Windows the SDL3 runtime DLL is copied next to the executable automatically.
The bundled font (`fonts/`) and the shared MANO face topology (`mano/`) are
copied next to the executable on every build as well, so the viewer finds them
relative to the binary.

## Building

Configure and build a Release binary:

```sh
cmake -S . -B build -G Ninja
cmake --build build --config Release
```

`-G Ninja` is optional; omit it to use your platform's default generator (e.g.
Visual Studio or Unix Makefiles). The first configure clones the dependencies,
so it takes longer than subsequent builds.

### Distribution builds

Two configure-time options help produce a release you can hand to others. Because
they are read at configure time, set them with `-D...` when configuring; the build
command is unchanged.

- **`RELEASE_STATIC`** — links everything (SDL3, the C/C++ runtime, libgcc /
  winpthread under GCC) into the executable so it runs with no bundled DLLs.
  `opengl32.dll` stays dynamic — it is the system GPU driver loader and is
  present on every Windows machine. Works under both GCC/MinGW (via `-static`)
  and MSVC (via the static CRT, `/MT`).
- **`PACKAGE_RELEASE`** — after each build, assembles only the shippable files
  (the executable plus the `fonts/` and `mano/` folders, and `SDL3.dll` for a
  non-static build) into `<build>/dist/`, then zips its contents into
  `<build>/dist.zip` for one-file distribution.

Example — a static, packaged Windows release:

```sh
cmake -S . -B build -DRELEASE_STATIC=ON -DPACKAGE_RELEASE=ON
cmake --build build --config Release
```

Then ship `build/dist.zip` (or the `build/dist/` folder). Note that with a
multi-config generator (e.g. Visual Studio) the executable itself lands under
`build/Release/`, but the packaged `dist/` and `dist.zip` are always at the build
root. The packaging step runs only when the executable is (re)linked.

## Running

```sh
build/hand_motion_viewer.exe
```

On non-Windows platforms the binary is `build/hand_motion_viewer`.

Open a mesh sequence folder from within the app using the file dialog. The folder
can live anywhere on disk.

## Data layout

A mesh sequence folder holds per-frame files:

- `frame_NNNN_<slot>.hmesh` — one compact binary hand mesh per detected hand per
  frame (778 MANO surface vertices + 21 joint positions)
- `frame_NNNN_all_keypoints.jpg` (or `.png`) — the keypoint overlay image for
  that frame

The MANO face topology shared by every hand lives in `mano/mano_faces.bin`
(`uint16` triangle indices, right-hand winding) and is loaded once at startup.

## Project layout

Sources live under `src/`, grouped into subfolders by domain. The build globs
them recursively, so adding a file (or a new subfolder) needs no CMake edit.
Includes are written relative to `src/` (e.g. `#include "data/geometry.h"`).

```
src/
├── app/                 Application shell
│   ├── main.cpp         Entry point, window/GL/ImGui setup, main loop
│   ├── window.*         SDL3 window + GL context, geometry persistence
│   └── config.*         JSON config load/save
├── graphics/            Rendering and GPU resources
│   ├── rendering.*      OpenGL 3.3 renderer, cameras, framebuffer
│   ├── image.*          Keypoint .jpg decoding (stb_image)
│   └── gl_loader.*      OpenGL function loading
├── ui/                  ImGui panels and controls
│   └── gui.*
├── data/                Hand-motion data model and loaders
│   ├── mesh_sequence.*  Per-frame mesh loading and playback state
│   └── geometry.*       Mesh/grid geometry + .hmesh decoding
└── util/                Shared utilities
    └── worker_queue.h   Background job queue (async keypoint image decode)
fonts/                   Bundled UI font (copied next to the exe)
mano/                    Shared MANO face topology (copied next to the exe)
```

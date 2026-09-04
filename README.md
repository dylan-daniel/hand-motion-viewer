# Hand Motion Viewer

A C++/OpenGL desktop viewer for inspecting reconstructed hand-motion sequences.
It plays back per-frame MANO hand meshes recovered from video and renders them in
an interactive 3D viewport.

## Features

- **Export playback** — opens a single `.hexport` binary file written by
  [infant_grasp_pipeline](https://github.com/dylan-daniel/infant_grasp_pipeline)'s
  `run_export_hands`: one row per tracked hand per frame, storing MANO
  shape/pose parameters (not the mesh itself). Loading a file runs every row
  through a MANO forward pass up front (fast — a handful of small matrix ops
  per hand) and plays the results back at 30 FPS. The face topology is shared
  globally (loaded once from `assets/mano/mano_faces.bin`) and the colored
  joint skeleton is regenerated procedurally.
- **3D viewport** — core OpenGL 3.3 renderer with an orbit camera and grid.
- **Frame View** — shows the plain source-video frame alongside the 3D scene,
  read from a `frames/<export-file-stem>/frame_%05d.jpg` (or `.png`) folder
  sitting next to the `.hexport` file. This is a viewer-side packaging
  convention, not something recorded inside the export itself; the pane stays
  empty if that folder isn't present.
- **Per-track coloring** (Settings menu) — off (default) shows only
  infant-labeled hands, tinted red (left) / blue (right); on shows every
  hand, each colored uniquely by its pipeline tracking ID.
- **Explorer pane** — a folder tree rooted at a chosen data folder, pruned to
  the folders that lead to a `.hexport` file at any nesting depth; the files
  themselves are openable leaves.
- **Dockable UI** — Dear ImGui (docking branch); dock layout and window geometry
  persist across launches.
- **Native file dialogs** — open a `.hexport` file via the OS file picker
  (menu bar) or the Explorer pane.

## Requirements

- A C++20 compiler (MSVC, Clang, or GCC)
- [CMake](https://cmake.org/) 3.21 or newer
- A GPU/driver supporting OpenGL 3.3 core
- [Ninja](https://ninja-build.org/) (optional, recommended)
- Git (CMake fetches all dependencies from source)

All third-party libraries — SDL3, glm, nlohmann/json, Dear ImGui,
portable-file-dialogs, stb, and miniz — are downloaded and built
automatically by CMake via `FetchContent`. There is nothing to install by hand.
On Windows the SDL3 runtime DLL is copied next to the executable automatically.
The bundled font and the shared MANO assets (`assets/fonts/`, `assets/mano/`)
are copied next to the executable on every build as well, so the viewer finds
them relative to the binary.

## Building

Configure and build a Release binary:

```sh
cmake -S . -B build -G Ninja
cmake --build build --config Release
```

`-G Ninja` is optional; omit it to use your platform's default generator (e.g.
Visual Studio or Unix Makefiles). The first configure clones the dependencies,
so it takes longer than subsequent builds — miniz is small, but this is still
the heaviest dependency set in the project (SDL3 + FreeType + Dear ImGui all
build from source too).

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
  (the executable plus the `assets/` folder, and `SDL3.dll` for a non-static
  build) into `<build>/dist/`, then zips its contents into `<build>/dist.zip`
  for one-file distribution.

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

Open a `.hexport` file from within the app via **File → Open Export File**, or
by picking a data folder in the Explorer pane and clicking a file it finds.

## Data layout

A `.hexport` file is a single self-contained binary (see
`src/data/hand_export.h` for the exact format: a small header, then one
zlib-compressed payload holding a name/dtype schema table and the column
data). To also populate the Frame View pane, place a sibling folder next to
it named after the export file:

```
GZ65_T1_BabyView.hexport
frames/
└── GZ65_T1_BabyView/
    ├── frame_00001.jpg
    ├── frame_00002.jpg
    └── ...
```

Naming the subfolder after the export file (rather than one shared `frames/`)
disambiguates multiple `.hexport` files sitting in the same directory. Both
`.jpg` and `.png` are supported; a missing folder or frame just means no
image for that frame; it is not an error.

The MANO face topology and model weights shared by every hand live in
`assets/mano/` (`mano_faces.bin`: `uint16` triangle indices, right-hand
winding; `mano_model.bin`: the MANO layer's weights) and are loaded once at
startup.

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
│   ├── image.*          Frame .jpg/.png decoding (stb_image)
│   └── gl_loader.*      OpenGL function loading
├── ui/                  ImGui panels and controls
│   ├── gui.*             Menu bar, viewport/image windows, transport bar
│   └── explorer.*        Data-folder tree browser for .hexport files
├── data/                Hand-motion data model and loaders
│   ├── hand_export.*    .hexport binary format reader (see the file for the byte layout)
│   ├── mano_model.*     MANO forward pass (shape/pose params -> verts + joints)
│   ├── mesh_sequence.*  Per-frame playback state, built from a loaded .hexport
│   └── geometry.*       Mesh/grid geometry, MANO face topology, track coloring
└── util/                Shared utilities
    └── worker_queue.h   Background job queue (async Explorer folder scan)
assets/
├── fonts/               Bundled UI font (copied next to the exe)
├── icons/                Explorer/transport icon PNGs (copied next to the exe)
└── mano/                 Shared MANO face topology + model weights (copied next to the exe)
```

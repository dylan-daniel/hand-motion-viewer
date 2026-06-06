# Hand Motion Viewer

A C++/OpenGL desktop viewer for inspecting reconstructed hand-motion sequences.
It plays back per-frame MANO hand meshes recovered from video, renders them in an
interactive 3D viewport, and can fold a second camera's reconstruction into the
first camera's coordinate space so both views line up.

## Features

- **Sequence playback** — loads a folder of per-frame `frame_NNNN_<slot>.hmesh`
  hand meshes (one compact binary mesh per detected hand per video frame) and
  plays them back at 30 FPS. Each `.hmesh` stores only the 778 MANO surface
  vertices and 21 joint positions; the face topology is shared globally (loaded
  once from `mano/mano_faces.bin`) and the colored joint skeleton is regenerated
  procedurally, so only the moving vertices are streamed per frame. Parsing runs
  on background worker threads to keep the UI responsive.
- **Cross-view overlay** — aligns an over-hand (OHView) camera's hands into a
  BabyView camera's frame using a per-frame scaled Kabsch fit over the 3D joints,
  so both reconstructions can be drawn together.
- **3D viewport** — core OpenGL 3.3 renderer with an orbit camera and grid.
- **Keypoint image view** — shows the per-frame `*_all_keypoints.jpg` alongside
  the 3D scene.
- **Dockable UI** — Dear ImGui (docking branch); dock layout and window geometry
  persist across launches.
- **Native file dialogs** — open a sequence folder via the OS file picker.

## Requirements

- A C++20 compiler (MSVC, Clang, or GCC)
- [CMake](https://cmake.org/) 3.21 or newer
- A GPU/driver supporting OpenGL 3.3 core
- [Ninja](https://ninja-build.org/) (optional, recommended)
- Git (CMake fetches all dependencies from source)

All third-party libraries — SDL2, glm, nlohmann/json, Dear ImGui,
portable-file-dialogs, and stb — are downloaded and built
automatically by CMake via `FetchContent`. There is nothing to install by hand.
On Windows the SDL2 runtime DLL is copied next to the executable automatically.
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

## Running

```sh
build/hand_motion_viewer.exe
```

On non-Windows platforms the binary is `build/hand_motion_viewer`.

Open a sequence folder from within the app using the file dialog. A sample
sequence ships under `data/SUBJECT/`.

## Data layout

A sequence folder holds per-frame files:

- `frame_NNNN_<slot>.hmesh` — one compact binary hand mesh per detected hand per
  frame (778 MANO surface vertices + 21 joint positions)
- `frame_NNNN_<slot>_joints3d.npy`, `_cam.npy`, `_pose.npy`, `_shape.npy` —
  per-hand sidecars (joints and camera are used for cross-view alignment)
- `frame_NNNN_all_keypoints.jpg` — the keypoint overlay image for that frame

The MANO face topology shared by every hand lives in `mano/mano_faces.bin`
(`uint16` triangle indices, right-hand winding) and is loaded once at startup.

## Project layout

```
src/
├── main.cpp        Entry point, window/GL/ImGui setup, main loop
├── window.*        SDL2 window + GL context, geometry persistence
├── config.*        JSON config load/save
├── sequence.*      Per-frame mesh loading and playback state
├── crossview.*     OHView → BabyView similarity alignment
├── geometry.*      Mesh/grid geometry helpers
├── rendering.*     OpenGL 3.3 renderer
├── image.*         Keypoint .jpg decoding (stb_image)
├── npy.*           NumPy .npy sidecar parsing
├── gui.*           ImGui panels and controls
├── gl_loader.*     OpenGL function loading
└── worker_queue.h  Background parsing job queue
fonts/              Bundled UI font (copied next to the exe)
mano/               Shared MANO face topology (copied next to the exe)
data/               Sample sequence(s)
```

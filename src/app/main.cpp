#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <SDL3/SDL_opengl.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl3.h>
#include <imgui_internal.h>

#include <portable-file-dialogs.h>

#include "app/config.h"
#include "app/window.h"
#include "data/mano_model.h"
#include "data/mesh_sequence.h"
#include "graphics/gl_loader.h"
#include "graphics/image.h"
#include "graphics/rendering.h"
#include "ui/explorer.h"
#include "ui/gui.h"

namespace {
    // Frames advanced per second while the sequence is playing.
    constexpr double PLAYBACK_FPS = 30.0;
} // namespace

int main(int, char**) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::printf("SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const std::string config_path = default_config_path();
    Config settings = load_config(config_path);

    // Load the shared MANO face topology every hand renders against (both .hmesh
    // and .hexport sources), plus the MANO model weights .hexport sources need to
    // regenerate hand geometry via a forward pass. Both live under assets/mano/,
    // consolidated there from the old top-level mano/ (see CMakeLists.txt).
    {
        std::string assets_mano_dir = "assets/mano/";
        const char* base = SDL_GetBasePath();
        if (base != nullptr) {
            assets_mano_dir = std::string(base) + "assets/mano/";
        }
        try {
            init_mano_topology(assets_mano_dir + "mano_faces.bin");
        } catch (const std::exception& error) {
            std::printf("Warning: %s — hand sequences will not render.\n", error.what());
        }
        try {
            init_mano_model(assets_mano_dir + "mano_model.bin");
        } catch (const std::exception& error) {
            std::printf("Warning: %s — .hexport files will not load.\n", error.what());
        }
    }

    WindowGeometry geometry = compute_initial_geometry(settings);
    WindowGeometry windowed_geometry = geometry;

    // Request a core OpenGL 3.3 context before creating the window. The renderer
    // and the ImGui OpenGL3 backend both target the programmable pipeline.
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

    // Create the window via a properties bag so its position is set before creation
    // rather than patched in afterward (SDL_CreateWindow takes no position). It is
    // created hidden and stays hidden through GL/ImGui init and the first rendered
    // frame (which builds the default dock layout) so the user never sees the UI
    // assemble itself — it is shown once after the first frame swaps, already fully
    // laid out, below.
    SDL_PropertiesID window_props = SDL_CreateProperties();
    SDL_SetStringProperty(window_props, SDL_PROP_WINDOW_CREATE_TITLE_STRING, "Hand Motion Viewer");
    SDL_SetNumberProperty(window_props, SDL_PROP_WINDOW_CREATE_X_NUMBER, geometry.x);
    SDL_SetNumberProperty(window_props, SDL_PROP_WINDOW_CREATE_Y_NUMBER, geometry.y);
    SDL_SetNumberProperty(window_props, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, geometry.width);
    SDL_SetNumberProperty(window_props, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, geometry.height);
    SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_CREATE_OPENGL_BOOLEAN, true);
    SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_CREATE_RESIZABLE_BOOLEAN, true);
    SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_CREATE_HIGH_PIXEL_DENSITY_BOOLEAN, true);
    SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_CREATE_HIDDEN_BOOLEAN, true);
    SDL_SetBooleanProperty(window_props, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN, settings.window_fullscreen);

    SDL_Window* app_window = SDL_CreateWindowWithProperties(window_props);
    SDL_DestroyProperties(window_props);

    if (app_window == nullptr) {
        std::printf("SDL_CreateWindowWithProperties failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(app_window);
    SDL_GL_MakeCurrent(app_window, gl_context);
    SDL_GL_SetSwapInterval(1); // vsync

    if (!glx::load()) {
        std::printf("Warning: some OpenGL entry points could not be loaded.\n");
    }

    // Pixel size (not logical size) drives the final glViewport so it stays correct
    // on high-DPI displays where the two differ.
    int win_width = 0;
    int win_height = 0;
    SDL_GetWindowSizeInPixels(app_window, &win_width, &win_height);

    glEnable(GL_DEPTH_TEST);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& imgui_io = ImGui::GetIO();
    imgui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Remove the docking/collapse menu button (the small triangle in each pane's
    // tab bar). Doing it via the style rather than the per-node NoWindowMenuButton
    // flag also drops the space the tab bar reserved for it: the tab-bar layout
    // only offsets for the button when WindowMenuButtonPosition is Left, so None
    // hides the button and reclaims the offset everywhere — including the docking
    // drag preview, which otherwise ignores the per-node flag and leaves a gap.
    ImGui::GetStyle().WindowMenuButtonPosition = ImGuiDir_None;

    // Persist the dock layout next to the executable (like config.json) instead of
    // imgui's default cwd-relative path, so the saved window arrangement is
    // restored no matter where the app is launched from. The string must outlive
    // the context: imgui keeps the pointer and writes the file on shutdown.
    static std::string imgui_ini_path;
    {
        const char* base = SDL_GetBasePath();
        imgui_ini_path = (base != nullptr ? std::string(base) : std::string()) + "imgui.ini";
    }
    imgui_io.IniFilename = imgui_ini_path.c_str();

    // On the very first run there is no saved layout, so build a default one: the
    // "Explorer" pane on the left, then "Frame View" and "Scene" splitting the
    // rest. Detected by the absence of the ini file imgui would have written on a
    // previous shutdown.
    bool build_default_layout = !std::filesystem::exists(imgui_ini_path);

    auto [ui_font, fps_font] = load_fonts(imgui_io);
    imgui_io.FontDefault = ui_font;

    ImGui_ImplSDL3_InitForOpenGL(app_window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 330");

    Framebuffer framebuffer;
    framebuffer.resize(win_width, win_height);

    // Decoded keypoint image for the current frame, shown in the "Frame View" pane.
    ImageTexture frame_image;
    // Which pane the playback transport rides on: 0 = Viewport, 1 = Frame View.
    // Tracks the focused pane so the player follows whichever the mouse last
    // interacted with; falls back to the viewport when none is focused. Seeded
    // from the persisted pane so the previously active tab is restored on launch.
    int active_pane = settings.active_pane;

    // imgui does not reliably restore which docked tab was selected (it defaults
    // to the last-submitted pane), so for the first few frames we re-assert focus
    // on the persisted pane by name once its window exists. Focusing a docked
    // window brings its tab to front, which is what selects it. It self-disables
    // afterward so it never fights the user's own tab clicks.
    const int restore_pane = settings.active_pane;
    int restore_focus_frames = 3;

    // ── Sequence state ─────────────────────────
    std::unique_ptr<MeshSequence> sequence;
    // GPU buffers for the frame currently on screen, rebuilt from disk whenever
    // the frame changes. loaded_frame tracks which frame current_gpu holds (-1 =
    // none) so we only re-read and re-upload when the frame actually advances.
    std::unique_ptr<FrameGpu> current_gpu;
    int loaded_frame = -1;
    std::optional<Transform> transform;
    std::optional<float> depth_reference;
    int current_frame = 0;
    bool playing = false;
    // Countdown until the held left/right arrow steps again. Negative means no
    // arrow is currently held, so the next press steps immediately.
    double scrub_repeat_timer = -1.0;
    // True while the user drags the scrubber; suspends auto-advance so the hands
    // don't jitter between the dragged frame and the next. Carried across frames
    // since the transport (which reports it) is drawn after playback is advanced.
    bool scrubbing = false;
    float playback_speed = settings.playback_speed;
    double playback_accumulator = 0.0;

    auto open_sequence = [&](const std::string& folder, int start_frame) {
        std::unique_ptr<MeshSequence> opened;
        bool has_frames = false;
        try {
            opened = std::make_unique<MeshSequence>(folder);
            has_frames = opened->frame_count() > 0;
            if (!has_frames) {
                std::printf("No frames found in %s\n", folder.c_str());
            }
        } catch (const std::exception& error) {
            // A .hexport file that fails to parse/decompress, or whose MANO
            // model wasn't loaded at startup, throws from the constructor
            // rather than the disk-mode path's "just found nothing" case.
            std::printf("Failed to open %s: %s\n", folder.c_str(), error.what());
        }
        // Reset the viewer to the requested folder either way: a folder with no
        // frames clears the scene rather than leaving the previous sequence on
        // screen (so opening from the Explorer always resets, like the menu does).
        sequence = has_frames ? std::move(opened) : nullptr;
        current_gpu.reset();
        loaded_frame = -1;
        transform.reset();
        depth_reference.reset();
        current_frame = has_frames ? std::clamp(start_frame, 0, sequence->frame_count() - 1) : 0;
    };

    // Reopen the last mesh sequence folder if present.
    namespace fs = std::filesystem;
    if (settings.last_folder && fs::is_directory(*settings.last_folder)) {
        open_sequence(*settings.last_folder, settings.last_frame);
    }

    // Explorer pane, rooted at the saved data folder. set_root validates the path and
    // kicks off the background scan that builds the pruned tree; it is a no-op when no
    // folder is saved.
    FileExplorer explorer;
    // Play/pause icons for the transport bar, loaded once here (GL context is current).
    TransportIcons transport_icons;
    {
        // Load icon textures from assets/icons/ next to the binary.
        const char* base = SDL_GetBasePath();
        const std::string icons_dir = (base != nullptr ? std::string(base) : std::string()) + "assets/icons";
        explorer.icons().load(icons_dir);
        transport_icons.load(icons_dir);
    }
    // Restore the tree's expanded folders from last session, then scan the root.
    explorer.set_saved_open(settings.expanded_folders);
    if (settings.data_folder) {
        explorer.set_root(*settings.data_folder);
    }

    // Native pickers: a folder for opening a .hmesh mesh sequence (menu / Explorer
    // click) or choosing the Explorer's data-folder root, and a single-file picker
    // for opening a .hexport binary export.
    std::unique_ptr<pfd::select_folder> folder_dialog;
    std::unique_ptr<pfd::select_folder> data_folder_dialog;
    std::unique_ptr<pfd::open_file> export_file_dialog;
    // A folder the Explorer asked to open, applied at the top of the next frame
    // rather than mid-frame: open_sequence frees current_gpu, and the render below
    // still holds a pointer to it, so opening inline would use freed memory.
    std::optional<std::string> pending_open_folder;

    // Both cameras kept alive; ``camera`` points at the active one.
    OrbitCamera orbit_cam;
    orbit_cam.set_state(settings.camera_azimuth, settings.camera_elevation, settings.camera_distance, settings.camera_target);
    FreeCamera free_cam;
    if (settings.free_camera) {
        free_cam.set_from_orbit(orbit_cam);
    }
    Camera* camera = settings.free_camera ? static_cast<Camera*>(&free_cam) : static_cast<Camera*>(&orbit_cam);

    bool orbiting = false;
    bool panning = false;
    bool viewport_hovered = false;

    Uint64 last_ticks = SDL_GetTicks();
    bool running = true;
    // The window is created hidden and revealed after the first frame swaps below.
    bool window_shown = false;
    // Tracks the main viewport width across frames so the one-time default dock
    // layout is built only once the size has settled (see the build block below).
    float last_viewport_width = 0.0f;
    while (running) {
        const Uint64 now_ticks = SDL_GetTicks();
        const double dt_seconds = static_cast<double>(now_ticks - last_ticks) / 1000.0;
        last_ticks = now_ticks;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL3_ProcessEvent(&event);

            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            } else if (event.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                win_width = event.window.data1;
                win_height = event.window.data2;
            } else if (event.type == SDL_EVENT_KEY_DOWN) {
                const SDL_Keycode key = event.key.key;
                if (key == SDLK_ESCAPE) {
                    running = false;
                } else if (key == SDLK_R) {
                    camera->reset();
                } else if (key == SDLK_H) {
                    settings.hand_translucent = !settings.hand_translucent;
                } else if (key == SDLK_SPACE && event.key.repeat == 0 && !imgui_io.WantTextInput) {
                    if (sequence && sequence->frame_count() > 0) {
                        playing = !playing;
                    }
                } else if (sequence && !playing && (key == SDLK_HOME || key == SDLK_END)) {
                    // Left/right scrubbing is handled by per-frame polling below so
                    // that holding an arrow keeps stepping even while another key is
                    // pressed; Home/End are one-shot jumps and stay event-driven.
                    if (key == SDLK_HOME) {
                        current_frame = 0;
                    } else {
                        current_frame = sequence->frame_count() - 1;
                    }
                } else if (key == SDLK_F11) {
                    if (!settings.window_fullscreen) {
                        windowed_geometry = read_current_geometry(app_window);
                        SDL_SetWindowFullscreen(app_window, true);
                    } else {
                        SDL_SetWindowFullscreen(app_window, false);
                    }
                    settings.window_fullscreen = !settings.window_fullscreen;
                    SDL_GetWindowSizeInPixels(app_window, &win_width, &win_height);
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && viewport_hovered) {
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    if (SDL_GetModState() & SDL_KMOD_SHIFT) {
                        panning = true;
                    } else {
                        orbiting = true;
                    }
                    set_relative_mouse(app_window, true);
                }
            } else if (event.type == SDL_EVENT_MOUSE_BUTTON_UP) {
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    orbiting = false;
                    panning = false;
                    set_relative_mouse(app_window, false);
                }
            } else if (event.type == SDL_EVENT_MOUSE_WHEEL && viewport_hovered) {
                if (event.wheel.y > 0) {
                    camera->zoom(1.0f);
                } else if (event.wheel.y < 0) {
                    camera->zoom(-1.0f);
                }
            } else if (event.type == SDL_EVENT_MOUSE_MOTION) {
                const float dx = static_cast<float>(event.motion.xrel);
                const float dy = static_cast<float>(event.motion.yrel);
                if (orbiting) {
                    camera->orbit(dx, dy);
                } else if (panning) {
                    camera->pan(dx, dy);
                }
            }
        }

        // WASD/QE continuous movement. Gated on WantTextInput (a focused text
        // field) rather than WantCaptureKeyboard: the latter is forced true while
        // any mouse button is held over an imgui window (the click grabs the
        // window move-id as the active item), which would otherwise freeze camera
        // movement whenever the left button is down over the viewport.
        if (!imgui_io.WantTextInput) {
            const bool* keys = SDL_GetKeyboardState(nullptr);
            const float forward = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
            const float right = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
            const float up = (keys[SDL_SCANCODE_E] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_Q] ? 1.0f : 0.0f);
            if (forward != 0.0f || right != 0.0f || up != 0.0f) {
                camera->move(forward, right, up, static_cast<float>(dt_seconds));
            }

            // Frame scrubbing by held left/right arrow. Polled per-frame rather
            // than driven by OS key-repeat events so that holding an arrow keeps
            // stepping even while another key is pressed (the OS only auto-repeats
            // the most recently pressed key). A typematic-style delay/interval
            // keeps held scrubbing from racing through frames at full framerate.
            constexpr double scrub_initial_delay = 0.25;
            constexpr double scrub_repeat_interval = 0.01;
            const int scrub_dir = (keys[SDL_SCANCODE_RIGHT] ? 1 : 0) - (keys[SDL_SCANCODE_LEFT] ? 1 : 0);
            if (sequence && !playing && scrub_dir != 0) {
                bool step_now = false;
                if (scrub_repeat_timer < 0.0) {
                    // First frame the arrow is held: step immediately, then wait
                    // the longer initial delay before auto-repeating.
                    step_now = true;
                    scrub_repeat_timer = scrub_initial_delay;
                } else {
                    scrub_repeat_timer -= dt_seconds;
                    if (scrub_repeat_timer <= 0.0) {
                        step_now = true;
                        scrub_repeat_timer = scrub_repeat_interval;
                    }
                }
                if (step_now) {
                    const int last = sequence->frame_count() - 1;
                    current_frame = std::clamp(current_frame + scrub_dir, 0, last);
                }
            } else {
                scrub_repeat_timer = -1.0;
            }
        }

        // ── Build the imgui frame ──────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL3_NewFrame();
        ImGui::NewFrame();

        const bool was_free_camera = settings.free_camera;

        // Label for the menu bar: the absolute path of the open sequence folder.
        std::string open_folder_label;
        if (sequence) {
            std::error_code abs_error;
            const fs::path absolute_folder = fs::absolute(sequence->folder(), abs_error);
            open_folder_label = abs_error ? sequence->folder() : absolute_folder.string();
        }

        const MenuResult menu = draw_menu_bar(
            MenuState{
                .hand_translucent = settings.hand_translucent,
                .show_camera_marker = settings.show_camera_marker,
                .free_camera = settings.free_camera,
                .open_folder = open_folder_label,
            }
        );
        settings.hand_translucent = menu.state.hand_translucent;
        settings.show_camera_marker = menu.state.show_camera_marker;
        settings.free_camera = menu.state.free_camera;
        if (settings.free_camera != was_free_camera) {
            if (settings.free_camera) {
                free_cam.set_from_orbit(orbit_cam);
                camera = &free_cam;
            } else {
                orbit_cam.set_from_free(free_cam);
                camera = &orbit_cam;
            }
        }

        // The docking menu button is hidden globally via WindowMenuButtonPosition
        // (set at startup), which also reclaims the tab-bar offset it left behind.
        const ImGuiID dock_id = ImGui::DockSpaceOverViewport();

        // Build the one-time default layout only after the window is shown and its
        // size has settled (the same width two frames running). A fullscreen window
        // reaches its final size a frame or two after being revealed; building before
        // then freezes the side panes at a tiny pixel size that doesn't scale up.
        const ImVec2 viewport_size = ImGui::GetMainViewport()->Size;
        const bool viewport_settled = window_shown && viewport_size.x > 0.0f && viewport_size.x == last_viewport_width;
        last_viewport_width = viewport_size.x;

        if (build_default_layout && viewport_settled) {
            build_default_layout = false;
            ImGui::DockBuilderRemoveNode(dock_id);
            ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dock_id, viewport_size);

            // Explorer takes the leftmost 25% and Frame View the rightmost 25%, with
            // the Scene filling the 50% between them. The Frame View split is taken
            // from the remaining 75%, so its fraction is 0.25 / 0.75 of that node.
            ImGuiID dock_explorer = 0;
            ImGuiID dock_rest = 0;
            ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left, 0.25f, &dock_explorer, &dock_rest);
            ImGuiID dock_frame = 0;
            ImGuiID dock_scene = 0;
            ImGui::DockBuilderSplitNode(dock_rest, ImGuiDir_Right, 0.25f / 0.75f, &dock_frame, &dock_scene);

            ImGui::DockBuilderDockWindow("Explorer", dock_explorer);
            ImGui::DockBuilderDockWindow("Scene", dock_scene);
            ImGui::DockBuilderDockWindow("Frame View", dock_frame);
            ImGui::DockBuilderFinish(dock_id);
        }

        // Apply the open-folder/open-file request before choosing what to draw,
        // since it frees GPU buffers the drawable below must reflect.
        if (menu.folder_requested && !folder_dialog) {
            folder_dialog = std::make_unique<pfd::select_folder>("Select mesh sequence folder");
        }
        if (menu.export_file_requested && !export_file_dialog) {
            export_file_dialog =
                std::make_unique<pfd::open_file>("Select hand-motion export file", "", std::vector<std::string>{"Hand export files", "*.hexport"});
        }

        // Poll with a zero timeout: pfd's ready() defaults to a 20ms wait that
        // blocks this thread every frame the dialog is open. A zero timeout
        // returns immediately and keeps idle frames.
        if (folder_dialog && folder_dialog->ready(0)) {
            const std::string folder = folder_dialog->result();
            if (!folder.empty()) {
                open_sequence(folder, 0);
            }
            folder_dialog.reset();
        }
        if (export_file_dialog && export_file_dialog->ready(0)) {
            const std::vector<std::string> chosen = export_file_dialog->result();
            if (!chosen.empty()) {
                open_sequence(chosen.front(), 0);
            }
            export_file_dialog.reset();
        }

        // A chosen data folder becomes the Explorer root (one scan here) and is
        // persisted so it reopens next launch.
        if (data_folder_dialog && data_folder_dialog->ready(0)) {
            const std::string folder = data_folder_dialog->result();
            if (!folder.empty()) {
                explorer.set_root(folder);
                settings.data_folder = folder;
            }
            data_folder_dialog.reset();
        }

        // Apply a deferred Explorer open here, before the current frame's GPU
        // buffers are read below, so opening never frees a buffer still in use.
        if (pending_open_folder) {
            open_sequence(*pending_open_folder, 0);
            pending_open_folder.reset();
        }

        // Advance playback at a fixed rate independent of the render frame rate.
        // Suspended while scrubbing so the dragged frame is not fought by auto-advance.
        if (playing && !scrubbing && sequence && sequence->frame_count() > 0) {
            playback_accumulator += dt_seconds;
            const int frame_step = static_cast<int>(playback_accumulator * PLAYBACK_FPS * playback_speed);
            if (frame_step > 0) {
                playback_accumulator -= frame_step / (PLAYBACK_FPS * playback_speed);
                current_frame = (current_frame + frame_step) % sequence->frame_count();
            }
        } else {
            playback_accumulator = 0.0;
        }

        // Decide what to draw and the status line. The current frame's meshes are
        // read from disk and uploaded synchronously here, but only when the frame
        // changes (loaded_frame tracks what current_gpu holds) so a paused frame
        // is not re-read every render tick.
        std::string status;
        if (sequence) {
            // The scene transform is derived from frame 0 so it stays fixed across
            // the whole sequence regardless of which frame playback starts on.
            if (!transform) {
                const Frame frame_zero = sequence->load_frame(0);
                transform = compute_transform(frame_zero);
                depth_reference = reference_depth(frame_zero);
            }
            if (current_frame != loaded_frame) {
                Frame hands = sequence->load_frame(current_frame);
                current_gpu = std::make_unique<FrameGpu>(prepare_frame(hands));
                loaded_frame = current_frame;
            }
            const std::size_t hand_count = sequence->frame_paths(current_frame).size();
            char buffer[128];
            std::snprintf(
                buffer, sizeof(buffer), "frame %d / %d - %zu hand%s", current_frame + 1, sequence->frame_count(), hand_count, hand_count == 1 ? "" : "s"
            );
            status = buffer;
        }

        FrameGpu* frame_ptr = current_gpu.get();
        const bool has_sequence = sequence != nullptr;
        const int frame_count = has_sequence ? sequence->frame_count() : 0;

        // Decode the current frame's modeled keypoint image (cached by path, so
        // this is a no-op unless the frame changed).
        if (has_sequence) {
            const std::vector<std::string>& hands = sequence->frame_paths(current_frame);
            const std::string image_path = hands.empty() ? std::string() : frame_image_path(hands.front());
            if (!image_path.empty()) {
                frame_image.load(image_path);
            } else {
                frame_image.clear();
            }
        } else {
            frame_image.clear();
        }

        // The transport rides on whichever pane was active (the one the mouse last
        // focused). Capture the pane that draws it this frame so we read its state
        // back from the matching window below.
        const int transport_pane = active_pane;
        // Shared player state; only show_transport differs between the two panes.
        const Transport transport_base{
            .has_sequence = has_sequence,
            .current_frame = current_frame,
            .frame_count = frame_count,
            .playing = playing,
            .speed = playback_speed,
            .play_icon = transport_icons.play,
            .pause_icon = transport_icons.pause,
            .speed_icon = transport_icons.speed,
        };
        Transport viewport_transport = transport_base;
        viewport_transport.show_transport = transport_pane == 0;
        Transport image_transport = transport_base;
        image_transport.show_transport = transport_pane == 1;

        const ViewportResult viewport =
            draw_viewport_window(framebuffer, dock_id, imgui_io.Framerate, fps_font, status, settings.show_controls, viewport_transport);
        settings.show_controls = viewport.show_controls;
        viewport_hovered = viewport.hovered;

        const ImageViewResult image_view =
            draw_image_window("Frame View", frame_image.texture(), frame_image.width(), frame_image.height(), dock_id, image_transport);

        // Explorer pane. Lazy: only expanding a folder touches the filesystem.
        const ExplorerResult explorer_result = draw_explorer_window(explorer, dock_id);
        if (explorer_result.choose_root_requested && !data_folder_dialog) {
            data_folder_dialog = std::make_unique<pfd::select_folder>("Select data folder");
        }
        if (explorer_result.open_folder) {
            pending_open_folder = explorer_result.open_folder;
        }

        if (has_sequence) {
            // Only the pane that drew the transport changed the state; read it back.
            const TransportState& echo = transport_pane == 1 ? image_view.transport : viewport.transport;
            current_frame = echo.current_frame;
            playing = echo.playing;
            scrubbing = echo.scrubbing;
            playback_speed = echo.speed;
        } else {
            playing = false;
            scrubbing = false;
        }

        // For the first few frames, re-assert focus on the persisted pane so its
        // tab is restored, ignoring imgui's default focus. Afterward, move the
        // transport to follow this frame's focus; keep the current pane when none
        // is focused so the player never disappears.
        if (restore_focus_frames > 0) {
            restore_focus_frames--;
            ImGui::SetWindowFocus(restore_pane == 1 ? "Frame View" : "Scene");
        } else if (viewport.focused) {
            active_pane = 0;
        } else if (image_view.focused) {
            active_pane = 1;
        }

        ImGui::Render();

        // ── Render scene into the offscreen texture, then the UI ──
        framebuffer.resize(viewport.width, viewport.height);
        render_scene(
            framebuffer,
            *camera,
            SceneRender{
                .frame = frame_ptr,
                .translucent = settings.hand_translucent,
                .transform = transform ? &*transform : nullptr,
                .reference_depth = depth_reference,
                .show_camera_marker = settings.show_camera_marker,
            }
        );

        glViewport(0, 0, win_width, win_height);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(static_cast<GLbitfield>(GL_COLOR_BUFFER_BIT) | static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(app_window);

        // Reveal the window only after the first frame has been rendered and
        // swapped, so it appears with the dock layout already built instead of
        // assembling itself on screen. One-shot: harmless to call once.
        if (!window_shown) {
            window_shown = true;
            SDL_ShowWindow(app_window);
        }
    }

    // ── Persist settings ───────────────────────
    if (sequence) {
        settings.last_folder = sequence->folder();
        settings.last_frame = current_frame;
    } else {
        settings.last_folder.reset();
    }
    settings.active_pane = active_pane;
    settings.playback_speed = playback_speed;
    settings.expanded_folders = explorer.expanded_paths(); // persist which tree folders are open
    if (settings.free_camera) {
        orbit_cam.set_from_free(free_cam);
    }
    orbit_cam.get_state(settings.camera_azimuth, settings.camera_elevation, settings.camera_distance, settings.camera_target);
    if (!settings.window_fullscreen) {
        windowed_geometry = read_current_geometry(app_window);
    }
    store_geometry(settings, windowed_geometry);
    save_config(config_path, settings);

    // Release GPU resources before tearing down the GL context.
    current_gpu.reset();
    sequence.reset();
    frame_image.clear();
    shutdown_renderer();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DestroyContext(gl_context);
    SDL_DestroyWindow(app_window);
    SDL_Quit();
    return 0;
}

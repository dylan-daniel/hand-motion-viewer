#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>

#include <SDL.h>
#include <SDL_opengl.h>

#include <imgui.h>
#include <imgui_impl_opengl3.h>
#include <imgui_impl_sdl2.h>
#include <imgui_internal.h>

#include <portable-file-dialogs.h>

#include "app/config.h"
#include "app/window.h"
#include "data/mesh_sequence.h"
#include "graphics/gl_loader.h"
#include "graphics/image.h"
#include "graphics/rendering.h"
#include "ui/gui.h"

namespace {
    // Frames advanced per second while the sequence is playing.
    constexpr double PLAYBACK_FPS = 30.0;
} // namespace

int main(int, char**) {
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::printf("SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const std::string config_path = default_config_path();
    Config settings = load_config(config_path);

    // Load the shared MANO face topology the .hmesh sequences render against. The
    // binary stores only vertices/joints, so faces come from this one file.
    {
        std::string faces_path = "mano/mano_faces.bin";
        char* base = SDL_GetBasePath();
        if (base != nullptr) {
            faces_path = std::string(base) + "mano/mano_faces.bin";
            SDL_free(base);
        }
        try {
            init_mano_topology(faces_path);
        } catch (const std::exception& error) {
            std::printf("Warning: %s — hand sequences will not render.\n", error.what());
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

    Uint32 window_flags = SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
    if (settings.window_fullscreen) {
        window_flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
    }
    SDL_Window* app_window = SDL_CreateWindow("Hand Motion Viewer", geometry.x, geometry.y, geometry.width, geometry.height, window_flags);
    if (app_window == nullptr) {
        std::printf("SDL_CreateWindow failed: %s\n", SDL_GetError());
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(app_window);
    SDL_GL_MakeCurrent(app_window, gl_context);
    SDL_GL_SetSwapInterval(1); // vsync

    if (!glx::load()) {
        std::printf("Warning: some OpenGL entry points could not be loaded.\n");
    }

    int win_width = 0;
    int win_height = 0;
    SDL_GetWindowSize(app_window, &win_width, &win_height);

    glEnable(GL_DEPTH_TEST);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& imgui_io = ImGui::GetIO();
    imgui_io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    // Persist the dock layout next to the executable (like config.json) instead of
    // imgui's default cwd-relative path, so the saved window arrangement is
    // restored no matter where the app is launched from. The string must outlive
    // the context: imgui keeps the pointer and writes the file on shutdown.
    static std::string imgui_ini_path;
    {
        char* base = SDL_GetBasePath();
        imgui_ini_path = (base != nullptr ? std::string(base) : std::string()) + "imgui.ini";
        if (base != nullptr) {
            SDL_free(base);
        }
    }
    imgui_io.IniFilename = imgui_ini_path.c_str();

    // On the very first run there is no saved layout, so build a default one:
    // the "Frame View" pane on the left half and the "Scene" viewport on the
    // right half. Detected by the absence of the ini file imgui would have
    // written on a previous shutdown.
    bool build_default_layout = !std::filesystem::exists(imgui_ini_path);

    auto [ui_font, fps_font] = load_fonts(imgui_io);
    imgui_io.FontDefault = ui_font;

    ImGui_ImplSDL2_InitForOpenGL(app_window, gl_context);
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
    // True while the user drags the scrubber; suspends auto-advance so the hands
    // don't jitter between the dragged frame and the next. Carried across frames
    // since the transport (which reports it) is drawn after playback is advanced.
    bool scrubbing = false;
    double playback_accumulator = 0.0;

    auto open_sequence = [&](const std::string& folder, int start_frame) {
        auto opened = std::make_unique<MeshSequence>(folder);
        if (opened->frame_count() == 0) {
            std::printf("No frame_*.hmesh files found in %s\n", folder.c_str());
            return;
        }
        sequence = std::move(opened);
        current_gpu.reset();
        loaded_frame = -1;
        transform.reset();
        depth_reference.reset();
        current_frame = std::clamp(start_frame, 0, sequence->frame_count() - 1);
    };

    // Reopen the last mesh sequence folder if present.
    namespace fs = std::filesystem;
    if (settings.last_folder && fs::is_directory(*settings.last_folder)) {
        open_sequence(*settings.last_folder, settings.last_frame);
    }

    std::unique_ptr<pfd::select_folder> folder_dialog;

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

    Uint64 last_ticks = SDL_GetTicks64();
    bool running = true;
    while (running) {
        const Uint64 now_ticks = SDL_GetTicks64();
        const double dt_seconds = static_cast<double>(now_ticks - last_ticks) / 1000.0;
        last_ticks = now_ticks;

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ImGui_ImplSDL2_ProcessEvent(&event);

            if (event.type == SDL_QUIT) {
                running = false;
            } else if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                win_width = event.window.data1;
                win_height = event.window.data2;
            } else if (event.type == SDL_KEYDOWN) {
                const SDL_Keycode key = event.key.keysym.sym;
                if (key == SDLK_ESCAPE) {
                    running = false;
                } else if (key == SDLK_r) {
                    camera->reset();
                } else if (key == SDLK_h) {
                    settings.hand_translucent = !settings.hand_translucent;
                } else if (key == SDLK_SPACE && event.key.repeat == 0 && !imgui_io.WantTextInput) {
                    if (sequence && sequence->frame_count() > 0) {
                        playing = !playing;
                    }
                } else if (sequence && !playing && (key == SDLK_LEFT || key == SDLK_RIGHT || key == SDLK_HOME || key == SDLK_END)) {
                    const int last = sequence->frame_count() - 1;
                    if (key == SDLK_LEFT) {
                        current_frame = std::max(0, current_frame - 1);
                    } else if (key == SDLK_RIGHT) {
                        current_frame = std::min(last, current_frame + 1);
                    } else if (key == SDLK_HOME) {
                        current_frame = 0;
                    } else {
                        current_frame = last;
                    }
                } else if (key == SDLK_F11) {
                    if (!settings.window_fullscreen) {
                        windowed_geometry = read_current_geometry(app_window);
                        SDL_SetWindowFullscreen(app_window, SDL_WINDOW_FULLSCREEN_DESKTOP);
                    } else {
                        SDL_SetWindowFullscreen(app_window, 0);
                    }
                    settings.window_fullscreen = !settings.window_fullscreen;
                    SDL_GetWindowSize(app_window, &win_width, &win_height);
                }
            } else if (event.type == SDL_MOUSEBUTTONDOWN && viewport_hovered) {
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    if (SDL_GetModState() & KMOD_SHIFT) {
                        panning = true;
                    } else {
                        orbiting = true;
                    }
                    set_relative_mouse(true);
                }
            } else if (event.type == SDL_MOUSEBUTTONUP) {
                if (event.button.button == SDL_BUTTON_RIGHT) {
                    orbiting = false;
                    panning = false;
                    set_relative_mouse(false);
                }
            } else if (event.type == SDL_MOUSEWHEEL && viewport_hovered) {
                if (event.wheel.y > 0) {
                    camera->zoom(1.0f);
                } else if (event.wheel.y < 0) {
                    camera->zoom(-1.0f);
                }
            } else if (event.type == SDL_MOUSEMOTION) {
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
            const Uint8* keys = SDL_GetKeyboardState(nullptr);
            const float forward = (keys[SDL_SCANCODE_W] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_S] ? 1.0f : 0.0f);
            const float right = (keys[SDL_SCANCODE_D] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_A] ? 1.0f : 0.0f);
            const float up = (keys[SDL_SCANCODE_E] ? 1.0f : 0.0f) - (keys[SDL_SCANCODE_Q] ? 1.0f : 0.0f);
            if (forward != 0.0f || right != 0.0f || up != 0.0f) {
                camera->move(forward, right, up, static_cast<float>(dt_seconds));
            }
        }

        // ── Build the imgui frame ──────────────
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        const bool was_free_camera = settings.free_camera;
        const MenuResult menu = draw_menu_bar(settings.hand_translucent, settings.show_camera_marker, settings.free_camera);
        settings.hand_translucent = menu.hand_translucent;
        settings.show_camera_marker = menu.show_camera_marker;
        settings.free_camera = menu.free_camera;
        if (settings.free_camera != was_free_camera) {
            if (settings.free_camera) {
                free_cam.set_from_orbit(orbit_cam);
                camera = &free_cam;
            } else {
                orbit_cam.set_from_free(free_cam);
                camera = &orbit_cam;
            }
        }

        const ImGuiID dock_id = ImGui::DockSpaceOverViewport();

        if (build_default_layout) {
            build_default_layout = false;
            ImGui::DockBuilderRemoveNode(dock_id);
            ImGui::DockBuilderAddNode(dock_id, ImGuiDockNodeFlags_DockSpace);
            ImGui::DockBuilderSetNodeSize(dock_id, ImGui::GetMainViewport()->Size);

            ImGuiID dock_left = 0;
            ImGuiID dock_right = 0;
            ImGui::DockBuilderSplitNode(dock_id, ImGuiDir_Left, 0.5f, &dock_left, &dock_right);

            ImGui::DockBuilderDockWindow("Frame View", dock_left);
            ImGui::DockBuilderDockWindow("Scene", dock_right);
            ImGui::DockBuilderFinish(dock_id);
        }

        // Apply the open-folder request before choosing what to draw, since it
        // frees GPU buffers the drawable below must reflect.
        if (menu.folder_requested && !folder_dialog) {
            folder_dialog = std::make_unique<pfd::select_folder>("Select mesh sequence folder");
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

        // Advance playback at a fixed rate independent of the render frame rate.
        // Suspended while scrubbing so the dragged frame is not fought by auto-advance.
        if (playing && !scrubbing && sequence && sequence->frame_count() > 0) {
            playback_accumulator += dt_seconds;
            const int frame_step = static_cast<int>(playback_accumulator * PLAYBACK_FPS);
            if (frame_step > 0) {
                playback_accumulator -= frame_step / PLAYBACK_FPS;
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
        const ViewportResult viewport = draw_viewport_window(
            framebuffer,
            dock_id,
            imgui_io.Framerate,
            fps_font,
            status,
            settings.show_controls,
            has_sequence,
            current_frame,
            frame_count,
            playing,
            transport_pane == 0
        );
        settings.show_controls = viewport.show_controls;
        viewport_hovered = viewport.hovered;

        const ImageViewResult image_view = draw_image_window(
            "Frame View",
            frame_image.texture(),
            frame_image.width(),
            frame_image.height(),
            dock_id,
            has_sequence,
            current_frame,
            frame_count,
            playing,
            transport_pane == 1
        );

        if (has_sequence) {
            // Only the pane that drew the transport changed the state; read it back.
            if (transport_pane == 1) {
                current_frame = image_view.current_frame;
                playing = image_view.playing;
                scrubbing = image_view.scrubbing;
            } else {
                current_frame = viewport.current_frame;
                playing = viewport.playing;
                scrubbing = viewport.scrubbing;
            }
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
            framebuffer, *camera, frame_ptr, settings.hand_translucent, transform ? &*transform : nullptr, depth_reference, settings.show_camera_marker
        );

        glViewport(0, 0, win_width, win_height);
        glClearColor(0.08f, 0.08f, 0.10f, 1.0f);
        glClear(static_cast<GLbitfield>(GL_COLOR_BUFFER_BIT) | static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT));
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        SDL_GL_SwapWindow(app_window);
    }

    // ── Persist settings ───────────────────────
    if (sequence) {
        settings.last_folder = sequence->folder();
        settings.last_frame = current_frame;
    } else {
        settings.last_folder.reset();
    }
    settings.active_pane = active_pane;
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
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();
    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(app_window);
    SDL_Quit();
    return 0;
}

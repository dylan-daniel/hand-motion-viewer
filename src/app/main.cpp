#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

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
#include "data/classification.h"
#include "data/kalman.h" // TEMP: KalmanParams for the debug smoothing slider
#include "data/mesh_sequence.h"
#include "graphics/gl_loader.h"
#include "graphics/image.h"
#include "graphics/rendering.h"
#include "ui/explorer.h"
#include "ui/gui.h"

namespace {
    // Frames advanced per second while the sequence is playing.
    constexpr double PLAYBACK_FPS = 30.0;

    // The per-hand model matrix the renderer applies, replicated here so picking
    // tests rays against the same world-space hand positions the scene shows. The
    // scene fit (scale-then-translate) is applied, then the per-hand depth-snap.
    glm::mat4 hand_model_matrix(const Transform* transform, float scale) {
        glm::mat4 model(1.0f);
        if (transform != nullptr) {
            model = glm::scale(model, glm::vec3(transform->scale));
            model = glm::translate(model, transform->translate);
        }
        if (scale != 1.0f) {
            model = glm::scale(model, glm::vec3(scale));
        }
        return model;
    }

    // Möller–Trumbore ray/triangle test. Writes the hit distance to ``out_t`` and
    // returns true only for a forward (t > 0) intersection.
    bool ray_triangle(const glm::vec3& origin, const glm::vec3& dir, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, float& out_t) {
        constexpr float epsilon = 1e-7f;
        const glm::vec3 edge1 = v1 - v0;
        const glm::vec3 edge2 = v2 - v0;
        const glm::vec3 pvec = glm::cross(dir, edge2);
        const float det = glm::dot(edge1, pvec);
        if (std::fabs(det) < epsilon) {
            return false;
        }
        const float inv_det = 1.0f / det;
        const glm::vec3 tvec = origin - v0;
        const float u = glm::dot(tvec, pvec) * inv_det;
        if (u < 0.0f || u > 1.0f) {
            return false;
        }
        const glm::vec3 qvec = glm::cross(tvec, edge1);
        const float v = glm::dot(dir, qvec) * inv_det;
        if (v < 0.0f || u + v > 1.0f) {
            return false;
        }
        const float t = glm::dot(edge2, qvec) * inv_det;
        if (t <= epsilon) {
            return false;
        }
        out_t = t;
        return true;
    }

    // Index of the hand whose surface the ray hits nearest the camera, or -1 if it
    // misses every hand. Mirrors the renderer's depth-snap so the picked hand is the
    // one actually drawn under the cursor.
    int pick_hand(const Frame& hands, const Transform* transform, std::optional<float> reference_depth, const glm::vec3& origin, const glm::vec3& dir) {
        int best = -1;
        float best_t = FLT_MAX;
        for (std::size_t index = 0; index < hands.size(); ++index) {
            const HandData& hand = hands[index];
            if (hand.verts.empty()) {
                continue;
            }
            double sum = 0.0;
            for (const glm::vec3& position : hand.verts) {
                sum += position.z;
            }
            const float depth = static_cast<float>(sum / hand.verts.size());
            const float scale = (reference_depth && depth != 0.0f) ? (*reference_depth / depth) : 1.0f;
            const glm::mat4 model = hand_model_matrix(transform, scale);
            std::vector<glm::vec3> world;
            world.reserve(hand.verts.size());
            for (const glm::vec3& position : hand.verts) {
                world.push_back(glm::vec3(model * glm::vec4(position, 1.0f)));
            }
            for (const glm::ivec3& face : mano_faces(hand.is_right)) {
                float t = 0.0f;
                if (ray_triangle(
                        origin,
                        dir,
                        world[static_cast<std::size_t>(face.x)],
                        world[static_cast<std::size_t>(face.y)],
                        world[static_cast<std::size_t>(face.z)],
                        t
                    ) &&
                    t < best_t) {
                    best_t = t;
                    best = static_cast<int>(index);
                }
            }
        }
        return best;
    }
} // namespace

int main(int, char**) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        std::printf("SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    const std::string config_path = default_config_path();
    Config settings = load_config(config_path);

    // Load the shared MANO face topology the .hmesh sequences render against. The
    // binary stores only vertices/joints, so faces come from this one file.
    {
        std::string faces_path = "mano/mano_faces.bin";
        const char* base = SDL_GetBasePath();
        if (base != nullptr) {
            faces_path = std::string(base) + "mano/mano_faces.bin";
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
    // TEMP: translucent overlay of the raw, unsmoothed hands so the smoothing can
    // be compared against the real per-frame positions. Built from disk alongside
    // the smoothed frame; remove once the smoothing is dialled in.
    std::unique_ptr<FrameGpu> raw_overlay_gpu;
    // TEMP: toggles the red raw-hands overlay from the Kalman pane (on by default).
    bool show_raw_overlay = true;
    // Adult (non-baby) hands, prepared in yellow and drawn translucent. Built from
    // the same frame as current_gpu, split out by the per-video labels. When
    // hide_adult_hands is set they are dropped instead of drawn.
    std::unique_ptr<FrameGpu> adult_gpu;
    bool hide_adult_hands = false;
    // Baby/adult labels for the loaded sequence (from data/labels_per_video). Lives
    // as long as the sequence so each frame's hands can be split as they load.
    HandClassification classification;
    // CPU-side hands of the loaded frame, kept so left-click picking can raycast
    // against the same geometry the GPU is drawing (the GPU buffers can't be read
    // back cheaply). Rebuilt in lockstep with current_gpu.
    Frame current_hands;
    int loaded_frame = -1;
    int frame_duplicate_count = 0; // duplicate hands in the loaded frame (for the Tracking pane)
    // Manual relabeling: the id the next picked hand is assigned, and the last
    // save's status message shown in the Manual Tracking pane.
    int manual_active_id = 0;
    bool manual_propagate = true; // relabel the rest of a track forward when assigning an id
    std::string manual_save_status;
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
    // TEMP: live Kalman tuning, seeded from the persisted config so a tuned value
    // survives a restart. The debug slider window below edits these and re-smooths
    // the loaded sequence on change; they are written back to config on shutdown.
    KalmanParams kalman_params;
    kalman_params.process_noise = settings.kalman_process_noise;
    kalman_params.measurement_noise = settings.kalman_measurement_noise;

    // Folder that labels each tracked hand as a baby hand or not. TEMP: hardcoded to
    // the moved data location; restore the working-dir/exe-dir search once the data
    // folder is settled.
    const std::string labels_dir = "E:/archive/hamer_data/classification_v2";

    auto open_sequence = [&](const std::string& folder, int start_frame) {
        // Load the per-take baby/adult labels CSV. The data layout is
        // data/<subject>/<take> (e.g. data/AE45/T1_BabyView) and the CSV is named
        // classification_<subject>_<take>.csv (classification_AE45_T1_BabyView.csv).
        // Stored in the sequence-scoped ``classification`` so frames can be split as
        // they load.
        classification = HandClassification{};
        if (!labels_dir.empty()) {
            std::filesystem::path folder_path(folder);
            if (folder_path.filename().empty()) {
                folder_path = folder_path.parent_path(); // tolerate a trailing separator
            }
            const std::string take = folder_path.filename().string();
            const std::string subject = folder_path.parent_path().filename().string();
            const std::string csv_name = subject.empty() ? "classification_" + take + ".csv" : "classification_" + subject + "_" + take + ".csv";
            const std::filesystem::path csv = std::filesystem::path(labels_dir) / csv_name;
            std::error_code error;
            if (std::filesystem::exists(csv, error)) {
                classification = HandClassification::load(csv.string());
            }
        }

        // Smooth the sequence on open: the source motion is recovered per 30fps
        // frame and is visibly jittery, so a forward-backward Kalman smoother
        // removes the noise while keeping the real motion (see kalman.h). The
        // smoother groups hands by their tracking-CSV id, so apply the remembered
        // tracking source before the smoothing reads the ids.
        auto opened = std::make_unique<MeshSequence>(folder, /*smooth=*/true);
        // Carry the remembered tracking source into the new sequence when it has
        // that source; otherwise the sequence keeps its own default. Under a smoothed
        // sequence this re-reads and re-smooths against the chosen source's ids.
        opened->prefer_tracking_source({settings.tracking_version, settings.tracking_linked});
        const bool has_frames = opened->frame_count() > 0;
        if (!has_frames) {
            std::printf("No frame_*.hmesh files found in %s\n", folder.c_str());
        }
        // Reset the viewer to the requested folder either way: a folder with no
        // frames clears the scene rather than leaving the previous sequence on
        // screen (so opening from the Explorer always resets, like the menu does).
        sequence = has_frames ? std::move(opened) : nullptr;
        if (sequence) {
            // The constructor smooths with the defaults; re-smooth with the saved
            // tuning so a reopened sequence reflects the persisted params.
            sequence->resmooth(kalman_params);
        }
        current_gpu.reset();
        raw_overlay_gpu.reset();
        adult_gpu.reset();
        current_hands.clear();
        manual_save_status.clear();
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

    // Two independent native folder pickers: one for opening a mesh sequence (menu
    // / Explorer click), one for choosing the Explorer's data-folder root.
    std::unique_ptr<pfd::select_folder> folder_dialog;
    std::unique_ptr<pfd::select_folder> data_folder_dialog;
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
                } else if (key == SDLK_Z && (SDL_GetModState() & SDL_KMOD_CTRL) && !imgui_io.WantTextInput) {
                    // Ctrl+Z undoes the last manual id assignment. Not while a text
                    // field is focused, where Ctrl+Z is the field's own text undo.
                    if (sequence && sequence->undo()) {
                        loaded_frame = -1; // reload so the reverted colours show
                    }
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
            // Tracking and Manual Tracking share the left column, tabbed behind Explorer.
            ImGui::DockBuilderDockWindow("Tracking", dock_explorer);
            ImGui::DockBuilderDockWindow("Manual Tracking", dock_explorer);
            ImGui::DockBuilderDockWindow("Scene", dock_scene);
            ImGui::DockBuilderDockWindow("Frame View", dock_frame);
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
                transform = compute_transform(*sequence);
                depth_reference = reference_depth(sequence->load_frame(0));
            }
            if (current_frame != loaded_frame) {
                Frame hands = sequence->load_frame(current_frame);
                frame_duplicate_count = 0;
                for (const HandData& hand : hands) {
                    if (hand.is_duplicate) {
                        ++frame_duplicate_count;
                    }
                }
                // Split the frame's hands into baby and adult by the per-track
                // labels (keyed by each hand's tracking id). Baby hands render
                // normally (coloured by track id); adult hands get a yellow surface,
                // drawn translucent (or dropped, per the toggle). Unlabeled hands
                // default to baby.
                //
                // Drop rarely-seen tracks first: the transformer occasionally emits a
                // fleeting duplicate hand that lives for only a handful of frames, so a
                // track classified in fewer than this many frames (n_votes) is treated
                // as noise and skipped. Tracks with no vote count (-1, absent from the
                // CSV or unlabeled) are kept so untracked data is never hidden.
                constexpr int min_track_frames = 15;
                Frame baby_hands;
                Frame adult_hands;
                for (const HandData& hand : hands) {
                    const int votes = classification.track_votes(hand.track_id);
                    if (votes >= 0 && votes < min_track_frames) {
                        continue;
                    }
                    (classification.is_baby(hand.track_id) ? baby_hands : adult_hands).push_back(hand);
                }
                current_gpu = std::make_unique<FrameGpu>(prepare_frame(baby_hands));
                const glm::vec4 adult_color(0.9f, 0.85f, 0.2f, 1.0f);
                adult_gpu = adult_hands.empty() ? nullptr : std::make_unique<FrameGpu>(prepare_frame(adult_hands, adult_color));

                // TEMP: prepare the raw frame in a contrasting red for the overlay.
                const Frame raw_hands = sequence->load_raw_frame(current_frame);
                const glm::vec4 raw_color(0.9f, 0.25f, 0.25f, 1.0f);
                raw_overlay_gpu = std::make_unique<FrameGpu>(prepare_frame(raw_hands, raw_color));

                // Keep the full frame (both baby and adult) for left-click picking,
                // which raycasts against every hand and keys overrides by hand.slot.
                current_hands = std::move(hands);
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

        // TEMP: Kalman tuning window. Only the ratio of the two noises sets the
        // amount of smoothing; both are log sliders since they span orders of
        // magnitude. Changing either re-smooths the loaded sequence (from the
        // cached raw frames, no disk) and invalidates the uploaded frame so the
        // new smoothing shows immediately. Remove with the rest of the temp UI.
        {
            ImGui::Begin("Kalman (temp)");
            bool params_changed = false;
            params_changed |= ImGui::SliderFloat("process noise", &kalman_params.process_noise, 1e-4f, 10.0f, "%.5f", ImGuiSliderFlags_Logarithmic);
            params_changed |= ImGui::SliderFloat("measurement noise", &kalman_params.measurement_noise, 1e-3f, 100.0f, "%.4f", ImGuiSliderFlags_Logarithmic);
            ImGui::TextDisabled("higher process / lower measurement = less smoothing");
            if (params_changed && sequence) {
                sequence->resmooth(kalman_params);
                loaded_frame = -1; // force the on-screen frame to rebuild from the re-smoothed data
            }
            ImGui::Separator();
            ImGui::Checkbox("show real hands overlay (red)", &show_raw_overlay);
            ImGui::End();
        }

        // TEMP: adult (non-baby) hand display. Unchecked draws them as a yellow
        // translucent overlay; checked hides them entirely. Driven by the per-video
        // labels (data/labels_per_video).
        {
            ImGui::Begin("Classification (temp)");
            ImGui::Checkbox("hide adult hands", &hide_adult_hands);
            ImGui::TextDisabled("unchecked: adult hands shown yellow + transparent");
            ImGui::End();
        }

        // Tracking pane: pick the tracking-CSV version, toggle hiding of duplicate
        // hands, and show this frame's duplicate count.
        const TrackingResult tracking = draw_tracking_window(
            dock_id,
            TrackingState{
                .hide_duplicates = settings.hide_duplicate_hands,
                .duplicate_count = has_sequence ? frame_duplicate_count : 0,
                .sources = has_sequence ? sequence->tracking_sources() : std::vector<TrackingSource>{},
                .selected_source = has_sequence ? sequence->tracking_source() : TrackingSource{},
            }
        );
        settings.hide_duplicate_hands = tracking.state.hide_duplicates;

        // Manual Tracking pane: choose the active colour id (swatches / number keys)
        // and save the relabeled CSV. Picking a hand in the Scene is handled below.
        const ManualTrackingResult manual = draw_manual_tracking_window(
            dock_id,
            ManualTrackingState{
                .has_sequence = has_sequence,
                .present_ids = has_sequence ? sequence->present_color_ids() : std::vector<int>{},
                .active_id = manual_active_id,
                .propagate = manual_propagate,
                .override_count = has_sequence ? sequence->override_count() : 0,
                .save_status = manual_save_status,
            }
        );
        manual_active_id = manual.active_id;
        manual_propagate = manual.propagate;
        if (manual.save_requested && has_sequence) {
            const std::string written = sequence->save_truth_csv();
            manual_save_status =
                written.empty() ? std::string("Save failed.") : "Saved " + std::to_string(sequence->override_count()) + " overrides to " + written;
        }

        // Hover a hand in the Scene to see its id (tooltip); left-click to assign it
        // the active id. Both use one ray cast from the cursor through the same
        // projection the scene is rendered with, hitting the nearest hand.
        if (has_sequence && viewport.hovered && !orbiting && !panning) {
            const ImVec2 mouse = ImGui::GetMousePos();
            const float local_x = mouse.x - viewport.image_pos.x;
            const float local_y = mouse.y - viewport.image_pos.y;
            const float view_width = static_cast<float>(viewport.width);
            const float view_height = static_cast<float>(viewport.height);
            if (view_width > 0.0f && view_height > 0.0f) {
                const float ndc_x = 2.0f * local_x / view_width - 1.0f;
                const float ndc_y = 1.0f - 2.0f * local_y / view_height;
                const float aspect = view_width / view_height;
                const glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
                const glm::mat4 inverse_vp = glm::inverse(projection * camera->view_matrix());
                glm::vec4 near_point = inverse_vp * glm::vec4(ndc_x, ndc_y, -1.0f, 1.0f);
                glm::vec4 far_point = inverse_vp * glm::vec4(ndc_x, ndc_y, 1.0f, 1.0f);
                near_point /= near_point.w;
                far_point /= far_point.w;
                const glm::vec3 ray_origin(near_point);
                const glm::vec3 ray_dir = glm::normalize(glm::vec3(far_point - near_point));
                const int picked = pick_hand(current_hands, transform ? &*transform : nullptr, depth_reference, ray_origin, ray_dir);
                if (picked >= 0) {
                    const HandData& hand = current_hands[static_cast<std::size_t>(picked)];
                    // Tooltip: the hovered hand's current id (its colour) and which
                    // hand it is, so the right active id can be chosen before clicking.
                    if (hand.track_id >= 0) {
                        ImGui::SetTooltip("id %d  (%s hand)", hand.track_id, hand.is_right ? "right" : "left");
                    } else {
                        ImGui::SetTooltip("unlabeled  (%s hand)", hand.is_right ? "right" : "left");
                    }
                    // Middle-click eyedrops the hovered hand's id into the active id,
                    // so an existing colour can be matched without hunting the swatches.
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Middle) && hand.track_id >= 0) {
                        manual_active_id = hand.track_id;
                    }
                    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                        // Overrides key by the on-disk frame number (what the CSV uses),
                        // not the 0-based playback index, so load_frame and the saved CSV
                        // both find them.
                        const int frame_number = sequence->frame_number(current_frame);
                        // Snapshot before the edit so one Ctrl+Z reverts this whole
                        // assignment (the single hand plus any forward propagation).
                        sequence->push_undo_state();
                        sequence->set_override(frame_number, hand.slot, manual_active_id);
                        // If enabled and the hand already had an id, carry the relabel
                        // to the rest of that track from here forward.
                        if (manual_propagate) {
                            sequence->propagate_id(frame_number, hand.track_id, manual_active_id);
                        }
                        loaded_frame = -1; // force a reload so the new colour shows
                    }
                }
            }
        }

        // Resolve the target tracking source from the pane's combo and, when the
        // Scene viewport is focused, the Up/Down arrows — Up steps to the next entry
        // in the source list, Down to the previous, clamped to the ones that exist.
        TrackingSource target_source = tracking.state.selected_source;
        if (has_sequence && viewport.focused) {
            const std::vector<TrackingSource>& sources = sequence->tracking_sources();
            int step = 0;
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) {
                step = 1;
            } else if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) {
                step = -1;
            }
            if (step != 0) {
                const auto current = std::find(sources.begin(), sources.end(), sequence->tracking_source());
                if (current != sources.end()) {
                    const auto last = static_cast<std::ptrdiff_t>(sources.size()) - 1;
                    const auto next = std::clamp((current - sources.begin()) + step, static_cast<std::ptrdiff_t>(0), last);
                    target_source = sources[static_cast<std::size_t>(next)];
                }
            }
        }

        // Switching source re-parses the colour ids; force a reload so the scene
        // re-renders with the new colours and duplicate flags.
        if (has_sequence && !(target_source == sequence->tracking_source())) {
            sequence->set_tracking_source(target_source);
            // Remember the choice so it carries to the next sequence and next run.
            settings.tracking_version = target_source.version;
            settings.tracking_linked = target_source.linked;
            loaded_frame = -1;
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
                .raw_overlay = show_raw_overlay ? raw_overlay_gpu.get() : nullptr, // TEMP: raw-position overlay
                .adult_overlay = hide_adult_hands ? nullptr : adult_gpu.get(),     // adult hands shown yellow/translucent unless hidden
                // Pulse the duplicate-hand glow so it reads as a soft throb rather
                // than a flat tint; derived from the wall clock so it animates even
                // when playback is paused.
                .duplicate_glow = 0.35f + 0.25f * static_cast<float>(std::sin(static_cast<double>(now_ticks) / 1000.0 * 5.0)),
                .hide_duplicates = settings.hide_duplicate_hands,
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
    // TEMP: persist the tuned Kalman noise params so they reload next launch.
    settings.kalman_process_noise = kalman_params.process_noise;
    settings.kalman_measurement_noise = kalman_params.measurement_noise;
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

#include <algorithm>
#include <cmath>
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
#include "data/k_metric_table.h"
#include "data/mano_model.h"
#include "data/mesh_sequence.h"
#include "data/production_log.h"
#include "data/object_sequence.h"
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

    // Load the MANO model and shared face topology that the CSV sequences are
    // reconstructed and rendered with. The CSV stores only generative parameters,
    // so the model weights regenerate the vertices and the faces complete them.
    {
        const char* base = SDL_GetBasePath();
        const std::string prefix = base != nullptr ? std::string(base) : std::string();
        try {
            init_mano_model(prefix + "assets/mano/mano_model.bin");
            init_mano_topology(prefix + "assets/mano/mano_faces.bin");
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

    // Shared unit-cube/unit-sphere meshes for the tracked object, built once;
    // which one is drawn is picked per frame by object_sequence's shape() (see
    // data/object_sequence.h). The per-frame pose is applied as a model matrix
    // at draw time, so this geometry never changes. Held by pointer so their GL
    // buffers can be freed before the context is torn down.
    auto cube_mesh = std::make_unique<GpuMesh>(build_unit_cube(glm::vec4(0.95f, 0.55f, 0.15f, 1.0f)));
    auto sphere_mesh = std::make_unique<GpuMesh>(build_unit_sphere(glm::vec4(0.95f, 0.55f, 0.15f, 1.0f)));

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
    // The trial's tracked-object (cube/sphere) pose sequence, fit from the SAM3
    // + DA3 caches under the SAM3/DA3 folders (see data/object_sequence.h).
    std::unique_ptr<TrackedObjectSequence> object_sequence;
    // Whole-trial smoothed per-hand wrist depth (see
    // MeshSequence::precompute_smoothed_wrist_depths), or null when smoothing
    // is off / not yet computed — load_frame then resolves depth live,
    // per-frame, as it always did before smoothing existed.
    std::unique_ptr<std::vector<std::vector<float>>> smoothed_hand_depths;
    // GPU buffers for the frame currently on screen, rebuilt from disk whenever
    // the frame changes. loaded_frame tracks which frame current_gpu holds (-1 =
    // none) so we only re-read and re-upload when the frame actually advances.
    std::unique_ptr<FrameGpu> current_gpu;
    int loaded_frame = -1;
    int current_frame = 0;
    // Whether the currently-open trial's auto-detected focal length (see
    // open_sequence) matches a focal length we actually have a calibrated
    // k_metric for. Purely a display hint for the Camera pane — true (no
    // warning) when a trial hasn't set it, e.g. before any CSV is opened.
    bool intrinsics_calibrated = true;
    // Whether the currently-open trial's object_shape/object_label/
    // object_size_m came from an auto-detected production-log match (see
    // open_sequence) rather than a manual/stale Object-pane setting. Purely
    // a display hint, like intrinsics_calibrated above.
    bool object_auto_detected = true;
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

    // The open trial's per-trial SAM3/DA3 ``arrays`` directory under a
    // configured root folder, e.g. ``<sam3_folder>/<subject>/<trial>/arrays``
    // — shared by reload_object (the tracked object) and the hand-depth
    // config built before every load_frame call. Empty when no sequence is
    // open, no root folder is configured, or the per-trial directory doesn't
    // exist on disk.
    auto trial_arrays_dir = [&](const std::optional<std::string>& folder) -> std::string {
        if (!sequence || !folder) {
            return {};
        }
        const std::filesystem::path dir = std::filesystem::path(*folder) / sequence->subject() / sequence->trial() / "arrays";
        return std::filesystem::is_directory(dir) ? dir.string() : std::string();
    };

    // (Re)fit the tracked object for the open sequence from the SAM3/DA3
    // folders. Missing folders, a missing per-trial ``arrays`` directory, or an
    // object_shape of None just leaves no object (the hands still render).
    // This backprojects+fits the WHOLE trial up front (see
    // data/object_sequence.h for why it can't be done frame-by-frame), so it
    // can take noticeably longer than opening the hand CSV for a long trial.
    auto reload_object = [&]() {
        object_sequence.reset();
        const std::string sam3_dir = trial_arrays_dir(settings.sam3_folder);
        const std::string da3_dir = trial_arrays_dir(settings.da3_folder);
        if (!sequence || sam3_dir.empty() || da3_dir.empty() || settings.object_shape == 0) {
            return;
        }
        std::vector<int> frame_numbers;
        frame_numbers.reserve(static_cast<std::size_t>(sequence->frame_count()));
        for (int index = 0; index < sequence->frame_count(); ++index) {
            frame_numbers.push_back(sequence->frame_number(index));
        }
        const CameraIntrinsics intrinsics{
            settings.intrinsics_fx, settings.intrinsics_fy, settings.intrinsics_cx, settings.intrinsics_cy, settings.intrinsics_k_metric
        };
        const ObjectShape shape = settings.object_shape == 1 ? ObjectShape::Cube : ObjectShape::Sphere;
        const float smooth_sigma_frames = settings.smoothing_enabled ? 3.0f : 0.0f;
        object_sequence = std::make_unique<TrackedObjectSequence>(
            sam3_dir, da3_dir, settings.object_label, shape, settings.object_size_m, intrinsics, frame_numbers, smooth_sigma_frames
        );
    };

    // (Re)compute the whole-trial smoothed hand wrist depths (see
    // MeshSequence::precompute_smoothed_wrist_depths). Off when smoothing is
    // disabled — load_frame then falls back to its original live per-frame
    // resolution. Same up-front cost profile as reload_object: worth calling
    // together whenever either might be stale.
    auto reload_hand_smoothing = [&]() {
        smoothed_hand_depths.reset();
        if (!sequence || !settings.smoothing_enabled) {
            return;
        }
        const CameraIntrinsics intrinsics{
            settings.intrinsics_fx, settings.intrinsics_fy, settings.intrinsics_cx, settings.intrinsics_cy, settings.intrinsics_k_metric
        };
        const HandDepthConfig depth_config{
            settings.hand_depth_source == 1 ? HandDepthSource::Hamer : HandDepthSource::Da3, trial_arrays_dir(settings.sam3_folder),
            trial_arrays_dir(settings.da3_folder)
        };
        smoothed_hand_depths =
            std::make_unique<std::vector<std::vector<float>>>(sequence->precompute_smoothed_wrist_depths(intrinsics, depth_config, 3.0f));
    };

    auto open_sequence = [&](const std::string& csv_path, int start_frame) {
        std::unique_ptr<MeshSequence> opened;
        const HandClassificationConfig classification_config{settings.tracking_folder.value_or(std::string()), settings.baby_hand_idx_folder.value_or(std::string())};
        try {
            opened = std::make_unique<MeshSequence>(csv_path, classification_config);
        } catch (const std::exception& error) {
            std::printf("%s\n", error.what());
        }
        const bool has_frames = opened && opened->frame_count() > 0;
        if (!has_frames) {
            std::printf("No hands reconstructed from %s\n", csv_path.c_str());
        }
        // Reset the viewer to the requested CSV either way: a CSV with no frames
        // clears the scene rather than leaving the previous sequence on screen (so
        // opening from the Explorer always resets, like the menu does).
        sequence = has_frames ? std::move(opened) : nullptr;
        current_gpu.reset();
        loaded_frame = -1;
        current_frame = has_frames ? std::clamp(start_frame, 0, sequence->frame_count() - 1) : 0;

        // Auto-detect this trial's own camera calibration (focal length, and
        // image size for the principal point) straight from the CSV, rather
        // than leaving whatever the Camera pane had from a previously-open
        // trial — different trials/exports can use different focal length
        // overrides (see hamer_vggt_focal_cache). k_metric is only auto-filled
        // when this focal length is one we actually have a calibrated value
        // for (known_k_metric_for_focal_length); otherwise it's left as-is
        // and flagged unverified in the Camera pane, since k_metric is not
        // portable across focal lengths.
        if (has_frames && sequence->focal_length_px() > 0.0f) {
            settings.intrinsics_fx = sequence->focal_length_px();
            settings.intrinsics_fy = sequence->focal_length_px();
            settings.intrinsics_cx = sequence->image_width() * 0.5f;
            settings.intrinsics_cy = sequence->image_height() * 0.5f;
            // Prefer the per-(subject, trial) calibrated k_metric table (keyed
            // by the actual trial, so two trials that happen to share a
            // VGGT-estimated focal length still get their own value) over the
            // hardcoded fx-based table.
            const KMetricTable k_metric_table(settings.k_metric_csv.value_or(std::string()));
            if (const std::optional<float> by_trial = k_metric_table.lookup(sequence->subject(), sequence->trial())) {
                settings.intrinsics_k_metric = *by_trial;
                intrinsics_calibrated = true;
            } else if (const std::optional<float> by_fx = known_k_metric_for_focal_length(settings.intrinsics_fx)) {
                settings.intrinsics_k_metric = *by_fx;
                intrinsics_calibrated = true;
            } else {
                intrinsics_calibrated = false;
            }
        } else {
            intrinsics_calibrated = true; // no CSV calibration info to check against
        }

        // Auto-detect the tracked object's shape/label/size from the study's
        // production log (data/production_log.h), keyed by this trial's own
        // subject/trial rather than left over from whatever trial was open
        // before. A trial not in the log (or no production log configured)
        // just keeps the Object pane's last manual setting, flagged
        // unverified rather than guessed.
        if (has_frames) {
            const ProductionLog production_log(settings.production_csv.value_or(std::string()));
            if (const std::optional<ObjectSizeInfo> info = production_log.lookup(sequence->subject(), sequence->trial())) {
                settings.object_shape = info->shape;
                settings.object_label = info->object_label;
                settings.object_size_m = info->size_m;
                object_auto_detected = true;
            } else {
                object_auto_detected = false;
            }
        } else {
            object_auto_detected = true; // no CSV to check against
        }

        reload_object();
        reload_hand_smoothing();
    };

    // Reopen the last trial CSV if present.
    namespace fs = std::filesystem;
    if (settings.last_folder && fs::is_regular_file(*settings.last_folder)) {
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

    // Native pickers: a file picker for opening a trial CSV (menu), a folder picker
    // for the Explorer's data-folder root, and folder pickers for the images/
    // SAM3/DA3 roots.
    std::unique_ptr<pfd::open_file> csv_dialog;
    std::unique_ptr<pfd::select_folder> data_folder_dialog;
    std::unique_ptr<pfd::select_folder> images_folder_dialog;
    std::unique_ptr<pfd::select_folder> sam3_folder_dialog;
    std::unique_ptr<pfd::select_folder> da3_folder_dialog;
    std::unique_ptr<pfd::select_folder> tracking_folder_dialog;
    std::unique_ptr<pfd::select_folder> baby_hand_idx_folder_dialog;
    std::unique_ptr<pfd::open_file> k_metric_csv_dialog;
    std::unique_ptr<pfd::open_file> production_csv_dialog;
    // A CSV the Explorer asked to open, applied at the top of the next frame rather
    // than mid-frame: open_sequence frees current_gpu, and the render below still
    // holds a pointer to it, so opening inline would use freed memory.
    std::optional<std::string> pending_open_csv;

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

        // Label for the menu bar: the absolute path of the open trial CSV.
        std::string open_path_label;
        if (sequence) {
            std::error_code abs_error;
            const fs::path absolute_path = fs::absolute(sequence->csv_path(), abs_error);
            open_path_label = abs_error ? sequence->csv_path() : absolute_path.string();
        }

        const MenuResult menu = draw_menu_bar(
            MenuState{
                .hand_translucent = settings.hand_translucent,
                .show_camera_marker = settings.show_camera_marker,
                .free_camera = settings.free_camera,
                .open_path = open_path_label,
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
            ImGui::DockBuilderDockWindow("Hands", dock_explorer);
            ImGui::DockBuilderDockWindow("Scene", dock_scene);
            ImGui::DockBuilderDockWindow("Frame View", dock_frame);
            ImGui::DockBuilderFinish(dock_id);
        }

        // Open the file/folder pickers when their menu items are chosen.
        if (menu.csv_requested && !csv_dialog) {
            csv_dialog = std::make_unique<pfd::open_file>("Select trial CSV", "", std::vector<std::string>{"CSV files", "*.csv", "All files", "*"});
        }
        if (menu.images_requested && !images_folder_dialog) {
            images_folder_dialog = std::make_unique<pfd::select_folder>("Select images folder");
        }
        if (menu.sam3_requested && !sam3_folder_dialog) {
            sam3_folder_dialog = std::make_unique<pfd::select_folder>("Select SAM3 cache folder");
        }
        if (menu.da3_requested && !da3_folder_dialog) {
            da3_folder_dialog = std::make_unique<pfd::select_folder>("Select DA3 cache folder");
        }
        if (menu.tracking_requested && !tracking_folder_dialog) {
            tracking_folder_dialog = std::make_unique<pfd::select_folder>("Select tracking folder (tracks3_*.csv)");
        }
        if (menu.baby_hand_idx_requested && !baby_hand_idx_folder_dialog) {
            baby_hand_idx_folder_dialog = std::make_unique<pfd::select_folder>("Select baby-hand-idx folder (*_hand_idx.csv)");
        }
        if (menu.k_metric_csv_requested && !k_metric_csv_dialog) {
            k_metric_csv_dialog =
                std::make_unique<pfd::open_file>("Select k_metric CSV", "", std::vector<std::string>{"CSV files", "*.csv", "All files", "*"});
        }
        if (menu.production_csv_requested && !production_csv_dialog) {
            production_csv_dialog =
                std::make_unique<pfd::open_file>("Select production log CSV", "", std::vector<std::string>{"CSV files", "*.csv", "All files", "*"});
        }

        // Poll with a zero timeout: pfd's ready() defaults to a 20ms wait that
        // blocks this thread every frame the dialog is open. A zero timeout
        // returns immediately and keeps idle frames.
        if (csv_dialog && csv_dialog->ready(0)) {
            const std::vector<std::string> chosen = csv_dialog->result();
            if (!chosen.empty() && !chosen.front().empty()) {
                open_sequence(chosen.front(), 0);
            }
            csv_dialog.reset();
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

        // The chosen images folder is the root the Frame View resolves per-trial
        // images under; persisted so it reopens next launch.
        if (images_folder_dialog && images_folder_dialog->ready(0)) {
            const std::string folder = images_folder_dialog->result();
            if (!folder.empty()) {
                settings.images_folder = folder;
            }
            images_folder_dialog.reset();
        }

        // The chosen SAM3/DA3 folders are where per-trial ``arrays`` caches live;
        // persisted, and the open trial's tracked object + smoothed hand depths
        // are refit from them immediately (see reload_object/reload_hand_smoothing).
        if (sam3_folder_dialog && sam3_folder_dialog->ready(0)) {
            const std::string folder = sam3_folder_dialog->result();
            if (!folder.empty()) {
                settings.sam3_folder = folder;
                reload_object();
                reload_hand_smoothing();
                loaded_frame = -1;
            }
            sam3_folder_dialog.reset();
        }
        if (da3_folder_dialog && da3_folder_dialog->ready(0)) {
            const std::string folder = da3_folder_dialog->result();
            if (!folder.empty()) {
                settings.da3_folder = folder;
                reload_object();
                reload_hand_smoothing();
                loaded_frame = -1;
            }
            da3_folder_dialog.reset();
        }

        // The tracking/baby-hand-idx folders feed MeshSequence's constructor
        // (see data/hand_classification.h), so a change reopens the current
        // trial from scratch rather than just invalidating the loaded frame.
        if (tracking_folder_dialog && tracking_folder_dialog->ready(0)) {
            const std::string folder = tracking_folder_dialog->result();
            if (!folder.empty()) {
                settings.tracking_folder = folder;
                if (sequence) {
                    open_sequence(sequence->csv_path(), current_frame);
                }
            }
            tracking_folder_dialog.reset();
        }
        if (baby_hand_idx_folder_dialog && baby_hand_idx_folder_dialog->ready(0)) {
            const std::string folder = baby_hand_idx_folder_dialog->result();
            if (!folder.empty()) {
                settings.baby_hand_idx_folder = folder;
                if (sequence) {
                    open_sequence(sequence->csv_path(), current_frame);
                }
            }
            baby_hand_idx_folder_dialog.reset();
        }

        // The k_metric CSV feeds open_sequence's intrinsics auto-detect (see
        // above), so a change reopens the current trial to re-run that lookup
        // against the newly-set table.
        if (k_metric_csv_dialog && k_metric_csv_dialog->ready(0)) {
            const std::vector<std::string> chosen = k_metric_csv_dialog->result();
            if (!chosen.empty() && !chosen.front().empty()) {
                settings.k_metric_csv = chosen.front();
                if (sequence) {
                    open_sequence(sequence->csv_path(), current_frame);
                }
            }
            k_metric_csv_dialog.reset();
        }

        // The production log feeds open_sequence's object auto-detect (see
        // above), so a change reopens the current trial to re-run that lookup.
        if (production_csv_dialog && production_csv_dialog->ready(0)) {
            const std::vector<std::string> chosen = production_csv_dialog->result();
            if (!chosen.empty() && !chosen.front().empty()) {
                settings.production_csv = chosen.front();
                if (sequence) {
                    open_sequence(sequence->csv_path(), current_frame);
                }
            }
            production_csv_dialog.reset();
        }

        // Apply a deferred Explorer open here, before the current frame's GPU
        // buffers are read below, so opening never frees a buffer still in use.
        if (pending_open_csv) {
            open_sequence(*pending_open_csv, 0);
            pending_open_csv.reset();
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
        // Which hands to hide, from the Hands pane (persisted). A change invalidates
        // the loaded frame below so it rebuilds filtered.
        const HandFilter filter{settings.hide_duplicates, settings.hide_adults};
        const CameraIntrinsics intrinsics{
            settings.intrinsics_fx, settings.intrinsics_fy, settings.intrinsics_cx, settings.intrinsics_cy, settings.intrinsics_k_metric
        };
        const HandDepthConfig depth_config{
            settings.hand_depth_source == 1 ? HandDepthSource::Hamer : HandDepthSource::Da3, trial_arrays_dir(settings.sam3_folder),
            trial_arrays_dir(settings.da3_folder)
        };

        std::string status;
        if (sequence) {
            if (current_frame != loaded_frame) {
                Frame hands = sequence->load_frame(current_frame, filter, intrinsics, depth_config, smoothed_hand_depths.get());
                current_gpu = std::make_unique<FrameGpu>(prepare_frame(hands));
                loaded_frame = current_frame;
            }
            const std::size_t hand_count = static_cast<std::size_t>(sequence->hand_count(current_frame, filter));
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
            const std::string image_path = sequence->frame_image_path(current_frame, settings.images_folder.value_or(std::string()));
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
        if (explorer_result.open_csv) {
            pending_open_csv = explorer_result.open_csv;
        }

        // Hands pane: filter toggles (hide duplicates / adults). A change reloads the
        // current frame so the filter applies; each hand is placed independently
        // (place_hand_metric), so hiding a hand never moves the ones that remain.
        const HandPaneState hands = draw_hand_pane(HandPaneState{settings.hide_duplicates, settings.hide_adults}, dock_id);
        if (hands.hide_duplicates != settings.hide_duplicates || hands.hide_adults != settings.hide_adults) {
            settings.hide_duplicates = hands.hide_duplicates;
            settings.hide_adults = hands.hide_adults;
            loaded_frame = -1;
        }

        // Camera pane: recording-camera intrinsics. A change reloads the current
        // frame, since hand placement (place_hand_metric) depends on them.
        const CameraPaneState camera_pane = draw_camera_pane(
            CameraPaneState{
                settings.intrinsics_fx, settings.intrinsics_fy, settings.intrinsics_cx, settings.intrinsics_cy, settings.intrinsics_k_metric,
                intrinsics_calibrated, settings.hand_depth_source == 0, settings.smoothing_enabled
            },
            dock_id
        );
        if (camera_pane.fx != settings.intrinsics_fx || camera_pane.fy != settings.intrinsics_fy || camera_pane.cx != settings.intrinsics_cx ||
            camera_pane.cy != settings.intrinsics_cy || camera_pane.k_metric != settings.intrinsics_k_metric) {
            settings.intrinsics_fx = camera_pane.fx;
            settings.intrinsics_fy = camera_pane.fy;
            settings.intrinsics_cx = camera_pane.cx;
            settings.intrinsics_cy = camera_pane.cy;
            settings.intrinsics_k_metric = camera_pane.k_metric;
            // Re-check the warning against a manual edit too: known only when
            // this fx has a calibrated k_metric AND the pane's k_metric matches
            // it (a hand-typed k_metric for an unrecognized fx isn't trusted).
            const std::optional<float> known = known_k_metric_for_focal_length(settings.intrinsics_fx);
            intrinsics_calibrated = known.has_value() && std::abs(*known - settings.intrinsics_k_metric) < 1e-4f;
            // The hands themselves reload cheaply every frame (load_frame just
            // takes the live intrinsics, no disk I/O), so this can happen on
            // every drag tick. Refitting the tracked object / hand-depth
            // smoothing, though, re-reads the whole trial's SAM3/DA3 caches
            // and re-runs pose fitting for every frame -- deferred to
            // intrinsics_committed (drag released) so a single drag gesture
            // doesn't trigger that dozens of times and freeze the UI.
            loaded_frame = -1;
            if (camera_pane.intrinsics_committed) {
                reload_object();
                reload_hand_smoothing();
            }
        }
        const int hand_depth_source = camera_pane.prefer_da3_hand_depth ? 0 : 1;
        if (hand_depth_source != settings.hand_depth_source) {
            settings.hand_depth_source = hand_depth_source;
            loaded_frame = -1;
            reload_hand_smoothing();
        }
        if (camera_pane.smoothing_enabled != settings.smoothing_enabled) {
            settings.smoothing_enabled = camera_pane.smoothing_enabled;
            loaded_frame = -1;
            reload_object();
            reload_hand_smoothing();
        }

        // Object pane: tracked-object shape/label/size. A change refits the
        // whole trial's object pose sequence (see reload_object).
        const ObjectPaneState object_pane =
            draw_object_pane(ObjectPaneState{settings.object_label, settings.object_shape, settings.object_size_m, object_auto_detected}, dock_id);
        if (object_pane.object_label != settings.object_label || object_pane.shape != settings.object_shape || object_pane.size_m != settings.object_size_m) {
            settings.object_label = object_pane.object_label;
            settings.object_shape = object_pane.shape;
            settings.object_size_m = object_pane.size_m;
            object_auto_detected = false; // a manual edit overrides the production-log match
            reload_object();
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

        // Place this frame's tracked object (cube or sphere, per object_sequence's
        // shape), if the trial has one and it was fit for this frame. Placed in
        // the same real-metric on-screen space the hands are (see
        // data/object_sequence.h), so it sits correctly relative to them.
        const GpuMesh* object_ptr = nullptr;
        glm::mat4 object_model(1.0f);
        if (object_sequence && has_sequence) {
            const std::optional<CubePose> pose = object_sequence->pose_for_frame(sequence->frame_number(current_frame));
            if (pose) {
                object_model = cube_model_matrix(*pose);
                object_ptr = object_sequence->shape() == ObjectShape::Sphere ? sphere_mesh.get() : cube_mesh.get();
            }
        }

        // NOTE: the interactive camera intentionally does NOT use the recording
        // camera's own FOV (2*atan(height/(2*fy)), ~9 degrees at this study's
        // fx=6900) — that's an extreme telephoto lens, correct for a fixed-
        // position comparison render but unusable for free-fly navigation (it
        // reads as near-orthographic and disorienting). SceneRender.fov_y_degrees
        // defaults to a normal 45 degrees for interactive use; only pass the
        // intrinsics-derived FOV when driving a fixed side-by-side comparison
        // against a render_scene_video.py still.

        // ── Render scene into the offscreen texture, then the UI ──
        framebuffer.resize(viewport.width, viewport.height);
        render_scene(
            framebuffer,
            *camera,
            SceneRender{
                .frame = frame_ptr,
                .translucent = settings.hand_translucent,
                .show_camera_marker = settings.show_camera_marker,
                .cube = object_ptr,
                .cube_model = object_model,
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
        settings.last_folder = sequence->csv_path();
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
    cube_mesh.reset();
    sphere_mesh.reset();
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

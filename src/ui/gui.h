#pragma once

// imgui UI: bundled fonts, the top menu bar, the dockable viewport window that
// displays the rendered scene texture, and the sequence playback transport bar.

#include <string>
#include <utility>

#include <imgui.h>

#include "graphics/rendering.h"

// Pixel sizes the bundled font is baked at (baking each size keeps text crisp).
inline constexpr float UI_FONT_SIZE = 22.0f;
inline constexpr float FPS_FONT_SIZE = 22.0f;

// Height of the sequence playback bar pinned to the window bottom.
inline constexpr float PLAYBACK_BAR_HEIGHT = 56.0f;

/// The menu bar's toggle state, passed in and echoed back with the user's edits.
/// New menu toggles are added here, not as another draw_menu_bar parameter.
struct MenuState {
    bool hand_translucent = false;
    bool show_camera_marker = false;
    bool free_camera = false;
    std::string open_path; // currently-loaded trial CSV path, shown right-aligned in the bar (empty = none)
};

struct MenuResult {
    MenuState state;               // the (possibly toggled) menu state
    bool csv_requested = false;    // user picked "Open Trial CSV"
    bool images_requested = false; // user picked "Set Images Folder"
    bool sam3_requested = false;             // user picked "Set SAM3 Folder"
    bool da3_requested = false;              // user picked "Set DA3 Folder"
    bool tracking_requested = false;         // user picked "Set Tracking Folder"
    bool baby_hand_idx_requested = false;    // user picked "Set Baby-Hand-Idx Folder"
    bool k_metric_csv_requested = false;     // user picked "Set K-Metric CSV"
    bool production_csv_requested = false;   // user picked "Set Production Log CSV"
};

/// Playback-transport state every pane that can host the player shares: passed in
/// to the window-drawing functions and echoed back (with the user's changes) in
/// their results. Grouped so a new transport control is one field here instead of
/// another parameter threaded through every drawing function and its result.
struct Transport {
    bool has_sequence = false; // no sequence => transport hidden, controls dimmed
    int current_frame = 0;
    int frame_count = 0;
    bool playing = false;
    float speed = 1.0f;          // playback rate multiplier (0.5x .. 4x)
    bool show_transport = false; // this pane carries the transport this frame
    unsigned int play_icon = 0;  // play button texture (0 => text-button fallback)
    unsigned int pause_icon = 0; // pause button texture (0 => text-button fallback)
    unsigned int speed_icon = 0; // speed (gauge) button texture (0 => text-button fallback)
};

/// GL textures for the transport's play/pause button. Loaded once after the GL
/// context exists (plain ``GL_TEXTURE_2D`` names) and freed on destruction. A
/// handle left at 0 (its PNG was missing) falls back to a text button.
struct TransportIcons {
    TransportIcons() = default;
    ~TransportIcons();

    TransportIcons(const TransportIcons&) = delete;
    TransportIcons& operator=(const TransportIcons&) = delete;

    /// Load the icon PNGs from ``icons_dir`` (the assets/icons/ folder next to the
    /// binary). Decodes synchronously and uploads to GL, so call once on the render
    /// thread after the GL context is current.
    void load(const std::string& icons_dir);

    unsigned int play = 0;
    unsigned int pause = 0;
    unsigned int speed = 0;
};

/// The transport's echoed-back state after a pane drew it: the (possibly
/// user-changed) frame and play state, plus whether the scrubber is being dragged.
struct TransportState {
    int current_frame = 0;
    bool playing = false;
    bool scrubbing = false;
    float speed = 1.0f; // echoed-back playback rate multiplier
};

struct ViewportResult {
    int width;
    int height;
    bool hovered;
    // True when this window is the one the mouse is interacting with (focused),
    // used to decide which pane carries the playback transport.
    bool focused;
    bool show_controls;
    TransportState transport; // echoed transport state when this pane drew it
};

struct ImageViewResult {
    bool hovered;
    bool focused;
    TransportState transport; // echoed transport state when this pane drew it
};

/// The "Hands" pane's filter toggles, passed in and echoed back with the user's
/// edits. Hands are also coloured by track id in the scene (always on); this pane
/// only controls which hands are hidden.
struct HandPaneState {
    bool hide_duplicates = true;
    bool hide_adults = true;
};

/// Bake the bundled font for the UI and the FPS overlay. Returns (ui, fps),
/// both the imgui default font if no .ttf is bundled in fonts/.
std::pair<ImFont*, ImFont*> load_fonts(ImGuiIO& io);

/// Draw the top menu bar; returns the (possibly updated) toggles and requests.
MenuResult draw_menu_bar(MenuState state);

/// Draw the dockable window that displays the rendered scene texture. ``status``
/// may be empty to omit the overlay status line. When ``transport.has_sequence``
/// and ``transport.show_transport`` are set, the video-player transport bar is
/// drawn at the bottom of this same window so it follows the viewport wherever it
/// is docked; the result's ``transport`` carries any changes the user made.
ViewportResult draw_viewport_window(
    const Framebuffer& framebuffer, ImGuiID dock_id, float fps, ImFont* fps_font, const std::string& status, bool show_controls, const Transport& transport
);

/// Draw the dockable "Image" window showing the modeled keypoint image for the
/// current frame. ``texture`` may be 0 (no image available) — a placeholder note
/// is shown instead. When ``transport.has_sequence`` and ``transport.show_transport``
/// are set the transport bar is drawn here too, so the player follows whichever
/// pane the mouse is on; the result's ``transport`` carries any user changes back.
ImageViewResult draw_image_window(const char* title, unsigned int texture, int texture_width, int texture_height, ImGuiID dock_id, const Transport& transport);

/// Draw the dockable "Hands" pane with the hide-duplicates / hide-adults filter
/// checkboxes. Returns the (possibly toggled) state.
HandPaneState draw_hand_pane(HandPaneState state, ImGuiID dock_id);

/// The recording-camera intrinsics editable from the "Camera" pane (mirrors
/// data/mano_model.h's CameraIntrinsics; kept as plain floats here so gui.h
/// doesn't need to depend on data/mano_model.h).
struct CameraPaneState {
    float fx = 6900.0f;
    float fy = 6900.0f;
    float cx = 960.0f;
    float cy = 540.0f;
    float k_metric = 0.354f;
    // Display-only hint (not user-editable, not persisted): false when the
    // currently-open trial's auto-detected focal length has no known-good
    // k_metric (see data/mano_model.h's known_k_metric_for_focal_length) —
    // draws a warning rather than silently trusting a stale/guessed value.
    bool calibrated = true;
    // Whether a hand's wrist depth prefers a real per-frame DA3 sample (true,
    // the default — mirrors data/mesh_sequence.h's HandDepthSource::Da3) or
    // always uses HaMeR's own k_metric * cam_t.z (false, HandDepthSource::Hamer).
    // Da3 needs the SAM3/DA3 folders set (File menu) to actually take effect.
    bool prefer_da3_hand_depth = true;
    // Whole-trial Gaussian temporal smoothing (sigma=3 frames, matching
    // render_scene_video.py's --smooth-sigma-frames default) for both hand
    // wrist depth and the tracked object's pose. Off = each frame's own raw
    // resolved value.
    bool smoothing_enabled = true;
    // Output-only: true on the one frame a fx/fy/cx/cy/k_metric drag just
    // finished (mouse released after an edit), false every other frame
    // including the ones while still dragging. The intrinsics values
    // themselves update live on every drag tick (so the pane always shows
    // the current number); callers that refit the tracked object/hand
    // smoothing from these intrinsics should gate that expensive whole-trial
    // work on this flag instead of "value changed", or a single drag gesture
    // triggers it dozens of times before the mouse is released.
    bool intrinsics_committed = false;
};

/// Draw the dockable "Camera" pane with the recording-camera intrinsics used to
/// backproject hands (and, once ported, the tracked object) into real metric
/// space. Returns the (possibly edited) state.
CameraPaneState draw_camera_pane(CameraPaneState state, ImGuiID dock_id);

/// The tracked-object pipeline's settings, editable from the "Object" pane
/// (mirrors app/config.h's sam3_folder/da3_folder/object_label/object_shape/
/// object_size_m; the SAM3/DA3 folders themselves are set via the File menu's
/// folder pickers, like Images/Objects). ``shape`` mirrors
/// data/object_sequence.h's ObjectShape: 0 = None, 1 = Cube, 2 = Sphere.
struct ObjectPaneState {
    std::string object_label = "small_ball";
    int shape = 0;
    float size_m = 0.04f;
    // Display-only hint (not user-editable, not persisted): true when
    // shape/label/size were just auto-detected for the open trial from the
    // production log (data/production_log.h), false when no production log
    // is configured or the open trial isn't in it — in which case these
    // values are whatever was last set manually.
    bool auto_detected = true;
};

/// Draw the dockable "Object" pane with the tracked-object shape/label/size
/// settings. A change here requires reopening the trial (or reselecting the
/// SAM3/DA3 folders) to take effect, since the whole pose sequence is fit
/// once at load time — see data/object_sequence.h. Returns the (possibly
/// edited) state.
ObjectPaneState draw_object_pane(ObjectPaneState state, ImGuiID dock_id);

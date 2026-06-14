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
    std::string open_folder; // currently-loaded sequence folder, shown right-aligned in the bar (empty = none)
};

struct MenuResult {
    MenuState state;               // the (possibly toggled) menu state
    bool folder_requested = false; // user picked "Open Mesh Sequence Folder"
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

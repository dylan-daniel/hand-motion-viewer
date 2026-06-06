#pragma once

// imgui UI: bundled fonts, the top menu bar, the dockable viewport window that
// displays the rendered scene texture, and the sequence playback transport bar.

#include <string>
#include <utility>

#include <imgui.h>

#include "rendering.h"

// Pixel sizes the bundled font is baked at (baking each size keeps text crisp).
inline constexpr float UI_FONT_SIZE = 18.0f;
inline constexpr float FPS_FONT_SIZE = 22.0f;

// Height of the sequence playback bar pinned to the window bottom.
inline constexpr float PLAYBACK_BAR_HEIGHT = 56.0f;

struct MenuResult {
    bool hand_translucent;
    bool show_camera_marker;
    bool free_camera;
    bool folder_requested;
};

struct ViewportResult {
    int width;
    int height;
    bool hovered;
    // True when this window is the one the mouse is interacting with (focused),
    // used to decide which pane carries the playback transport.
    bool focused;
    bool show_controls;
    // Playback transport state, echoed back when a sequence is loaded.
    int current_frame;
    bool playing;
    // True while the user is dragging the scrubber on this pane's transport.
    bool scrubbing;
};

struct ImageViewResult {
    bool hovered;
    bool focused;
    // Playback transport state, echoed back when the transport is drawn here.
    int current_frame;
    bool playing;
    // True while the user is dragging the scrubber on this pane's transport.
    bool scrubbing;
};

/// Bake the bundled font for the UI and the FPS overlay. Returns (ui, fps),
/// both the imgui default font if no .ttf is bundled in fonts/.
std::pair<ImFont*, ImFont*> load_fonts(ImGuiIO& io);

/// Draw the top menu bar; returns the (possibly updated) toggles and requests.
MenuResult draw_menu_bar(bool translucent, bool show_marker, bool free_camera);

/// Draw the dockable window that displays the rendered scene texture. ``status``
/// may be empty to omit the overlay status line. When ``has_sequence`` and
/// ``show_transport`` are set, the video-player transport bar is drawn at the
/// bottom of this same window so it follows the viewport wherever it is docked;
/// ``current_frame`` / ``playing`` are returned with any changes the user made.
ViewportResult draw_viewport_window(
    const Framebuffer& framebuffer,
    ImGuiID dock_id,
    float fps,
    ImFont* fps_font,
    const std::string& status,
    bool show_controls,
    bool has_sequence,
    int current_frame,
    int frame_count,
    bool playing,
    bool show_transport
);

/// Draw the dockable "Image" window showing the modeled keypoint image for the
/// current frame. ``texture`` may be 0 (no image available) — a placeholder note
/// is shown instead. When ``has_sequence`` and ``show_transport`` are set the
/// transport bar is drawn here too, so the player follows whichever pane the
/// mouse is on; ``current_frame`` / ``playing`` carry any user changes back.
ImageViewResult draw_image_window(
    const char* title,
    unsigned int texture,
    int texture_width,
    int texture_height,
    ImGuiID dock_id,
    bool has_sequence,
    int current_frame,
    int frame_count,
    bool playing,
    bool show_transport
);

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
    bool load_requested;
    bool folder_requested;
};

struct ViewportResult {
    int width;
    int height;
    bool hovered;
    bool show_controls;
};

struct PlaybackResult {
    int current_frame;
    bool playing;
};

/// Bake the bundled font for the UI and the FPS overlay. Returns (ui, fps),
/// both the imgui default font if no .ttf is bundled in fonts/.
std::pair<ImFont*, ImFont*> load_fonts(ImGuiIO& io);

/// Draw the top menu bar; returns the (possibly updated) toggles and requests.
MenuResult draw_menu_bar(bool translucent, bool show_marker, bool free_camera);

/// Draw the dockable window that displays the rendered scene texture. ``status``
/// may be empty to omit the overlay status line.
ViewportResult draw_viewport_window(
    const Framebuffer& framebuffer, ImGuiID dock_id, float fps, ImFont* fps_font, const std::string& status, bool has_sequence, bool show_controls
);

/// Draw the video-player transport bar pinned to the bottom of the window.
PlaybackResult draw_playback_bar(int win_width, int win_height, int current_frame, int frame_count, bool playing);

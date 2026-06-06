#pragma once

// Persistent user settings, stored as JSON next to the executable. Load once at
// startup into a Config, mutate it during the session, and save() on quit.

#include <array>
#include <optional>
#include <string>

struct Config {
    std::optional<std::string> last_folder; // path of the most recently loaded mesh sequence folder
    int last_frame = 0;                     // frame index to reopen the sequence on
    bool hand_translucent = true;           // whether the hand is drawn see-through
    bool show_overlay_hands = true;         // whether the OHView over-hand meshes are drawn
    bool show_controls = true;              // whether the viewport controls help panel is open
    bool show_camera_marker = false;        // whether the pink look-at marker ball is drawn
    bool free_camera = true;                // use the first-person free camera instead of the orbit camera

    // Window placement. Position is stored relative to ``window_display``'s
    // top-left corner so the window returns to the right monitor next launch.
    std::optional<int> window_x;
    std::optional<int> window_y;
    int window_width = 1024;
    int window_height = 720;
    bool window_fullscreen = true;
    int window_display = 0;

    // Orbit camera pose.
    float camera_azimuth = 30.0f;
    float camera_elevation = 20.0f;
    float camera_distance = 7.0f;
    std::array<float, 3> camera_target = {0.0f, 0.0f, 0.0f};
};

/// Read the config file, falling back to defaults if missing or invalid.
Config load_config(const std::string& path);

/// Write ``config`` back to the config file.
void save_config(const std::string& path, const Config& config);

/// Absolute path of the config file, next to the running executable.
std::string default_config_path();

#pragma once

// Persistent user settings, stored as JSON next to the executable. Load once at
// startup into a Config, mutate it during the session, and save() on quit.

#include <array>
#include <optional>
#include <string>
#include <vector>

struct Config {
    std::optional<std::string> last_folder;   // path of the most recently loaded trial CSV
    std::optional<std::string> images_folder; // root folder holding the per-trial frame images
    int last_frame = 0;                       // frame index to reopen the sequence on
    float playback_speed = 1.0f;               // playback rate multiplier (0.5x .. 4x)
    bool hand_translucent = true;              // whether the hand is drawn see-through
    bool show_controls = true;                 // whether the viewport controls help panel is open
    bool show_camera_marker = false;           // whether the pink look-at marker ball is drawn
    bool free_camera = true;                   // use the first-person free camera instead of the orbit camera
    int active_pane = 0;                       // last focused pane / selected tab: 0 = Scene, 1 = Frame View
    bool hide_duplicates = true;               // hide hands flagged as duplicate detections
    bool hide_adults = true;                   // hide non-baby (adult) hands

    std::optional<std::string> data_folder;    // root folder the Explorer pane browses, if chosen
    std::vector<std::string> expanded_folders; // Explorer tree folders left expanded, restored on launch

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

    // Recording-camera intrinsics (see data/mano_model.h's CameraIntrinsics),
    // used to backproject a hand's observed wrist pixel into a real metric 3D
    // position. Defaults match scene_placement.py's CLI defaults for this study.
    float intrinsics_fx = 6900.0f;
    float intrinsics_fy = 6900.0f;
    float intrinsics_cx = 960.0f;
    float intrinsics_cy = 540.0f;
    float intrinsics_k_metric = 0.354f;

    // Where a hand's per-frame wrist depth comes from (see
    // data/mesh_sequence.h's HandDepthSource): 0 = Da3 (default, matches
    // render_scene_video.py's own default), 1 = Hamer. Da3 needs sam3_folder/
    // da3_folder to actually have effect; with neither set it silently
    // behaves like Hamer.
    int hand_depth_source = 0;

    // Whole-trial Gaussian temporal smoothing (sigma=3 frames, matching
    // render_scene_video.py's --smooth-sigma-frames default), applied to both
    // hand wrist depth (data/mesh_sequence.h's precompute_smoothed_wrist_depths)
    // and the tracked object's pose (data/object_sequence.h's
    // TrackedObjectSequence). Off = each frame's own raw resolved value, same
    // as this viewer's original per-frame-only behavior.
    bool smoothing_enabled = true;

    // Tracked-object pipeline (data/object_sequence.h): root folders holding
    // per-trial SAM3/DA3 ``arrays`` caches, i.e.
    // ``<sam3_folder>/<subject>/<trial>/arrays/frame_XXXX_<object_label>_masks.npy``
    // and ``<da3_folder>/<subject>/<trial>/arrays/frame_XXXX_depth.npy``.
    std::optional<std::string> sam3_folder;
    std::optional<std::string> da3_folder;
    std::string object_label = "small_ball"; // matches the SAM3 mask filename
    int object_shape = 0;                    // ObjectShape: 0 = None, 1 = Cube, 2 = Sphere
    float object_size_m = 0.04f;             // sphere diameter or cube edge length, meters

    // Per-hand classification sidecars (data/hand_classification.h): root
    // folders holding, respectively,
    // ``tracking_folder/tracks3_<subject>_<trial>.csv`` (stable track ids +
    // duplicate-detection flags) and
    // ``baby_hand_idx_folder/<subject>_<trial>_hand_idx.csv`` (per-track
    // is_baby classification). Either unset falls back to hamer_cache's own
    // (usually absent) track_id/is_duplicate/is_baby columns, in which case
    // every hand is untracked (default colour) and shown.
    std::optional<std::string> tracking_folder;
    std::optional<std::string> baby_hand_idx_folder;

    // Per-(subject, trial) calibrated k_metric CSV (data/k_metric_table.h),
    // e.g. ``outputs/hamer_vggt_k_metric/k_metric.csv``. Checked before the
    // hardcoded fx-based table (data/mano_model.h's
    // known_k_metric_for_focal_length) when a trial is opened, since it's
    // keyed by the actual trial rather than an incidental focal-length match.
    std::optional<std::string> k_metric_csv;

    // The study's production log (data/production_log.h), e.g.
    // ``seattle_production_data.csv``. When set, opening a trial
    // auto-detects the tracked object's shape/label/size from it (see
    // main.cpp's open_sequence), same as intrinsics are auto-detected from
    // the trial CSV. object_shape/object_label/object_size_m above still
    // hold whatever was last resolved (auto or manual), and remain editable
    // in the Object pane as a manual override.
    std::optional<std::string> production_csv;
};

/// Read the config file, falling back to defaults if missing or invalid.
Config load_config(const std::string& path);

/// Write ``config`` back to the config file.
void save_config(const std::string& path, const Config& config);

/// Absolute path of the config file, next to the running executable.
std::string default_config_path();

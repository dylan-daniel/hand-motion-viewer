#pragma once

// Looks up a trial's real tracked-object shape/size straight from the
// study's own production log (this repo's copy of
// dylan/hamer/scene_renderer/scene_placement.py's object_size_meters), so
// the Object pane's shape/label/size don't need to be set by hand per trial
// — see main.cpp's open_sequence.

#include <optional>
#include <string>
#include <unordered_map>

/// One trial's tracked-object shape + real physical size, matching
/// scene_placement.py's SIZE_CM mapping (Large/Medium/Small -> 4.0/2.5/1.0
/// cm; the same value is a sphere's diameter or a cube's edge length).
struct ObjectSizeInfo {
    // 1 = Cube, 2 = Sphere (matches data/object_sequence.h's ObjectShape,
    // kept as a plain int here so this header has no dependency on it).
    int shape = 0;
    float size_m = 0.0f;
    // The SAM3 mask filename convention this study's captures use for each
    // shape (frame_XXXX_<label>_masks.npy) — "small_ball" for Sphere,
    // "small_cube" for Cube, regardless of the production log's Large/
    // Medium/Small size category (that's the physical size, not the SAM3
    // prompt name).
    std::string object_label;
};

/// The study's production log (Baby ID/Trial/Size/Shape columns), loaded
/// once and queried per (subject, trial_view) — e.g. ("AE45", "T1_BabyView").
class ProductionLog {
public:
    ProductionLog() = default;
    explicit ProductionLog(const std::string& csv_path);

    /// This trial's object info, or nullopt if the CSV has no row for this
    /// (subject, trial number) or the trial_view's leading trial number
    /// couldn't be parsed.
    std::optional<ObjectSizeInfo> lookup(const std::string& subject, const std::string& trial_view) const;

private:
    // Key: subject + "/" + trial number (e.g. "AE45/1").
    std::unordered_map<std::string, ObjectSizeInfo> info_by_subject_trial_;
};

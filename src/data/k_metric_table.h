#pragma once

// A per-(subject, trial) calibrated k_metric table, loaded from a CSV
// generated externally (this repo's own analog of scene_placement.py's
// per-trial calibration pass — see outputs/hamer_vggt_k_metric/k_metric.csv
// on the A6000 box). More precise than mano_model.h's
// known_k_metric_for_focal_length: that table only recognizes a handful of
// hardcoded focal lengths, while this one is keyed by the actual trial, so
// two trials that happen to share a VGGT-estimated focal length (which
// varies per trial, unlike this study's fixed fx=6900 rig) still get their
// own calibrated value.

#include <optional>
#include <string>
#include <unordered_map>

/// Loaded from a CSV with columns ``subject,trial,fx,n_hands,k_metric_median,
/// k_metric_mean,k_metric_std`` (only subject/trial/k_metric_median are
/// read). Empty (every lookup misses) if the CSV path is empty or the file
/// can't be opened/parsed.
class KMetricTable {
public:
    KMetricTable() = default;
    explicit KMetricTable(const std::string& csv_path);

    /// This trial's calibrated k_metric (the CSV's k_metric_median column),
    /// or nullopt if the CSV has no row for this exact (subject, trial).
    std::optional<float> lookup(const std::string& subject, const std::string& trial) const;

private:
    std::unordered_map<std::string, float> k_metric_by_subject_trial_;
};

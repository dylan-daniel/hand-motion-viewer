#pragma once

// Joins two sidecar CSVs (produced by the Python pipeline, outside this repo)
// onto a trial's hamer_cache rows to recover a stable per-hand track id and the
// duplicate/baby classification hamer_cache itself does not carry:
//
//  - the tracker CSV (``tracking_dir/tracks3_<subject>_<trial>.csv``) maps a
//    frame's ``idx`` (hamer_cache's own ``hand_idx``) to a stable ``track_id``
//    and flags redundant detections via ``is_duplicate``.
//  - the baby-hand classification CSV
//    (``baby_hand_idx_dir/<subject>_<trial>_hand_idx.csv``) maps ``track_id``
//    to ``is_baby`` (majority-vote classification over the whole trial).
//
// hamer_cache has no ``track_id`` column at all, so the join must go through
// the tracker CSV's ``idx``/frame pair — see mesh_sequence.cpp's use of this.

#include <string>
#include <unordered_map>

/// One (frame, hand_idx) row's classification, resolved via the tracker CSV's
/// track_id and, for is_baby, the baby_hand_idx CSV keyed by that track_id —
/// a track only counts as the baby's if it has an explicit is_baby==1 row
/// there; a track absent from that CSV entirely (e.g. never voted on) is
/// treated as not-baby, mirroring the Python reference join exactly.
struct HandClassificationEntry {
    int track_id = -1;
    bool is_duplicate = false;
    bool is_baby = false;
};

/// The joined classification for one trial, keyed by (frame_number, hand_idx).
/// Empty (lookup always misses) if either sidecar CSV was missing/unreadable.
class HandClassification {
public:
    HandClassification() = default;

    /// Load and join the tracker CSV and baby-hand classification CSV for
    /// ``subject``/``trial`` (trial as it appears in hamer_cache, e.g.
    /// "T1_BabyView"). Either directory empty, or either file missing, leaves
    /// an empty (always-miss) classification.
    HandClassification(const std::string& tracking_dir, const std::string& baby_hand_idx_dir, const std::string& subject, const std::string& trial);

    /// True if both sidecar CSVs were found and parsed for this trial.
    bool loaded() const { return loaded_; }

    /// The classification for one hamer_cache row, or nullptr if this
    /// (frame_number, hand_idx) pair has no tracker-CSV row (caller keeps its
    /// own default/CSV-column value).
    const HandClassificationEntry* lookup(int frame_number, int hand_idx) const;

private:
    bool loaded_ = false;
    // Key: frame_number * 100000 + hand_idx (hamer_cache never has anywhere
    // near 100000 hands in one frame).
    std::unordered_map<long long, HandClassificationEntry> by_frame_hand_;
};

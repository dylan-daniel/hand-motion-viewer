#pragma once

// Per-hmesh "baby hand" labels loaded from a data/labels_per_video CSV. Each row
// (columns: subject,trial,view,frame,hand_id,is_baby,match_iou,gemma_side) labels
// one hand as a baby hand (is_baby = 1) or not (0). The CSV's ``frame`` is 0-based
// while the hmesh files are 1-based, so the matching hmesh basename is
// ``frame_<frame+1>_<hand_id>.hmesh``. The viewer uses this to single out non-baby
// hands (drawn yellow/translucent, or hidden).

#include <string>
#include <unordered_map>

class HandClassification {
public:
    /// Load labels from a labels_per_video CSV. A missing or unreadable file yields
    /// an empty classification (has_data() == false), which callers treat as
    /// "label everything as shown".
    static HandClassification load(const std::string& csv_path);

    /// Whether any labels were loaded.
    bool has_data() const { return !is_baby_.empty(); }

    /// True if the given hmesh file (basename, e.g. "frame_0001_0.hmesh") is a baby
    /// hand. Files absent from the CSV default to true (shown), so unlabeled data
    /// is never silently hidden — only explicit is_baby = 0 rows are non-baby.
    bool is_baby(const std::string& hmesh_filename) const;

private:
    std::unordered_map<std::string, bool> is_baby_;
};

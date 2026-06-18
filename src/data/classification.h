#pragma once

// Per-track "baby hand" labels loaded from a data/classification_v2 CSV. Each row
// (columns: subject,trial,view,track_id,n_votes,n_baby,n_adult,frac_baby,label)
// labels one tracked hand as a baby hand (label = "baby") or an adult hand
// (label = "adult"). Unlike the older per-frame labels, this version keys off the
// persistent ``track_id`` from the tracking CSV, so a hand keeps its baby/adult
// label across the whole take regardless of detection order. The viewer uses this
// to single out adult hands (drawn yellow/translucent, or hidden).

#include <string>
#include <unordered_map>

class HandClassification {
public:
    /// Load labels from a classification_v2 CSV. A missing or unreadable file yields
    /// an empty classification (has_data() == false), which callers treat as
    /// "label everything as shown".
    static HandClassification load(const std::string& csv_path);

    /// Whether any labels were loaded.
    bool has_data() const { return !is_baby_.empty(); }

    /// True if the hand with the given tracking ``track_id`` is a baby hand. A
    /// track absent from the CSV (or an unlabeled hand, track_id < 0) defaults to
    /// true (shown), so untracked data is never silently hidden — only tracks
    /// explicitly labelled "adult" are non-baby.
    bool is_baby(int track_id) const;

    /// Number of frames the given ``track_id`` was classified in (the CSV's
    /// ``n_votes``). Fleeting duplicate hands the transformer briefly emits show up
    /// as low-vote tracks. Returns -1 for a track absent from the CSV (or an
    /// unlabeled hand, track_id < 0) so callers can leave those shown rather than
    /// filtering data with no vote count.
    int track_votes(int track_id) const;

private:
    std::unordered_map<int, bool> is_baby_;
    std::unordered_map<int, int> votes_;
};

#pragma once

// Loading and holding one trial's hand motion for playback. A trial is a single
// ``<trial>.csv`` (one row per detected hand per frame; see data/SAVE_ALL.md)
// holding the ``subject``/``trial`` keys and the generative MANO parameters, not
// baked geometry, so each hand's 778 surface vertices and 21 joints are
// regenerated on demand through the MANO layer (see data/mano_model.h). The
// per-frame images live in a separate images tree, located at
// ``<images_root>/<subject>/<trial>/frame_NNNN.jpg`` (see frame_image_path). The
// whole CSV is parsed once in the constructor; ``load_frame`` then runs the MANO
// forward pass for a frame.

#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "data/mano_model.h"

/// One hand in one frame: the regenerated MANO surface vertices and joint
/// positions, in the viewer's on-screen space. The face topology is shared
/// globally (see mano_faces) so it is not stored here; only the handedness needed
/// to pick the right winding.
struct HandData {
    std::vector<glm::vec3> verts;  // MANO surface vertices (778)
    std::vector<glm::vec3> joints; // joint positions (21)
    bool is_right = true;          // handedness, picks the shared face winding
    int track_id = -1;             // tracking id, drives the per-hand surface colour
};

/// Per-hand tracking/classification flags read from the CSV, used to filter which
/// hands are drawn and to colour them by track.
struct HandMeta {
    int track_id = -1;
    bool is_duplicate = false; // a redundant detection of an already-tracked hand
    bool is_baby = false;      // the infant's hand (vs. an adult's)
};

/// Which hands to hide. A hand is shown unless an enabled filter excludes it.
struct HandFilter {
    bool hide_duplicates = true;
    bool hide_adults = true;
};

/// Whether ``meta`` passes ``filter`` (i.e. the hand should be drawn).
inline bool hand_visible(const HandMeta& meta, const HandFilter& filter) {
    if (filter.hide_duplicates && meta.is_duplicate) {
        return false;
    }
    if (filter.hide_adults && !meta.is_baby) {
        return false;
    }
    return true;
}

using Frame = std::vector<HandData>;

/// Where a hand's per-frame wrist depth comes from — mirrors
/// render_scene_video.py's ``--hand-depth-source`` flag. ``Da3`` (the
/// Python default) prefers a DA3 depth-map sample at the wrist pixel,
/// falling back to HaMeR's own ``k_metric * cam_t.z`` when no robust sample
/// exists (missing DA3/SAM3 data, or too few valid pixels in the sampling
/// disk — see object_fit.h's sample_wrist_depth_da3). ``Hamer`` always uses
/// ``k_metric * cam_t.z``, ignoring DA3 entirely for hand placement (the
/// tracked object is unaffected either way — it always uses DA3).
enum class HandDepthSource { Da3, Hamer };

/// Per-trial DA3/SAM3 ``arrays`` directories plus which depth source to
/// prefer, passed to MeshSequence::load_frame. An empty ``sam3_dir``/
/// ``da3_dir`` makes ``Da3`` silently behave like ``Hamer`` (nothing to
/// sample), the same missing-folder tolerance data/object_sequence.h's
/// tracked-object pipeline has.
struct HandDepthConfig {
    HandDepthSource source = HandDepthSource::Da3;
    std::string sam3_dir;
    std::string da3_dir;
};

/// Per-trial sidecar folders used to recover a stable track id and the
/// duplicate/baby classification hamer_cache's own CSV doesn't carry (see
/// data/hand_classification.h). Either left empty falls back to whatever
/// hamer_cache's own track_id/is_duplicate/is_baby columns say (usually
/// absent, in which case every hand is untracked and treated as a
/// non-duplicate baby hand — see MeshSequence's constructor).
struct HandClassificationConfig {
    std::string tracking_dir;
    std::string baby_hand_idx_dir;
};

/// One trial's hand motion, parsed from its CSV once at construction. Holds the
/// per-frame MANO parameters; ``load_frame`` regenerates a frame's hand geometry
/// from them each time it is called (no caching, no background threads). Each
/// hand is placed independently in real metric space (see
/// data/mano_model.h's place_hand_metric) — there is no scene-wide fit/transform.
class MeshSequence {
public:
    /// ``csv_path`` is a trial's raw ``hamer_collect.py`` CSV. Throws if it cannot
    /// be opened; leaves an empty sequence (frame_count 0) if the CSV has no hands.
    /// ``classification_config``, if its directories are set, overrides each
    /// hand's track_id/is_duplicate/is_baby with the tracker-CSV/baby-hand-idx
    /// join (see data/hand_classification.h) instead of the CSV's own (usually
    /// absent) columns.
    explicit MeshSequence(const std::string& csv_path, const HandClassificationConfig& classification_config = {});

    int frame_count() const { return static_cast<int>(frames_.size()); }

    /// The original 1-indexed CSV frame number for dense frame ``index`` (frame
    /// numbers may have gaps; this maps a playback index back to it). Returns -1 for
    /// an out-of-range index. Used to look up sidecar data keyed by frame number,
    /// such as the tracked-cube poses.
    int frame_number(int index) const {
        if (index < 0 || index >= frame_count()) {
            return -1;
        }
        return frame_numbers_[static_cast<std::size_t>(index)];
    }

    const std::string& csv_path() const { return csv_path_; }

    /// The ``subject`` / ``trial`` keys read from the CSV, used to locate the
    /// matching per-frame images under the images root.
    const std::string& subject() const { return subject_; }
    const std::string& trial() const { return trial_; }

    /// Number of detected hands in frame ``index`` (0 for out-of-range), counting
    /// only those that pass ``filter``.
    int hand_count(int index, const HandFilter& filter = {}) const;

    /// Path to frame ``index``'s image under ``images_root``, i.e.
    /// ``<images_root>/<subject>/<trial>/frame_NNNN.{jpg,png}``. Prefers ``.jpg``
    /// then ``.png``; returns empty for an out-of-range index or an empty root.
    std::string frame_image_path(int index, const std::string& images_root) const;

    /// Regenerate the hands for frame ``index`` through the MANO layer and metric
    /// placement (place_hand_metric), keeping only those that pass ``filter``.
    /// ``depth_config`` selects (and locates) the per-frame wrist-depth source;
    /// defaults to DA3 with no folders set, i.e. the HaMeR fallback.
    /// ``smoothed_depths`` (see precompute_smoothed_wrist_depths), if given,
    /// overrides live per-frame depth resolution with a whole-trial-smoothed
    /// value — pass the exact table precompute_smoothed_wrist_depths(intrinsics,
    /// depth_config, ...) returned for THIS csv/intrinsics/depth_config, or the
    /// two will silently disagree. Returns an empty frame for an out-of-range
    /// index.
    Frame load_frame(
        int index, const HandFilter& filter, const CameraIntrinsics& intrinsics, const HandDepthConfig& depth_config = {},
        const std::vector<std::vector<float>>* smoothed_depths = nullptr
    ) const;

    /// Whole-trial smoothed per-hand wrist depth, mirroring
    /// render_scene_video.precompute_smoothed_depths's hand-only path: each
    /// hand's raw depth is resolved exactly like load_frame's live per-frame
    /// path (DA3 sample or HaMeR fallback, per ``depth_config``), then hands
    /// are tracked across frames (object_fit.h's track_hand_sequences — a
    /// CSV's hand_idx is not stable identity), DA3-less frames are filled
    /// from that same track's own calibrated ratio
    /// (fill_missing_depths_by_track), and the result is Gaussian-smoothed
    /// per track by real frame-number distance (smooth_sequence).
    /// ``sigma_frames`` <= 0 skips the final smoothing pass (fill only).
    /// The returned table is indexed [frame_index][hand_position_in_that_frame]
    /// — the same positional order load_frame iterates frames_ in, BEFORE
    /// HandFilter is applied — and must be passed back to load_frame's
    /// ``smoothed_depths`` verbatim.
    std::vector<std::vector<float>> precompute_smoothed_wrist_depths(
        const CameraIntrinsics& intrinsics, const HandDepthConfig& depth_config, float sigma_frames
    ) const;

    /// The tracking/classification metadata of frame ``index``'s hands (empty for
    /// an out-of-range index).
    const std::vector<HandMeta>& frame_meta(int index) const;

    /// This trial's own ``scaled_focal_length`` (pixels) that hamer_collect.py
    /// was run with — the CSV's own camera calibration, read straight from the
    /// data rather than a Config default. 0 if the CSV predates this column.
    float focal_length_px() const { return focal_length_px_; }

    /// This trial's frame width/height (pixels), the CSV assumes the principal
    /// point sits at the center of (see HAMER_COLLECT.md). 0 if the CSV
    /// predates these columns.
    float image_width() const { return image_width_; }
    float image_height() const { return image_height_; }

private:
    std::string csv_path_;
    std::string subject_; // ``subject`` key from the CSV (for locating images)
    std::string trial_;   // ``trial`` key from the CSV (for locating images)
    // One entry per frame, each a list of the frame's hands' MANO parameters.
    std::vector<std::vector<ManoParams>> frames_;
    // Parallel to frames_: each hand's tracking/classification metadata.
    std::vector<std::vector<HandMeta>> meta_;
    // The original frame number per frame index, used to locate the image on disk.
    std::vector<int> frame_numbers_;
    // This trial's own camera calibration, captured from the first real row.
    float focal_length_px_ = 0.0f;
    float image_width_ = 0.0f;
    float image_height_ = 0.0f;
};

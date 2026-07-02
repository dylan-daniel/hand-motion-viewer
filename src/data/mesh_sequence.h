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

/// A fixed translate-then-scale that frames the whole sequence on the grid.
struct Transform {
    glm::vec3 translate;
    float scale;
};

using Frame = std::vector<HandData>;

/// Per-hand placed-space stats precomputed in the CSV: the axis-aligned bounding
/// box and mean depth of the hand's placed vertices (post the place() transform,
/// pre depth-stabilisation). These let the scene fit (compute_transform) and the
/// stabilisation reference be derived without running the MANO forward pass on
/// every frame. See data/SAVE_ALL.md for the ``placed_bbox_*`` / ``placed_mean_z``
/// columns.
struct HandBounds {
    glm::vec3 placed_min{0.0f};
    glm::vec3 placed_max{0.0f};
    float placed_mean_z = 0.0f;
};

/// One trial's hand motion, parsed from its CSV once at construction. Holds the
/// per-frame MANO parameters; ``load_frame`` regenerates a frame's hand geometry
/// from them each time it is called (no caching, no background threads).
class MeshSequence {
public:
    /// ``csv_path`` is a trial's ``*.csv`` file. Throws if it cannot be opened;
    /// leaves an empty sequence (frame_count 0) if the CSV has no hands.
    explicit MeshSequence(const std::string& csv_path);

    int frame_count() const { return static_cast<int>(frames_.size()); }

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

    /// The original frame number (the CSV's ``frame`` column value, which may
    /// have gaps) for frame index ``index`` — the same numbering
    /// frame_image_path uses to build ``frame_NNNN``, and what the
    /// sam3_cache/da3_cache trees are keyed by. Returns -1 for an out-of-range
    /// index.
    int frame_number(int index) const;

    /// Regenerate the hands for frame ``index`` through the MANO layer, keeping
    /// only those that pass ``filter``. Returns an empty frame for an out-of-range
    /// index.
    Frame load_frame(int index, const HandFilter& filter = {}) const;

    /// The precomputed placed bounds of frame ``index``'s hands (empty for an
    /// out-of-range index). Read straight from the CSV, no MANO forward pass.
    const std::vector<HandBounds>& frame_bounds(int index) const;

    /// The tracking/classification metadata of frame ``index``'s hands, parallel to
    /// frame_bounds (empty for an out-of-range index).
    const std::vector<HandMeta>& frame_meta(int index) const;

    /// Mean placed depth (z) of frame 0's hands — the common plane every other
    /// frame's hands are depth-snapped onto. Derived from the CSV at construction.
    float reference_depth() const { return reference_depth_; }

private:
    std::string csv_path_;
    std::string subject_; // ``subject`` key from the CSV (for locating images)
    std::string trial_;   // ``trial`` key from the CSV (for locating images)
    // One entry per frame, each a list of the frame's hands' MANO parameters.
    std::vector<std::vector<ManoParams>> frames_;
    // Parallel to frames_: the precomputed placed bounds for each frame's hands.
    std::vector<std::vector<HandBounds>> bounds_;
    // Parallel to frames_: each hand's tracking/classification metadata.
    std::vector<std::vector<HandMeta>> meta_;
    // Mean placed depth of frame 0, cached for the stabilisation reference.
    float reference_depth_ = 0.0f;
    // The original frame number per frame index, used to locate the image on disk.
    std::vector<int> frame_numbers_;
};

/// Build the fixed transform that sits the sequence's hands on the grid and
/// scales them to a comfortable size. Centering X/Z and the fit scale come from
/// frame 0 so they stay put across playback, but the floor (Y) is the lowest
/// point across *every* frame so no frame ever dips below the grid plane.
Transform compute_transform(const MeshSequence& sequence);

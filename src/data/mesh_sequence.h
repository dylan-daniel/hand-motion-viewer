#pragma once

// Loading and holding a folder of per-frame hand meshes for motion playback. A
// mesh sequence folder holds files named ``frame_NNNN_<slot>.hmesh`` — one compact
// binary mesh per detected hand per video frame, storing just the MANO surface
// vertices and 21 joint positions. The face topology is identical for every hand
// and shared globally (see geometry's mano_faces), so only the moving vertices
// are streamed in per frame. Each frame is parsed synchronously on demand when
// playback reaches it (no background preloading).

#include <map>
#include <string>
#include <utility>
#include <vector>

#include <glm/glm.hpp>

/// One hand in one frame, decoded from a ``.hmesh``: the moving MANO surface
/// vertices and joint positions. The face topology is shared globally (see
/// mano_faces) so it is not stored here; only the handedness needed to pick it.
struct HandData {
    std::vector<glm::vec3> verts;  // MANO surface vertices (778)
    std::vector<glm::vec3> joints; // joint positions (21)
    bool is_right = true;          // handedness, picks the shared face winding
    int track_id = -1;             // persistent hand id from the tracking CSV, -1 if unknown
    bool is_duplicate = false;     // another hand in the same frame shares this track id
    int slot = -1;                 // hand slot within the frame (the ``_<slot>`` in the file name)
};

/// Per-detection facts read from a hand's row in the sibling tracking CSV.
struct TrackInfo {
    int color_id = -1;         // id the hand is coloured by (track_id, or hand_id when linked)
    bool is_duplicate = false; // the tracker flagged this detection as a duplicate
};

/// Identifies one tracking CSV beside a sequence: a numbered version, optionally
/// the ``_linked`` variant that colours hands by the linked ``hand_id`` instead of
/// the raw ``track_id``.
struct TrackingSource {
    int version = 1;
    bool linked = false;

    bool operator==(const TrackingSource& other) const { return version == other.version && linked == other.linked; }
};

/// A fixed translate-then-scale that frames the whole sequence on the grid.
struct Transform {
    glm::vec3 translate;
    float scale;
};

using Frame = std::vector<HandData>;

/// Group a folder's ``frame_NNNN_<slot>.hmesh`` files into ordered frames.
std::vector<std::vector<std::string>> discover_frames(const std::string& folder);

/// Map a frame hand's mesh path (``frame_NNNN_<slot>.hmesh``) to the modeled image
/// that sits beside it (``frame_NNNN_all_keypoints.jpg``). Returns empty if the
/// path does not match the expected ``frame_NNNN_<slot>`` shape.
std::string frame_image_path(const std::string& mesh_path);

/// Mean camera-space depth (z) of a frame's hands — the common plane depth
/// stabilisation snaps every other frame's hands onto.
float reference_depth(const Frame& hands);

/// A folder of per-frame hand meshes, parsed synchronously on demand. Holds only
/// the discovered file paths; ``load_frame`` reads and decodes a frame's hands
/// from disk each time it is called (no caching, no background threads).
class MeshSequence {
public:
    explicit MeshSequence(const std::string& folder);

    int frame_count() const { return frame_count_; }

    const std::string& folder() const { return folder_; }

    const std::vector<std::string>& frame_paths(int index) const { return frame_paths_[static_cast<std::size_t>(index)]; }

    /// The on-disk frame number (the ``frame_NNNN`` in the file names, as used by
    /// the tracking CSV) for the sequence position ``index``. This differs from the
    /// 0-based ``index`` whenever the numbering does not start at 0, so override and
    /// CSV lookups must key by this, not by ``index``. Returns -1 if unavailable.
    int frame_number(int index) const;

    /// Read and decode the hands for frame ``index`` from disk. Returns an empty
    /// frame for an out-of-range index.
    Frame load_frame(int index) const;

    /// Tracking CSVs found beside this sequence (numbered versions plus any
    /// ``_linked`` variants), sorted by version then linked-last. Empty when no
    /// tracking files exist for the folder.
    const std::vector<TrackingSource>& tracking_sources() const { return tracking_sources_; }

    /// The tracking source currently applied to ``load_frame``'s colour ids.
    TrackingSource tracking_source() const { return tracking_source_; }

    /// Re-parse the colour ids from ``source``'s tracking CSV. A no-op if it is
    /// already the active source; otherwise the next ``load_frame`` reflects it.
    void set_tracking_source(TrackingSource source);

    /// Apply ``source`` only if it exists for this sequence; otherwise keep the
    /// current one. Lets a remembered preference carry across sequences while still
    /// falling back gracefully when a sequence lacks that source.
    void prefer_tracking_source(TrackingSource source);

    // ── Manual id overrides ────────────────────
    // A manual labeling pass over the active source: the user picks a hand in the
    // scene and assigns it a colour id, overriding whatever the CSV said. Overrides
    // are keyed by (frame number, hand slot) so they survive switching the source,
    // and they feed ``load_frame``'s colour id and the saved truth CSV.

    /// Override the colour id of the hand at (``frame``, ``slot``). Takes effect on
    /// the next ``load_frame`` for that frame.
    void set_override(int frame, int slot, int hand_id);

    /// Drop a manual override, falling back to the source CSV's colour id.
    void clear_override(int frame, int slot);

    /// Snapshot the current overrides onto the undo stack. Call once before an edit
    /// (a pick, which may set one override and propagate many) so a single ``undo``
    /// reverts that whole edit.
    void push_undo_state();

    /// Restore the most recent snapshot pushed by ``push_undo_state``. Returns true
    /// if an edit was undone, false if there was nothing to undo.
    bool undo();

    /// Re-label a whole track forward: every hand from ``from_frame`` onward whose
    /// current colour id equals ``old_id`` is overridden to ``new_id``. This is how
    /// relabeling one detection carries to the rest of that track. A negative
    /// ``old_id`` is a no-op (an unlabeled hand has no track to follow).
    void propagate_id(int from_frame, int old_id, int new_id);

    /// Number of manual overrides recorded so far (for a UI readout).
    int override_count() const { return static_cast<int>(overrides_.size()); }

    /// Distinct colour ids present across the whole active source (overrides
    /// applied), sorted ascending, negatives dropped. Drives the labeller's
    /// colour-swatch row.
    std::vector<int> present_color_ids() const;

    /// Write a drop-in copy of the active source CSV into the sibling
    /// ``tracking/truth`` folder, with the colour column replaced by the manual
    /// overrides where present. Returns the written path, or empty on failure.
    std::string save_truth_csv() const;

private:
    std::string folder_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    std::vector<TrackingSource> tracking_sources_; // available CSVs for this folder, sorted
    TrackingSource tracking_source_;               // source currently parsed into track_ids_
    // (frame number, hand slot) → tracking facts. Empty when no tracking file is
    // found for this folder.
    std::map<std::pair<int, int>, TrackInfo> track_ids_;
    // (frame number, hand slot) → manually assigned colour id, overriding the CSV.
    std::map<std::pair<int, int>, int> overrides_;
    // Snapshots of overrides_ taken before each edit, for Ctrl+Z undo (newest last).
    std::vector<std::map<std::pair<int, int>, int>> undo_stack_;
};

/// Build the fixed transform that sits the sequence's hands on the grid and
/// scales them to a comfortable size. Centering X/Z and the fit scale come from
/// frame 0 so they stay put across playback, but the floor (Y) is the lowest
/// point across *every* frame so no frame ever dips below the grid plane.
Transform compute_transform(const MeshSequence& sequence);

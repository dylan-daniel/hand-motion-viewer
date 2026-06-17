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

private:
    std::string folder_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    std::vector<TrackingSource> tracking_sources_; // available CSVs for this folder, sorted
    TrackingSource tracking_source_;               // source currently parsed into track_ids_
    // (frame number, hand slot) → tracking facts. Empty when no tracking file is
    // found for this folder.
    std::map<std::pair<int, int>, TrackInfo> track_ids_;
};

/// Build the fixed transform that sits the sequence's hands on the grid and
/// scales them to a comfortable size. Centering X/Z and the fit scale come from
/// frame 0 so they stay put across playback, but the floor (Y) is the lowest
/// point across *every* frame so no frame ever dips below the grid plane.
Transform compute_transform(const MeshSequence& sequence);

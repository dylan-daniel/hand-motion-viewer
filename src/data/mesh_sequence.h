#pragma once

// Loading and holding per-frame hand meshes for motion playback, from either of
// two sources: a folder of ``frame_NNNN_<slot>.hmesh`` files (one compact binary
// mesh per detected hand per video frame, parsed synchronously on demand as
// playback reaches it), or a single infant_grasp_pipeline ``.hexport`` binary
// export (read and MANO-forward-passed entirely up front at open time -- see
// data/hand_export.h -- since it's a single compressed stream with no way to
// decompress just one frame's worth without decompressing everything before
// it anyway). Either way the face topology is identical for every hand and
// shared globally (see geometry's mano_faces), so only the moving vertices are
// held here.

#include <string>
#include <vector>

#include <glm/glm.hpp>

/// One hand in one frame, decoded from a ``.hmesh``: the moving MANO surface
/// vertices and joint positions. The face topology is shared globally (see
/// mano_faces) so it is not stored here; only the handedness needed to pick it.
struct HandData {
    std::vector<glm::vec3> verts;  // MANO surface vertices (778)
    std::vector<glm::vec3> joints; // joint positions (21)
    bool is_right = true;          // handedness, picks the shared face winding
    // Populated only in .hexport mode, from the pipeline's tracking/
    // classification output; left at their defaults (-1, empty) for a
    // .hmesh-sourced hand, which carries no such data.
    int hand_track_id = -1;
    std::string label; // "infant" | "adult" | "unknown" | "unclassified" | ""
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

/// Build the fixed transform that sits a frame's hands on the grid and scales
/// them to a comfortable size (mirrors center_model but kept as a transform so
/// every frame shares it and motion shows).
Transform compute_transform(const Frame& hands);

/// Mean camera-space depth (z) of a frame's hands — the common plane depth
/// stabilisation snaps every other frame's hands onto.
float reference_depth(const Frame& hands);

/// Either a folder of per-frame hand meshes (parsed synchronously on demand) or
/// a single ``.hexport`` file (fully loaded and MANO-forward-passed up front).
/// ``path`` is a directory in the first case, a regular file in the second --
/// detected once in the constructor. ``frame_paths`` has no per-hand file paths
/// in export mode (nothing on disk to point at); it still reports the correct
/// per-frame hand count via placeholder entries, since callers use its size for
/// that, and frame_image_path() gracefully reports "no image" for a path that
/// doesn't match the .hmesh naming pattern.
class MeshSequence {
public:
    explicit MeshSequence(const std::string& path);

    int frame_count() const { return frame_count_; }

    const std::string& folder() const { return folder_; }

    const std::vector<std::string>& frame_paths(int index) const { return frame_paths_[static_cast<std::size_t>(index)]; }

    /// Read/decode (disk mode) or return the precomputed (export mode) hands for
    /// frame ``index``. Returns an empty frame for an out-of-range index.
    Frame load_frame(int index) const;

private:
    std::string folder_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    bool is_export_mode_ = false;
    std::vector<Frame> export_frames_;
};

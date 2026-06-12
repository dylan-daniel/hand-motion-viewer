#pragma once

// Loading and holding a folder of per-frame hand meshes for motion playback. A
// mesh sequence folder holds files named ``frame_NNNN_<slot>.hmesh`` — one compact
// binary mesh per detected hand per video frame, storing just the MANO surface
// vertices and 21 joint positions. The face topology is identical for every hand
// and shared globally (see geometry's mano_faces), so only the moving vertices
// are streamed in per frame. Each frame is parsed synchronously on demand when
// playback reaches it (no background preloading).

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

/// A folder of per-frame hand meshes. With ``smooth`` disabled a frame's hands
/// are parsed synchronously from disk each time ``load_frame`` is called (no
/// caching). With ``smooth`` enabled every frame is read once at construction and
/// run through a forward-backward Kalman smoother (see kalman.h) to remove the
/// 30fps per-frame jitter; ``load_frame`` then returns the cached smoothed hands.
class MeshSequence {
public:
    explicit MeshSequence(const std::string& folder, bool smooth = false);

    int frame_count() const { return frame_count_; }

    const std::string& folder() const { return folder_; }

    const std::vector<std::string>& frame_paths(int index) const { return frame_paths_[static_cast<std::size_t>(index)]; }

    /// Return the hands for frame ``index``: the cached smoothed hands when
    /// smoothing is enabled, otherwise read and decoded from disk on the spot.
    /// Returns an empty frame for an out-of-range index.
    Frame load_frame(int index) const;

    /// Read the raw, unsmoothed hands for frame ``index`` straight from disk,
    /// bypassing the smoothing cache. Used by the temporary raw-position overlay.
    Frame load_raw_frame(int index) const { return load_frame_from_disk(index); }

private:
    /// Read and decode the hands for frame ``index`` straight from disk.
    Frame load_frame_from_disk(int index) const;

    std::string folder_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    // Populated only when smoothing is enabled: the whole sequence is read and
    // smoothed once at construction, and load_frame serves frames from here.
    bool smoothed_ = false;
    std::vector<Frame> smoothed_frames_;
};

/// Build the fixed transform that sits the sequence's hands on the grid and
/// scales them to a comfortable size. Centering X/Z and the fit scale come from
/// frame 0 so they stay put across playback, but the floor (Y) is the lowest
/// point across *every* frame so no frame ever dips below the grid plane.
Transform compute_transform(const MeshSequence& sequence);

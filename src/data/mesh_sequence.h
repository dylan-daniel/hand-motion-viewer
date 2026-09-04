#pragma once

// Loading and holding per-frame hand meshes for motion playback from a single
// infant_grasp_pipeline ``.hexport`` binary export (see data/hand_export.h):
// read and MANO-forward-passed entirely up front at open time, since it's a
// single compressed stream with no way to decompress just one frame's worth
// without decompressing everything before it anyway. The face topology is
// identical for every hand and shared globally (see geometry's mano_faces),
// so only the moving vertices are held here.

#include <string>
#include <vector>

#include <glm/glm.hpp>

/// One hand in one frame: the moving MANO surface vertices and joint
/// positions, plus the pipeline's own tracking/classification identity. The
/// face topology is shared globally (see mano_faces) so it is not stored
/// here; only the handedness needed to pick it.
struct HandData {
    std::vector<glm::vec3> verts;  // MANO surface vertices (778)
    std::vector<glm::vec3> joints; // joint positions (21)
    bool is_right = true;          // handedness, picks the shared face winding
    int hand_track_id = -1;        // from the pipeline's tracking stage; -1 if untracked
    std::string label;             // from the pipeline's classification stage: "infant" | "adult" | "unknown" | "unclassified"
};

/// A fixed translate-then-scale that frames the whole sequence on the grid.
struct Transform {
    glm::vec3 translate;
    float scale;
};

using Frame = std::vector<HandData>;

/// Build the fixed transform that sits a frame's hands on the grid and scales
/// them to a comfortable size (mirrors center_model but kept as a transform so
/// every frame shares it and motion shows).
Transform compute_transform(const Frame& hands);

/// Mean camera-space depth (z) of a frame's hands — the common plane depth
/// stabilisation snaps every other frame's hands onto.
float reference_depth(const Frame& hands);

/// A ``.hexport`` file's hands, fully loaded and MANO-forward-passed up front
/// at construction (see build_export_frames in the .cpp).
class MeshSequence {
public:
    explicit MeshSequence(const std::string& path);

    int frame_count() const { return frame_count_; }

    const std::string& path() const { return path_; }

    std::size_t hand_count(int index) const { return index < 0 || index >= frame_count_ ? 0 : frames_[static_cast<std::size_t>(index)].size(); }

    /// The precomputed hands for frame ``index``. Returns an empty frame for
    /// an out-of-range index.
    Frame load_frame(int index) const;

    /// Path of the plain source-video frame image for playback frame
    /// ``index``, or empty if none is found (see the .cpp for the
    /// frames/<export-stem>/ convention this looks for).
    std::string frame_image_path(int index) const;

private:
    std::string path_;
    std::vector<Frame> frames_;
    std::vector<int> frame_numbers_; // frame_numbers_[i] = the pipeline's own frame number for playback index i
    int frame_count_;
};

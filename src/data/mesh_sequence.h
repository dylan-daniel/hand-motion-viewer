#pragma once

// Loading and holding a folder of per-frame hand meshes for motion playback. A
// mesh sequence folder holds files named ``frame_NNNN_<slot>.hmesh`` — one compact
// binary mesh per detected hand per video frame, storing just the MANO surface
// vertices and 21 joint positions. The face topology is identical for every hand
// and shared globally (see geometry's mano_faces), so only the moving vertices
// are streamed in per frame. Parsing runs on background worker threads so the UI
// stays responsive.

#include <atomic>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "data/crossview.h"

/// One hand in one frame, decoded from a ``.hmesh``: the moving MANO surface
/// vertices and joint positions. The face topology is shared globally (see
/// mano_faces) so it is not stored here; only the handedness needed to pick it.
struct HandData {
    std::vector<glm::vec3> verts;                        // MANO surface vertices (778)
    std::vector<glm::vec3> joints;                       // joint positions (21)
    bool is_right = true;                                // handedness, picks the shared face winding
    glm::vec4 surface_color = {0.6f, 0.75f, 0.9f, 1.0f}; // hand-surface tint (overlay hands differ)
    bool is_overlay = false;                             // true for OHView hands folded in via the cross-view overlay
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

/// Streams every frame's hand positions into memory on background threads.
class MeshSequenceLoader {
public:
    explicit MeshSequenceLoader(const std::string& folder, int workers = 1, std::shared_ptr<const CrossViewOverlay> overlay = nullptr);
    ~MeshSequenceLoader();

    MeshSequenceLoader(const MeshSequenceLoader&) = delete;
    MeshSequenceLoader& operator=(const MeshSequenceLoader&) = delete;

    int frame_count() const { return frame_count_; }

    /// Total number of hand meshes across every frame, and how many have finished
    /// parsing so far (a frame may hold several hands, each its own .obj).
    int mesh_total() const { return mesh_total_; }
    int meshes_loaded() const { return meshes_loaded_.load(); }

    const std::string& folder() const { return folder_; }

    /// The cross-view overlay folded into this sequence, or nullptr if this is a
    /// plain single-view sequence (used to locate the OHView keypoint images).
    const std::shared_ptr<const CrossViewOverlay>& overlay() const { return overlay_; }
    const std::vector<std::string>& frame_paths(int index) const { return frame_paths_[static_cast<std::size_t>(index)]; }

    /// True once frame ``index`` has finished parsing.
    bool frame_ready(int index) const;

    /// The frame's hands, or nullptr if it has not finished parsing yet.
    std::shared_ptr<const Frame> get(int index) const;

    /// Ask the workers to load ``index`` ahead of the sequential backlog.
    void prioritize(int index);

    /// Signal workers to stop and join them (also called by the destructor).
    void stop();

private:
    int claim_next();
    void worker();

    std::string folder_;
    std::shared_ptr<const CrossViewOverlay> overlay_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    int mesh_total_ = 0;
    std::atomic<int> meshes_loaded_{0};

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<const Frame>> frames_; // nullptr until parsed
    int cursor_ = 0;
    std::deque<int> priority_;
    std::unordered_set<int> claimed_;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
};

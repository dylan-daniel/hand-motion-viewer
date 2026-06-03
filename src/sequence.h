#pragma once

// Loading and holding a folder of per-frame hand meshes for motion playback. A
// sequence folder holds files named ``frame_NNNN_<slot>.obj`` — one mesh per
// detected hand per video frame. Every mesh shares the same MANO topology
// (faces + per-vertex colours), so the topology is parsed once and interned
// while just the positions are streamed in for every frame. Parsing runs on
// background worker threads so the UI stays responsive.

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

/// Shared per-vertex/face data for one MANO hand mesh, identical across frames.
struct Topology {
    std::vector<glm::ivec3> faces;            // 0-based vertex indices
    std::vector<glm::vec4> colors;            // per-vertex RGBA in 0..1
    std::vector<std::uint8_t> hand_face_mask; // 1 for hand-surface faces, 0 for joints
};

/// One hand in one frame: moving vertex positions plus its shared topology.
struct HandData {
    std::vector<glm::vec3> positions;
    std::shared_ptr<const Topology> topology;
};

/// A fixed translate-then-scale that frames the whole sequence on the grid.
struct Transform {
    glm::vec3 translate;
    float scale;
};

using Frame = std::vector<HandData>;

/// Group a folder's ``frame_NNNN_<slot>.obj`` files into ordered frames.
std::vector<std::vector<std::string>> discover_frames(const std::string& folder);

/// Map a frame hand's mesh path (``frame_NNNN_<slot>.obj``) to the modeled image
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
class SequenceLoader {
public:
    explicit SequenceLoader(const std::string& folder, int workers = 1);
    ~SequenceLoader();

    SequenceLoader(const SequenceLoader&) = delete;
    SequenceLoader& operator=(const SequenceLoader&) = delete;

    int frame_count() const { return frame_count_; }

    /// Total number of hand meshes across every frame, and how many have finished
    /// parsing so far (a frame may hold several hands, each its own .obj).
    int mesh_total() const { return mesh_total_; }
    int meshes_loaded() const { return meshes_loaded_.load(); }

    const std::string& folder() const { return folder_; }
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
    std::shared_ptr<const Topology> get_topology(std::uint64_t key, const std::string& path);
    void worker();

    std::string folder_;
    std::vector<std::vector<std::string>> frame_paths_;
    int frame_count_;
    int mesh_total_ = 0;
    std::atomic<int> meshes_loaded_{0};

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<const Frame>> frames_; // nullptr until parsed
    std::unordered_map<std::uint64_t, std::shared_ptr<const Topology>> topologies_;
    int cursor_ = 0;
    std::deque<int> priority_;
    std::unordered_set<int> claimed_;
    std::atomic<bool> stop_{false};
    std::vector<std::thread> threads_;
};

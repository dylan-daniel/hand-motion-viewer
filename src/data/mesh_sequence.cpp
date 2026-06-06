#include "data/mesh_sequence.h"

#include "data/geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <regex>

namespace {
    namespace fs = std::filesystem;

    // frame_0001_0.hmesh → frame number 0001, hand slot 0.
    const std::regex kFrameRe(R"(frame_(\d+)_(\d+)\.hmesh$)", std::regex::icase);
} // namespace

std::vector<std::vector<std::string>> discover_frames(const std::string& folder) {
    // Group by frame number, each frame's hands ordered by slot; frame numbers
    // may have gaps, so the result is densified by sorting the keys.
    std::map<int, std::vector<std::pair<int, std::string>>> by_frame;
    std::error_code error;
    for (const fs::directory_entry& entry : fs::directory_iterator(folder, error)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        std::smatch match;
        if (!std::regex_search(name, match, kFrameRe)) {
            continue;
        }
        const int frame_number = std::stoi(match[1].str());
        const int slot = std::stoi(match[2].str());
        by_frame[frame_number].emplace_back(slot, entry.path().string());
    }

    std::vector<std::vector<std::string>> frames;
    frames.reserve(by_frame.size());
    for (auto& [frame_number, hands] : by_frame) {
        std::sort(hands.begin(), hands.end());
        std::vector<std::string> paths;
        paths.reserve(hands.size());
        for (auto& [slot, path] : hands) {
            paths.push_back(path);
        }
        frames.push_back(std::move(paths));
    }
    return frames;
}

std::string frame_image_path(const std::string& mesh_path) {
    const fs::path mesh(mesh_path);
    const std::string name = mesh.filename().string();
    std::smatch match;
    if (!std::regex_search(name, match, kFrameRe)) {
        return {};
    }
    // Keep the original zero-padded frame number so the jpg name matches on disk.
    const std::string image_name = "frame_" + match[1].str() + "_all_keypoints.jpg";
    return (mesh.parent_path() / image_name).string();
}

Transform compute_transform(const Frame& hands) {
    // Whole-scene bounds: used only to centre the scene on the grid.
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const HandData& hand : hands) {
        for (const glm::vec3& position : hand.verts) {
            low = glm::min(low, position);
            high = glm::max(high, position);
        }
    }
    const float center_x = (high.x + low.x) / 2.0f;
    const float floor_y = low.y; // sit on the grid
    const float center_z = (high.z + low.z) / 2.0f;

    // Scale from a single reference hand's size, not the whole-scene extent. The
    // meshes are metric, so one hand is a stable size cue, whereas the full-scene
    // bounding box balloons when hands sit far apart, which would shrink every
    // hand to nothing. This keeps a hand a consistent on-screen size across takes.
    glm::vec3 hand_low(std::numeric_limits<float>::max());
    glm::vec3 hand_high(std::numeric_limits<float>::lowest());
    if (!hands.empty()) {
        for (const glm::vec3& position : hands.front().verts) {
            hand_low = glm::min(hand_low, position);
            hand_high = glm::max(hand_high, position);
        }
    }
    const glm::vec3 hand_extent = hand_high - hand_low;
    float span = std::max({hand_extent.x, hand_extent.y, hand_extent.z});
    if (!(span > 0.0f)) {
        span = 1.0f;
    }
    return Transform{glm::vec3(-center_x, -floor_y, -center_z), MODEL_FIT_SPAN / span};
}

float reference_depth(const Frame& hands) {
    double sum = 0.0;
    std::size_t count = 0;
    for (const HandData& hand : hands) {
        for (const glm::vec3& position : hand.verts) {
            sum += position.z;
            ++count;
        }
    }
    return count == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(count));
}

MeshSequenceLoader::MeshSequenceLoader(const std::string& folder, int workers) : folder_(folder) {
    frame_paths_ = discover_frames(folder);
    frame_count_ = static_cast<int>(frame_paths_.size());
    frames_.assign(static_cast<std::size_t>(frame_count_), nullptr);
    for (const std::vector<std::string>& hands : frame_paths_) {
        mesh_total_ += static_cast<int>(hands.size());
    }

    const int worker_count = std::max(1, std::min(workers, frame_count_));
    threads_.reserve(static_cast<std::size_t>(worker_count));
    for (int index = 0; index < worker_count; ++index) {
        threads_.emplace_back([this] { worker(); });
    }
}

MeshSequenceLoader::~MeshSequenceLoader() { stop(); }

void MeshSequenceLoader::stop() {
    stop_.store(true);
    for (std::thread& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}

bool MeshSequenceLoader::frame_ready(int index) const {
    if (index < 0 || index >= frame_count_) {
        return false;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    return frames_[static_cast<std::size_t>(index)] != nullptr;
}

std::shared_ptr<const Frame> MeshSequenceLoader::get(int index) const {
    if (index < 0 || index >= frame_count_) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    return frames_[static_cast<std::size_t>(index)];
}

void MeshSequenceLoader::prioritize(int index) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (index >= 0 && index < frame_count_ && frames_[static_cast<std::size_t>(index)] == nullptr && claimed_.find(index) == claimed_.end()) {
        priority_.push_front(index);
    }
}

int MeshSequenceLoader::claim_next() {
    std::lock_guard<std::mutex> guard(mutex_);
    while (!priority_.empty()) {
        const int index = priority_.front();
        priority_.pop_front();
        if (frames_[static_cast<std::size_t>(index)] == nullptr && claimed_.find(index) == claimed_.end()) {
            claimed_.insert(index);
            return index;
        }
    }
    while (cursor_ < frame_count_) {
        const int index = cursor_++;
        if (frames_[static_cast<std::size_t>(index)] == nullptr && claimed_.find(index) == claimed_.end()) {
            claimed_.insert(index);
            return index;
        }
    }
    return -1;
}

void MeshSequenceLoader::worker() {
    while (!stop_.load()) {
        const int index = claim_next();
        if (index < 0) {
            return; // everything is claimed or loaded; nothing left to do
        }
        auto hands = std::make_shared<Frame>();
        for (const std::string& path : frame_paths_[static_cast<std::size_t>(index)]) {
            HMesh mesh = load_hmesh(path);
            HandData hand;
            hand.verts = std::move(mesh.verts);
            hand.joints = std::move(mesh.joints);
            hand.is_right = mesh.is_right;
            hands->push_back(std::move(hand));
            meshes_loaded_.fetch_add(1);
        }
        std::lock_guard<std::mutex> guard(mutex_);
        frames_[static_cast<std::size_t>(index)] = std::move(hands);
    }
}

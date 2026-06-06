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
    // Keep the original zero-padded frame number so the name matches on disk.
    // Both jpg and png are supported; prefer jpg, fall back to png.
    const std::string base = "frame_" + match[1].str() + "_all_keypoints";
    const fs::path dir = mesh.parent_path();
    const fs::path jpg = dir / (base + ".jpg");
    if (fs::exists(jpg)) {
        return jpg.string();
    }
    const fs::path png = dir / (base + ".png");
    if (fs::exists(png)) {
        return png.string();
    }
    // Default to the jpg path; the caller treats a missing file as no image.
    return jpg.string();
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

MeshSequence::MeshSequence(const std::string& folder) : folder_(folder) {
    frame_paths_ = discover_frames(folder);
    frame_count_ = static_cast<int>(frame_paths_.size());
}

Frame MeshSequence::load_frame(int index) const {
    Frame hands;
    if (index < 0 || index >= frame_count_) {
        return hands;
    }
    const std::vector<std::string>& paths = frame_paths_[static_cast<std::size_t>(index)];
    hands.reserve(paths.size());
    for (const std::string& path : paths) {
        HMesh mesh = load_hmesh(path);
        HandData hand;
        hand.verts = std::move(mesh.verts);
        hand.joints = std::move(mesh.joints);
        hand.is_right = mesh.is_right;
        hands.push_back(std::move(hand));
    }
    return hands;
}

#include "data/mesh_sequence.h"

#include "data/geometry.h"
#include "data/hand_export.h"
#include "data/mano_model.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>

namespace {
    namespace fs = std::filesystem;
} // namespace

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

namespace {
    // Runs the MANO forward pass for every row of an export file and groups the
    // results by frame number (sorted; a frame number can be absent, e.g. every
    // hand in it was filtered out upstream, so this densifies by sorting the
    // keys rather than assuming a dense 1..N range). frame_numbers[i] is the
    // pipeline's own frame number for playback index i -- not generally i+1,
    // since gaps mean the two can diverge -- which is what frame_image_path
    // needs to find the matching source frame on disk. One-time cost at open:
    // cheap even for a full trial's worth of hands, since each forward pass is
    // just a handful of small matrix ops.
    void build_export_frames(const std::string& path, std::vector<Frame>& frames, std::vector<int>& frame_numbers) {
        std::vector<HandExportRow> rows = load_hand_export(path);

        std::map<int, Frame> by_frame;
        for (const HandExportRow& row : rows) {
            const ManoHand hand = mano_forward(row.params);
            HandData data;
            data.verts = hand.verts;
            data.joints = hand.joints;
            data.is_right = row.params.is_right;
            data.hand_track_id = row.hand_track_id;
            data.label = row.label;
            by_frame[row.frame].push_back(std::move(data));
        }

        frames.reserve(by_frame.size());
        frame_numbers.reserve(by_frame.size());
        for (auto& [frame_number, hands] : by_frame) {
            frame_numbers.push_back(frame_number);
            frames.push_back(std::move(hands));
        }
    }
} // namespace

MeshSequence::MeshSequence(const std::string& path) : path_(path) {
    build_export_frames(path, frames_, frame_numbers_);
    frame_count_ = static_cast<int>(frames_.size());
}

Frame MeshSequence::load_frame(int index) const {
    if (index < 0 || index >= frame_count_) {
        return {};
    }
    return frames_[static_cast<std::size_t>(index)];
}

std::string MeshSequence::frame_image_path(int index) const {
    if (index < 0 || index >= frame_count_) {
        return {};
    }
    // Convention: a plain source-video frame for a hand-motion export lives in
    // a sibling ``frames/<export-file-stem>/`` folder, named ``frame_%05d.jpg``
    // (matching infant_grasp_pipeline's own frame-extraction naming) -- e.g.
    // GZ65_T1_BabyView.hexport looks in frames/GZ65_T1_BabyView/ next to it.
    // Naming the subfolder after the export file (rather than one shared
    // frames/ folder) disambiguates multiple .hexport files sitting in the
    // same directory. Nothing in the .hexport file itself records this path --
    // it is purely a packaging convention between however the export and its
    // frames are delivered together, so a missing folder or frame is not an
    // error, just "no image" (the caller treats an empty path that way).
    const fs::path export_file(path_);
    const fs::path frames_dir = export_file.parent_path() / "frames" / export_file.stem();
    const int frame_number = frame_numbers_[static_cast<std::size_t>(index)];

    char name[32];
    std::snprintf(name, sizeof(name), "frame_%05d.jpg", frame_number);
    std::error_code error;
    fs::path candidate = frames_dir / name;
    if (fs::exists(candidate, error)) {
        return candidate.string();
    }
    std::snprintf(name, sizeof(name), "frame_%05d.png", frame_number);
    candidate = frames_dir / name;
    if (fs::exists(candidate, error)) {
        return candidate.string();
    }
    return {};
}

#include "data/mesh_sequence.h"

#include "data/geometry.h"
#include "data/hand_export.h"
#include "data/mano_model.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <map>

namespace {
    namespace fs = std::filesystem;

    // Index into HandData-adjacent per-frame flag arrays <-> the .hexport
    // column it comes from. Order must match kFlagLayerCount in the header and
    // the FlagLayer color table in ui/gui.cpp.
    constexpr std::array<std::int8_t HandExportRow::*, kFlagLayerCount> FLAG_COLUMNS = {
        &HandExportRow::flag_same_side_infant_conflict,
        &HandExportRow::flag_same_side_infant_unknown_conflict,
        &HandExportRow::flag_translation_jump,
        &HandExportRow::flag_pose_rotation_jump,
        &HandExportRow::flag_scale_jump,
        &HandExportRow::flag_track_contaminated,
        &HandExportRow::flag_track_fragmented,
    };
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
    void build_export_frames(
        const std::string& path,
        std::vector<Frame>& frames,
        std::vector<int>& frame_numbers,
        std::vector<std::array<bool, kFlagLayerCount>>& frame_flags_all,
        std::vector<std::array<bool, kFlagLayerCount>>& frame_flags_infant_only
    ) {
        std::vector<HandExportRow> rows = load_hand_export(path);

        std::map<int, Frame> by_frame;
        // Flags active for any hand in the frame (for per-track coloring mode)
        std::map<int, std::array<bool, kFlagLayerCount>> flags_all_by_frame;
        // Flags active only for infant-classified hands (when per-track coloring is off)
        std::map<int, std::array<bool, kFlagLayerCount>> flags_infant_by_frame;
        for (const HandExportRow& row : rows) {
            const ManoHand hand = mano_forward(row.params);
            HandData data;
            data.verts = hand.verts;
            data.joints = hand.joints;
            data.is_right = row.params.is_right;
            data.hand_track_id = row.hand_track_id;
            data.label = row.label;
            by_frame[row.frame].push_back(std::move(data));

            std::array<bool, kFlagLayerCount>& flags_all = flags_all_by_frame[row.frame];
            std::array<bool, kFlagLayerCount>& flags_infant = flags_infant_by_frame[row.frame];
            const bool is_infant = (row.label == "infant");

            for (int i = 0; i < kFlagLayerCount; ++i) {
                if (row.*FLAG_COLUMNS[static_cast<std::size_t>(i)] != 0) {
                    flags_all[static_cast<std::size_t>(i)] = true;
                    if (is_infant) {
                        flags_infant[static_cast<std::size_t>(i)] = true;
                    }
                }
            }
        }

        frames.reserve(by_frame.size());
        frame_numbers.reserve(by_frame.size());
        frame_flags_all.reserve(by_frame.size());
        frame_flags_infant_only.reserve(by_frame.size());
        for (auto& [frame_number, hands] : by_frame) {
            frame_numbers.push_back(frame_number);
            frames.push_back(std::move(hands));
            frame_flags_all.push_back(flags_all_by_frame[frame_number]);
            frame_flags_infant_only.push_back(flags_infant_by_frame[frame_number]);
        }
    }

    fs::path resolve_frames_dir(const fs::path& export_file) {
        const std::string stem = export_file.stem().string();
        std::string frames_key = stem;
        const auto last_sep = stem.rfind("__");
        if (last_sep != std::string::npos) {
            frames_key = stem.substr(0, last_sep);
        }

        const fs::path parent = export_file.parent_path();
        const fs::path grandparent = parent.parent_path();

        const std::vector<fs::path> candidates = {
            grandparent / "frames" / frames_key,
            parent / "frames" / frames_key,
            parent / frames_key,
            parent / "frames" / stem,
            grandparent / "frames" / stem,
        };

        for (const auto& candidate : candidates) {
            std::error_code ec;
            if (fs::is_directory(candidate, ec)) {
                return candidate;
            }
        }

        // Default candidate if directory does not exist yet (e.g. pending remote download)
        return parent / "frames" / frames_key;
    }
} // namespace

MeshSequence::MeshSequence(const std::string& path) : path_(path) {
    frames_dir_ = resolve_frames_dir(fs::path(path)).string();
    build_export_frames(path, frames_, frame_numbers_, frame_flags_all_, frame_flags_infant_only_);
    frame_count_ = static_cast<int>(frames_.size());
}

Frame MeshSequence::load_frame(int index) const {
    if (index < 0 || index >= frame_count_) {
        return {};
    }
    return frames_[static_cast<std::size_t>(index)];
}

bool MeshSequence::is_flagged(int index, int flag_index, bool per_track_coloring) const {
    if (index < 0 || index >= frame_count_ || flag_index < 0 || flag_index >= kFlagLayerCount) {
        return false;
    }
    const auto& flags = per_track_coloring ? frame_flags_all_ : frame_flags_infant_only_;
    return flags[static_cast<std::size_t>(index)][static_cast<std::size_t>(flag_index)];
}

std::string MeshSequence::frame_image_path(int index) const {
    if (index < 0 || index >= frame_count_ || frames_dir_.empty()) {
        return {};
    }
    const int frame_number = frame_numbers_[static_cast<std::size_t>(index)];
    const fs::path frames_path(frames_dir_);

    char name[64];
    constexpr const char* formats[] = {"frame_%05d.jpg", "frame_%05d.jpeg", "frame_%05d.png", "frame_%04d.jpg", "frame_%04d.png"};
    for (const char* fmt : formats) {
        std::snprintf(name, sizeof(name), fmt, frame_number);
        const fs::path candidate = frames_path / name;
        std::error_code error;
        if (fs::exists(candidate, error)) {
            return candidate.string();
        }
    }
    return {};
}

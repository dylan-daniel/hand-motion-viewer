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

    // A mesh folder ``.../data/<subject>/<view>`` is tracked by a sibling CSV
    // ``.../data/tracking/tracks_<subject>_<view>.csv``. Returns an empty path if
    // the layout does not match.
    fs::path tracking_csv_path(const std::string& folder) {
        const fs::path mesh_folder(folder);
        const fs::path subject_dir = mesh_folder.parent_path();
        if (subject_dir.empty() || subject_dir.filename().empty()) {
            return {};
        }
        const std::string view = mesh_folder.filename().string();
        const std::string subject = subject_dir.filename().string();
        const std::string name = "tracks3_" + subject + "_" + view + ".csv";
        return subject_dir.parent_path().parent_path() / "tracking" / name;
    }

    // Split a CSV line on commas.
    std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> fields;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = line.find(',', start);
            fields.push_back(line.substr(start, comma - start));
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return fields;
    }

    // Parse ``tracks_*.csv`` into (frame, idx) → TrackInfo. Columns are located by
    // their header name rather than a fixed position, since some files carry an
    // extra ``is_duplicate`` column and others do not. ``is_duplicate`` comes
    // straight from the CSV; a missing column leaves every detection non-duplicate.
    std::map<std::pair<int, int>, TrackInfo> load_track_ids(const std::string& folder) {
        std::map<std::pair<int, int>, TrackInfo> track_ids;
        const fs::path csv = tracking_csv_path(folder);
        if (csv.empty() || !fs::exists(csv)) {
            return track_ids;
        }
        std::ifstream stream(csv);
        if (!stream) {
            return track_ids;
        }
        std::string line;
        if (!std::getline(stream, line)) {
            return track_ids;
        }
        // Map the columns we need from the header; -1 means the column is absent.
        const std::vector<std::string> header = split_csv(line);
        int frame_col = -1;
        int idx_col = -1;
        int track_col = -1;
        int duplicate_col = -1;
        for (int column = 0; column < static_cast<int>(header.size()); ++column) {
            const std::string& name = header[static_cast<std::size_t>(column)];
            if (name == "frame") {
                frame_col = column;
            } else if (name == "idx") {
                idx_col = column;
            } else if (name == "track_id") {
                track_col = column;
            } else if (name == "is_duplicate") {
                duplicate_col = column;
            }
        }
        if (frame_col < 0 || idx_col < 0 || track_col < 0) {
            return track_ids;
        }
        const int min_fields = std::max({frame_col, idx_col, track_col, duplicate_col}) + 1;
        while (std::getline(stream, line)) {
            if (line.empty()) {
                continue;
            }
            const std::vector<std::string> fields = split_csv(line);
            if (static_cast<int>(fields.size()) < min_fields) {
                continue;
            }
            try {
                const int frame_number = std::stoi(fields[static_cast<std::size_t>(frame_col)]);
                const int slot = std::stoi(fields[static_cast<std::size_t>(idx_col)]);
                TrackInfo info;
                info.track_id = std::stoi(fields[static_cast<std::size_t>(track_col)]);
                info.is_duplicate = duplicate_col >= 0 && std::stoi(fields[static_cast<std::size_t>(duplicate_col)]) != 0;
                track_ids[{frame_number, slot}] = info;
            } catch (const std::exception&) {
                continue;
            }
        }
        return track_ids;
    }
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

// Per-hand depth-stabilization factor used by the renderer: each hand is scaled
// about the origin by reference_depth / its-own-mean-depth. The scene transform
// must measure the *stabilized* geometry, so it applies the same factor here.
static float stabilization_scale(const HandData& hand, float reference) {
    double sum = 0.0;
    for (const glm::vec3& position : hand.verts) {
        sum += position.z;
    }
    const float depth = hand.verts.empty() ? 0.0f : static_cast<float>(sum / hand.verts.size());
    return depth != 0.0f ? reference / depth : 1.0f;
}

Transform compute_transform(const MeshSequence& sequence) {
    // Centering X/Z and the fit scale come from frame 0 so they stay fixed across
    // playback. The renderer scales each hand about the origin by reference_depth /
    // depth before this transform's offset is applied, so everything here is
    // measured on those stabilized positions, not the raw verts. The reference
    // depth is frame 0's, matching the value the renderer caches and reuses.
    const Frame frame_zero = sequence.load_frame(0);
    const float reference = reference_depth(frame_zero);

    // Whole-scene X/Z bounds of frame 0's stabilized hands: used only to centre.
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const HandData& hand : frame_zero) {
        const float scale = stabilization_scale(hand, reference);
        for (const glm::vec3& position : hand.verts) {
            const glm::vec3 stabilized = position * scale;
            low = glm::min(low, stabilized);
            high = glm::max(high, stabilized);
        }
    }
    const float center_x = (high.x + low.x) / 2.0f;
    const float center_z = (high.z + low.z) / 2.0f;

    // Floor (Y): the lowest stabilized point across *every* frame, so no frame
    // ever sinks below the grid plane (only frame 0 would sit on the grid if we
    // used its bounds alone, and a later frame could reach lower).
    float floor_y = std::numeric_limits<float>::max();
    for (int index = 0; index < sequence.frame_count(); ++index) {
        const Frame hands = sequence.load_frame(index);
        for (const HandData& hand : hands) {
            const float scale = stabilization_scale(hand, reference);
            for (const glm::vec3& position : hand.verts) {
                floor_y = std::min(floor_y, position.y * scale);
            }
        }
    }
    if (floor_y == std::numeric_limits<float>::max()) {
        floor_y = 0.0f; // no verts in any frame
    }

    // Scale from a single reference hand's size, not the whole-scene extent. The
    // meshes are metric, so one hand is a stable size cue, whereas the full-scene
    // bounding box balloons when hands sit far apart, which would shrink every
    // hand to nothing. This keeps a hand a consistent on-screen size across takes.
    // Measure the stabilized reference hand so the fit matches what's drawn.
    glm::vec3 hand_low(std::numeric_limits<float>::max());
    glm::vec3 hand_high(std::numeric_limits<float>::lowest());
    if (!frame_zero.empty()) {
        const float scale = stabilization_scale(frame_zero.front(), reference);
        for (const glm::vec3& position : frame_zero.front().verts) {
            const glm::vec3 stabilized = position * scale;
            hand_low = glm::min(hand_low, stabilized);
            hand_high = glm::max(hand_high, stabilized);
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
    track_ids_ = load_track_ids(folder);
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
        // Resolve the track id and duplicate flag from the parsed tracking CSV
        // using the file's frame number and hand slot (frame_NNNN_<slot>.hmesh).
        std::smatch match;
        const std::string name = fs::path(path).filename().string();
        if (std::regex_search(name, match, kFrameRe)) {
            const int frame_number = std::stoi(match[1].str());
            const int slot = std::stoi(match[2].str());
            const auto found = track_ids_.find({frame_number, slot});
            if (found != track_ids_.end()) {
                hand.track_id = found->second.track_id;
                hand.is_duplicate = found->second.is_duplicate;
            }
        }
        hands.push_back(std::move(hand));
    }
    return hands;
}

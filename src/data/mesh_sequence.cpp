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
#include <set>

namespace {
    namespace fs = std::filesystem;

    // frame_0001_0.hmesh → frame number 0001, hand slot 0.
    const std::regex kFrameRe(R"(frame_(\d+)_(\d+)\.hmesh$)", std::regex::icase);

    // Where a mesh folder's tracking CSVs live and the subject/view that name them.
    // A mesh folder ``.../data/<subject>/<view>`` is tracked by sibling CSVs in
    // ``.../tracking`` named ``tracks<N>_<subject>_<view>.csv``.
    struct TrackingLocation {
        fs::path dir;        // the tracking/ directory
        std::string subject; // subject component of the file name
        std::string view;    // view component of the file name
        bool valid = false;
    };

    TrackingLocation tracking_location(const std::string& folder) {
        const fs::path mesh_folder(folder);
        const fs::path subject_dir = mesh_folder.parent_path();
        if (subject_dir.empty() || subject_dir.filename().empty()) {
            return {};
        }
        TrackingLocation location;
        location.view = mesh_folder.filename().string();
        location.subject = subject_dir.filename().string();
        location.dir = subject_dir.parent_path().parent_path() / "tracking";
        location.valid = true;
        return location;
    }

    // File-name prefix for a tracking version: version 1 is the unnumbered
    // ``tracks_`` set; later versions carry the number (``tracks2_``, ``tracks3_``).
    std::string tracking_prefix(int version) { return version <= 1 ? "tracks_" : "tracks" + std::to_string(version) + "_"; }

    // Absolute path of the source's tracking CSV for ``folder``. Empty if the mesh
    // folder layout does not match the expected ``.../<subject>/<view>`` shape. The
    // ``_linked`` variant carries an extra suffix before ``.csv``.
    fs::path tracking_csv_path(const std::string& folder, const TrackingSource& source) {
        const TrackingLocation location = tracking_location(folder);
        if (!location.valid) {
            return {};
        }
        std::string name = tracking_prefix(source.version) + location.subject + "_" + location.view;
        if (source.linked) {
            name += "_linked";
        }
        name += ".csv";
        return location.dir / name;
    }

    // The tracking sources available for ``folder``, sorted by version then with the
    // ``_linked`` variant after its plain counterpart. A file
    // ``tracks<N>_<subject>_<view>[_linked].csv`` contributes version N (1 for the
    // unnumbered ``tracks_`` set), linked when the ``_linked`` suffix is present.
    std::vector<TrackingSource> discover_tracking_sources(const std::string& folder) {
        std::vector<TrackingSource> sources;
        const TrackingLocation location = tracking_location(folder);
        if (!location.valid) {
            return sources;
        }
        const std::string base = "_" + location.subject + "_" + location.view; // between prefix and (_linked)?.csv
        const std::regex prefix_re(R"(^tracks(\d*)$)");
        std::error_code error;
        for (const fs::directory_entry& entry : fs::directory_iterator(location.dir, error)) {
            if (!entry.is_regular_file()) {
                continue;
            }
            std::string stem = entry.path().filename().string();
            if (!stem.ends_with(".csv")) {
                continue;
            }
            stem.resize(stem.size() - 4); // drop ".csv"
            const bool linked = stem.ends_with("_linked");
            if (linked) {
                stem.resize(stem.size() - 7); // drop "_linked"
            }
            if (!stem.ends_with(base)) {
                continue;
            }
            const std::string prefix = stem.substr(0, stem.size() - base.size()); // "tracks<N>"
            std::smatch match;
            if (!std::regex_match(prefix, match, prefix_re)) {
                continue;
            }
            sources.push_back({match[1].str().empty() ? 1 : std::stoi(match[1].str()), linked});
        }
        std::sort(sources.begin(), sources.end(), [](const TrackingSource& a, const TrackingSource& b) {
            return a.version != b.version ? a.version < b.version : (!a.linked && b.linked);
        });
        sources.erase(std::unique(sources.begin(), sources.end()), sources.end());
        return sources;
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
    // their header name rather than a fixed position, since files vary in which
    // columns they carry. The colour id is ``hand_id`` for a linked source and
    // ``track_id`` otherwise; ``is_duplicate`` comes straight from the CSV (a
    // missing column leaves every detection non-duplicate).
    std::map<std::pair<int, int>, TrackInfo> load_track_ids(const std::string& folder, const TrackingSource& source) {
        std::map<std::pair<int, int>, TrackInfo> track_ids;
        const fs::path csv = tracking_csv_path(folder, source);
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
        const std::string color_column = source.linked ? "hand_id" : "track_id";
        int frame_col = -1;
        int idx_col = -1;
        int color_col = -1;
        int duplicate_col = -1;
        for (int column = 0; column < static_cast<int>(header.size()); ++column) {
            const std::string& name = header[static_cast<std::size_t>(column)];
            if (name == "frame") {
                frame_col = column;
            } else if (name == "idx") {
                idx_col = column;
            } else if (name == color_column) {
                color_col = column;
            } else if (name == "is_duplicate") {
                duplicate_col = column;
            }
        }
        if (frame_col < 0 || idx_col < 0 || color_col < 0) {
            return track_ids;
        }
        const int min_fields = std::max({frame_col, idx_col, color_col, duplicate_col}) + 1;
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
                info.color_id = std::stoi(fields[static_cast<std::size_t>(color_col)]);
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
    tracking_sources_ = discover_tracking_sources(folder);
    // Default to the first available source (lowest version, non-linked); fall back
    // to {1, false} when none are found so the path logic still resolves a
    // (possibly missing) ``tracks_`` file.
    tracking_source_ = tracking_sources_.empty() ? TrackingSource{} : tracking_sources_.front();
    track_ids_ = load_track_ids(folder, tracking_source_);
}

void MeshSequence::set_tracking_source(TrackingSource source) {
    if (source == tracking_source_) {
        return;
    }
    tracking_source_ = source;
    track_ids_ = load_track_ids(folder_, source);
}

void MeshSequence::prefer_tracking_source(TrackingSource source) {
    if (std::find(tracking_sources_.begin(), tracking_sources_.end(), source) != tracking_sources_.end()) {
        set_tracking_source(source);
    }
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
            hand.slot = slot;
            const auto found = track_ids_.find({frame_number, slot});
            if (found != track_ids_.end()) {
                hand.track_id = found->second.color_id;
            }
            // A manual override wins over the CSV colour id for this hand.
            const auto overridden = overrides_.find({frame_number, slot});
            if (overridden != overrides_.end()) {
                hand.track_id = overridden->second;
            }
        }
        hands.push_back(std::move(hand));
    }

    // Recompute duplicates from the final colour ids (CSV ids with manual overrides
    // applied) instead of trusting the CSV's own flag, so manual relabeling stays
    // consistent. As the CSV does, the first hand (in slot order) to use a colour id
    // is the canonical one and stays non-duplicate; any later hand reusing that id is
    // the duplicate. Flagging only the extras means one hand survives "hide
    // duplicates", while assigning a hand an id already used in the frame still marks
    // it (and relabeling away from a clash clears it).
    std::set<int> seen_ids;
    for (HandData& hand : hands) {
        hand.is_duplicate = hand.track_id >= 0 && !seen_ids.insert(hand.track_id).second;
    }
    return hands;
}

int MeshSequence::frame_number(int index) const {
    if (index < 0 || index >= frame_count_) {
        return -1;
    }
    const std::vector<std::string>& paths = frame_paths_[static_cast<std::size_t>(index)];
    if (paths.empty()) {
        return -1;
    }
    std::smatch match;
    const std::string name = fs::path(paths.front()).filename().string();
    if (!std::regex_search(name, match, kFrameRe)) {
        return -1;
    }
    return std::stoi(match[1].str());
}

void MeshSequence::set_override(int frame, int slot, int hand_id) { overrides_[{frame, slot}] = hand_id; }

void MeshSequence::clear_override(int frame, int slot) { overrides_.erase({frame, slot}); }

void MeshSequence::push_undo_state() {
    undo_stack_.push_back(overrides_);
    // Cap the history so a long session can't grow it without bound.
    constexpr std::size_t max_undo = 200;
    if (undo_stack_.size() > max_undo) {
        undo_stack_.erase(undo_stack_.begin());
    }
}

bool MeshSequence::undo() {
    if (undo_stack_.empty()) {
        return false;
    }
    overrides_ = std::move(undo_stack_.back());
    undo_stack_.pop_back();
    return true;
}

void MeshSequence::propagate_id(int from_frame, int old_id, int new_id) {
    if (old_id < 0 || old_id == new_id) {
        return;
    }
    // Walk every detection from this frame onward; one whose current colour id
    // (override if set, else the CSV's) matches the old id joins the relabel. We
    // read overrides_ but only ever add keys, so iterating track_ids_ stays valid.
    for (const auto& [key, info] : track_ids_) {
        if (key.first < from_frame) {
            continue;
        }
        const auto overridden = overrides_.find(key);
        const int current_id = overridden != overrides_.end() ? overridden->second : info.color_id;
        if (current_id == old_id) {
            overrides_[key] = new_id;
        }
    }
}

std::vector<int> MeshSequence::present_color_ids() const {
    // Union of the source CSV's colour ids and the manual overrides, with each
    // override masking the CSV id for its (frame, slot). Negatives are dropped.
    std::set<int> ids;
    for (const auto& [key, info] : track_ids_) {
        if (overrides_.find(key) == overrides_.end() && info.color_id >= 0) {
            ids.insert(info.color_id);
        }
    }
    for (const auto& [key, hand_id] : overrides_) {
        if (hand_id >= 0) {
            ids.insert(hand_id);
        }
    }
    return {ids.begin(), ids.end()};
}

std::string MeshSequence::save_truth_csv() const {
    // Re-read the active source CSV and rewrite it row-for-row into the sibling
    // tracking/truth folder, replacing the colour column (track_id, or hand_id for
    // a linked source) with the manual override for any row that has one. Keeping
    // the full schema makes the output a drop-in replacement for the source.
    const fs::path source = tracking_csv_path(folder_, tracking_source_);
    if (source.empty() || !fs::exists(source)) {
        return {};
    }
    std::ifstream input(source);
    if (!input) {
        return {};
    }
    std::string header_line;
    if (!std::getline(input, header_line)) {
        return {};
    }
    const std::vector<std::string> header = split_csv(header_line);
    const std::string color_column = tracking_source_.linked ? "hand_id" : "track_id";
    int frame_col = -1;
    int idx_col = -1;
    int color_col = -1;
    for (int column = 0; column < static_cast<int>(header.size()); ++column) {
        const std::string& name = header[static_cast<std::size_t>(column)];
        if (name == "frame") {
            frame_col = column;
        } else if (name == "idx") {
            idx_col = column;
        } else if (name == color_column) {
            color_col = column;
        }
    }
    if (frame_col < 0 || idx_col < 0 || color_col < 0) {
        return {};
    }

    const fs::path out_path = source.parent_path() / "truth" / source.filename();
    std::error_code error;
    fs::create_directories(out_path.parent_path(), error);
    std::ofstream output(out_path);
    if (!output) {
        return {};
    }
    output << header_line << "\n";
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= std::max({frame_col, idx_col, color_col})) {
            output << line << "\n";
            continue;
        }
        try {
            const int frame_number = std::stoi(fields[static_cast<std::size_t>(frame_col)]);
            const int slot = std::stoi(fields[static_cast<std::size_t>(idx_col)]);
            const auto overridden = overrides_.find({frame_number, slot});
            if (overridden != overrides_.end()) {
                fields[static_cast<std::size_t>(color_col)] = std::to_string(overridden->second);
            }
        } catch (const std::exception&) {
            // Leave a malformed row untouched.
        }
        for (std::size_t column = 0; column < fields.size(); ++column) {
            output << (column == 0 ? "" : ",") << fields[column];
        }
        output << "\n";
    }
    return out_path.string();
}

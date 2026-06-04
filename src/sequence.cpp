#include "sequence.h"

#include "geometry.h"

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

    // frame_0001_0.obj → frame number 0001, hand slot 0.
    const std::regex kFrameRe(R"(frame_(\d+)_(\d+)\.obj$)", std::regex::icase);

    // Hand colour as a 0..1 float triple, matched against parsed vertex colours.
    const glm::vec3 kHandColorFloat = {HAND_COLOR_8BIT.r / 255.0f, HAND_COLOR_8BIT.g / 255.0f, HAND_COLOR_8BIT.b / 255.0f};

    /// Read a whole file into a string in one pass (the string stays null
    /// terminated, so the strtof/strtol scanners below never run past the buffer).
    std::string read_file(const std::string& path) {
        std::ifstream obj_file(path, std::ios::binary | std::ios::ate);
        if (!obj_file) {
            return {};
        }
        const std::streamsize size = obj_file.tellg();
        if (size <= 0) {
            return {};
        }
        std::string data(static_cast<std::size_t>(size), '\0');
        obj_file.seekg(0);
        obj_file.read(data.data(), size);
        return data;
    }

    /// 64-bit FNV-1a hash of a byte range, used to intern identical topologies.
    std::uint64_t fnv1a(const char* begin, const char* end) {
        std::uint64_t hash = 1469598103934665603ull;
        for (const char* cursor = begin; cursor < end; ++cursor) {
            hash ^= static_cast<std::uint8_t>(*cursor);
            hash *= 1099511628211ull;
        }
        return hash;
    }

    /// Advance ``cursor`` past the current line's newline (handles the final line
    /// with no trailing newline).
    void skip_line(const char*& cursor, const char* end) {
        while (cursor < end && *cursor != '\n') {
            ++cursor;
        }
        if (cursor < end) {
            ++cursor;
        }
    }

    /// Read an obj's vertex positions plus a key identifying its face block. The key
    /// is a hash of the raw face bytes, so identical topologies intern without
    /// re-parsing their faces. Relies on the trimesh export layout: a comment, then
    /// all ``v`` lines, then all ``f`` lines.
    ///
    /// Parsing walks the raw buffer with strtof — no per-line std::string or
    /// std::istringstream allocation, which is what made the first cut slow.
    std::pair<std::vector<glm::vec3>, std::uint64_t> parse_positions(const std::string& path) {
        const std::string data = read_file(path);
        const char* begin = data.data();
        const char* end = begin + data.size();

        const std::size_t face_pos = data.find("\nf ");
        const char* vertex_end = face_pos == std::string::npos ? end : begin + face_pos;
        const char* face_begin = face_pos == std::string::npos ? end : begin + face_pos + 1;
        const std::uint64_t key = fnv1a(face_begin, end);

        std::vector<glm::vec3> positions;
        positions.reserve(800); // a MANO hand is ~778 vertices
        const char* cursor = begin;
        while (cursor < vertex_end) {
            if (cursor[0] == 'v' && cursor + 1 < vertex_end && cursor[1] == ' ') {
                char* scan = const_cast<char*>(cursor + 2);
                const float x = std::strtof(scan, &scan);
                const float y = std::strtof(scan, &scan);
                const float z = std::strtof(scan, &scan);
                positions.emplace_back(x, y, z);
                cursor = scan;
            }
            skip_line(cursor, vertex_end);
        }
        return {std::move(positions), key};
    }

    /// Fully parse one obj's faces, vertex colours and hand/joint split. Run only
    /// once per distinct topology, so this stays off the per-frame hot path.
    Topology parse_topology(const std::string& path) {
        const std::string data = read_file(path);
        const char* begin = data.data();
        const char* end = begin + data.size();

        Topology topology;
        std::vector<std::uint8_t> is_hand_vertex;
        topology.colors.reserve(800);
        is_hand_vertex.reserve(800);
        topology.faces.reserve(1600);

        const char* cursor = begin;
        while (cursor < end) {
            if (cursor[0] == 'v' && cursor + 1 < end && cursor[1] == ' ') {
                char* scan = const_cast<char*>(cursor + 2);
                std::strtof(scan, &scan); // x
                std::strtof(scan, &scan); // y
                std::strtof(scan, &scan); // z
                const float r = std::strtof(scan, &scan);
                const float g = std::strtof(scan, &scan);
                const float b = std::strtof(scan, &scan);
                topology.colors.emplace_back(r, g, b, 1.0f);
                const bool is_hand =
                    std::fabs(r - kHandColorFloat.r) < 1e-3f && std::fabs(g - kHandColorFloat.g) < 1e-3f && std::fabs(b - kHandColorFloat.b) < 1e-3f;
                is_hand_vertex.push_back(is_hand ? 1 : 0);
                cursor = scan;
            } else if (cursor[0] == 'f' && cursor + 1 < end && cursor[1] == ' ') {
                char* scan = const_cast<char*>(cursor + 2);
                int indices[3] = {0, 0, 0};
                for (int corner = 0; corner < 3; ++corner) {
                    indices[corner] = static_cast<int>(std::strtol(scan, &scan, 10)) - 1; // obj is 1-based
                    // Skip any /vt/vn part so the next strtol starts at the next index.
                    while (scan < end && *scan != ' ' && *scan != '\n' && *scan != '\r' && *scan != '\0') {
                        ++scan;
                    }
                }
                topology.faces.emplace_back(indices[0], indices[1], indices[2]);
                cursor = scan;
            }
            skip_line(cursor, end);
        }

        topology.hand_face_mask.reserve(topology.faces.size());
        for (const glm::ivec3& face : topology.faces) {
            const bool is_hand = is_hand_vertex[static_cast<std::size_t>(face[0])] || is_hand_vertex[static_cast<std::size_t>(face[1])] ||
                is_hand_vertex[static_cast<std::size_t>(face[2])];
            topology.hand_face_mask.push_back(is_hand ? 1 : 0);
        }
        return topology;
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
    // Keep the original zero-padded frame number so the jpg name matches on disk.
    const std::string image_name = "frame_" + match[1].str() + "_all_keypoints.jpg";
    return (mesh.parent_path() / image_name).string();
}

Transform compute_transform(const Frame& hands) {
    // Whole-scene bounds: used only to centre the scene on the grid.
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const HandData& hand : hands) {
        for (const glm::vec3& position : hand.positions) {
            low = glm::min(low, position);
            high = glm::max(high, position);
        }
    }
    const float center_x = (high.x + low.x) / 2.0f;
    const float floor_y = low.y; // sit on the grid
    const float center_z = (high.z + low.z) / 2.0f;

    // Scale from a single reference hand's size, not the whole-scene extent. The
    // meshes are metric, so one hand is a stable size cue, whereas the full-scene
    // bounding box balloons when hands sit far apart (e.g. the OHView overlay's
    // monocular depth disagreement), which would shrink every hand to nothing.
    // This keeps a hand the same on-screen size as the single-mesh view
    // (center_model) and consistent across takes.
    glm::vec3 hand_low(std::numeric_limits<float>::max());
    glm::vec3 hand_high(std::numeric_limits<float>::lowest());
    if (!hands.empty()) {
        for (const glm::vec3& position : hands.front().positions) {
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
        for (const glm::vec3& position : hand.positions) {
            sum += position.z;
            ++count;
        }
    }
    return count == 0 ? 0.0f : static_cast<float>(sum / static_cast<double>(count));
}

SequenceLoader::SequenceLoader(const std::string& folder, int workers, std::shared_ptr<const CrossViewOverlay> overlay) :
    folder_(folder), overlay_(std::move(overlay)) {
    frame_paths_ = discover_frames(folder);
    frame_count_ = static_cast<int>(frame_paths_.size());
    frames_.assign(static_cast<std::size_t>(frame_count_), nullptr);
    for (const std::vector<std::string>& hands : frame_paths_) {
        mesh_total_ += static_cast<int>(hands.size());
    }
    // The folded-in OHView hands are parsed alongside each frame's BabyView hands,
    // so count them toward the load totals too.
    if (overlay_) {
        mesh_total_ += overlay_->mesh_total();
    }

    const int worker_count = std::max(1, std::min(workers, frame_count_));
    threads_.reserve(static_cast<std::size_t>(worker_count));
    for (int index = 0; index < worker_count; ++index) {
        threads_.emplace_back([this] { worker(); });
    }
}

SequenceLoader::~SequenceLoader() { stop(); }

void SequenceLoader::stop() {
    stop_.store(true);
    for (std::thread& thread : threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    threads_.clear();
}

bool SequenceLoader::frame_ready(int index) const {
    if (index < 0 || index >= frame_count_) {
        return false;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    return frames_[static_cast<std::size_t>(index)] != nullptr;
}

std::shared_ptr<const Frame> SequenceLoader::get(int index) const {
    if (index < 0 || index >= frame_count_) {
        return nullptr;
    }
    std::lock_guard<std::mutex> guard(mutex_);
    return frames_[static_cast<std::size_t>(index)];
}

void SequenceLoader::prioritize(int index) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (index >= 0 && index < frame_count_ && frames_[static_cast<std::size_t>(index)] == nullptr && claimed_.find(index) == claimed_.end()) {
        priority_.push_front(index);
    }
}

int SequenceLoader::claim_next() {
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

std::shared_ptr<const Topology> SequenceLoader::get_topology(std::uint64_t key, const std::string& path) {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found = topologies_.find(key);
        if (found != topologies_.end()) {
            return found->second;
        }
    }
    auto parsed = std::make_shared<const Topology>(parse_topology(path));
    std::lock_guard<std::mutex> guard(mutex_);
    auto [iterator, inserted] = topologies_.emplace(key, parsed);
    return iterator->second;
}

std::shared_ptr<const Topology> SequenceLoader::get_tinted_topology(std::uint64_t key, const std::string& path) {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found = tinted_topologies_.find(key);
        if (found != tinted_topologies_.end()) {
            return found->second;
        }
    }
    // Start from the untinted topology, then recolour the hand-surface vertices
    // with the overlay tint (joint-skeleton vertices keep their vivid colours).
    auto base = get_topology(key, path);
    auto tinted = std::make_shared<Topology>(*base);
    const glm::vec4 tint = overlay_->tint();
    for (std::size_t vertex = 0; vertex < tinted->colors.size(); ++vertex) {
        const glm::vec4& original = tinted->colors[vertex];
        const bool is_hand = std::fabs(original.r - kHandColorFloat.r) < 1e-3f && std::fabs(original.g - kHandColorFloat.g) < 1e-3f &&
            std::fabs(original.b - kHandColorFloat.b) < 1e-3f;
        if (is_hand) {
            tinted->colors[vertex] = tint;
        }
    }
    std::lock_guard<std::mutex> guard(mutex_);
    auto [iterator, inserted] = tinted_topologies_.emplace(key, std::move(tinted));
    return iterator->second;
}

void SequenceLoader::worker() {
    while (!stop_.load()) {
        const int index = claim_next();
        if (index < 0) {
            return; // everything is claimed or loaded; nothing left to do
        }
        auto hands = std::make_shared<Frame>();
        for (const std::string& path : frame_paths_[static_cast<std::size_t>(index)]) {
            auto [positions, key] = parse_positions(path);
            HandData hand;
            hand.positions = std::move(positions);
            hand.topology = get_topology(key, path);
            hands->push_back(std::move(hand));
            meshes_loaded_.fetch_add(1);
        }
        // Fold in the matching OHView hands, mapped into this BabyView frame's
        // coordinate space by the estimated similarity transform and tinted apart.
        if (overlay_) {
            const SimilarityTransform& transform = overlay_->transform();
            for (const std::string& path : overlay_->oh_paths(index)) {
                auto [positions, key] = parse_positions(path);
                for (glm::vec3& position : positions) {
                    position = transform.apply(position);
                }
                HandData hand;
                hand.positions = std::move(positions);
                hand.topology = get_tinted_topology(key, path);
                hand.is_overlay = true;
                hands->push_back(std::move(hand));
                meshes_loaded_.fetch_add(1);
            }
        }
        std::lock_guard<std::mutex> guard(mutex_);
        frames_[static_cast<std::size_t>(index)] = std::move(hands);
    }
}

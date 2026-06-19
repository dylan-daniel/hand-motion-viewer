#include "data/mesh_sequence.h"

#include "data/geometry.h"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
    namespace fs = std::filesystem;

    /// Split a CSV line into fields on commas. The data has no quoted fields, so a
    /// plain split is sufficient; empty fields (e.g. empty-frame marker rows) are
    /// preserved as empty strings.
    std::vector<std::string> split_csv(const std::string& line) {
        std::vector<std::string> fields;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = line.find(',', start);
            if (comma == std::string::npos) {
                fields.push_back(line.substr(start));
                break;
            }
            fields.push_back(line.substr(start, comma - start));
            start = comma + 1;
        }
        return fields;
    }

    float to_float(const std::string& text) { return text.empty() ? 0.0f : std::stof(text); }
} // namespace

MeshSequence::MeshSequence(const std::string& csv_path) : csv_path_(csv_path) {
    std::ifstream file(csv_path);
    if (!file) {
        throw std::runtime_error("Could not open CSV " + csv_path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        return; // empty file: no frames
    }
    // Map column name -> index from the header so the parse is order-independent.
    std::unordered_map<std::string, int> column;
    {
        const std::vector<std::string> headers = split_csv(line);
        for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
            column[headers[index]] = index;
        }
    }
    auto column_index = [&](const std::string& name) -> int {
        const auto found = column.find(name);
        if (found == column.end()) {
            throw std::runtime_error("CSV missing column '" + name + "' in " + csv_path);
        }
        return found->second;
    };
    // The beta/gorient/pose blocks are contiguous, so one base index each suffices.
    const int subject_column = column_index("subject");
    const int trial_column = column_index("trial");
    const int frame_column = column_index("frame");
    const int hand_column = column_index("hand_idx");
    const int right_column = column_index("is_right");
    const int beta_base = column_index("beta_0");
    const int gorient_base = column_index("gorient_0");
    const int pose_base = column_index("pose_0");
    const int cam_x = column_index("cam_t_x");
    const int cam_y = column_index("cam_t_y");
    const int cam_z = column_index("cam_t_z");
    // Precomputed placed-space bounds (see HandBounds): the placed bounding box and
    // mean depth, so the scene fit needs no MANO forward pass.
    const int bbox_min_x = column_index("placed_bbox_min_x");
    const int bbox_max_x = column_index("placed_bbox_max_x");
    const int bbox_min_y = column_index("placed_bbox_min_y");
    const int bbox_max_y = column_index("placed_bbox_max_y");
    const int bbox_min_z = column_index("placed_bbox_min_z");
    const int bbox_max_z = column_index("placed_bbox_max_z");
    const int mean_z = column_index("placed_mean_z");

    // A row must reach every column we read; placed_mean_z is the last of them.
    const int last_needed = mean_z;

    // Group hands by frame number; frame numbers may have gaps, so the map keeps
    // them sorted and the result is densified into contiguous frame indices. Each
    // hand carries its MANO parameters alongside its precomputed placed bounds.
    struct FrameHand {
        ManoParams params;
        HandBounds bounds;
    };
    std::map<int, std::vector<FrameHand>> by_frame;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= last_needed) {
            continue; // malformed / short row
        }
        // Empty-frame marker rows carry hand_idx == -1 (and otherwise-blank fields).
        if (fields[hand_column].empty() || std::stoi(fields[hand_column]) < 0) {
            continue;
        }
        // The subject/trial keys are constant across the file; capture them once
        // from the first real row to locate this trial's images later.
        if (subject_.empty() && trial_.empty()) {
            subject_ = fields[subject_column];
            trial_ = fields[trial_column];
        }
        ManoParams params;
        params.is_right = std::stoi(fields[right_column]) != 0;
        for (int index = 0; index < 10; ++index) {
            params.betas[index] = to_float(fields[beta_base + index]);
        }
        for (int index = 0; index < 9; ++index) {
            params.global_orient[index] = to_float(fields[gorient_base + index]);
        }
        for (int index = 0; index < 135; ++index) {
            params.hand_pose[index] = to_float(fields[pose_base + index]);
        }
        params.cam_t = glm::vec3(to_float(fields[cam_x]), to_float(fields[cam_y]), to_float(fields[cam_z]));

        HandBounds bounds;
        bounds.placed_min = glm::vec3(to_float(fields[bbox_min_x]), to_float(fields[bbox_min_y]), to_float(fields[bbox_min_z]));
        bounds.placed_max = glm::vec3(to_float(fields[bbox_max_x]), to_float(fields[bbox_max_y]), to_float(fields[bbox_max_z]));
        bounds.placed_mean_z = to_float(fields[mean_z]);

        by_frame[std::stoi(fields[frame_column])].push_back(FrameHand{std::move(params), bounds});
    }

    frames_.reserve(by_frame.size());
    bounds_.reserve(by_frame.size());
    frame_numbers_.reserve(by_frame.size());
    for (auto& [frame_number, hands] : by_frame) {
        frame_numbers_.push_back(frame_number);
        std::vector<ManoParams> frame_params;
        std::vector<HandBounds> frame_bounds;
        frame_params.reserve(hands.size());
        frame_bounds.reserve(hands.size());
        for (FrameHand& hand : hands) {
            frame_params.push_back(std::move(hand.params));
            frame_bounds.push_back(hand.bounds);
        }
        frames_.push_back(std::move(frame_params));
        bounds_.push_back(std::move(frame_bounds));
    }

    // Stabilisation reference: frame 0's mean placed depth. Each hand contributes
    // an equal vertex count, so the per-vertex mean equals the mean of the hands'
    // placed_mean_z values.
    if (!bounds_.empty() && !bounds_.front().empty()) {
        double sum = 0.0;
        for (const HandBounds& hand : bounds_.front()) {
            sum += hand.placed_mean_z;
        }
        reference_depth_ = static_cast<float>(sum / static_cast<double>(bounds_.front().size()));
    }
}

int MeshSequence::hand_count(int index) const {
    if (index < 0 || index >= frame_count()) {
        return 0;
    }
    return static_cast<int>(frames_[static_cast<std::size_t>(index)].size());
}

const std::vector<HandBounds>& MeshSequence::frame_bounds(int index) const {
    static const std::vector<HandBounds> empty;
    if (index < 0 || index >= frame_count()) {
        return empty;
    }
    return bounds_[static_cast<std::size_t>(index)];
}

std::string MeshSequence::frame_image_path(int index, const std::string& images_root) const {
    if (index < 0 || index >= frame_count() || images_root.empty()) {
        return {};
    }
    char name[32];
    std::snprintf(name, sizeof(name), "frame_%04d", frame_numbers_[static_cast<std::size_t>(index)]);
    const fs::path directory = fs::path(images_root) / subject_ / trial_;
    const fs::path jpg = directory / (std::string(name) + ".jpg");
    if (fs::exists(jpg)) {
        return jpg.string();
    }
    const fs::path png = directory / (std::string(name) + ".png");
    if (fs::exists(png)) {
        return png.string();
    }
    return jpg.string(); // caller treats a missing file as no image
}

Frame MeshSequence::load_frame(int index) const {
    Frame hands;
    if (index < 0 || index >= frame_count()) {
        return hands;
    }
    const std::vector<ManoParams>& params = frames_[static_cast<std::size_t>(index)];
    hands.reserve(params.size());
    for (const ManoParams& hand_params : params) {
        ManoHand mesh = mano_forward(hand_params);
        HandData hand;
        hand.verts = std::move(mesh.verts);
        hand.joints = std::move(mesh.joints);
        hand.is_right = hand_params.is_right;
        hands.push_back(std::move(hand));
    }
    return hands;
}

// Per-hand depth-stabilization factor used by the renderer: each hand is scaled
// about the origin by reference_depth / its-own-mean-depth. The scene transform
// measures the *stabilized* geometry, so it applies the same factor here — using
// the placed mean depth precomputed in the CSV rather than a loaded mesh.
static float stabilization_scale(const HandBounds& hand, float reference) { return hand.placed_mean_z != 0.0f ? reference / hand.placed_mean_z : 1.0f; }

Transform compute_transform(const MeshSequence& sequence) {
    // Centering X/Z and the fit scale come from frame 0 so they stay fixed across
    // playback. The renderer scales each hand about the origin by reference_depth /
    // depth before this transform's offset is applied, so everything here is
    // measured on those stabilized positions. The bounds are the placed AABBs read
    // from the CSV, so this needs no MANO forward pass (scale > 0, so a stabilized
    // AABB is just the placed AABB scaled — its min/max stay the min/max).
    const float reference = sequence.reference_depth();

    // Whole-scene X/Z bounds of frame 0's stabilized hands: used only to centre.
    glm::vec3 low(std::numeric_limits<float>::max());
    glm::vec3 high(std::numeric_limits<float>::lowest());
    for (const HandBounds& hand : sequence.frame_bounds(0)) {
        const float scale = stabilization_scale(hand, reference);
        low = glm::min(low, hand.placed_min * scale);
        high = glm::max(high, hand.placed_max * scale);
    }
    const float center_x = (high.x + low.x) / 2.0f;
    const float center_z = (high.z + low.z) / 2.0f;

    // Floor (Y): the lowest stabilized point across *every* frame, so no frame
    // ever sinks below the grid plane (only frame 0 would sit on the grid if we
    // used its bounds alone, and a later frame could reach lower).
    float floor_y = std::numeric_limits<float>::max();
    for (int index = 0; index < sequence.frame_count(); ++index) {
        for (const HandBounds& hand : sequence.frame_bounds(index)) {
            floor_y = std::min(floor_y, hand.placed_min.y * stabilization_scale(hand, reference));
        }
    }
    if (floor_y == std::numeric_limits<float>::max()) {
        floor_y = 0.0f; // no hands in any frame
    }

    // Scale from a single reference hand's size, not the whole-scene extent. The
    // meshes are metric, so one hand is a stable size cue, whereas the full-scene
    // bounding box balloons when hands sit far apart, which would shrink every
    // hand to nothing. This keeps a hand a consistent on-screen size across takes.
    // Measure the stabilized reference hand so the fit matches what's drawn.
    float span = 1.0f;
    const std::vector<HandBounds>& frame_zero = sequence.frame_bounds(0);
    if (!frame_zero.empty()) {
        const HandBounds& reference_hand = frame_zero.front();
        const glm::vec3 extent = (reference_hand.placed_max - reference_hand.placed_min) * stabilization_scale(reference_hand, reference);
        const float largest = std::max({extent.x, extent.y, extent.z});
        if (largest > 0.0f) {
            span = largest;
        }
    }
    return Transform{glm::vec3(-center_x, -floor_y, -center_z), MODEL_FIT_SPAN / span};
}

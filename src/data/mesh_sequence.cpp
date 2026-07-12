#include "data/mesh_sequence.h"

#include "data/geometry.h"
#include "data/hand_classification.h"
#include "data/npy_reader.h"
#include "data/object_fit.h"

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

    /// Path to one frame's cached array file, matching object_sequence.cpp's
    /// own ``frame_file`` helper (kept separate since the two have no other
    /// shared dependency).
    std::string frame_array_file(const std::string& dir, int frame_number, const std::string& suffix) {
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d_", frame_number);
        return (fs::path(dir) / (std::string(name) + suffix)).string();
    }

    /// One frame's DA3 depth map + SAM3 hand-mask stacks (mirrors
    /// render_scene_video.load_depth_frame / load_hand_mask_stacks), loaded
    /// once and shared by every hand in the frame. Everything stays empty
    /// (a silent HaMeR fallback everywhere it's used) unless the caller asked
    /// for DA3 and both folders are set.
    struct FrameDepthData {
        std::optional<NpyDepthMap> depth;
        std::optional<NpyMaskStack> right_masks;
        std::optional<NpyMaskStack> left_masks;
    };

    FrameDepthData load_frame_depth_data(const HandDepthConfig& depth_config, int frame_number) {
        FrameDepthData data;
        if (depth_config.source == HandDepthSource::Da3 && !depth_config.sam3_dir.empty() && !depth_config.da3_dir.empty()) {
            data.depth = read_npy_depth(frame_array_file(depth_config.da3_dir, frame_number, "depth.npy"));
            data.right_masks = read_npy_mask_stack(frame_array_file(depth_config.sam3_dir, frame_number, "right_hand_masks.npy"));
            data.left_masks = read_npy_mask_stack(frame_array_file(depth_config.sam3_dir, frame_number, "left_hand_masks.npy"));
        }
        return data;
    }

    /// One hand's raw DA3-sampled wrist depth from already-loaded frame data,
    /// or nullopt if no robust sample exists (caller falls back to HaMeR's
    /// own k_metric * cam_t.z). Mirrors
    /// scene_placement.sample_wrist_depth_da3(..., select_hand_mask(...)).
    std::optional<float> resolve_da3_wrist_depth(const FrameDepthData& frame_data, const ManoParams& hand_params) {
        if (!frame_data.depth) {
            return std::nullopt;
        }
        const std::optional<NpyMaskStack>& stack = hand_params.is_right ? frame_data.right_masks : frame_data.left_masks;
        const std::uint8_t* hand_mask = nullptr;
        if (stack && stack->count > 0) {
            const int selected = select_hand_mask_instance(stack->masks, stack->count, stack->height, stack->width, hand_params.wrist_uv);
            if (selected >= 0) {
                const std::size_t plane = static_cast<std::size_t>(stack->height) * static_cast<std::size_t>(stack->width);
                hand_mask = stack->masks.data() + static_cast<std::size_t>(selected) * plane;
            }
        }
        return sample_wrist_depth_da3(frame_data.depth->depth, frame_data.depth->height, frame_data.depth->width, hand_params.wrist_uv, hand_mask);
    }

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

    // Parse an integer field, falling back to ``fallback`` when the cell is empty.
    // The track/classification columns are blank in trials whose classification
    // pass hasn't run yet, so parsing them must not throw.
    int to_int(const std::string& text, int fallback) { return text.empty() ? fallback : std::stoi(text); }
} // namespace

MeshSequence::MeshSequence(const std::string& csv_path, const HandClassificationConfig& classification_config) : csv_path_(csv_path) {
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
    // Like column_index but tolerates the column being entirely absent (-1),
    // not just individual blank cells: hamer_collect.py's raw CSV carries no
    // tracking/classification columns at all (they live in separate per-trial
    // sidecar files — baby_hand_idx/baby_classification — keyed differently
    // from hand_idx, not merged into this CSV), so a trial with no
    // classification pass yet must still load.
    auto optional_column_index = [&](const std::string& name) -> int {
        const auto found = column.find(name);
        return found == column.end() ? -1 : found->second;
    };
    // The beta/gorient/pose blocks are contiguous, so one base index each suffices.
    const int subject_column = column_index("subject");
    const int trial_column = column_index("trial");
    const int frame_column = column_index("frame");
    const int hand_column = column_index("hand_idx");
    const int track_column = optional_column_index("track_id");
    const int duplicate_column = optional_column_index("is_duplicate");
    const int baby_column = optional_column_index("is_baby");
    const int right_column = column_index("is_right");
    const int beta_base = column_index("beta_0");
    const int gorient_base = column_index("gorient_0");
    const int pose_base = column_index("pose_0");
    const int cam_x = column_index("cam_t_x");
    const int cam_y = column_index("cam_t_y");
    const int cam_z = column_index("cam_t_z");
    // The wrist's observed 2D pixel (HaMeR's own j0 reprojection), needed to
    // backproject the wrist into real metric space (see place_hand_metric).
    const int wrist_u = column_index("j0_u");
    const int wrist_v = column_index("j0_v");
    // This trial's own camera calibration (see HAMER_COLLECT.md's "Camera"
    // column block): the focal length hamer_collect.py was run with, and the
    // image size the principal point (cx, cy) is assumed centred in. Optional
    // so a CSV missing this block (an older export) still loads — just without
    // auto-detected intrinsics.
    const int focal_column = optional_column_index("scaled_focal_length");
    const int width_column = optional_column_index("img_w");
    const int height_column = optional_column_index("img_h");

    // A row must reach every column we read; j0_v is the last of them (the
    // focal/width/height columns sit earlier in the schema, before it).
    const int last_needed = wrist_v;

    // Group hands by frame number; frame numbers may have gaps, so the map keeps
    // them sorted and the result is densified into contiguous frame indices.
    struct FrameHand {
        ManoParams params;
        HandMeta meta;
        int hand_idx = -1; // hamer_cache's own per-frame detection index, for the classification join
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
        // The subject/trial keys (and this trial's camera calibration) are
        // constant across the file; capture them once from the first real row.
        if (subject_.empty() && trial_.empty()) {
            subject_ = fields[subject_column];
            trial_ = fields[trial_column];
            if (focal_column >= 0) {
                focal_length_px_ = to_float(fields[focal_column]);
            }
            if (width_column >= 0) {
                image_width_ = to_float(fields[width_column]);
            }
            if (height_column >= 0) {
                image_height_ = to_float(fields[height_column]);
            }
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
        params.wrist_uv = glm::vec2(to_float(fields[wrist_u]), to_float(fields[wrist_v]));

        // When a classification column is blank OR entirely absent (not
        // generated yet, or a raw hamer_collect.py CSV with no merged
        // classification pass), default to an untracked, non-duplicate, shown
        // hand so the trial still displays: id -1 takes the default colour, and
        // is_baby defaults true so "hide adults" keeps it visible rather than
        // hiding every unclassified hand.
        HandMeta meta;
        meta.track_id = track_column >= 0 ? to_int(fields[track_column], -1) : -1;
        meta.is_duplicate = duplicate_column >= 0 ? to_int(fields[duplicate_column], 0) != 0 : false;
        meta.is_baby = baby_column >= 0 ? to_int(fields[baby_column], 1) != 0 : true;

        const int hand_idx = std::stoi(fields[hand_column]);
        by_frame[std::stoi(fields[frame_column])].push_back(FrameHand{std::move(params), meta, hand_idx});
    }

    // Override each hand's track_id/is_duplicate/is_baby with the
    // tracker-CSV/baby-hand-idx join when both sidecar folders are configured
    // (see data/hand_classification.h) — otherwise keep whatever the CSV's own
    // (usually absent) columns produced above.
    const HandClassification classification =
        subject_.empty()
            ? HandClassification()
            : HandClassification(classification_config.tracking_dir, classification_config.baby_hand_idx_dir, subject_, trial_);

    frames_.reserve(by_frame.size());
    meta_.reserve(by_frame.size());
    frame_numbers_.reserve(by_frame.size());
    for (auto& [frame_number, hands] : by_frame) {
        frame_numbers_.push_back(frame_number);
        std::vector<ManoParams> frame_params;
        std::vector<HandMeta> frame_meta;
        frame_params.reserve(hands.size());
        frame_meta.reserve(hands.size());
        for (FrameHand& hand : hands) {
            if (classification.loaded()) {
                if (const HandClassificationEntry* entry = classification.lookup(frame_number, hand.hand_idx)) {
                    hand.meta.track_id = entry->track_id;
                    hand.meta.is_duplicate = entry->is_duplicate;
                    hand.meta.is_baby = entry->is_baby;
                }
            }
            frame_params.push_back(std::move(hand.params));
            frame_meta.push_back(hand.meta);
        }
        frames_.push_back(std::move(frame_params));
        meta_.push_back(std::move(frame_meta));
    }
}

int MeshSequence::hand_count(int index, const HandFilter& filter) const {
    int count = 0;
    for (const HandMeta& meta : frame_meta(index)) {
        if (hand_visible(meta, filter)) {
            ++count;
        }
    }
    return count;
}

const std::vector<HandMeta>& MeshSequence::frame_meta(int index) const {
    static const std::vector<HandMeta> empty;
    if (index < 0 || index >= frame_count()) {
        return empty;
    }
    return meta_[static_cast<std::size_t>(index)];
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

Frame MeshSequence::load_frame(
    int index, const HandFilter& filter, const CameraIntrinsics& intrinsics, const HandDepthConfig& depth_config,
    const std::vector<std::vector<float>>* smoothed_depths
) const {
    Frame hands;
    if (index < 0 || index >= frame_count()) {
        return hands;
    }
    const std::vector<ManoParams>& params = frames_[static_cast<std::size_t>(index)];
    const std::vector<HandMeta>& metas = meta_[static_cast<std::size_t>(index)];
    hands.reserve(params.size());

    // Only load this frame's DA3 depth map + SAM3 hand-mask stacks when they
    // might actually be used below: precomputed smoothed depths already
    // resolved them (across the whole trial), so live per-frame resolution
    // would be redundant work repeated on every frame change.
    const FrameDepthData frame_data =
        smoothed_depths ? FrameDepthData{} : load_frame_depth_data(depth_config, frame_numbers_[static_cast<std::size_t>(index)]);

    for (std::size_t hand_index = 0; hand_index < params.size(); ++hand_index) {
        const HandMeta& meta = metas[hand_index];
        if (!hand_visible(meta, filter)) {
            continue;
        }
        const ManoParams& hand_params = params[hand_index];

        // Local mesh (cam_t=0), then position + size correction into real metric
        // space — mirrors scene_placement.place_hand_metric exactly (both are
        // derived from the single ratio z_metric / raw cam_t.z; see its doc
        // comment). z_metric is either the precomputed whole-trial-smoothed
        // depth (see precompute_smoothed_wrist_depths) when given, or resolved
        // live: a real per-frame DA3 sample at the wrist pixel when
        // frame_data's depth map loaded and the sample is robust enough,
        // otherwise (or under HandDepthSource::Hamer) k_metric * cam_t.z.
        const ManoHand local = mano_forward_local(hand_params);
        const glm::vec3& local_wrist = local.joints[0];
        float z_metric;
        if (smoothed_depths) {
            z_metric = (*smoothed_depths)[static_cast<std::size_t>(index)][hand_index];
        } else {
            z_metric = intrinsics.k_metric * hand_params.cam_t.z;
            if (const std::optional<float> da3_z = resolve_da3_wrist_depth(frame_data, hand_params)) {
                z_metric = *da3_z;
            }
        }

        HandData hand;
        hand.verts = place_hand_metric(local.verts, local_wrist, hand_params.wrist_uv, hand_params.cam_t.z, z_metric, intrinsics);
        hand.joints = place_hand_metric(local.joints, local_wrist, hand_params.wrist_uv, hand_params.cam_t.z, z_metric, intrinsics);
        hand.is_right = hand_params.is_right;
        hand.track_id = meta.track_id;
        hands.push_back(std::move(hand));
    }
    return hands;
}

std::vector<std::vector<float>> MeshSequence::precompute_smoothed_wrist_depths(
    const CameraIntrinsics& intrinsics, const HandDepthConfig& depth_config, float sigma_frames
) const {
    const std::size_t n = frames_.size();
    std::vector<std::vector<glm::vec2>> wrist_uv_by_frame(n);
    std::vector<std::vector<bool>> is_right_by_frame(n);
    std::vector<std::vector<std::optional<float>>> raw_z_by_frame(n);
    std::vector<std::vector<float>> raw_cam_t_z_by_frame(n);

    for (std::size_t index = 0; index < n; ++index) {
        const std::vector<ManoParams>& params = frames_[index];
        wrist_uv_by_frame[index].reserve(params.size());
        is_right_by_frame[index].reserve(params.size());
        raw_z_by_frame[index].reserve(params.size());
        raw_cam_t_z_by_frame[index].reserve(params.size());

        const FrameDepthData frame_data = load_frame_depth_data(depth_config, frame_numbers_[index]);
        for (const ManoParams& hand_params : params) {
            wrist_uv_by_frame[index].push_back(hand_params.wrist_uv);
            is_right_by_frame[index].push_back(hand_params.is_right);
            raw_cam_t_z_by_frame[index].push_back(hand_params.cam_t.z);
            raw_z_by_frame[index].push_back(resolve_da3_wrist_depth(frame_data, hand_params));
        }
    }

    const std::vector<std::vector<int>> track_ids = track_hand_sequences(wrist_uv_by_frame, is_right_by_frame);
    const std::vector<std::vector<float>> filled = fill_missing_depths_by_track(raw_z_by_frame, raw_cam_t_z_by_frame, track_ids, intrinsics.k_metric);
    if (sigma_frames <= 0.0f) {
        return filled;
    }

    // Group entries by track, Gaussian-smooth each track's depth sequence
    // independently by real frame-number distance (smooth_by_track), then
    // scatter the smoothed values back into frame/hand-position order.
    struct Entry {
        std::size_t frame_index;
        std::size_t hand_position;
        int frame_number;
    };
    std::unordered_map<int, std::vector<Entry>> by_track;
    for (std::size_t index = 0; index < n; ++index) {
        for (std::size_t position = 0; position < track_ids[index].size(); ++position) {
            by_track[track_ids[index][position]].push_back({index, position, frame_numbers_[index]});
        }
    }

    std::vector<std::vector<float>> result = filled;
    for (const auto& [track_id, entries] : by_track) {
        std::vector<int> track_frame_numbers;
        std::vector<float> track_values;
        track_frame_numbers.reserve(entries.size());
        track_values.reserve(entries.size());
        for (const Entry& entry : entries) {
            track_frame_numbers.push_back(entry.frame_number);
            track_values.push_back(filled[entry.frame_index][entry.hand_position]);
        }
        const std::vector<float> smoothed = smooth_sequence(track_frame_numbers, track_values, sigma_frames);
        for (std::size_t i = 0; i < entries.size(); ++i) {
            result[entries[i].frame_index][entries[i].hand_position] = smoothed[i];
        }
    }
    return result;
}

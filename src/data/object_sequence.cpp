#include "data/object_sequence.h"

#include "data/npy_reader.h"
#include "data/object_fit.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <random>

#include <glm/gtc/matrix_transform.hpp>

namespace {
    namespace fs = std::filesystem;

    std::string frame_file(const std::string& dir, int frame_number, const std::string& suffix) {
        char name[64];
        std::snprintf(name, sizeof(name), "frame_%04d_", frame_number);
        return (fs::path(dir) / (std::string(name) + suffix)).string();
    }

    // Matches mano_numpy.mano_forward's / scene_placement.flip_to_onscreen's
    // convention exactly: negate Y and Z to turn camera-space (OpenCV, +Y
    // down/+Z forward) into the on-screen space (Y up, camera looking down
    // -Z) the rest of the viewer's placed geometry (hands) is already in.
    glm::vec3 flip_to_onscreen(const glm::vec3& point) { return {point.x, -point.y, -point.z}; }

    glm::mat3 flip_to_onscreen(const glm::mat3& rotation) {
        return glm::mat3(flip_to_onscreen(rotation[0]), flip_to_onscreen(rotation[1]), flip_to_onscreen(rotation[2]));
    }

    // A frame's object depth-outlier margin before fitting — matches
    // render_scene_video.load_object_points's own filter (a coarser pre-pass
    // than fit_sphere_center's own tighter internal filter).
    constexpr float OBJECT_DEPTH_OUTLIER_MARGIN = 0.10f;

    // Loads this frame's DA3 depth map and SAM3 object mask (first instance
    // only, matching render_scene_video.load_object_points — SAM3's cache can
    // hold multiple candidate instances per label, but the object pipeline
    // doesn't disambiguate them the way the hand pipeline's select_hand_mask
    // does), backprojects, and depth-outlier-filters. Empty on any missing
    // file / undetected frame.
    std::vector<glm::vec3> load_frame_object_points(
        const std::string& sam3_dir, const std::string& da3_dir, const std::string& object_label, int frame_number, const CameraIntrinsics& intrinsics
    ) {
        const std::optional<NpyDepthMap> depth = read_npy_depth(frame_file(da3_dir, frame_number, "depth.npy"));
        if (!depth) {
            return {};
        }
        const std::optional<NpyMaskStack> masks = read_npy_mask_stack(frame_file(sam3_dir, frame_number, object_label + "_masks.npy"));
        if (!masks || masks->count == 0) {
            return {};
        }
        const std::size_t plane = static_cast<std::size_t>(masks->height) * static_cast<std::size_t>(masks->width);
        const std::vector<std::uint8_t> first_instance(masks->masks.begin(), masks->masks.begin() + static_cast<std::ptrdiff_t>(plane));
        std::vector<glm::vec3> points = backproject_mask(depth->depth, depth->height, depth->width, first_instance, masks->height, masks->width, intrinsics);
        if (points.empty()) {
            return {};
        }
        return filter_depth_outliers(points, OBJECT_DEPTH_OUTLIER_MARGIN);
    }

    float median(std::vector<float> values) {
        std::sort(values.begin(), values.end());
        return values[values.size() / 2];
    }
} // namespace

TrackedObjectSequence::TrackedObjectSequence(
    const std::string& sam3_dir, const std::string& da3_dir, const std::string& object_label, ObjectShape shape, float object_size_m,
    const CameraIntrinsics& intrinsics, const std::vector<int>& frame_numbers, float smooth_sigma_frames
) :
    shape_(shape) {
    if (shape == ObjectShape::None || sam3_dir.empty() || da3_dir.empty()) {
        return;
    }

    if (shape == ObjectShape::Sphere) {
        const float radius = object_size_m * 0.5f;
        std::vector<int> raw_frames;
        std::vector<glm::vec3> raw_centers;
        for (int frame_number : frame_numbers) {
            const std::vector<glm::vec3> points = load_frame_object_points(sam3_dir, da3_dir, object_label, frame_number, intrinsics);
            if (points.empty()) {
                continue;
            }
            const std::optional<glm::vec3> center = fit_sphere_center(points, radius);
            if (!center) {
                continue;
            }
            raw_frames.push_back(frame_number);
            raw_centers.push_back(*center);
        }
        if (raw_frames.empty()) {
            return;
        }
        std::vector<float> raw_z;
        raw_z.reserve(raw_centers.size());
        for (const glm::vec3& center : raw_centers) {
            raw_z.push_back(center.z);
        }
        const std::vector<float> smoothed_z = smooth_sequence(raw_frames, raw_z, smooth_sigma_frames);
        for (std::size_t i = 0; i < raw_frames.size(); ++i) {
            glm::vec3 center = raw_centers[i];
            center.z = smoothed_z[i];
            CubePose pose;
            pose.center = flip_to_onscreen(center);
            pose.half_extent = glm::vec3(radius);
            pose.basis = glm::mat3(1.0f);
            poses_[raw_frames[i]] = pose;
        }
        return;
    }

    // ObjectShape::Cube — see class-level comment for what's intentionally
    // not ported (silhouette refinement, accept/reject gating).
    const float edge = object_size_m;
    std::unordered_map<int, std::vector<glm::vec3>> raw_points;
    for (int frame_number : frame_numbers) {
        std::vector<glm::vec3> points = load_frame_object_points(sam3_dir, da3_dir, object_label, frame_number, intrinsics);
        if (points.empty()) {
            continue;
        }
        if (points.size() > 2500) {
            std::mt19937 rng(0);
            std::shuffle(points.begin(), points.end(), rng);
            points.resize(2500);
        }
        raw_points[frame_number] = std::move(points);
    }
    if (raw_points.empty()) {
        return;
    }

    // Trial-level size-scale prior (see object_fit.h's estimate_cube_scale):
    // the median implied scale across every frame with a cleanly-visible face.
    std::vector<float> clean_scales;
    for (const auto& [frame_number, points] : raw_points) {
        if (const std::optional<float> scale = estimate_cube_scale(points, edge)) {
            clean_scales.push_back(*scale);
        }
    }
    const float scale_prior = clean_scales.empty() ? 1.0f : median(clean_scales);

    std::vector<int> sorted_frames;
    sorted_frames.reserve(raw_points.size());
    for (const auto& [frame_number, points] : raw_points) {
        sorted_frames.push_back(frame_number);
    }
    std::sort(sorted_frames.begin(), sorted_frames.end());

    // Sequential per-frame fit, each initialized from the previous frame's
    // pose (helps a heavily-occluded frame's cloud converge to something
    // consistent rather than a fresh, unconstrained plane+silhouette guess).
    std::unordered_map<int, CubeFit> fits;
    std::optional<CubeFit> last_pose;
    for (int frame_number : sorted_frames) {
        const std::optional<CubeFit> fit = fit_cube_pose(raw_points[frame_number], edge, last_pose ? &*last_pose : nullptr, scale_prior);
        if (!fit) {
            continue;
        }
        last_pose = fit;
        fits[frame_number] = *fit;
    }
    if (fits.empty()) {
        return;
    }

    std::vector<int> fitted_frames;
    for (int frame_number : sorted_frames) {
        if (fits.count(frame_number) != 0) {
            fitted_frames.push_back(frame_number);
        }
    }

    std::vector<glm::mat3> raw_rotations;
    raw_rotations.reserve(fitted_frames.size());
    for (int frame_number : fitted_frames) {
        raw_rotations.push_back(fits[frame_number].rotation);
    }
    const std::vector<glm::mat3> canonical = canonicalize_cube_rotations(raw_rotations);
    const std::vector<glm::mat3> smoothed_rotations = smooth_rotations(fitted_frames, canonical, smooth_sigma_frames);

    std::vector<float> raw_z, raw_scales;
    raw_z.reserve(fitted_frames.size());
    raw_scales.reserve(fitted_frames.size());
    for (int frame_number : fitted_frames) {
        raw_z.push_back(fits[frame_number].center.z);
        raw_scales.push_back(fits[frame_number].scale);
    }
    const std::vector<float> smoothed_z = smooth_sequence(fitted_frames, raw_z, smooth_sigma_frames);
    const std::vector<float> smoothed_scales = smooth_sequence(fitted_frames, raw_scales, smooth_sigma_frames);

    for (std::size_t i = 0; i < fitted_frames.size(); ++i) {
        glm::vec3 center = fits[fitted_frames[i]].center;
        center.z = smoothed_z[i];
        CubePose pose;
        pose.center = flip_to_onscreen(center);
        pose.half_extent = glm::vec3(edge * smoothed_scales[i] * 0.5f);
        pose.basis = flip_to_onscreen(smoothed_rotations[i]);
        poses_[fitted_frames[i]] = pose;
    }
}

std::optional<CubePose> TrackedObjectSequence::pose_for_frame(int frame_number) const {
    const auto found = poses_.find(frame_number);
    if (found == poses_.end()) {
        return std::nullopt;
    }
    return found->second;
}

glm::mat4 cube_model_matrix(const CubePose& pose) {
    glm::mat4 model(1.0f);
    model = glm::translate(model, pose.center);
    model = model * glm::mat4(pose.basis);
    model = glm::scale(model, pose.half_extent);
    return model;
}

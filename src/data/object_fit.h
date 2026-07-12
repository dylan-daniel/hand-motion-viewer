#pragma once

// Real-metric tracked-object geometry: backprojecting a SAM3 mask through a
// DA3 depth map into a camera-space point cloud, then fitting either a
// fixed-radius sphere or a known-edge cube's full 6-DOF pose to that cloud.
// Ports the core of scene_placement.py's object-placement math (see the
// A6000 render script this viewer mirrors) into C++ so the interactive
// viewer can place the tracked object the same way, without depending on
// SAM3/DA3/hamer being installed anywhere near this app.
//
// All points here are in OpenCV camera-space convention (+X right, +Y down,
// +Z forward/away from camera) — the same convention data/mano_model.h's
// place_hand_metric works in before its own on-screen flip; callers apply
// that same flip (negate Y, Z) once a final pose is chosen, not before.
//
// NOT ported from the Python reference (documented simplifications, kept
// deliberately out of scope for this pass): the 2D-silhouette refinement
// pass (refine_cube_pose_silhouette) and its reprojection-containment
// accept/reject gating with slerp interpolation across rejected frames.
// Every frame whose SDF fit succeeds is accepted outright here. Revisit if
// cube orientation proves too noisy without it.

#include <cstdint>
#include <optional>
#include <vector>

#include <glm/glm.hpp>

#include "data/mano_model.h" // CameraIntrinsics

/// Backprojects every set (nonzero) pixel of a SAM3 mask through a DA3 depth
/// map into camera-space points (meters), using the study's camera
/// intrinsics. Mirrors scene_placement.backproject_mask exactly. Skips
/// pixels with non-positive depth.
std::vector<glm::vec3> backproject_mask(
    const std::vector<float>& depth, int depth_height, int depth_width, const std::vector<std::uint8_t>& mask, int mask_height, int mask_width,
    const CameraIntrinsics& intrinsics
);

/// Drops points whose depth is more than ``margin`` meters from the cloud's
/// median depth — the same DA3 silhouette-edge outlier filter
/// render_scene_video.load_object_points applies before fitting.
std::vector<glm::vec3> filter_depth_outliers(const std::vector<glm::vec3>& points, float margin);

/// Gauss-Newton fit of a fixed-radius sphere's center to a partial point
/// cloud (mirrors scene_placement.fit_sphere_center). Returns nullopt if
/// there are too few points to fit reliably.
std::optional<glm::vec3> fit_sphere_center(const std::vector<glm::vec3>& points, float radius);

/// A cube's full pose: ``rotation``'s columns are the cube's local axes in
/// camera space (local->camera), ``center`` is its centroid, and
/// ``edge * scale`` is its true world-space edge length (see
/// estimate_cube_scale for why a size scale is needed at all).
struct CubeFit {
    glm::mat3 rotation{1.0f};
    glm::vec3 center{0.0f};
    float scale = 1.0f;
    float rms = 0.0f;
};

/// The size scale implied by one frame's cleanly-visible cube face (a
/// trial-level pre-pass; see scene_placement.estimate_cube_scale), or
/// nullopt if this frame's face isn't clean enough to trust.
std::optional<float> estimate_cube_scale(const std::vector<glm::vec3>& points, float edge);

/// Fits the known-edge cube's pose — position, orientation, and a size scale
/// — to a backprojected partial cloud (mirrors scene_placement.fit_cube_pose).
/// ``init`` is the previous frame's fit, used as a fallback refinement start
/// when this frame's cloud is too sparse to build a fresh plane+silhouette
/// construction (heavy occlusion mid-grasp). ``scale_prior`` is the trial's
/// estimate_cube_scale median, softly pulling an occluded frame's scale
/// toward it.
std::optional<CubeFit> fit_cube_pose(const std::vector<glm::vec3>& points, float edge, const CubeFit* init, float scale_prior);

/// Snaps each fit's rotation to the cube-symmetry equivalent (of the 24
/// signed-permutation rotations) closest to the PREVIOUS frame's rotation,
/// so a sequence of per-frame fits becomes temporally continuous. The first
/// entry is returned unchanged.
std::vector<glm::mat3> canonicalize_cube_rotations(const std::vector<glm::mat3>& rotations);

/// Gaussian temporal smoothing of a (already symmetry-canonicalized)
/// rotation sequence via sign-aligned quaternion averaging, weighted by real
/// frame-number distance. ``sigma_frames`` <= 0 returns the input unchanged.
std::vector<glm::mat3> smooth_rotations(const std::vector<int>& frame_numbers, const std::vector<glm::mat3>& rotations, float sigma_frames);

/// Gaussian temporal smoothing of a scalar sequence (e.g. depth), weighted
/// by real frame-number distance. ``sigma_frames`` <= 0 returns the input
/// unchanged.
std::vector<float> smooth_sequence(const std::vector<int>& frame_numbers, const std::vector<float>& values, float sigma_frames);

// ── Hand wrist depth (DA3 depth source) ─────────────────────────────────
//
// The tracked object's placement always uses DA3's own depth map (see
// backproject_mask above); these two mirror scene_placement's SEPARATE
// hand-side machinery, used only when the "hand depth source" is DA3 rather
// than HaMeR's own cam_t.z (see data/mesh_sequence.h's HandDepthSource).

/// Index of the mask-stack instance (0-based, into a ``count``-deep stack of
/// ``height`` x ``width`` masks) that plausibly belongs to a hand detection
/// at ``wrist_uv``: the instance containing the wrist pixel wins; otherwise
/// the instance whose bounding-box center is nearest, within
/// ``max_snap_px`` (SAM3's hand labels deliberately keep every candidate
/// instance — a baby's hand AND a reaching adult's can both match "baby left
/// hand" — so this disambiguates against the detection actually being
/// placed). Returns -1 if no instance plausibly corresponds. Mirrors
/// scene_placement.select_hand_mask.
int select_hand_mask_instance(const std::vector<std::uint8_t>& masks, int count, int height, int width, const glm::vec2& wrist_uv, float max_snap_px = 100.0f);

/// The wrist's metric depth (meters, camera-space +Z) sampled ROBUSTLY from a
/// DA3 depth map: the median of the valid (>0) depths in a small disk of
/// radius ``radius_px`` around ``wrist_uv``, restricted to ``hand_mask`` (a
/// ``depth_height`` x ``depth_width`` plane — e.g. one instance sliced out of
/// a mask stack via select_hand_mask_instance — or ``nullptr`` for no
/// restriction) when given, so the sample can't straddle the silhouette edge.
/// Returns nullopt when the disk has fewer than ``min_valid_px`` valid
/// samples (caller falls back to HaMeR-derived depth). Mirrors
/// scene_placement.sample_wrist_depth_da3.
std::optional<float> sample_wrist_depth_da3(
    const std::vector<float>& depth, int depth_height, int depth_width, const glm::vec2& wrist_uv, const std::uint8_t* hand_mask, int radius_px = 25,
    int min_valid_px = 9
);

// ── Whole-trial hand-depth smoothing ────────────────────────────────────
//
// Everything below operates on a WHOLE trial's hands at once (one entry per
// frame, aligned with whatever per-frame hand list the caller is describing —
// data/mesh_sequence.h's MeshSequence::precompute_smoothed_wrist_depths is
// the only caller), unlike backproject_mask/sample_wrist_depth_da3 above
// which are single-frame primitives. Mirrors
// render_scene_video.precompute_smoothed_depths's hand-only path.

/// Greedy nearest-neighbor association of each frame's hands into persistent
/// per-hand tracks across the whole trial, needed because a CSV's hand_idx is
/// NOT a stable identity frame-to-frame (a single trial can have more/fewer
/// hands detected per frame, combined with both handedness values). Tracks
/// are built separately per handedness, matching each frame's hands to the
/// nearest (by wrist_uv pixel distance) still-unmatched hand from the
/// previous frame, within ``max_track_dist_px``; otherwise a new track
/// starts. ``wrist_uv_by_frame``/``is_right_by_frame`` must be the same
/// shape (one inner list per frame, aligned entries). Mirrors
/// scene_placement.track_hand_sequences. Returns a same-shape list of track
/// ids.
std::vector<std::vector<int>> track_hand_sequences(
    const std::vector<std::vector<glm::vec2>>& wrist_uv_by_frame, const std::vector<std::vector<bool>>& is_right_by_frame, float max_track_dist_px = 150.0f
);

/// Fills frames where no robust DA3 wrist depth existed (nullopt in
/// ``raw_z_by_frame``) with a HaMeR-derived estimate calibrated PER TRACK,
/// not by the global ``k_metric`` alone: each track's own median ratio
/// (da3_z / cam_t_z) over its DA3-backed frames, falling back to the global
/// ``k_metric`` when a track has fewer than ``min_ratio_samples`` DA3-backed
/// frames. Keeps a DA3-less frame consistent with the SAME hand's
/// DA3-measured neighbors instead of teleporting on the raw
/// ``k_metric * cam_t_z`` average. Mirrors
/// scene_placement.fill_missing_depths_by_track. All three by-frame
/// arguments (plus ``track_ids_by_frame``) must be the same shape. Returns a
/// same-shape list with every nullopt replaced.
std::vector<std::vector<float>> fill_missing_depths_by_track(
    const std::vector<std::vector<std::optional<float>>>& raw_z_by_frame, const std::vector<std::vector<float>>& raw_cam_t_z_by_frame,
    const std::vector<std::vector<int>>& track_ids_by_frame, float k_metric, int min_ratio_samples = 5
);

#pragma once

// MANO hand reconstruction from generative parameters. A trial CSV stores the
// MANO shape/pose parameters per detected hand (see data/SAVE_ALL.md); feeding
// them through the MANO layer here regenerates the 778 surface vertices and 21
// joints exactly, which replaces the precomputed .hmesh vertex dumps.
//
// The model weights (template mesh, shape/pose blendshapes, joint regressor,
// linear-blend-skinning weights, kinematic tree) are loaded once at startup from
// a binary exported from MANO_RIGHT.pkl. Only the right-hand model is stored; a
// left hand is produced by mirroring the result on X (the same flip HaMeR bakes
// into its outputs).

#include <array>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

/// The generative parameters for one detected hand, as read from a CSV row.
/// Rotations are flattened 3x3 matrices in row-major order (the model runs MANO
/// with ``pose2rot=False``), matching the CSV's ``gorient_*`` / ``pose_*`` layout.
struct ManoParams {
    std::array<float, 10> betas{};        // MANO shape coefficients
    std::array<float, 9> global_orient{}; // wrist orientation, one 3x3 (row-major)
    std::array<float, 135> hand_pose{};   // 15 joints x (3x3), row-major
    bool is_right = true;                 // handedness (drives the X mirror)
    glm::vec3 cam_t{0.0f};                // raw HaMeR camera-space translation (not yet metric)
    glm::vec2 wrist_uv{0.0f};             // j0_u/j0_v: the wrist's observed 2D pixel
};

/// The study's physical camera model, used to backproject a hand's observed 2D
/// wrist pixel into a real metric 3D position (see place_hand_metric) and, once
/// the tracked-object pipeline is ported, to backproject SAM3+DA3 object points
/// the same way. Defaults mirror scene_placement.py's CLI defaults, calibrated
/// for this study's rig (fx=fy=6900, k_metric=0.354 — see METRIC_SCENE_PIPELINE.md
/// on the A6000 render-script side).
struct CameraIntrinsics {
    float fx = 6900.0f;
    float fy = 6900.0f;
    float cx = 960.0f;
    float cy = 540.0f;
    // Empirical scale correcting HaMeR's depth (cam_t.z) to real meters — and,
    // as of place_hand_metric's current formula, indirectly the mesh size too
    // (see place_hand_metric's doc comment for why they share one root cause).
    // Must be re-derived if fx/fy change — it is NOT portable across focal
    // lengths. See METRIC_SCENE_PIPELINE.md.
    float k_metric = 0.354f;
};

/// The trial-independent k_metric known to be correct for a given focal
/// length, from calibrations done so far (fx=6900 -> 0.354, the study-wide
/// override; fx=3963.732421875 -> 0.579, AE45 T1_BabyView's VGGT-omega-
/// estimated focal length). Matched within 1 pixel to absorb float rounding
/// in a CSV's own scaled_focal_length column. Returns nullopt for any other
/// focal length — k_metric is NOT portable across focal lengths (see
/// CameraIntrinsics::k_metric), so an unrecognized one has no safe default.
std::optional<float> known_k_metric_for_focal_length(float fx);

/// One reconstructed hand in the viewer's on-screen space: the 778 MANO surface
/// vertices and 21 joints, already mirrored for handedness, offset by ``cam_t``,
/// and rotated 180 degrees about X (negated y/z) to match the pose the rest of
/// the viewer draws in.
struct ManoHand {
    std::vector<glm::vec3> verts;
    std::vector<glm::vec3> joints;
};

/// Load the MANO model weights from ``model_path`` (the exported
/// ``mano_model.bin``) once at startup. Throws std::runtime_error if the file is
/// missing or malformed.
void init_mano_model(const std::string& model_path);

/// True once init_mano_model has successfully loaded the weights.
bool mano_model_loaded();

/// Run the MANO forward pass for one hand. init_mano_model must have run first.
/// ``params.cam_t`` is applied as a flat translation before the on-screen flip —
/// pass a zeroed cam_t (see mano_forward_local) to get the hand's LOCAL mesh for
/// metric placement instead.
ManoHand mano_forward(const ManoParams& params);

/// The hand's LOCAL mesh: mano_forward with cam_t forced to zero, i.e. mirrored
/// for handedness and on-screen-flipped but not translated. This is the input
/// place_hand_metric expects (mirrors scene_placement.py's local_verts/local_wrist,
/// which come from the same "mano_forward with cam_t=(0,0,0)" call).
ManoHand mano_forward_local(const ManoParams& params);

/// Places a hand's LOCAL points (verts or joints) at their true metric position
/// AND size: the wrist is repositioned via real pinhole backprojection of
/// ``wrist_uv`` at depth ``z_metric``, and the whole mesh is scaled toward that
/// wrist by ``z_metric / raw_cam_t_z``.
///
/// That scale ratio corrects mesh size and wrist position from the SAME root
/// cause: HaMeR's MANO shape (beta) regression is trained mostly on adult
/// hands and doesn't shrink correctly for infant subjects, so its weak-
/// perspective solver compensates for the oversized mesh by placing it
/// proportionally farther away to keep the 2D silhouette's apparent size
/// correct — i.e. ``raw_cam_t_z`` is inflated by the same factor the mesh is
/// oversized by. Scaling by THIS frame's actual ratio (rather than a fixed
/// target span, or the trial-wide average k_metric) keeps size and position
/// consistent frame-by-frame: when ``z_metric`` is a real per-frame DA3
/// measurement, it can differ from ``k_metric * raw_cam_t_z`` by +/-30-40% on
/// any given frame, and using the fixed average instead of this frame's own
/// ratio would move the wrist to the correct depth while leaving the mesh
/// sized as if it were still at the k_metric-implied depth. When ``z_metric``
/// falls back to ``k_metric * raw_cam_t_z`` (no robust DA3 sample), this
/// ratio reduces exactly to ``k_metric``, so that fallback case is
/// unaffected. Mirrors scene_placement.place_hand_metric exactly.
std::vector<glm::vec3> place_hand_metric(
    const std::vector<glm::vec3>& local_points, const glm::vec3& local_wrist, const glm::vec2& wrist_uv, float raw_cam_t_z, float z_metric,
    const CameraIntrinsics& intrinsics
);

#pragma once

// A rigid tracked object (cube or sphere) reconstructed alongside a trial's
// hands, from the SAME real-metric SAM3+DA3 pipeline the A6000 render script
// (render_scene_video.py / scene_placement.py) uses — see data/object_fit.h
// for the backprojection/fitting math this ports, and data/npy_reader.h for
// reading the SAM3 mask / DA3 depth caches themselves. Unlike the hands
// (data/mano_model.h's place_hand_metric, one MANO forward pass per frame),
// the whole trial's object pose sequence is computed once at construction —
// cube fitting needs the previous frame's pose to initialize from, and
// temporal smoothing needs every frame's raw fit before it can smooth any of
// them, so there's no way to fit "just frame N" in isolation.

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>

#include "data/mano_model.h" // CameraIntrinsics

enum class ObjectShape { None, Cube, Sphere };

/// One frame's tracked-object pose, already in the viewer's on-screen
/// ("placed") space — the same Y-up, camera-looking-down-(-Z) convention
/// place_hand_metric produces. ``basis`` maps the object's local axes into
/// that space (identity for a sphere, since it's rotation-invariant); the
/// renderer builds ``translate(center) * mat4(basis) * scale(half_extent)``
/// around a [-1, 1] unit cube or unit sphere mesh.
struct CubePose {
    glm::vec3 center{0.0f};
    glm::vec3 half_extent{0.0f};
    glm::mat3 basis{1.0f};
};

/// One trial's tracked-object motion, computed once at construction by
/// backprojecting SAM3 masks through DA3 depth maps and fitting ``shape`` to
/// the resulting point cloud, frame by frame (mirrors
/// render_scene_video.precompute_smoothed_depths' object-only paths).
///
/// NOT ported from the Python reference (see data/object_fit.h's file-level
/// comment for why): 2D-silhouette pose refinement and reprojection-gated
/// accept/reject of individual frames' cube fits. Every frame whose fit
/// succeeds is accepted outright, so a badly-occluded frame's cube can be
/// noisier here than in the Python renderer's output.
class TrackedObjectSequence {
public:
    /// ``sam3_dir``/``da3_dir`` are the trial's ``arrays`` directories (e.g.
    /// ``<sam3_folder>/<subject>/<trial>/arrays``). ``object_label`` matches
    /// the SAM3 mask filename (e.g. "small_ball", "small_cube").
    /// ``object_size_m`` is the object's true diameter (sphere) or edge
    /// length (cube). ``frame_numbers`` are the CSV's 1-indexed frame
    /// numbers to attempt (missing SAM3/DA3 files for a given frame are
    /// simply skipped, same as a frame the object wasn't detected in).
    /// A ``shape`` of ``None`` (or an empty ``sam3_dir``/``da3_dir``)
    /// leaves an empty sequence — cheap to construct, safe to call
    /// unconditionally.
    TrackedObjectSequence(
        const std::string& sam3_dir, const std::string& da3_dir, const std::string& object_label, ObjectShape shape, float object_size_m,
        const CameraIntrinsics& intrinsics, const std::vector<int>& frame_numbers, float smooth_sigma_frames = 3.0f
    );

    ObjectShape shape() const { return shape_; }

    /// The object's pose for the given 1-indexed frame number, or nullopt if
    /// it wasn't detected/fit in that frame.
    std::optional<CubePose> pose_for_frame(int frame_number) const;

    /// Number of frames with an object pose.
    int pose_count() const { return static_cast<int>(poses_.size()); }

private:
    ObjectShape shape_ = ObjectShape::None;
    std::unordered_map<int, CubePose> poses_;
};

/// The model matrix that places a unit ([-1, 1]) cube or sphere mesh at
/// ``pose`` in the scene.
glm::mat4 cube_model_matrix(const CubePose& pose);

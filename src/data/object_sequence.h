#pragma once

// A rigid tracked object (currently a cube) reconstructed alongside a trial's
// hands. The object lives in the same viewer "placed" space as the hands (see
// data/mano_model.cpp's place transform), so it is drawn with the same scene fit
// and depth stabilisation the hands use — letting the cube sit correctly in the
// baby's hand. The per-frame pose is parsed once from a sidecar CSV
// (``cube_3d_<subject>_<trial>.csv``); the geometry itself is a unit cube the
// renderer scales and orients with each frame's pose.

#include <optional>
#include <string>
#include <unordered_map>

#include <glm/glm.hpp>

#include "data/mesh_sequence.h" // Transform

/// One frame's cube pose in placed space. ``basis`` already maps the cube's local
/// axes into placed space (the CSV's camera-space orientation composed with the
/// camera→placed flip), so the renderer just builds
/// ``translate(center) * mat4(basis) * scale(half_extent)`` around a [-1, 1] cube.
struct CubePose {
    glm::vec3 center{0.0f};      // placed-space centre (placed_center_*)
    glm::vec3 half_extent{0.0f}; // OBB half-extents (obb_half_*)
    glm::mat3 basis{1.0f};       // local→placed rotation (placed-space OBB axes)
    float placed_mean_z = 0.0f;  // mean placed depth, for the same depth snap the hands use
};

/// One trial's tracked-cube motion, parsed from its sidecar CSV at construction.
/// Poses are keyed by the CSV's 1-indexed ``frame`` number (matching the hand
/// CSV's frame column); frames where the cube was not visible simply have no
/// entry. Throws if the file cannot be opened.
class CubeSequence {
public:
    explicit CubeSequence(const std::string& csv_path);

    /// The cube's pose for the given 1-indexed frame number, or nullopt if the
    /// cube was not detected in that frame.
    std::optional<CubePose> pose_for_frame(int frame_number) const;

    /// Number of frames with a cube pose.
    int pose_count() const { return static_cast<int>(poses_.size()); }

private:
    std::unordered_map<int, CubePose> poses_;
};

/// The model matrix that places a unit ([-1, 1]) cube at ``pose`` in the scene.
/// Mirrors the hand placement (FrameGpu::hand_matrix): the same scene fit
/// ``transform`` and the same about-origin depth snap to ``reference_depth`` are
/// applied, so the cube stays glued to the hand it sits in. ``transform`` may be
/// null (no scene fit) and ``reference_depth`` empty (no depth snap).
glm::mat4 cube_model_matrix(const CubePose& pose, const Transform* transform, std::optional<float> reference_depth);

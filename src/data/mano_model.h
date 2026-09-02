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
    glm::vec3 cam_t{0.0f};                // full-frame camera translation
};

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
ManoHand mano_forward(const ManoParams& params);

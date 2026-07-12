#include "data/mano_model.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace {
    // Fixed MANO dimensions. The exporter writes these in the header too and they
    // are validated on load, but the forward pass relies on them being constant.
    constexpr int kNumVerts = 778;
    constexpr int kNumBetas = 10;
    constexpr int kNumJoints = 16; // wrist + 15 articulated joints
    constexpr int kNumPose = 135;  // 15 joints x 3x3
    constexpr int kNumOutJoints = 21;
    constexpr int kNumTips = 5;

    // The whole MANO model, loaded once. Arrays mirror the numpy layouts dumped by
    // the exporter; see the index helpers in the forward pass for the orderings.
    struct ManoModel {
        bool loaded = false;
        std::vector<glm::vec3> v_template;     // [v]
        std::vector<float> shapedirs;          // [v][3][beta]
        std::vector<float> posedirs;           // [v][3][pose]
        std::vector<float> joint_regressor;    // [joint][v]
        std::vector<float> weights;            // [v][joint]
        std::array<int, kNumJoints> parents{}; // kinematic parent (-1 for wrist)
        std::array<int, kNumTips> tip_verts{}; // fingertip vertex ids
    };

    ManoModel g_model;

    template <typename T>
    void read_into(std::ifstream& file, T* dest, std::size_t count) {
        file.read(reinterpret_cast<char*>(dest), static_cast<std::streamsize>(count * sizeof(T)));
        if (!file) {
            throw std::runtime_error("Truncated MANO model file");
        }
    }

    // Build a rigid transform from a row-major 3x3 rotation and a translation. The
    // result is a standard math matrix (glm operator* is correct regardless of the
    // column-major storage), so composition and point transforms behave normally.
    glm::mat4 rigid_transform(const float* rotation, const glm::vec3& translation) {
        glm::mat4 result(1.0f);
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                result[col][row] = rotation[row * 3 + col]; // math element (row, col)
            }
            result[3][row] = translation[row];
        }
        return result;
    }
} // namespace

void init_mano_model(const std::string& model_path) {
    std::ifstream file(model_path, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Could not open MANO model " + model_path);
    }

    char magic[4];
    read_into(file, magic, 4);
    if (std::memcmp(magic, "MANO", 4) != 0) {
        throw std::runtime_error("Not a MANO model file: " + model_path);
    }
    std::uint32_t header[5];
    read_into(file, header, 5); // version, num_verts, num_betas, num_joints, num_pose
    if (header[1] != kNumVerts || header[2] != kNumBetas || header[3] != kNumJoints || header[4] != kNumPose) {
        throw std::runtime_error("Unexpected MANO model dimensions in " + model_path);
    }

    ManoModel model;
    std::vector<float> verts(static_cast<std::size_t>(kNumVerts) * 3);
    read_into(file, verts.data(), verts.size());
    model.v_template.resize(kNumVerts);
    for (int vertex = 0; vertex < kNumVerts; ++vertex) {
        model.v_template[vertex] = glm::vec3(verts[vertex * 3 + 0], verts[vertex * 3 + 1], verts[vertex * 3 + 2]);
    }

    model.shapedirs.resize(static_cast<std::size_t>(kNumVerts) * 3 * kNumBetas);
    read_into(file, model.shapedirs.data(), model.shapedirs.size());
    model.posedirs.resize(static_cast<std::size_t>(kNumVerts) * 3 * kNumPose);
    read_into(file, model.posedirs.data(), model.posedirs.size());
    model.joint_regressor.resize(static_cast<std::size_t>(kNumJoints) * kNumVerts);
    read_into(file, model.joint_regressor.data(), model.joint_regressor.size());
    model.weights.resize(static_cast<std::size_t>(kNumVerts) * kNumJoints);
    read_into(file, model.weights.data(), model.weights.size());

    std::array<std::int32_t, kNumJoints> parents{};
    read_into(file, parents.data(), parents.size());
    for (int joint = 0; joint < kNumJoints; ++joint) {
        model.parents[joint] = parents[joint];
    }
    std::array<std::int32_t, kNumTips> tips{};
    read_into(file, tips.data(), tips.size());
    for (int tip = 0; tip < kNumTips; ++tip) {
        model.tip_verts[tip] = tips[tip];
    }

    model.loaded = true;
    g_model = std::move(model);
}

bool mano_model_loaded() { return g_model.loaded; }

ManoHand mano_forward(const ManoParams& params) {
    if (!g_model.loaded) {
        throw std::runtime_error("MANO model not initialised (call init_mano_model first)");
    }
    const ManoModel& model = g_model;

    // The 16 rotation matrices: wrist orientation then the 15 articulated joints,
    // all kept row-major as read from the CSV.
    std::array<std::array<float, 9>, kNumJoints> rotations{};
    rotations[0] = {
        params.global_orient[0],
        params.global_orient[1],
        params.global_orient[2],
        params.global_orient[3],
        params.global_orient[4],
        params.global_orient[5],
        params.global_orient[6],
        params.global_orient[7],
        params.global_orient[8]
    };
    for (int joint = 1; joint < kNumJoints; ++joint) {
        for (int element = 0; element < 9; ++element) {
            rotations[joint][element] = params.hand_pose[(joint - 1) * 9 + element];
        }
    }

    // Shape: v_shaped = v_template + shapedirs . betas.
    std::vector<glm::vec3> v_shaped(kNumVerts);
    for (int vertex = 0; vertex < kNumVerts; ++vertex) {
        glm::vec3 offset(0.0f);
        for (int axis = 0; axis < 3; ++axis) {
            const std::size_t base = (static_cast<std::size_t>(vertex) * 3 + axis) * kNumBetas;
            float sum = 0.0f;
            for (int beta = 0; beta < kNumBetas; ++beta) {
                sum += model.shapedirs[base + beta] * params.betas[beta];
            }
            offset[axis] = sum;
        }
        v_shaped[vertex] = model.v_template[vertex] + offset;
    }

    // Rest-pose joints regressed from the shaped mesh.
    std::array<glm::vec3, kNumJoints> rest_joints{};
    for (int joint = 0; joint < kNumJoints; ++joint) {
        glm::vec3 sum(0.0f);
        const std::size_t base = static_cast<std::size_t>(joint) * kNumVerts;
        for (int vertex = 0; vertex < kNumVerts; ++vertex) {
            sum += model.joint_regressor[base + vertex] * v_shaped[vertex];
        }
        rest_joints[joint] = sum;
    }

    // Pose blendshapes: feature is (R - I) for the 15 articulated joints, flattened
    // row-major (matching numpy's (R[1:] - I).reshape(-1)).
    std::array<float, kNumPose> pose_feature{};
    for (int joint = 1; joint < kNumJoints; ++joint) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                const float identity = (row == col) ? 1.0f : 0.0f;
                pose_feature[(joint - 1) * 9 + row * 3 + col] = rotations[joint][row * 3 + col] - identity;
            }
        }
    }
    std::vector<glm::vec3> v_posed(kNumVerts);
    for (int vertex = 0; vertex < kNumVerts; ++vertex) {
        glm::vec3 offset(0.0f);
        for (int axis = 0; axis < 3; ++axis) {
            const std::size_t base = (static_cast<std::size_t>(vertex) * 3 + axis) * kNumPose;
            float sum = 0.0f;
            for (int pose = 0; pose < kNumPose; ++pose) {
                sum += model.posedirs[base + pose] * pose_feature[pose];
            }
            offset[axis] = sum;
        }
        v_posed[vertex] = v_shaped[vertex] + offset;
    }

    // Walk the kinematic tree to build each joint's global transform.
    std::array<glm::mat4, kNumJoints> global{};
    global[0] = rigid_transform(rotations[0].data(), rest_joints[0]);
    for (int joint = 1; joint < kNumJoints; ++joint) {
        const int parent = model.parents[joint];
        const glm::vec3 local_offset = rest_joints[joint] - rest_joints[parent];
        global[joint] = global[parent] * rigid_transform(rotations[joint].data(), local_offset);
    }

    // Strip the rest pose so each transform maps rest vertices to posed ones.
    std::array<glm::mat4, kNumJoints> skinning{};
    for (int joint = 0; joint < kNumJoints; ++joint) {
        skinning[joint] = global[joint] * rigid_transform(std::array<float, 9>{1, 0, 0, 0, 1, 0, 0, 0, 1}.data(), -rest_joints[joint]);
    }

    // Linear blend skinning.
    ManoHand hand;
    hand.verts.resize(kNumVerts);
    for (int vertex = 0; vertex < kNumVerts; ++vertex) {
        glm::mat4 blended(0.0f);
        const std::size_t base = static_cast<std::size_t>(vertex) * kNumJoints;
        for (int joint = 0; joint < kNumJoints; ++joint) {
            blended += model.weights[base + joint] * skinning[joint];
        }
        const glm::vec4 posed = blended * glm::vec4(v_posed[vertex], 1.0f);
        hand.verts[vertex] = glm::vec3(posed);
    }

    // Assemble the 21 OpenPose joints: wrist, then four per finger in thumb, index,
    // middle, ring, pinky order. The first three of each finger come from the MANO
    // regressed joints; the fourth (tip) is a posed surface vertex.
    static constexpr std::array<int, 5> kFingerRoots = {13, 1, 4, 10, 7}; // thumb..pinky
    hand.joints.reserve(kNumOutJoints);
    hand.joints.push_back(glm::vec3(global[0][3]));
    for (int finger = 0; finger < 5; ++finger) {
        const int root = kFingerRoots[finger];
        hand.joints.push_back(glm::vec3(global[root][3]));
        hand.joints.push_back(glm::vec3(global[root + 1][3]));
        hand.joints.push_back(glm::vec3(global[root + 2][3]));
        hand.joints.push_back(hand.verts[model.tip_verts[finger]]);
    }

    // Mirror for handedness, offset into camera space, then rotate 180 degrees
    // about X (negate y/z) into the on-screen pose the rest of the viewer uses.
    const float sign = params.is_right ? 1.0f : -1.0f;
    auto place = [&](glm::vec3 point) {
        point.x = sign * point.x + params.cam_t.x;
        point.y = -(point.y + params.cam_t.y);
        point.z = -(point.z + params.cam_t.z);
        return point;
    };
    for (glm::vec3& vertex : hand.verts) {
        vertex = place(vertex);
    }
    for (glm::vec3& joint : hand.joints) {
        joint = place(joint);
    }
    return hand;
}

std::optional<float> known_k_metric_for_focal_length(float fx) {
    static constexpr std::array<std::pair<float, float>, 2> known = {{
        {6900.0f, 0.354f},
        {3963.732421875f, 0.579f},
    }};
    for (const auto& [known_fx, k_metric] : known) {
        if (std::abs(fx - known_fx) < 1.0f) {
            return k_metric;
        }
    }
    return std::nullopt;
}

ManoHand mano_forward_local(const ManoParams& params) {
    ManoParams local = params;
    local.cam_t = glm::vec3(0.0f);
    return mano_forward(local);
}

std::vector<glm::vec3> place_hand_metric(
    const std::vector<glm::vec3>& local_points, const glm::vec3& local_wrist, const glm::vec2& wrist_uv, float raw_cam_t_z, float z_metric,
    const CameraIntrinsics& intrinsics
) {
    const float x_metric = (wrist_uv.x - intrinsics.cx) * z_metric / intrinsics.fx;
    const float y_metric = (wrist_uv.y - intrinsics.cy) * z_metric / intrinsics.fy;
    const glm::vec3 metric_wrist(x_metric, -y_metric, -z_metric);
    const float scale = raw_cam_t_z != 0.0f ? z_metric / raw_cam_t_z : 1.0f;

    std::vector<glm::vec3> placed;
    placed.reserve(local_points.size());
    for (const glm::vec3& point : local_points) {
        placed.push_back(scale * (point - local_wrist) + metric_wrist);
    }
    return placed;
}

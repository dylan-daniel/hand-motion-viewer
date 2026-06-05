#include "crossview.h"

#include "npy.h"
#include "sequence.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <map>
#include <regex>

namespace {
    namespace fs = std::filesystem;

    // HaMeR camera-space joints come out in a vision frame (y/z down/forward);
    // the exported mesh flips y and z to the GL frame. Applying the same flip to
    // both views' joints keeps the estimate in the mesh space the viewer renders.
    const glm::vec3 kObjAxisFlip = {1.0f, -1.0f, -1.0f};

    constexpr int kJointCount = 21;
    constexpr int kRefineIterations = 8;

    // frame_0001_0.hmesh → frame number 0001.
    const std::regex kFrameNumberRe(R"(frame_(\d+)_\d+\.hmesh$)", std::regex::icase);

    int frame_number_of(const std::string& mesh_path) {
        const std::string name = fs::path(mesh_path).filename().string();
        std::smatch match;
        if (!std::regex_search(name, match, kFrameNumberRe)) {
            return -1;
        }
        return std::stoi(match[1].str());
    }

    /// One hand's 21 mesh-space 3D joints, read from the npy sidecars beside its
    /// ``.hmesh``. Empty if either sidecar is missing or the wrong shape.
    std::vector<glm::vec3> read_hand_joints(const std::string& mesh_path) {
        const std::string base = mesh_path.substr(0, mesh_path.size() - 6); // strip ".hmesh"
        const NpyArray joints = load_npy_f32(base + "_joints3d.npy");
        const NpyArray cam = load_npy_f32(base + "_cam.npy");
        if (joints.size() != static_cast<std::size_t>(kJointCount) * 3 || cam.size() != 3) {
            return {};
        }
        const glm::vec3 translation(cam.data[0], cam.data[1], cam.data[2]);
        std::vector<glm::vec3> result;
        result.reserve(kJointCount);
        for (int joint = 0; joint < kJointCount; ++joint) {
            const glm::vec3 local(joints.data[joint * 3 + 0], joints.data[joint * 3 + 1], joints.data[joint * 3 + 2]);
            result.push_back(kObjAxisFlip * (local + translation));
        }
        return result;
    }

    // All hands (each a 21-joint set) for a folder, grouped by frame number.
    using HandSet = std::vector<std::vector<glm::vec3>>;
    std::map<int, HandSet> load_view_joints(const std::string& folder) {
        std::map<int, HandSet> by_frame;
        for (const std::vector<std::string>& frame : discover_frames(folder)) {
            for (const std::string& obj_path : frame) {
                std::vector<glm::vec3> joints = read_hand_joints(obj_path);
                if (joints.size() == static_cast<std::size_t>(kJointCount)) {
                    by_frame[frame_number_of(obj_path)].push_back(std::move(joints));
                }
            }
        }
        return by_frame;
    }

    // ── 3×3 symmetric eigensolver (cyclic Jacobi) ──
    // Returns eigenvalues (descending) and eigenvectors as the columns of vectors.
    void jacobi_eigen(const glm::mat3& symmetric, glm::vec3& values, glm::mat3& vectors) {
        glm::mat3 matrix = symmetric;
        vectors = glm::mat3(1.0f);
        for (int sweep = 0; sweep < 64; ++sweep) {
            // Largest off-diagonal magnitude; stop once negligible.
            int axis_p = 0;
            int axis_q = 1;
            float largest = 0.0f;
            for (int row = 0; row < 3; ++row) {
                for (int col = row + 1; col < 3; ++col) {
                    const float magnitude = std::fabs(matrix[col][row]);
                    if (magnitude > largest) {
                        largest = magnitude;
                        axis_p = row;
                        axis_q = col;
                    }
                }
            }
            if (largest < 1e-12f) {
                break;
            }
            const float app = matrix[axis_p][axis_p];
            const float aqq = matrix[axis_q][axis_q];
            const float apq = matrix[axis_q][axis_p];
            const float phi = 0.5f * std::atan2(2.0f * apq, aqq - app);
            const float cosine = std::cos(phi);
            const float sine = std::sin(phi);

            glm::mat3 rotation(1.0f);
            rotation[axis_p][axis_p] = cosine;
            rotation[axis_q][axis_q] = cosine;
            rotation[axis_q][axis_p] = sine;
            rotation[axis_p][axis_q] = -sine;
            // matrix = rotationᵀ * matrix * rotation; vectors accumulate rotation.
            matrix = glm::transpose(rotation) * matrix * rotation;
            vectors = vectors * rotation;
        }
        values = glm::vec3(matrix[0][0], matrix[1][1], matrix[2][2]);

        // Sort eigenpairs by eigenvalue, descending.
        int order[3] = {0, 1, 2};
        std::sort(order, order + 3, [&](int lhs, int rhs) { return values[lhs] > values[rhs]; });
        const glm::vec3 sorted_values(values[order[0]], values[order[1]], values[order[2]]);
        const glm::mat3 sorted_vectors(vectors[order[0]], vectors[order[1]], vectors[order[2]]);
        values = sorted_values;
        vectors = sorted_vectors;
    }

    /// Scaled Kabsch fit: best ``scale * R * src + t`` mapping src onto dst.
    SimilarityTransform fit_similarity(const std::vector<glm::vec3>& src, const std::vector<glm::vec3>& dst) {
        SimilarityTransform result;
        const std::size_t count = src.size();
        if (count == 0) {
            return result;
        }
        glm::vec3 mean_src(0.0f);
        glm::vec3 mean_dst(0.0f);
        for (std::size_t index = 0; index < count; ++index) {
            mean_src += src[index];
            mean_dst += dst[index];
        }
        mean_src /= static_cast<float>(count);
        mean_dst /= static_cast<float>(count);

        glm::mat3 covariance(0.0f);
        float src_variance = 0.0f;
        for (std::size_t index = 0; index < count; ++index) {
            const glm::vec3 centered_src = src[index] - mean_src;
            const glm::vec3 centered_dst = dst[index] - mean_dst;
            // outer product centered_src ⊗ centered_dst, accumulated.
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    covariance[col][row] += centered_src[row] * centered_dst[col];
                }
            }
            src_variance += glm::dot(centered_src, centered_src);
        }

        // SVD of covariance via eigen-decomposition of covarianceᵀ·covariance.
        const glm::mat3 normal = glm::transpose(covariance) * covariance;
        glm::vec3 eigenvalues;
        glm::mat3 right_vectors; // columns are V
        jacobi_eigen(normal, eigenvalues, right_vectors);
        glm::vec3 singular(std::sqrt(std::max(eigenvalues[0], 0.0f)), std::sqrt(std::max(eigenvalues[1], 0.0f)), std::sqrt(std::max(eigenvalues[2], 0.0f)));

        glm::mat3 left_vectors(1.0f); // U = covariance · V · Σ⁻¹
        for (int col = 0; col < 3; ++col) {
            const glm::vec3 v_col = right_vectors[col];
            const glm::vec3 mapped = covariance * v_col;
            left_vectors[col] = singular[col] > 1e-8f ? mapped / singular[col] : mapped;
        }

        // R = V · diag(1,1,d) · Uᵀ, with d fixing any reflection.
        glm::mat3 rotation = right_vectors * glm::transpose(left_vectors);
        float reflection = glm::determinant(rotation) < 0.0f ? -1.0f : 1.0f;
        if (reflection < 0.0f) {
            glm::mat3 corrected = right_vectors;
            corrected[2] = -corrected[2];
            rotation = corrected * glm::transpose(left_vectors);
        }

        const float trace = singular[0] + singular[1] + reflection * singular[2];
        const float scale = src_variance > 1e-12f ? trace / src_variance : 1.0f;
        result.scale = scale;
        result.rotation = rotation;
        result.translation = mean_dst - scale * (rotation * mean_src);
        return result;
    }

    float squared_distance(const glm::vec3& lhs, const glm::vec3& rhs) {
        const glm::vec3 diff = lhs - rhs;
        return glm::dot(diff, diff);
    }
} // namespace

std::shared_ptr<CrossViewOverlay> CrossViewOverlay::create(const std::string& baby_folder) {
    // Resolve the sibling OHView folder name.
    const std::string marker = "BabyView";
    const std::size_t at = baby_folder.rfind(marker);
    if (at == std::string::npos) {
        return nullptr;
    }
    std::string oh_folder = baby_folder;
    oh_folder.replace(at, marker.size(), "OHView");
    if (!fs::is_directory(oh_folder)) {
        return nullptr;
    }

    const std::map<int, HandSet> baby = load_view_joints(baby_folder);
    const std::map<int, HandSet> over = load_view_joints(oh_folder);
    if (baby.empty() || over.empty()) {
        return nullptr;
    }

    // Iterative closest-hand matching: match each BabyView hand to the OHView hand
    // in the same frame that best fits the current transform, then refit.
    SimilarityTransform transform;
    transform.scale = 0.5f; // OHView tends to sit at a larger metric scale
    float residual_rms = 0.0f;
    for (int iteration = 0; iteration < kRefineIterations; ++iteration) {
        std::vector<glm::vec3> src; // OHView joints
        std::vector<glm::vec3> dst; // BabyView joints
        for (const auto& [frame_number, baby_hands] : baby) {
            const auto over_it = over.find(frame_number);
            if (over_it == over.end()) {
                continue;
            }
            const HandSet& over_hands = over_it->second;
            for (const std::vector<glm::vec3>& baby_hand : baby_hands) {
                const std::vector<glm::vec3>* best = nullptr;
                float best_cost = std::numeric_limits<float>::max();
                for (const std::vector<glm::vec3>& over_hand : over_hands) {
                    float cost = 0.0f;
                    for (int joint = 0; joint < kJointCount; ++joint) {
                        cost += squared_distance(transform.apply(over_hand[joint]), baby_hand[joint]);
                    }
                    if (cost < best_cost) {
                        best_cost = cost;
                        best = &over_hand;
                    }
                }
                if (best != nullptr) {
                    for (int joint = 0; joint < kJointCount; ++joint) {
                        src.push_back((*best)[joint]);
                        dst.push_back(baby_hand[joint]);
                    }
                }
            }
        }
        if (src.empty()) {
            return nullptr;
        }
        transform = fit_similarity(src, dst);

        double sum_squared = 0.0;
        for (std::size_t index = 0; index < src.size(); ++index) {
            sum_squared += squared_distance(transform.apply(src[index]), dst[index]);
        }
        residual_rms = static_cast<float>(std::sqrt(sum_squared / static_cast<double>(src.size())));
    }

    auto overlay = std::shared_ptr<CrossViewOverlay>(new CrossViewOverlay());
    overlay->transform_ = transform;
    overlay->residual_rms_ = residual_rms;
    overlay->oh_folder_ = oh_folder;

    // Align OHView obj paths to the BabyView frame ordering the loader will use.
    std::map<int, std::vector<std::string>> over_paths;
    for (const std::vector<std::string>& frame : discover_frames(oh_folder)) {
        if (!frame.empty()) {
            over_paths[frame_number_of(frame.front())] = frame;
        }
    }
    for (const std::vector<std::string>& frame : discover_frames(baby_folder)) {
        const int number = frame.empty() ? -1 : frame_number_of(frame.front());
        const auto found = over_paths.find(number);
        overlay->oh_paths_.push_back(found != over_paths.end() ? found->second : std::vector<std::string>{});
        overlay->mesh_total_ += static_cast<int>(overlay->oh_paths_.back().size());
    }
    return overlay;
}

const std::vector<std::string>& CrossViewOverlay::oh_paths(int index) const {
    if (index < 0 || index >= static_cast<int>(oh_paths_.size())) {
        return empty_;
    }
    return oh_paths_[static_cast<std::size_t>(index)];
}

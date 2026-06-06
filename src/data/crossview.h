#pragma once

// Cross-view alignment: place an over-hand (OHView) camera's hand meshes into a
// BabyView camera's coordinate space so both can be drawn together.
//
// Each view reconstructs hands monocularly in its OWN camera frame, so the raw
// meshes do not share an origin. Because both cameras film the same subject at
// the same time, a single similarity transform (scale + rotation + translation)
// relates the two frames. We recover it from the per-frame 3D joints (the
// ``*_joints3d.npy`` + ``*_cam.npy`` sidecars) via iterative closest-hand
// matching plus a scaled Kabsch fit. Depth is the weak axis (monocular depth is
// unreliable), so the fit is honest but not metrically exact in z.

#include <memory>
#include <string>
#include <vector>

#include <glm/glm.hpp>

/// A uniform-scale rigid-plus-scale map: ``p' = scale * rotation * p + translation``.
struct SimilarityTransform {
    float scale = 1.0f;
    glm::mat3 rotation = glm::mat3(1.0f);
    glm::vec3 translation = glm::vec3(0.0f);

    glm::vec3 apply(const glm::vec3& point) const { return scale * (rotation * point) + translation; }
};

/// Recovered alignment for one take plus the OHView meshes to fold in.
class CrossViewOverlay {
public:
    /// Build the overlay for a BabyView mesh sequence folder by locating its sibling
    /// OHView folder (``*_BabyView`` → ``*_OHView``) and estimating the transform.
    /// Returns nullptr when there is no sibling folder or estimation fails.
    static std::shared_ptr<CrossViewOverlay> create(const std::string& baby_folder);

    const SimilarityTransform& transform() const { return transform_; }

    /// OHView ``.obj`` paths for BabyView frame ``index`` (same ordering the
    /// BabyView MeshSequenceLoader assigns), empty if that frame has no OHView hands.
    const std::vector<std::string>& oh_paths(int index) const;

    /// RGBA tint applied to the folded-in OHView hand surface so the two views
    /// read apart visually.
    const glm::vec4& tint() const { return tint_; }

    /// RMS per-joint alignment residual in BabyView mesh units (fit quality).
    float residual_rms() const { return residual_rms_; }
    const std::string& oh_folder() const { return oh_folder_; }

    /// Total OHView hand meshes folded across every frame (for load counters).
    int mesh_total() const { return mesh_total_; }

    /// OHView hand count for BabyView frame ``index`` (0 if none).
    int hand_count(int index) const { return static_cast<int>(oh_paths(index).size()); }

private:
    SimilarityTransform transform_;
    std::vector<std::vector<std::string>> oh_paths_;
    glm::vec4 tint_ = {0.95f, 0.55f, 0.25f, 1.0f}; // warm orange vs. the blue-grey BabyView hand
    float residual_rms_ = 0.0f;
    int mesh_total_ = 0;
    std::string oh_folder_;
    std::vector<std::string> empty_;
};

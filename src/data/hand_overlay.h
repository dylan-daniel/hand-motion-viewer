#pragma once

// SAM3 + DA3 point-cloud overlay: rescales an independently-estimated depth
// point cloud (SAM3 hand masks unprojected through DA3's per-frame metric
// depth/intrinsics) into the coordinate frame of a specific MANO hand, as a
// visual cross-check of the MANO reconstruction against an independent depth
// estimate. See dylan/_tmp_experiments_do_not_use/combined_3d_render.py in the
// hamer repo for the derivation this mirrors (align_hand_to_da3, inverted).
//
// HaMeR's mesh uses a fixed, assumed focal length baked into its training
// convention (``scaled_focal_length = 5000/256 * max(img_w, img_h)``),
// independent of the real camera; DA3 estimates a real per-frame pinhole
// camera. That assumed focal length is only used to *locate* the hand in the
// image (projecting the mesh's mean vertex to find the pixel to sample DA3's
// depth at) — it does not affect the mesh's own scale. MANO's vertex
// *positions* (cam_t) are in HaMeR's own inflated, uncalibrated units, but
// each vertex's *offset from the mesh's own mean* is already real-world
// metric scale (confirmed against the CSV's placed_bbox extents, which stay
// ~0.1-0.2 units regardless of a hand's absolute depth). So DA3's real
// metric offsets from its own anchor pixel need no rescaling to match —
// they're added directly to the mesh's mean vertex. This is a practical
// alignment tied to DA3's own depth accuracy, not a calibration, and the
// depth anchor is a single-pixel sample (noisier than a mask average, but
// needs no identity/track matching to work).

#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "data/mesh_sequence.h"

/// Resolved cache directories for one archive root (see resolve_cache_roots).
struct CacheRoots {
    std::string sam3_root; // <hand_cache_root>/sam3_cache
    std::string da3_root;  // <hand_cache_root>/da3_cache
};

/// Split ``hand_cache_root`` into its sam3_cache/da3_cache subdirectories.
/// Empty roots in the result mean ``hand_cache_root`` was empty.
CacheRoots resolve_cache_roots(const std::string& hand_cache_root);

/// True if both caches have a ``<subject>/<trial>`` directory, i.e. there is
/// overlay data to try loading for this trial. Cheap (existence checks only).
bool cache_available_for(const CacheRoots& roots, const std::string& subject, const std::string& trial);

/// Build the DA3+SAM3 point cloud for one hand in one frame, rescaled into
/// that hand's raw HaMeR vertex space (the same space ``HandData::verts``
/// lives in, i.e. matching ``mano_forward``'s ``place()`` output before the
/// scene's depth stabilisation and fit Transform). Returns an empty vector if
/// the hand's projected pixel falls outside the frame, lands on invalid DA3
/// depth, or the cache has no matching mask/depth for this frame.
///
/// ``img_width``/``img_height`` are the source video frame's pixel
/// dimensions (HaMeR's assumed focal length is derived from these).
/// ``reference_depth``, when set, applies the same per-hand depth
/// stabilisation the renderer applies to the mesh (see
/// ``FrameGpu::hand_matrix`` / ``stabilization_scale`` in mesh_sequence.cpp),
/// so the overlay lines up with the drawn (stabilised) hand.
std::vector<glm::vec3> build_hand_overlay_points(
    const CacheRoots& roots,
    const std::string& subject,
    const std::string& trial,
    int frame_number,
    const HandData& hand,
    int img_width,
    int img_height,
    std::optional<float> reference_depth
);

#pragma once

// Mesh data types and loading. A HaMeR mesh is split into the hand surface and
// the joint skeleton by vertex colour, and expanded into flat per-vertex arrays
// ready for a GPU vertex buffer.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

// HaMeR colours the MANO hand surface a uniform light blue-grey; the joint
// skeleton uses vivid per-finger colours. This lets us split the two by colour.
inline constexpr glm::ivec3 HAND_COLOR_8BIT = {166, 189, 219};

// Tint applied to a part that carries no per-vertex colours of its own.
inline constexpr glm::vec4 DEFAULT_COLOR = {0.6f, 0.75f, 0.9f, 1.0f};

/// A drawable chunk of geometry: shared vertices plus per-face indices/normals.
struct MeshPart {
    std::vector<glm::vec3> verts;
    std::vector<glm::ivec3> faces;
    std::vector<glm::vec3> normals; // one per face
    std::vector<glm::vec4> colors;  // per-vertex RGBA in 0..1 (may be empty)
};

/// Flat per-vertex arrays for one mesh part, ready for a GPU vertex buffer.
/// Triangles are expanded (no index buffer): every face contributes three
/// consecutive vertices, so positions/normals/colors share the same length.
struct MeshArrays {
    std::vector<float> positions; // 3 floats per vertex
    std::vector<float> normals;   // 3 floats per vertex
    std::vector<float> colors;    // 4 floats per vertex
    int vertex_count = 0;
};

/// GPU-ready vertex arrays for a loaded scene plus a little metadata.
struct PreparedMesh {
    MeshArrays joints;
    MeshArrays hand_solid;
    MeshArrays hand_translucent;
    int hand_triangle_count = 0;
    int joint_triangle_count = 0;
};

/// Expand a MeshPart into flat per-vertex arrays for GL_TRIANGLES drawing. If
/// ``alpha`` is given it overrides every vertex's alpha (the translucent hand).
MeshArrays build_arrays(const MeshPart& part, std::optional<float> alpha = std::nullopt);

/// Load an obj file and split it into (hand, joints) by vertex colour.
std::pair<MeshPart, MeshPart> load_mesh(const std::string& path);

/// Per-face unit normals for triangles, recomputed each frame for a sequence.
std::vector<glm::vec3> face_normals(const std::vector<glm::vec3>& positions, const std::vector<glm::ivec3>& faces);

/// Build GPU-ready arrays for one sequence hand from shared topology + positions.
PreparedMesh prepare_hand(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::ivec3>& faces,
    const std::vector<glm::vec4>& colors,
    const std::vector<std::uint8_t>& hand_face_mask,
    float alpha = 0.30f
);

/// Translate + scale verts so the model sits on the grid (y=0) and fits nicely;
/// returns the scale factor and centres ``verts`` in place.
float center_model(std::vector<glm::vec3>& verts);

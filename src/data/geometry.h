#pragma once

// Mesh data types and loading. A HaMeR mesh is split into the hand surface and
// the joint skeleton by vertex colour, and expanded into flat per-vertex arrays
// ready for a GPU vertex buffer.

#include <array>
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

// Target world-unit span a reference hand's largest extent is scaled to fit.
// The grid spans +/-10 units, so a hand of this size sits comfortably on it.
// Used by the sequence fit (compute_transform) as the one knob for the overall
// scene scale.
inline constexpr float MODEL_FIT_SPAN = 1.0f;

// ── MANO joint skeleton ────────────────────────
// The .hmesh format stores only the 21 joint positions, not the colored sphere/
// bone geometry the OBJ baked in, so we regenerate that visualization here. The
// definitions below mirror the authoritative ones in HaMeR's demo.py.

/// 20 (parent, child) bone pairs in standard MANO/OpenPose joint order.
inline constexpr std::array<glm::ivec2, 20> HAND_BONES = {{{0, 1},   {1, 2},   {2, 3},  {3, 4},   {0, 5},   {5, 6},   {6, 7},  {7, 8},   {0, 9},   {9, 10},
                                                           {10, 11}, {11, 12}, {0, 13}, {13, 14}, {14, 15}, {15, 16}, {0, 17}, {17, 18}, {18, 19}, {19, 20}}};

/// Per-finger colour: 0 wrist (white), 1 thumb, 2 index, 3 middle, 4 ring, 5 pinky.
inline constexpr std::array<glm::vec4, 6> FINGER_COLORS = {
    {{1.0f, 1.0f, 1.0f, 1.0f},
     {0.90f, 0.13f, 0.13f, 1.0f},
     {0.13f, 0.80f, 0.13f, 1.0f},
     {0.20f, 0.40f, 1.00f, 1.0f},
     {0.95f, 0.85f, 0.10f, 1.0f},
     {0.70f, 0.20f, 0.85f, 1.0f}}
};

/// Finger group a joint belongs to: 0 = wrist, 1 = thumb, …, 5 = pinky.
inline int finger_of(int joint) { return joint == 0 ? 0 : (joint - 1) / 4 + 1; }

/// A drawable chunk of geometry: shared vertices plus per-face indices/normals.
struct MeshPart {
    std::vector<glm::vec3> verts;
    std::vector<glm::ivec3> faces;
    std::vector<glm::vec3> normals; // one per face
    std::vector<glm::vec4> colors;  // per-vertex RGBA in 0..1 (may be empty)
};

/// Flat per-vertex arrays for one mesh part, ready for a GPU vertex buffer. When
/// ``indices`` is empty the triangles are expanded (every three consecutive
/// vertices form a face, drawn with glDrawArrays). When ``indices`` is set the
/// vertices are shared and the triangles are drawn indexed (glDrawElements),
/// which is how the hand surface is stored to avoid 6x-duplicating each vertex.
struct MeshArrays {
    std::vector<float> positions;       // 3 floats per vertex
    std::vector<float> normals;         // 3 floats per vertex
    std::vector<float> colors;          // 4 floats per vertex
    std::vector<std::uint16_t> indices; // triangle indices (empty => non-indexed)
    int vertex_count = 0;               // number of vertices (positions.size() / 3)
};

/// GPU-ready vertex arrays for a loaded scene plus a little metadata. The hand
/// surface is one mesh drawn either opaque or translucent (the alpha is a shader
/// uniform, so a single buffer serves both).
struct PreparedMesh {
    MeshArrays joints;
    MeshArrays hand;
    int hand_triangle_count = 0;
    int joint_triangle_count = 0;
};

/// Expand a MeshPart into flat per-vertex arrays for GL_TRIANGLES drawing. If
/// ``alpha`` is given it overrides every vertex's alpha.
MeshArrays build_arrays(const MeshPart& part, std::optional<float> alpha = std::nullopt);

/// Build an indexed surface mesh: one vertex per entry in ``verts`` (no
/// duplication), with smooth area-weighted per-vertex normals and a uniform
/// ``color``, plus a triangle index buffer from ``faces``. ~5x smaller than the
/// expanded form for a closed mesh like the MANO hand.
MeshArrays build_indexed_surface(const std::vector<glm::vec3>& verts, const std::vector<glm::ivec3>& faces, const glm::vec4& color);

// ── Binary .hmesh format ───────────────────────

/// One hand decoded from a ``.hmesh`` file: the 778 MANO surface vertices and 21
/// joint positions, both rotated 180° about X into the on-screen pose the OBJ
/// export used (so they line up with the rest of the viewer).
struct HMesh {
    bool is_right = true;
    std::vector<glm::vec3> verts;  // MANO surface vertices (778 for MANO)
    std::vector<glm::vec3> joints; // joint positions (21 for MANO)
};

/// Read a ``.hmesh`` binary file. Throws std::runtime_error on a bad header.
HMesh load_hmesh(const std::string& path);

/// Load the shared MANO face topology from ``mano_faces.bin`` once at startup.
/// Builds both the right-hand winding and the left-hand (flipped) winding so a
/// hand only needs its handedness to pick a face list. Throws if the file is
/// missing or malformed.
void init_mano_topology(const std::string& faces_path);

/// The shared MANO triangle indices for the given handedness. init_mano_topology
/// must have run first.
const std::vector<glm::ivec3>& mano_faces(bool is_right);

/// Build the colored joint skeleton (a sphere per joint, a cylinder per bone)
/// from the 21 joint positions. ``hand_radius`` is the hand's bounding-box
/// diagonal; markers scale to ~2% of it (matching the OBJ export).
MeshArrays build_joint_mesh(const std::vector<glm::vec3>& joints, float hand_diagonal);

/// Per-face unit normals for triangles, recomputed each frame for a sequence.
std::vector<glm::vec3> face_normals(const std::vector<glm::vec3>& positions, const std::vector<glm::ivec3>& faces);

/// Build GPU-ready arrays for one sequence hand: the indexed surface from
/// ``verts`` plus the shared MANO ``faces`` painted a uniform ``surface_color``,
/// and a regenerated joint skeleton from ``joints``.
PreparedMesh
prepare_hand(const std::vector<glm::vec3>& verts, const std::vector<glm::ivec3>& faces, const glm::vec4& surface_color, const std::vector<glm::vec3>& joints);

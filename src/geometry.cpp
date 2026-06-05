#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include <glm/gtc/constants.hpp>
#include <rapidobj/rapidobj.hpp>

MeshArrays build_arrays(const MeshPart& part, std::optional<float> alpha) {
    MeshArrays out;
    const std::size_t face_count = part.faces.size();
    if (face_count == 0) {
        return out;
    }
    out.vertex_count = static_cast<int>(face_count * 3);
    out.positions.reserve(face_count * 9);
    out.normals.reserve(face_count * 9);
    out.colors.reserve(face_count * 12);

    const bool has_colors = !part.colors.empty();
    for (std::size_t face = 0; face < face_count; ++face) {
        const glm::ivec3 indices = part.faces[face];
        const glm::vec3 normal = part.normals[face];
        for (int corner = 0; corner < 3; ++corner) {
            const int vertex = indices[corner];
            const glm::vec3& position = part.verts[static_cast<std::size_t>(vertex)];
            out.positions.push_back(position.x);
            out.positions.push_back(position.y);
            out.positions.push_back(position.z);

            out.normals.push_back(normal.x);
            out.normals.push_back(normal.y);
            out.normals.push_back(normal.z);

            glm::vec4 color = has_colors ? part.colors[static_cast<std::size_t>(vertex)] : DEFAULT_COLOR;
            if (alpha) {
                color.a = *alpha;
            }
            out.colors.push_back(color.r);
            out.colors.push_back(color.g);
            out.colors.push_back(color.b);
            out.colors.push_back(color.a);
        }
    }
    return out;
}

std::vector<glm::vec3> face_normals(const std::vector<glm::vec3>& positions, const std::vector<glm::ivec3>& faces) {
    std::vector<glm::vec3> normals;
    normals.reserve(faces.size());
    for (const glm::ivec3& face : faces) {
        const glm::vec3& a = positions[static_cast<std::size_t>(face[0])];
        const glm::vec3& b = positions[static_cast<std::size_t>(face[1])];
        const glm::vec3& c = positions[static_cast<std::size_t>(face[2])];
        glm::vec3 raw = glm::cross(b - a, c - a);
        float length = glm::length(raw);
        normals.push_back(length == 0.0f ? glm::vec3(0.0f) : raw / length);
    }
    return normals;
}

namespace {
    /// Bounding-box diagonal of a point set (0 if empty), the size cue the joint
    /// markers scale against.
    float bounding_diagonal(const std::vector<glm::vec3>& points) {
        if (points.empty()) {
            return 0.0f;
        }
        glm::vec3 low = points.front();
        glm::vec3 high = points.front();
        for (const glm::vec3& point : points) {
            low = glm::min(low, point);
            high = glm::max(high, point);
        }
        return glm::length(high - low);
    }
} // namespace

MeshArrays build_indexed_surface(const std::vector<glm::vec3>& verts, const std::vector<glm::ivec3>& faces, const glm::vec4& color) {
    MeshArrays out;
    const std::size_t vertex_count = verts.size();
    out.vertex_count = static_cast<int>(vertex_count);

    // Smooth per-vertex normals: sum each face's normal (its magnitude is twice
    // the triangle area, so larger faces weigh more) into its three vertices,
    // then normalize. This is what lets the vertices be shared.
    std::vector<glm::vec3> normals(vertex_count, glm::vec3(0.0f));
    out.indices.reserve(faces.size() * 3);
    for (const glm::ivec3& face : faces) {
        const glm::vec3& a = verts[static_cast<std::size_t>(face[0])];
        const glm::vec3& b = verts[static_cast<std::size_t>(face[1])];
        const glm::vec3& c = verts[static_cast<std::size_t>(face[2])];
        const glm::vec3 weighted = glm::cross(b - a, c - a);
        normals[static_cast<std::size_t>(face[0])] += weighted;
        normals[static_cast<std::size_t>(face[1])] += weighted;
        normals[static_cast<std::size_t>(face[2])] += weighted;
        out.indices.push_back(static_cast<std::uint16_t>(face[0]));
        out.indices.push_back(static_cast<std::uint16_t>(face[1]));
        out.indices.push_back(static_cast<std::uint16_t>(face[2]));
    }

    out.positions.reserve(vertex_count * 3);
    out.normals.reserve(vertex_count * 3);
    out.colors.reserve(vertex_count * 4);
    for (std::size_t vertex = 0; vertex < vertex_count; ++vertex) {
        const glm::vec3& position = verts[vertex];
        out.positions.push_back(position.x);
        out.positions.push_back(position.y);
        out.positions.push_back(position.z);

        const float length = glm::length(normals[vertex]);
        const glm::vec3 normal = length == 0.0f ? glm::vec3(0.0f) : normals[vertex] / length;
        out.normals.push_back(normal.x);
        out.normals.push_back(normal.y);
        out.normals.push_back(normal.z);

        out.colors.push_back(color.r);
        out.colors.push_back(color.g);
        out.colors.push_back(color.b);
        out.colors.push_back(color.a);
    }
    return out;
}

PreparedMesh
prepare_hand(const std::vector<glm::vec3>& verts, const std::vector<glm::ivec3>& faces, const glm::vec4& surface_color, const std::vector<glm::vec3>& joints) {
    PreparedMesh prepared;
    prepared.hand = build_indexed_surface(verts, faces, surface_color);
    prepared.joints = build_joint_mesh(joints, bounding_diagonal(verts));
    prepared.hand_triangle_count = static_cast<int>(faces.size());
    prepared.joint_triangle_count = static_cast<int>(prepared.joints.vertex_count / 3);
    return prepared;
}

float center_model(std::vector<glm::vec3>& verts) {
    if (verts.empty()) {
        return 1.0f;
    }
    glm::vec3 low = verts.front();
    glm::vec3 high = verts.front();
    for (const glm::vec3& vertex : verts) {
        low = glm::min(low, vertex);
        high = glm::max(high, vertex);
    }
    const float center_x = (high.x + low.x) / 2.0f;
    const float floor_y = low.y; // sit on floor
    const float center_z = (high.z + low.z) / 2.0f;
    glm::vec3 extent = high - low;
    float span = std::max({extent.x, extent.y, extent.z});
    if (span == 0.0f) {
        span = 1.0f;
    }
    const float scale = MODEL_FIT_SPAN / span;
    for (glm::vec3& vertex : verts) {
        vertex = glm::vec3((vertex.x - center_x) * scale, (vertex.y - floor_y) * scale, (vertex.z - center_z) * scale);
    }
    return scale;
}

std::pair<MeshPart, MeshPart> load_mesh(const std::string& path) {
    rapidobj::Result result = rapidobj::ParseFile(path);
    if (result.error) {
        throw std::runtime_error("Could not load mesh " + path + ": " + result.error.code.message());
    }
    // The mesh may carry quads/n-gons; triangulate so faces are uniform triangles.
    rapidobj::Triangulate(result);
    if (result.error) {
        throw std::runtime_error("Could not triangulate mesh " + path + ": " + result.error.code.message());
    }

    // rapidobj stores positions/colours as flat float arrays, 3 per vertex; colours
    // are present only when the obj's ``v`` lines carry r g b (matching extent).
    const rapidobj::Array<float>& position_floats = result.attributes.positions;
    const rapidobj::Array<float>& color_floats = result.attributes.colors;

    std::vector<glm::vec3> verts;
    verts.reserve(position_floats.size() / 3);
    for (std::size_t offset = 0; offset + 3 <= position_floats.size(); offset += 3) {
        verts.emplace_back(position_floats[offset], position_floats[offset + 1], position_floats[offset + 2]);
    }

    std::vector<glm::vec4> colors; // one per vertex when the obj carries colours
    if (!color_floats.empty() && color_floats.size() == position_floats.size()) {
        colors.reserve(color_floats.size() / 3);
        for (std::size_t offset = 0; offset + 3 <= color_floats.size(); offset += 3) {
            colors.emplace_back(color_floats[offset], color_floats[offset + 1], color_floats[offset + 2], 1.0f);
        }
    }

    std::vector<glm::ivec3> faces;
    for (const rapidobj::Shape& shape : result.shapes) {
        const rapidobj::Array<rapidobj::Index>& indices = shape.mesh.indices;
        faces.reserve(faces.size() + indices.size() / 3);
        for (std::size_t corner = 0; corner + 3 <= indices.size(); corner += 3) {
            faces.emplace_back(indices[corner].position_index, indices[corner + 1].position_index, indices[corner + 2].position_index);
        }
    }

    const std::vector<glm::vec3> normals = face_normals(verts, faces);

    // Split faces by vertex colour: any face touching a hand-coloured vertex is
    // the hand, the rest are joints. With no colours, everything is the hand.
    MeshPart hand;
    MeshPart joints;
    hand.verts = verts;
    hand.colors = colors;
    joints.verts = verts;
    joints.colors = colors;

    if (colors.empty()) {
        hand.faces = faces;
        hand.normals = normals;
        return {hand, joints};
    }

    std::vector<std::uint8_t> is_hand_vertex(verts.size(), 0);
    for (std::size_t vertex = 0; vertex < verts.size(); ++vertex) {
        const glm::vec4& color = colors[vertex];
        is_hand_vertex[vertex] =
            (static_cast<int>(std::lround(color.r * 255.0f)) == HAND_COLOR_8BIT.r && static_cast<int>(std::lround(color.g * 255.0f)) == HAND_COLOR_8BIT.g &&
             static_cast<int>(std::lround(color.b * 255.0f)) == HAND_COLOR_8BIT.b)
            ? 1
            : 0;
    }

    for (std::size_t face = 0; face < faces.size(); ++face) {
        const glm::ivec3& indices = faces[face];
        const bool is_hand = is_hand_vertex[static_cast<std::size_t>(indices[0])] || is_hand_vertex[static_cast<std::size_t>(indices[1])] ||
            is_hand_vertex[static_cast<std::size_t>(indices[2])];
        if (is_hand) {
            hand.faces.push_back(indices);
            hand.normals.push_back(normals[face]);
        } else {
            joints.faces.push_back(indices);
            joints.normals.push_back(normals[face]);
        }
    }
    return {hand, joints};
}

// ── Binary .hmesh format ───────────────────────

namespace {
    /// Read ``count`` little-endian float32 vec3s from ``data`` at ``offset``,
    /// rotating each 180° about X (negate y and z) into the OBJ-export pose.
    std::vector<glm::vec3> read_posed_vec3(const std::string& data, std::size_t offset, std::size_t count) {
        std::vector<glm::vec3> points;
        points.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            float xyz[3];
            std::memcpy(xyz, data.data() + offset + index * 3 * sizeof(float), 3 * sizeof(float));
            points.emplace_back(xyz[0], -xyz[1], -xyz[2]);
        }
        return points;
    }
} // namespace

HMesh load_hmesh(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open hmesh " + path);
    }
    const std::streamsize size = file.tellg();
    if (size < 8) {
        throw std::runtime_error("Truncated hmesh " + path);
    }
    std::string data(static_cast<std::size_t>(size), '\0');
    file.seekg(0);
    file.read(data.data(), size);

    if (std::memcmp(data.data(), "HMH", 3) != 0) {
        throw std::runtime_error("Not an hmesh file: " + path);
    }
    HMesh mesh;
    mesh.is_right = static_cast<std::uint8_t>(data[3]) != 0;
    std::uint16_t n_verts = 0;
    std::uint16_t n_joints = 0;
    std::memcpy(&n_verts, data.data() + 4, sizeof(n_verts));
    std::memcpy(&n_joints, data.data() + 6, sizeof(n_joints));

    const std::size_t expected = 8 + (static_cast<std::size_t>(n_verts) + n_joints) * 3 * sizeof(float);
    if (static_cast<std::size_t>(size) < expected) {
        throw std::runtime_error("Truncated hmesh body: " + path);
    }
    mesh.verts = read_posed_vec3(data, 8, n_verts);
    mesh.joints = read_posed_vec3(data, 8 + static_cast<std::size_t>(n_verts) * 3 * sizeof(float), n_joints);
    return mesh;
}

// ── Shared MANO face topology ──────────────────

namespace {
    std::vector<glm::ivec3> g_faces_right;
    std::vector<glm::ivec3> g_faces_left;
} // namespace

void init_mano_topology(const std::string& faces_path) {
    std::ifstream file(faces_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Could not open MANO faces " + faces_path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0 || size % (3 * sizeof(std::uint16_t)) != 0) {
        throw std::runtime_error("Malformed MANO faces " + faces_path);
    }
    std::vector<std::uint16_t> indices(static_cast<std::size_t>(size) / sizeof(std::uint16_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(indices.data()), size);

    const std::size_t face_count = indices.size() / 3;
    g_faces_right.clear();
    g_faces_left.clear();
    g_faces_right.reserve(face_count);
    g_faces_left.reserve(face_count);
    for (std::size_t face = 0; face < face_count; ++face) {
        const int a = indices[face * 3 + 0];
        const int b = indices[face * 3 + 1];
        const int c = indices[face * 3 + 2];
        g_faces_right.emplace_back(a, b, c);
        g_faces_left.emplace_back(a, c, b); // flip winding for a left hand
    }
}

const std::vector<glm::ivec3>& mano_faces(bool is_right) {
    const std::vector<glm::ivec3>& faces = is_right ? g_faces_right : g_faces_left;
    if (faces.empty()) {
        throw std::runtime_error("MANO topology not initialised (call init_mano_topology first)");
    }
    return faces;
}

// ── Procedural joint skeleton ──────────────────

namespace {
    using Triangle = std::array<glm::vec3, 3>;

    /// A unit sphere (radius 1, centred at origin) as a triangle soup, built once:
    /// a bare icosahedron (20 triangles) — low-poly markers keep the joint mesh
    /// small, which dominates per-hand memory.
    const std::vector<Triangle>& unit_sphere() {
        static const std::vector<Triangle> sphere = [] {
            const float phi = (1.0f + std::sqrt(5.0f)) * 0.5f;
            std::vector<glm::vec3> verts = {
                {-1, phi, 0},
                {1, phi, 0},
                {-1, -phi, 0},
                {1, -phi, 0},
                {0, -1, phi},
                {0, 1, phi},
                {0, -1, -phi},
                {0, 1, -phi},
                {phi, 0, -1},
                {phi, 0, 1},
                {-phi, 0, -1},
                {-phi, 0, 1}
            };
            for (glm::vec3& vertex : verts) {
                vertex = glm::normalize(vertex);
            }
            const int ico[20][3] = {{0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
                                    {3, 9, 4},  {3, 4, 2}, {3, 2, 6}, {3, 6, 8},  {3, 8, 9},   {4, 9, 5}, {2, 4, 11}, {6, 2, 10},  {8, 6, 7},  {9, 8, 1}};
            std::vector<Triangle> tris;
            tris.reserve(20);
            for (const auto& face : ico) {
                tris.push_back({verts[face[0]], verts[face[1]], verts[face[2]]});
            }
            return tris;
        }();
        return sphere;
    }

    /// A unit cylinder (radius 1, spanning y in [0, 1]) as a triangle soup, built
    /// once. Sides plus end caps.
    const std::vector<Triangle>& unit_cylinder() {
        static const std::vector<Triangle> cylinder = [] {
            constexpr int segments = 10;
            std::vector<Triangle> tris;
            for (int segment = 0; segment < segments; ++segment) {
                const float a0 = glm::two_pi<float>() * segment / segments;
                const float a1 = glm::two_pi<float>() * (segment + 1) / segments;
                const glm::vec3 p0(std::cos(a0), 0.0f, std::sin(a0));
                const glm::vec3 p1(std::cos(a1), 0.0f, std::sin(a1));
                const glm::vec3 t0 = p0 + glm::vec3(0, 1, 0);
                const glm::vec3 t1 = p1 + glm::vec3(0, 1, 0);
                tris.push_back({p0, p1, t1}); // side quad
                tris.push_back({p0, t1, t0});
                tris.push_back({glm::vec3(0, 0, 0), p1, p0}); // bottom cap
                tris.push_back({glm::vec3(0, 1, 0), t0, t1}); // top cap
            }
            return tris;
        }();
        return cylinder;
    }

    /// An orthonormal basis whose Y axis points along ``direction`` (unit length).
    glm::mat3 basis_from_y(const glm::vec3& direction) {
        const glm::vec3 reference = std::fabs(direction.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        const glm::vec3 x_axis = glm::normalize(glm::cross(reference, direction));
        const glm::vec3 z_axis = glm::cross(direction, x_axis);
        return glm::mat3(x_axis, direction, z_axis);
    }

    /// Append a transformed primitive (``origin + transform * local``) to ``out``,
    /// recomputing a flat per-triangle normal in world space.
    void
    append_primitive(MeshArrays& out, const std::vector<Triangle>& primitive, const glm::vec3& origin, const glm::mat3& transform, const glm::vec4& color) {
        for (const Triangle& local : primitive) {
            const glm::vec3 a = origin + transform * local[0];
            const glm::vec3 b = origin + transform * local[1];
            const glm::vec3 c = origin + transform * local[2];
            const glm::vec3 raw = glm::cross(b - a, c - a);
            const float length = glm::length(raw);
            const glm::vec3 normal = length == 0.0f ? glm::vec3(0.0f) : raw / length;
            for (const glm::vec3& position : {a, b, c}) {
                out.positions.push_back(position.x);
                out.positions.push_back(position.y);
                out.positions.push_back(position.z);
                out.normals.push_back(normal.x);
                out.normals.push_back(normal.y);
                out.normals.push_back(normal.z);
                out.colors.push_back(color.r);
                out.colors.push_back(color.g);
                out.colors.push_back(color.b);
                out.colors.push_back(color.a);
            }
        }
    }
} // namespace

MeshArrays build_joint_mesh(const std::vector<glm::vec3>& joints, float hand_diagonal) {
    MeshArrays out;
    if (joints.empty()) {
        return out;
    }
    const float joint_radius = (hand_diagonal > 0.0f ? hand_diagonal : 1.0f) * 0.02f;
    const float bone_radius = joint_radius * 0.45f;

    for (std::size_t joint = 0; joint < joints.size(); ++joint) {
        const glm::vec4& color = FINGER_COLORS[static_cast<std::size_t>(finger_of(static_cast<int>(joint)))];
        append_primitive(out, unit_sphere(), joints[joint], glm::mat3(joint_radius), color);
    }
    for (const glm::ivec2& bone : HAND_BONES) {
        if (bone[0] >= static_cast<int>(joints.size()) || bone[1] >= static_cast<int>(joints.size())) {
            continue;
        }
        const glm::vec3& parent = joints[static_cast<std::size_t>(bone[0])];
        const glm::vec3& child = joints[static_cast<std::size_t>(bone[1])];
        const glm::vec3 along = child - parent;
        const float length = glm::length(along);
        if (length <= 0.0f) {
            continue;
        }
        // Bone takes its child joint's finger colour so the wrist bones inherit
        // the finger they lead to (matching the OBJ skeleton).
        const glm::vec4& color = FINGER_COLORS[static_cast<std::size_t>(finger_of(bone[1]))];
        glm::mat3 transform = basis_from_y(along / length);
        transform[0] *= bone_radius;
        transform[1] *= length;
        transform[2] *= bone_radius;
        append_primitive(out, unit_cylinder(), parent, transform, color);
    }
    out.vertex_count = static_cast<int>(out.positions.size() / 3);
    return out;
}

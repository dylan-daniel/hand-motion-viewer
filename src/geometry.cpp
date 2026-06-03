#include "geometry.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace {
    /// One vertex of an obj face: its index in the shared vertex list (0-based).
    int parse_face_index(const std::string& token) {
        // obj face tokens are ``v``, ``v/vt``, ``v/vt/vn`` or ``v//vn``; only the
        // leading vertex index matters here. obj indices are 1-based.
        std::size_t slash = token.find('/');
        const std::string vertex = slash == std::string::npos ? token : token.substr(0, slash);
        return std::atoi(vertex.c_str()) - 1;
    }
} // namespace

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

PreparedMesh prepare_hand(
    const std::vector<glm::vec3>& positions,
    const std::vector<glm::ivec3>& faces,
    const std::vector<glm::vec4>& colors,
    const std::vector<std::uint8_t>& hand_face_mask,
    float alpha
) {
    const std::vector<glm::vec3> normals = face_normals(positions, faces);

    MeshPart hand_part;
    MeshPart joint_part;
    hand_part.verts = positions;
    hand_part.colors = colors;
    joint_part.verts = positions;
    joint_part.colors = colors;

    int hand_triangle_count = 0;
    for (std::size_t face = 0; face < faces.size(); ++face) {
        if (hand_face_mask[face]) {
            hand_part.faces.push_back(faces[face]);
            hand_part.normals.push_back(normals[face]);
            ++hand_triangle_count;
        } else {
            joint_part.faces.push_back(faces[face]);
            joint_part.normals.push_back(normals[face]);
        }
    }

    PreparedMesh prepared;
    prepared.joints = build_arrays(joint_part);
    prepared.hand_solid = build_arrays(hand_part);
    prepared.hand_translucent = build_arrays(hand_part, alpha);
    prepared.hand_triangle_count = hand_triangle_count;
    prepared.joint_triangle_count = static_cast<int>(faces.size()) - hand_triangle_count;
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
    const float scale = 4.0f / span;
    for (glm::vec3& vertex : verts) {
        vertex = glm::vec3((vertex.x - center_x) * scale, (vertex.y - floor_y) * scale, (vertex.z - center_z) * scale);
    }
    return scale;
}

std::pair<MeshPart, MeshPart> load_mesh(const std::string& path) {
    std::ifstream mesh_file(path);
    if (!mesh_file) {
        throw std::runtime_error("Could not open mesh file: " + path);
    }

    std::vector<glm::vec3> verts;
    std::vector<glm::vec4> colors; // one per vertex when the obj carries colours
    std::vector<glm::ivec3> faces;
    bool has_colors = true;

    std::string line;
    while (std::getline(mesh_file, line)) {
        if (line.size() < 2) {
            continue;
        }
        if (line[0] == 'v' && line[1] == ' ') {
            std::istringstream stream(line.substr(2));
            float x = 0.0f, y = 0.0f, z = 0.0f;
            stream >> x >> y >> z;
            verts.emplace_back(x, y, z);
            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (stream >> r >> g >> b) {
                colors.emplace_back(r, g, b, 1.0f);
            } else {
                has_colors = false;
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            std::istringstream stream(line.substr(2));
            std::string a, b, c;
            stream >> a >> b >> c;
            if (!a.empty() && !b.empty() && !c.empty()) {
                faces.emplace_back(parse_face_index(a), parse_face_index(b), parse_face_index(c));
            }
        }
    }

    if (!has_colors || colors.size() != verts.size()) {
        colors.clear();
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

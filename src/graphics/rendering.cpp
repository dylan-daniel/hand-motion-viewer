#include "graphics/rendering.h"

#include "graphics/gl_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include <SDL3/SDL_opengl.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

// ── Shared GLSL program + draw helpers ─────────
//
// One program serves every draw. Vertex attributes are fixed at locations
// 0 = position, 1 = normal, 2 = colour, matching every VAO built below. The
// ``uLighting`` flag switches between the lit meshes and the flat-coloured grid,
// axes, and camera marker. Lighting is computed per-vertex in view space with
// two fixed lights, reproducing the legacy fixed-function look.

namespace {
    constexpr float kPi = 3.14159265358979323846f;

    // Attribute locations shared by the shader and every VAO.
    constexpr GLuint ATTRIB_POSITION = 0;
    constexpr GLuint ATTRIB_NORMAL = 1;
    constexpr GLuint ATTRIB_COLOR = 2;

    const char* const VERTEX_SHADER = R"(#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec4 aColor;

uniform mat4 uModel;
uniform mat4 uView;
uniform mat4 uProj;

out vec3 vFragPos;   // view-space position; its screen-space derivatives give the face normal
out vec3 vNormalRef; // smooth view-space normal, used only to orient the derived facet normal
out vec4 vColor;

void main() {
    vec4 view_pos = uView * uModel * vec4(aPos, 1.0);
    gl_Position = uProj * view_pos;
    vFragPos = view_pos.xyz;
    // Model/view here only translate and uniformly scale, so the upper 3x3 needs
    // no inverse-transpose to keep the reference normal pointing outward.
    vNormalRef = mat3(uView * uModel) * aNormal;
    vColor = aColor;
}
)";

    // Lighting is per-fragment and uses a *flat* normal derived from the
    // view-space position's screen-space derivatives, so each triangle shades as
    // one facet even though the vertices are shared/indexed (smooth vNormalRef is
    // only used to pick the facet normal's sign). This restores the faceted look
    // without un-sharing vertices.
    const char* const FRAGMENT_SHADER = R"(#version 330 core
in vec3 vFragPos;
in vec3 vNormalRef;
in vec4 vColor;

uniform bool uLighting;
uniform float uAlpha; // multiplies alpha; lets one buffer draw opaque or translucent

out vec4 FragColor;

// View-space lights matching the legacy setup_lighting() values.
const vec3 LIGHT0_POS = vec3(5.0, 10.0, 5.0);
const vec3 LIGHT0_DIFFUSE = vec3(0.85, 0.85, 0.85);
const vec3 LIGHT0_AMBIENT = vec3(0.15, 0.15, 0.15);
const vec3 LIGHT1_POS = vec3(-5.0, 6.0, -8.0);
const vec3 LIGHT1_DIFFUSE = vec3(0.3, 0.35, 0.5);
const vec3 SPECULAR = vec3(0.4, 0.4, 0.4);
const float SHININESS = 32.0;

void main() {
    if (!uLighting) {
        FragColor = vec4(vColor.rgb, vColor.a * uAlpha);
        return;
    }

    // Flat per-triangle normal: derivatives of the linearly-interpolated position
    // are constant across a triangle, so this is the geometric facet normal.
    vec3 normal = normalize(cross(dFdx(vFragPos), dFdy(vFragPos)));
    // Orient it outward using the interpolated vertex normal as a sign reference.
    if (dot(normal, vNormalRef) < 0.0) {
        normal = -normal;
    }

    vec3 frag = vFragPos;
    vec3 eye = normalize(-frag);

    vec3 lit = vColor.rgb * LIGHT0_AMBIENT;

    vec3 dir0 = normalize(LIGHT0_POS - frag);
    lit += vColor.rgb * LIGHT0_DIFFUSE * max(dot(normal, dir0), 0.0);
    vec3 half0 = normalize(dir0 + eye);
    lit += SPECULAR * LIGHT0_DIFFUSE * pow(max(dot(normal, half0), 0.0), SHININESS);

    vec3 dir1 = normalize(LIGHT1_POS - frag);
    lit += vColor.rgb * LIGHT1_DIFFUSE * max(dot(normal, dir1), 0.0);

    FragColor = vec4(lit, vColor.a * uAlpha);
}
)";

    // Shared GL resources, created lazily on the first render and freed by
    // shutdown_renderer() before the context goes away.
    GLuint g_program = 0;
    GLint g_loc_model = -1;
    GLint g_loc_view = -1;
    GLint g_loc_proj = -1;
    GLint g_loc_lighting = -1;
    GLint g_loc_alpha = -1;
    GLuint g_dynamic_vao = 0; // pos+colour stream for grid / axes / marker
    GLuint g_dynamic_vbo = 0;

    GLuint compile_shader(GLenum type, const char* source) {
        const GLuint shader = glx::CreateShader(type);
        glx::ShaderSource(shader, 1, &source, nullptr);
        glx::CompileShader(shader);
        GLint ok = GL_FALSE;
        glx::GetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (ok == GL_FALSE) {
            char log[512];
            glx::GetShaderInfoLog(shader, sizeof(log), nullptr, log);
            std::printf("Shader compile failed: %s\n", log);
        }
        return shader;
    }

    void ensure_resources() {
        if (g_program != 0) {
            return;
        }
        const GLuint vertex = compile_shader(GL_VERTEX_SHADER, VERTEX_SHADER);
        const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        g_program = glx::CreateProgram();
        glx::AttachShader(g_program, vertex);
        glx::AttachShader(g_program, fragment);
        glx::LinkProgram(g_program);
        GLint ok = GL_FALSE;
        glx::GetProgramiv(g_program, GL_LINK_STATUS, &ok);
        if (ok == GL_FALSE) {
            char log[512];
            glx::GetProgramInfoLog(g_program, sizeof(log), nullptr, log);
            std::printf("Program link failed: %s\n", log);
        }
        glx::DeleteShader(vertex);
        glx::DeleteShader(fragment);

        g_loc_model = glx::GetUniformLocation(g_program, "uModel");
        g_loc_view = glx::GetUniformLocation(g_program, "uView");
        g_loc_proj = glx::GetUniformLocation(g_program, "uProj");
        g_loc_lighting = glx::GetUniformLocation(g_program, "uLighting");
        g_loc_alpha = glx::GetUniformLocation(g_program, "uAlpha");

        // Dynamic stream VAO: interleaved position (3) + colour (4); the normal
        // attribute stays disabled (a constant value) since this stream is unlit.
        glx::GenVertexArrays(1, &g_dynamic_vao);
        glx::GenBuffers(1, &g_dynamic_vbo);
        glx::BindVertexArray(g_dynamic_vao);
        glx::BindBuffer(GL_ARRAY_BUFFER, g_dynamic_vbo);
        const GLsizei stride = 7 * sizeof(float);
        glx::EnableVertexAttribArray(ATTRIB_POSITION);
        glx::VertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, stride, nullptr);
        glx::EnableVertexAttribArray(ATTRIB_COLOR);
        glx::VertexAttribPointer(ATTRIB_COLOR, 4, GL_FLOAT, GL_FALSE, stride, reinterpret_cast<const void*>(3 * sizeof(float)));
        glx::VertexAttrib3f(ATTRIB_NORMAL, 0.0f, 0.0f, 1.0f);
        glx::BindVertexArray(0);
        glx::BindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void set_model(const glm::mat4& model) { glx::UniformMatrix4fv(g_loc_model, 1, GL_FALSE, glm::value_ptr(model)); }

    void set_lighting(bool enabled) { glx::Uniform1i(g_loc_lighting, enabled ? 1 : 0); }

    void set_alpha(float alpha) { glx::Uniform1f(g_loc_alpha, alpha); }

    /// Draw an interleaved position(3)+colour(4) vertex stream (unlit) under
    /// ``model``. Used for the grid, axes, and camera marker.
    void draw_colored(GLenum mode, const std::vector<float>& data, const glm::mat4& model) {
        if (data.empty()) {
            return;
        }
        set_model(model);
        set_lighting(false);
        glx::BindVertexArray(g_dynamic_vao);
        glx::BindBuffer(GL_ARRAY_BUFFER, g_dynamic_vbo);
        glx::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_DYNAMIC_DRAW);
        glDrawArrays(mode, 0, static_cast<GLsizei>(data.size() / 7));
        glx::BindVertexArray(0);
        glx::BindBuffer(GL_ARRAY_BUFFER, 0);
    }

    /// Append one position+colour vertex to an interleaved stream.
    void push_vertex(std::vector<float>& out, const glm::vec3& position, const glm::vec4& color) {
        out.insert(out.end(), {position.x, position.y, position.z, color.r, color.g, color.b, color.a});
    }

    /// Append a UV sphere of triangles (centred on the origin) to ``out``.
    void append_sphere(std::vector<float>& out, float radius, int slices, int stacks, const glm::vec4& color) {
        for (int stack = 0; stack < stacks; ++stack) {
            const float phi0 = kPi * static_cast<float>(stack) / static_cast<float>(stacks);
            const float phi1 = kPi * static_cast<float>(stack + 1) / static_cast<float>(stacks);
            for (int slice = 0; slice < slices; ++slice) {
                const float theta0 = 2.0f * kPi * static_cast<float>(slice) / static_cast<float>(slices);
                const float theta1 = 2.0f * kPi * static_cast<float>(slice + 1) / static_cast<float>(slices);

                auto point = [radius](float phi, float theta) {
                    return glm::vec3(radius * std::sin(phi) * std::cos(theta), radius * std::cos(phi), radius * std::sin(phi) * std::sin(theta));
                };
                const glm::vec3 a = point(phi0, theta0);
                const glm::vec3 b = point(phi1, theta0);
                const glm::vec3 c = point(phi1, theta1);
                const glm::vec3 d = point(phi0, theta1);

                push_vertex(out, a, color);
                push_vertex(out, b, color);
                push_vertex(out, c, color);
                push_vertex(out, a, color);
                push_vertex(out, c, color);
                push_vertex(out, d, color);
            }
        }
    }
} // namespace

void shutdown_renderer() {
    if (g_program != 0) {
        glx::DeleteProgram(g_program);
        g_program = 0;
    }
    if (g_dynamic_vbo != 0) {
        glx::DeleteBuffers(1, &g_dynamic_vbo);
        g_dynamic_vbo = 0;
    }
    if (g_dynamic_vao != 0) {
        glx::DeleteVertexArrays(1, &g_dynamic_vao);
        g_dynamic_vao = 0;
    }
}

// ── Grid + marker ──────────────────────────────

void draw_grid(int size, float step, float y) {
    std::vector<float> lines;
    for (int index = -size; index <= size; ++index) {
        const glm::vec4 color = (index == 0) ? glm::vec4(0.9f, 0.9f, 0.9f, 1.0f) : glm::vec4(0.35f, 0.35f, 0.35f, 1.0f);
        push_vertex(lines, glm::vec3(index * step, y, size * step), color);
        push_vertex(lines, glm::vec3(index * step, y, -size * step), color);
        push_vertex(lines, glm::vec3(size * step, y, index * step), color);
        push_vertex(lines, glm::vec3(-size * step, y, index * step), color);
    }
    glLineWidth(1.0f);
    draw_colored(GL_LINES, lines, glm::mat4(1.0f));

    // Axis arrows on the floor; depth test off so they always win over the
    // coincident grid lines instead of z-fighting.
    std::vector<float> axes;
    const glm::vec4 x_color(1.0f, 0.2f, 0.2f, 1.0f); // X – red
    const glm::vec4 z_color(0.2f, 0.4f, 1.0f, 1.0f); // Z – blue
    const glm::vec4 y_color(0.2f, 0.9f, 0.2f, 1.0f); // Y – green (up)
    push_vertex(axes, glm::vec3(0.0f, y, 0.0f), x_color);
    push_vertex(axes, glm::vec3(2.0f, y, 0.0f), x_color);
    push_vertex(axes, glm::vec3(0.0f, y, 0.0f), z_color);
    push_vertex(axes, glm::vec3(0.0f, y, 2.0f), z_color);
    push_vertex(axes, glm::vec3(0.0f, y, 0.0f), y_color);
    push_vertex(axes, glm::vec3(0.0f, y + 2.0f, 0.0f), y_color);

    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    draw_colored(GL_LINES, axes, glm::mat4(1.0f));
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
}

void draw_camera_marker(const glm::vec3& center, float radius) {
    std::vector<float> sphere;
    append_sphere(sphere, radius, 24, 16, glm::vec4(1.0f, 0.4f, 0.7f, 0.35f));

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Don't write depth so the ball never occludes the scene behind it.
    glDepthMask(GL_FALSE);
    draw_colored(GL_TRIANGLES, sphere, glm::translate(glm::mat4(1.0f), center));
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

// ── GpuMesh ────────────────────────────────────

namespace {
    void upload_buffer(unsigned int vbo, const std::vector<float>& data) {
        glx::BindBuffer(GL_ARRAY_BUFFER, vbo);
        glx::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STATIC_DRAW);
    }
} // namespace

GpuMesh::GpuMesh(const MeshArrays& arrays) : vertex_count_(arrays.vertex_count), index_count_(static_cast<int>(arrays.indices.size())) {
    glx::GenBuffers(1, &position_vbo_);
    glx::GenBuffers(1, &normal_vbo_);
    glx::GenBuffers(1, &color_vbo_);
    upload_buffer(position_vbo_, arrays.positions);
    upload_buffer(normal_vbo_, arrays.normals);
    upload_buffer(color_vbo_, arrays.colors);

    // Bake the attribute layout into a VAO so draw() is a bind + draw call.
    glx::GenVertexArrays(1, &vao_);
    glx::BindVertexArray(vao_);
    glx::BindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    glx::EnableVertexAttribArray(ATTRIB_POSITION);
    glx::VertexAttribPointer(ATTRIB_POSITION, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glx::BindBuffer(GL_ARRAY_BUFFER, normal_vbo_);
    glx::EnableVertexAttribArray(ATTRIB_NORMAL);
    glx::VertexAttribPointer(ATTRIB_NORMAL, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
    glx::BindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    glx::EnableVertexAttribArray(ATTRIB_COLOR);
    glx::VertexAttribPointer(ATTRIB_COLOR, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
    // Indexed meshes keep an element buffer bound in the VAO; draw() uses it.
    if (index_count_ > 0) {
        glx::GenBuffers(1, &index_ebo_);
        glx::BindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_ebo_);
        glx::BufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(arrays.indices.size() * sizeof(std::uint16_t)), arrays.indices.data(), GL_STATIC_DRAW);
    }
    glx::BindVertexArray(0);
    glx::BindBuffer(GL_ARRAY_BUFFER, 0);
    glx::BindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept :
    vertex_count_(other.vertex_count_), index_count_(other.index_count_), vao_(other.vao_), position_vbo_(other.position_vbo_), normal_vbo_(other.normal_vbo_),
    color_vbo_(other.color_vbo_), index_ebo_(other.index_ebo_) {
    other.vertex_count_ = 0;
    other.index_count_ = 0;
    other.vao_ = 0;
    other.position_vbo_ = other.normal_vbo_ = other.color_vbo_ = other.index_ebo_ = 0;
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept {
    if (this != &other) {
        release();
        vertex_count_ = other.vertex_count_;
        index_count_ = other.index_count_;
        vao_ = other.vao_;
        position_vbo_ = other.position_vbo_;
        normal_vbo_ = other.normal_vbo_;
        color_vbo_ = other.color_vbo_;
        index_ebo_ = other.index_ebo_;
        other.vertex_count_ = 0;
        other.index_count_ = 0;
        other.vao_ = 0;
        other.position_vbo_ = other.normal_vbo_ = other.color_vbo_ = other.index_ebo_ = 0;
    }
    return *this;
}

GpuMesh::~GpuMesh() { release(); }

void GpuMesh::release() {
    if (vao_ != 0) {
        glx::DeleteVertexArrays(1, &vao_);
    }
    if (position_vbo_ != 0 || normal_vbo_ != 0 || color_vbo_ != 0 || index_ebo_ != 0) {
        unsigned int buffers[4] = {position_vbo_, normal_vbo_, color_vbo_, index_ebo_};
        glx::DeleteBuffers(4, buffers);
    }
    vertex_count_ = 0;
    index_count_ = 0;
    vao_ = 0;
    position_vbo_ = normal_vbo_ = color_vbo_ = index_ebo_ = 0;
}

void GpuMesh::draw() const {
    glx::BindVertexArray(vao_);
    if (index_count_ > 0) {
        glDrawElements(GL_TRIANGLES, index_count_, GL_UNSIGNED_SHORT, nullptr);
    } else if (vertex_count_ > 0) {
        glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
    }
    glx::BindVertexArray(0);
}

// ── FrameGpu ───────────────────────────────────

PreparedFrame prepare_frame(const Frame& hands) {
    PreparedFrame prepared;
    prepared.reserve(hands.size());
    for (const HandData& hand : hands) {
        PreparedMesh arrays = prepare_hand(hand.verts, mano_faces(hand.is_right), track_color(hand.track_id), hand.joints);
        double sum = 0.0;
        for (const glm::vec3& position : hand.verts) {
            sum += position.z;
        }
        const float depth = hand.verts.empty() ? 0.0f : static_cast<float>(sum / hand.verts.size());
        prepared.push_back({std::move(arrays), depth});
    }
    return prepared;
}

FrameGpu::FrameGpu(const PreparedFrame& prepared_hands) {
    hands_.reserve(prepared_hands.size());
    depths_.reserve(prepared_hands.size());
    for (const PreparedHand& prepared : prepared_hands) {
        HandGpu hand;
        hand.joints = std::make_unique<GpuMesh>(prepared.mesh.joints);
        hand.hand = std::make_unique<GpuMesh>(prepared.mesh.hand);
        hands_.push_back(std::move(hand));
        depths_.push_back(prepared.depth);
    }
}

glm::mat4 FrameGpu::hand_matrix(const Transform* transform, float scale) const {
    glm::mat4 model(1.0f);
    if (transform != nullptr) {
        model = glm::scale(model, glm::vec3(transform->scale));
        model = glm::translate(model, transform->translate);
    }
    if (scale != 1.0f) {
        model = glm::scale(model, glm::vec3(scale));
    }
    return model;
}

void FrameGpu::draw(bool translucent, const Transform* transform, std::optional<float> reference_depth) const {
    // Stabilise depth: snap every hand to the common reference plane.
    std::vector<float> scales;
    scales.reserve(depths_.size());
    for (float depth : depths_) {
        scales.push_back((reference_depth && depth != 0.0f) ? (*reference_depth / depth) : 1.0f);
    }

    set_lighting(true);
    // The joint skeleton only shows in translucent mode — it reads through the
    // see-through hand, and an opaque hand would hide it. Drawn first so the
    // translucent hand blends correctly over it.
    if (translucent) {
        for (std::size_t index = 0; index < hands_.size(); ++index) {
            set_model(hand_matrix(transform, scales[index]));
            hands_[index].joints->draw();
        }
    }

    if (translucent) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        set_alpha(0.30f);
    }
    for (std::size_t index = 0; index < hands_.size(); ++index) {
        set_model(hand_matrix(transform, scales[index]));
        hands_[index].hand->draw();
    }
    if (translucent) {
        set_alpha(1.0f);
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
}

// ── Framebuffer ────────────────────────────────

Framebuffer::Framebuffer() {
    glx::GenFramebuffers(1, &fbo_);
    glGenTextures(1, &texture_);
    glx::GenRenderbuffers(1, &depth_);
}

Framebuffer::~Framebuffer() {
    glx::DeleteFramebuffers(1, &fbo_);
    glDeleteTextures(1, &texture_);
    glx::DeleteRenderbuffers(1, &depth_);
}

void Framebuffer::resize(int width, int height) {
    if (width == width_ && height == height_) {
        return;
    }
    width_ = width;
    height_ = height;

    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glx::BindRenderbuffer(GL_RENDERBUFFER, depth_);
    glx::RenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);

    glx::BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    glx::FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture_, 0);
    glx::FramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_);
    glx::BindFramebuffer(GL_FRAMEBUFFER, 0);
}

// ── Cameras ────────────────────────────────────

OrbitCamera::OrbitCamera(float distance) : distance_(distance) {}

glm::mat4 OrbitCamera::view_matrix() const {
    const float az = glm::radians(azimuth_);
    const float el = glm::radians(elevation_);
    const glm::vec3 target(target_[0], target_[1], target_[2]);
    const glm::vec3 eye(
        target.x + distance_ * std::cos(el) * std::sin(az), target.y + distance_ * std::sin(el), target.z + distance_ * std::cos(el) * std::cos(az)
    );
    return glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void OrbitCamera::orbit(float dx, float dy) {
    azimuth_ -= dx * LOOK_SENSITIVITY;
    elevation_ += dy * LOOK_SENSITIVITY;
    elevation_ = std::clamp(elevation_, -89.0f, 89.0f);
}

void OrbitCamera::pan(float dx, float dy) {
    const float az = glm::radians(azimuth_);
    const float rx = std::cos(az);
    const float rz = -std::sin(az);
    const float scale = distance_ * 0.002f;
    target_[0] -= (rx * dx) * scale;
    target_[1] += dy * scale;
    target_[2] -= (rz * dx) * scale;
}

void OrbitCamera::move(float forward, float right, float up, float dt) {
    const float az = glm::radians(azimuth_);
    const float el = glm::radians(elevation_);
    const float fx = -std::cos(el) * std::sin(az);
    const float fy = -std::sin(el);
    const float fz = -std::cos(el) * std::cos(az);
    const float rx = std::cos(az);
    const float rz = -std::sin(az);
    const float ux = -rz * fy;
    const float uy = rz * fx - rx * fz;
    const float uz = rx * fy;
    const float step = distance_ * 0.9f * dt;
    target_[0] += (fx * forward + rx * right + ux * up) * step;
    target_[1] += (fy * forward + uy * up) * step;
    target_[2] += (fz * forward + rz * right + uz * up) * step;
}

void OrbitCamera::zoom(float delta) { distance_ = std::max(0.5f, distance_ - delta * 0.5f); }

void OrbitCamera::reset() {
    const float distance = distance_;
    *this = OrbitCamera(distance);
}

void OrbitCamera::get_state(float& azimuth, float& elevation, float& distance, std::array<float, 3>& target) const {
    azimuth = azimuth_;
    elevation = elevation_;
    distance = distance_;
    target = target_;
}

void OrbitCamera::set_state(float azimuth, float elevation, float distance, const std::array<float, 3>& target) {
    azimuth_ = azimuth;
    elevation_ = elevation;
    distance_ = distance;
    target_ = target;
}

void OrbitCamera::set_from_free(const FreeCamera& free) {
    azimuth_ = free.yaw();
    elevation_ = std::clamp(free.pitch(), -89.0f, 89.0f);
    const glm::vec3 dir = free.forward();
    const std::array<float, 3>& eye = free.position();
    target_ = {eye[0] + dir.x * distance_, eye[1] + dir.y * distance_, eye[2] + dir.z * distance_};
}

FreeCamera::FreeCamera() = default;

glm::vec3 FreeCamera::forward() const {
    const float yaw = glm::radians(yaw_);
    const float pitch = glm::radians(pitch_);
    return glm::vec3(-std::cos(pitch) * std::sin(yaw), -std::sin(pitch), -std::cos(pitch) * std::cos(yaw));
}

glm::mat4 FreeCamera::view_matrix() const {
    const glm::vec3 dir = forward();
    const glm::vec3 eye(position_[0], position_[1], position_[2]);
    return glm::lookAt(eye, eye + dir, glm::vec3(0.0f, 1.0f, 0.0f));
}

void FreeCamera::orbit(float dx, float dy) {
    yaw_ -= dx * LOOK_SENSITIVITY;
    pitch_ += dy * LOOK_SENSITIVITY;
    pitch_ = std::clamp(pitch_, -89.0f, 89.0f);
}

void FreeCamera::pan(float dx, float dy) {
    const float yaw = glm::radians(yaw_);
    const float rx = std::cos(yaw);
    const float rz = -std::sin(yaw);
    const float scale = 0.01f;
    position_[0] -= rx * dx * scale;
    position_[1] += dy * scale;
    position_[2] -= rz * dx * scale;
}

void FreeCamera::move(float forward_amount, float right, float up, float dt) {
    const glm::vec3 dir = forward();
    const float yaw = glm::radians(yaw_);
    const float rx = std::cos(yaw);
    const float rz = -std::sin(yaw);
    const float ux = -rz * dir.y;
    const float uy = rz * dir.x - rx * dir.z;
    const float uz = rx * dir.y;
    const float step = 4.0f * dt;
    position_[0] += (dir.x * forward_amount + rx * right + ux * up) * step;
    position_[1] += (dir.y * forward_amount + uy * up) * step;
    position_[2] += (dir.z * forward_amount + rz * right + uz * up) * step;
}

void FreeCamera::zoom(float delta) {
    const glm::vec3 dir = forward();
    const float step = delta * 0.5f;
    position_[0] += dir.x * step;
    position_[1] += dir.y * step;
    position_[2] += dir.z * step;
}

void FreeCamera::reset() { *this = FreeCamera(); }

void FreeCamera::set_from_orbit(const OrbitCamera& orbit) {
    const float az = glm::radians(orbit.azimuth());
    const float el = glm::radians(orbit.elevation());
    const std::array<float, 3>& target = orbit.target();
    const float distance = orbit.distance();
    position_ = {target[0] + distance * std::cos(el) * std::sin(az), target[1] + distance * std::sin(el), target[2] + distance * std::cos(el) * std::cos(az)};
    yaw_ = orbit.azimuth();
    pitch_ = orbit.elevation();
}

// ── Scene render ───────────────────────────────

void render_scene(const Framebuffer& framebuffer, const Camera& camera, const SceneRender& scene) {
    ensure_resources();

    glx::BindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo());
    glViewport(0, 0, framebuffer.width(), framebuffer.height());
    glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
    glClear(static_cast<GLbitfield>(GL_COLOR_BUFFER_BIT) | static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT));

    const float aspect = framebuffer.height() != 0 ? static_cast<float>(framebuffer.width()) / static_cast<float>(framebuffer.height()) : 1.0f;
    const glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);

    glx::UseProgram(g_program);
    glx::UniformMatrix4fv(g_loc_proj, 1, GL_FALSE, glm::value_ptr(projection));
    glx::UniformMatrix4fv(g_loc_view, 1, GL_FALSE, glm::value_ptr(camera.view_matrix()));
    set_alpha(1.0f); // opaque by default; translucent hand draws flip it transiently

    // Grid (flat-coloured; draw_grid keeps lighting off).
    draw_grid(10, 1.0f, 0.0f);

    if (scene.frame != nullptr) {
        scene.frame->draw(scene.translucent, scene.transform, scene.reference_depth);
    }

    // SAM3+DA3 overlay: already stabilised per-hand, so only the scene fit
    // Transform (scale + translate) is applied here, matching FrameGpu's model
    // matrix at scale 1. Drawn as unlit points, colour-coded to stand out
    // against the hand mesh.
    if (scene.overlay_points != nullptr && !scene.overlay_points->empty()) {
        glm::mat4 model(1.0f);
        if (scene.transform != nullptr) {
            model = glm::scale(model, glm::vec3(scene.transform->scale));
            model = glm::translate(model, scene.transform->translate);
        }
        std::vector<float> points;
        points.reserve(scene.overlay_points->size() * 7);
        const glm::vec4 overlay_color(1.0f, 0.1f, 0.6f, 1.0f);
        for (const glm::vec3& point : *scene.overlay_points) {
            push_vertex(points, point, overlay_color);
        }
        glPointSize(2.5f);
        draw_colored(GL_POINTS, points, model);
        glPointSize(1.0f);
    }

    // Marker for the orbit camera's look-at point; drawn last so its
    // translucency blends over the scene.
    if (scene.show_camera_marker && camera.draws_marker()) {
        draw_camera_marker(camera.marker_target());
    }

    glx::UseProgram(0);
    glx::BindFramebuffer(GL_FRAMEBUFFER, 0);
}

#pragma once

// OpenGL rendering: grid, mesh vertex buffers, cameras, lighting, and the
// offscreen framebuffer the scene is drawn into. Uses a modern core-profile
// pipeline (GL 3.3): a single GLSL program with per-vertex lighting, VAOs/VBOs
// for all geometry, and glm-supplied projection/view/model matrices fed as
// uniforms.

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <glm/glm.hpp>

#include "data/geometry.h"
#include "data/mesh_sequence.h"

// ── Grid + marker ──────────────────────────────
void draw_grid(int size = 10, float step = 1.0f, float y = 0.0f);
void draw_camera_marker(const glm::vec3& center, float radius = 0.1f);

// ── Mesh drawing ───────────────────────────────

/// One mesh part uploaded to GPU vertex buffers, drawn with glDrawArrays.
class GpuMesh {
public:
    explicit GpuMesh(const MeshArrays& arrays);
    ~GpuMesh();

    GpuMesh(const GpuMesh&) = delete;
    GpuMesh& operator=(const GpuMesh&) = delete;
    GpuMesh(GpuMesh&& other) noexcept;
    GpuMesh& operator=(GpuMesh&& other) noexcept;

    void draw() const;

private:
    void release();

    int vertex_count_ = 0; // vertices in the buffers
    int index_count_ = 0;  // indices to draw (0 => non-indexed, draw vertex_count_)
    unsigned int vao_ = 0;
    unsigned int position_vbo_ = 0;
    unsigned int normal_vbo_ = 0;
    unsigned int color_vbo_ = 0;
    unsigned int index_ebo_ = 0; // element buffer, 0 when non-indexed
};

/// One hand's GPU buffers within a sequence frame. The hand surface is a single
/// mesh; opaque vs. translucent is a draw-time alpha uniform, not a second copy.
struct HandGpu {
    std::unique_ptr<GpuMesh> joints;
    std::unique_ptr<GpuMesh> hand;
};

// A frame's prepared CPU arrays for one hand: GPU-ready mesh.
struct PreparedHand {
    PreparedMesh mesh;
};
using PreparedFrame = std::vector<PreparedHand>;

/// GPU buffers for every hand of one sequence frame, drawn as a unit. Each hand
/// is already placed in real metric scene space (see data/mano_model.h's
/// place_hand_metric), so drawing applies no further scene-wide transform.
class FrameGpu {
public:
    explicit FrameGpu(const PreparedFrame& prepared_hands);

    /// Draw the frame's hands as a unit.
    void draw(bool translucent) const;

private:
    std::vector<HandGpu> hands_;
};

/// Expand a frame's hands into GPU-ready arrays plus each hand's mean depth.
PreparedFrame prepare_frame(const Frame& hands);

// ── Offscreen render target ────────────────────

/// Offscreen render target (color texture + depth) the scene is drawn into.
class Framebuffer {
public:
    Framebuffer();
    ~Framebuffer();

    Framebuffer(const Framebuffer&) = delete;
    Framebuffer& operator=(const Framebuffer&) = delete;

    void resize(int width, int height);

    unsigned int fbo() const { return fbo_; }
    unsigned int texture() const { return texture_; }
    int width() const { return width_; }
    int height() const { return height_; }

private:
    unsigned int fbo_ = 0;
    unsigned int texture_ = 0;
    unsigned int depth_ = 0;
    int width_ = 0;
    int height_ = 0;
};

// ── Cameras ────────────────────────────────────

// Degrees of view rotation per pixel of mouse movement while looking around.
inline constexpr float LOOK_SENSITIVITY = 0.15f;

class FreeCamera; // forward declaration for the orbit→free conversion

/// Shared interface for the two cameras the renderer can drive.
class Camera {
public:
    virtual ~Camera() = default;
    virtual glm::mat4 view_matrix() const = 0; // the modelview (view) matrix
    virtual void orbit(float dx, float dy) = 0;
    virtual void pan(float dx, float dy) = 0;
    virtual void move(float forward, float right, float up, float dt) = 0;
    virtual void zoom(float delta) = 0;
    virtual void reset() = 0;
    virtual bool draws_marker() const { return false; }
    virtual glm::vec3 marker_target() const { return glm::vec3(0.0f); }
};

class OrbitCamera : public Camera {
public:
    explicit OrbitCamera(float distance = 6.0f);

    glm::mat4 view_matrix() const override;
    void orbit(float dx, float dy) override;
    void pan(float dx, float dy) override;
    void move(float forward, float right, float up, float dt) override;
    void zoom(float delta) override;
    void reset() override;
    bool draws_marker() const override { return true; }
    glm::vec3 marker_target() const override { return glm::vec3(target_[0], target_[1], target_[2]); }

    void get_state(float& azimuth, float& elevation, float& distance, std::array<float, 3>& target) const;
    void set_state(float azimuth, float elevation, float distance, const std::array<float, 3>& target);
    void set_from_free(const FreeCamera& free);

    float azimuth() const { return azimuth_; }
    float elevation() const { return elevation_; }
    float distance() const { return distance_; }
    const std::array<float, 3>& target() const { return target_; }

private:
    float azimuth_ = 30.0f;
    float elevation_ = 20.0f;
    float distance_;
    std::array<float, 3> target_ = {0.0f, 0.0f, 0.0f};
};

class FreeCamera : public Camera {
public:
    FreeCamera();

    glm::mat4 view_matrix() const override;
    void orbit(float dx, float dy) override;
    void pan(float dx, float dy) override;
    void move(float forward, float right, float up, float dt) override;
    void zoom(float delta) override;
    void reset() override;

    glm::vec3 forward() const;
    void set_from_orbit(const OrbitCamera& orbit);

    const std::array<float, 3>& position() const { return position_; }
    float yaw() const { return yaw_; }
    float pitch() const { return pitch_; }

private:
    // Defaults to the recording camera's own origin, looking down -Z (yaw=0,
    // pitch=0 -> forward (0,0,-1)) — see render_scene_video.py's default
    // ``--position 0 0 0 --target 0 0 -1``, so the viewer's initial view lines
    // up with the real recording camera.
    std::array<float, 3> position_ = {0.0f, 0.0f, 0.0f};
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
};

// ── Scene render ───────────────────────────────

/// Free the renderer's shared GL resources (shader program, helper VAO). Call
/// once on the main thread before the GL context is destroyed.
void shutdown_renderer();

/// What to draw in one frame and how — the per-render inputs that vary frame to
/// frame, grouped so render_scene takes a single descriptor instead of a long and
/// ever-growing parameter list. Every field has a default, so new draw options
/// are added here (not as another render_scene argument) and unused ones can be
/// left out at the call site via designated initializers.
struct SceneRender {
    const FrameGpu* frame = nullptr; // the hands to draw; null draws just the grid
    bool translucent = false;        // draw the hand surface see-through
    bool show_camera_marker = false; // draw the orbit camera's look-at marker
    const GpuMesh* cube = nullptr;   // tracked-object cube to draw; null draws none
    glm::mat4 cube_model{1.0f};      // model matrix placing the cube this frame (see cube_model_matrix)
    // Vertical FOV in degrees. Default (45) is an arbitrary viewer default; pass
    // the recording camera's own FOV (2*atan(height/(2*fy)), matching
    // render_scene_video.py's ``--fov-y`` auto default) so the viewer's default
    // camera view lines up with the real recording camera's framing.
    float fov_y_degrees = 45.0f;
};

/// Render the grid and the active frame into the offscreen framebuffer. The
/// ``framebuffer`` and ``camera`` are the stable where/viewpoint; ``scene``
/// carries what to draw this frame (see SceneRender).
void render_scene(const Framebuffer& framebuffer, const Camera& camera, const SceneRender& scene);

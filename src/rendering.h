#pragma once

// OpenGL rendering: grid, mesh vertex buffers, cameras, lighting, and the
// offscreen framebuffer the scene is drawn into. Uses a modern core-profile
// pipeline (GL 3.3): a single GLSL program with per-vertex lighting, VAOs/VBOs
// for all geometry, and glm-supplied projection/view/model matrices fed as
// uniforms.

#include <array>
#include <atomic>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <glm/glm.hpp>

#include "geometry.h"
#include "sequence.h"
#include "worker_queue.h"

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

    int vertex_count_ = 0;
    unsigned int vao_ = 0;
    unsigned int position_vbo_ = 0;
    unsigned int normal_vbo_ = 0;
    unsigned int color_vbo_ = 0;
};

/// Owns the GPU vertex buffers for the currently loaded single mesh.
class SceneMesh {
public:
    void upload(const std::string& path, const PreparedMesh& prepared);
    void release();
    void draw(bool translucent) const;
    bool has_mesh() const { return hand_solid_ != nullptr; }

private:
    std::optional<std::string> path_;
    std::unique_ptr<GpuMesh> joints_;
    std::unique_ptr<GpuMesh> hand_solid_;
    std::unique_ptr<GpuMesh> hand_translucent_;
};

/// One hand's GPU buffers within a sequence frame.
struct HandGpu {
    std::unique_ptr<GpuMesh> joints;
    std::unique_ptr<GpuMesh> hand_solid;
    std::unique_ptr<GpuMesh> hand_translucent;
};

// A frame's prepared CPU arrays: one (PreparedMesh, mean depth) pair per hand.
using PreparedFrame = std::vector<std::pair<PreparedMesh, float>>;

/// GPU buffers for every hand of one sequence frame, drawn as a unit.
class FrameGpu {
public:
    explicit FrameGpu(const PreparedFrame& prepared_hands);

    void draw(bool translucent, const Transform* transform, std::optional<float> reference_depth) const;

private:
    glm::mat4 hand_matrix(const Transform* transform, float scale) const;

    std::vector<HandGpu> hands_;
    std::vector<float> depths_;
};

/// Expand a frame's hands into GPU-ready arrays plus each hand's mean depth.
PreparedFrame prepare_frame(const Frame& hands);

// ── Frame cache ────────────────────────────────

// Cap on how many frames are kept GPU-resident at once (see FrameCache).
inline constexpr int MAX_RESIDENT_FRAMES = 1024;

/// Turns the loader's in-memory frame positions into GPU buffers, keeping the
/// whole sequence GPU-resident so playback never allocates on the hot path.
class FrameCache {
public:
    explicit FrameCache(SequenceLoader& loader, int gpu_capacity = MAX_RESIDENT_FRAMES);
    ~FrameCache();

    FrameCache(const FrameCache&) = delete;
    FrameCache& operator=(const FrameCache&) = delete;

    /// Return the uploaded FrameGpu for ``index`` (building it if needed), or
    /// nullptr if the frame's positions have not finished parsing. Main thread
    /// only — it creates GPU buffers — and advances the background warm-up.
    FrameGpu* ensure(int index);

private:
    FrameGpu* build_now(int index);
    void warm();
    void prefetch_neighbors(int index);
    void build_prepared(int index);
    FrameGpu* insert_gpu(int index, std::unique_ptr<FrameGpu> frame);
    void touch(int index);

    SequenceLoader& loader_;
    int gpu_capacity_;
    bool fully_resident_;

    // LRU of uploaded frames: list front = least-recently-used.
    std::list<int> lru_;
    std::unordered_map<int, std::pair<std::unique_ptr<FrameGpu>, std::list<int>::iterator>> gpu_;

    std::mutex mutex_; // guards prepared_ / prefetching_
    std::unordered_map<int, PreparedFrame> prepared_;
    std::unordered_set<int> prefetching_;
    int warm_cursor_ = 0;
    WorkerQueue prefetch_pool_;
};

// ── Background single-mesh loading ─────────────

/// Parse a mesh file and expand it into GPU-ready vertex arrays (CPU only).
PreparedMesh prepare_mesh(const std::string& path);

/// A pending background parse of a single mesh file.
class MeshLoadJob {
public:
    explicit MeshLoadJob(const std::string& path);

    const std::string& path() const { return path_; }
    bool done() const;
    PreparedMesh result(); // re-raises any error from the worker

private:
    std::string path_;
    std::shared_future<PreparedMesh> future_;
};

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
    std::array<float, 3> position_ = {0.0f, 2.0f, 6.0f};
    float yaw_ = 0.0f;
    float pitch_ = 0.0f;
};

// ── Scene render ───────────────────────────────

/// Free the renderer's shared GL resources (shader program, helper VAO). Call
/// once on the main thread before the GL context is destroyed.
void shutdown_renderer();

/// Render the grid and the active drawable into the offscreen framebuffer.
/// Exactly one of ``scene`` / ``frame`` is drawn (frame takes precedence);
/// either may be null to draw just the grid.
void render_scene(
    const Framebuffer& framebuffer,
    const Camera& camera,
    const SceneMesh* scene,
    const FrameGpu* frame,
    bool translucent,
    const Transform* transform,
    std::optional<float> reference_depth,
    bool show_camera_marker
);

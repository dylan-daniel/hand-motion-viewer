#include "rendering.h"

#include "gl_loader.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <SDL_opengl.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {
    constexpr float kPi = 3.14159265358979323846f;

    // How many prepared frames to upload per warm() call and how many CPU-array
    // builds to keep queued ahead. Small budgets spread the one-time warm-up over
    // many render frames so no single frame stalls.
    constexpr int WARM_UPLOAD_BUDGET = 4;
    constexpr int WARM_PREP_AHEAD = 3;

    /// Draw a UV sphere of triangles centred on the origin (lighting is off when
    /// this is used, so no normals are emitted).
    void draw_sphere(float radius, int slices, int stacks) {
        glBegin(GL_TRIANGLES);
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

                glVertex3f(a.x, a.y, a.z);
                glVertex3f(b.x, b.y, b.z);
                glVertex3f(c.x, c.y, c.z);

                glVertex3f(a.x, a.y, a.z);
                glVertex3f(c.x, c.y, c.z);
                glVertex3f(d.x, d.y, d.z);
            }
        }
        glEnd();
    }
} // namespace

// ── Grid + marker ──────────────────────────────

void draw_grid(int size, float step, float y) {
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    for (int index = -size; index <= size; ++index) {
        if (index == 0) {
            glColor3f(0.9f, 0.9f, 0.9f);
        } else {
            glColor3f(0.35f, 0.35f, 0.35f);
        }
        glVertex3f(index * step, y, size * step);
        glVertex3f(index * step, y, -size * step);
        glVertex3f(size * step, y, index * step);
        glVertex3f(-size * step, y, index * step);
    }
    glEnd();

    // Axis arrows on the floor; depth test off so they always win over the
    // coincident grid lines instead of z-fighting.
    glDisable(GL_DEPTH_TEST);
    glLineWidth(2.5f);
    glBegin(GL_LINES);
    glColor3f(1.0f, 0.2f, 0.2f); // X – red
    glVertex3f(0.0f, y, 0.0f);
    glVertex3f(2.0f, y, 0.0f);
    glColor3f(0.2f, 0.4f, 1.0f); // Z – blue
    glVertex3f(0.0f, y, 0.0f);
    glVertex3f(0.0f, y, 2.0f);
    glColor3f(0.2f, 0.9f, 0.2f); // Y – green (up)
    glVertex3f(0.0f, y, 0.0f);
    glVertex3f(0.0f, y + 2.0f, 0.0f);
    glEnd();
    glLineWidth(1.0f);
    glEnable(GL_DEPTH_TEST);
}

void draw_camera_marker(const glm::vec3& center, float radius) {
    glPushAttrib(static_cast<GLbitfield>(GL_ENABLE_BIT) | static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT));
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    // Don't write depth so the ball never occludes the scene behind it.
    glDepthMask(GL_FALSE);

    glPushMatrix();
    glTranslatef(center.x, center.y, center.z);
    glColor4f(1.0f, 0.4f, 0.7f, 0.35f);
    draw_sphere(radius, 24, 16);
    glPopMatrix();

    glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
    glPopAttrib();
}

// ── GpuMesh ────────────────────────────────────

namespace {
    void upload_buffer(unsigned int vbo, const std::vector<float>& data) {
        glx::BindBuffer(GL_ARRAY_BUFFER, vbo);
        glx::BufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(data.size() * sizeof(float)), data.data(), GL_STATIC_DRAW);
    }
} // namespace

GpuMesh::GpuMesh(const MeshArrays& arrays) : vertex_count_(arrays.vertex_count) {
    glx::GenBuffers(1, &position_vbo_);
    glx::GenBuffers(1, &normal_vbo_);
    glx::GenBuffers(1, &color_vbo_);
    upload_buffer(position_vbo_, arrays.positions);
    upload_buffer(normal_vbo_, arrays.normals);
    upload_buffer(color_vbo_, arrays.colors);
    glx::BindBuffer(GL_ARRAY_BUFFER, 0);
}

GpuMesh::GpuMesh(GpuMesh&& other) noexcept :
    vertex_count_(other.vertex_count_), position_vbo_(other.position_vbo_), normal_vbo_(other.normal_vbo_), color_vbo_(other.color_vbo_) {
    other.vertex_count_ = 0;
    other.position_vbo_ = other.normal_vbo_ = other.color_vbo_ = 0;
}

GpuMesh& GpuMesh::operator=(GpuMesh&& other) noexcept {
    if (this != &other) {
        release();
        vertex_count_ = other.vertex_count_;
        position_vbo_ = other.position_vbo_;
        normal_vbo_ = other.normal_vbo_;
        color_vbo_ = other.color_vbo_;
        other.vertex_count_ = 0;
        other.position_vbo_ = other.normal_vbo_ = other.color_vbo_ = 0;
    }
    return *this;
}

GpuMesh::~GpuMesh() { release(); }

void GpuMesh::release() {
    if (position_vbo_ != 0 || normal_vbo_ != 0 || color_vbo_ != 0) {
        unsigned int buffers[3] = {position_vbo_, normal_vbo_, color_vbo_};
        glx::DeleteBuffers(3, buffers);
    }
    vertex_count_ = 0;
    position_vbo_ = normal_vbo_ = color_vbo_ = 0;
}

void GpuMesh::draw() const {
    if (vertex_count_ == 0) {
        return;
    }
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glx::BindBuffer(GL_ARRAY_BUFFER, position_vbo_);
    glVertexPointer(3, GL_FLOAT, 0, nullptr);
    glx::BindBuffer(GL_ARRAY_BUFFER, normal_vbo_);
    glNormalPointer(GL_FLOAT, 0, nullptr);
    glx::BindBuffer(GL_ARRAY_BUFFER, color_vbo_);
    glColorPointer(4, GL_FLOAT, 0, nullptr);
    glDrawArrays(GL_TRIANGLES, 0, vertex_count_);
    glx::BindBuffer(GL_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
}

// ── SceneMesh ──────────────────────────────────

void SceneMesh::upload(const std::string& path, const PreparedMesh& prepared) {
    release();
    joints_ = std::make_unique<GpuMesh>(prepared.joints);
    hand_solid_ = std::make_unique<GpuMesh>(prepared.hand_solid);
    hand_translucent_ = std::make_unique<GpuMesh>(prepared.hand_translucent);
    path_ = path;
}

void SceneMesh::release() {
    joints_.reset();
    hand_solid_.reset();
    hand_translucent_.reset();
    path_.reset();
}

void SceneMesh::draw(bool translucent) const {
    // Joints first (opaque) so a translucent hand blends correctly over them.
    if (joints_) {
        joints_->draw();
    }
    if (translucent && hand_translucent_) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
        hand_translucent_->draw();
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    } else if (hand_solid_) {
        hand_solid_->draw();
    }
}

// ── FrameGpu ───────────────────────────────────

PreparedFrame prepare_frame(const Frame& hands) {
    PreparedFrame prepared;
    prepared.reserve(hands.size());
    for (const HandData& hand : hands) {
        PreparedMesh arrays = prepare_hand(hand.positions, hand.topology->faces, hand.topology->colors, hand.topology->hand_face_mask);
        double sum = 0.0;
        for (const glm::vec3& position : hand.positions) {
            sum += position.z;
        }
        const float depth = hand.positions.empty() ? 0.0f : static_cast<float>(sum / hand.positions.size());
        prepared.emplace_back(std::move(arrays), depth);
    }
    return prepared;
}

FrameGpu::FrameGpu(const PreparedFrame& prepared_hands) {
    hands_.reserve(prepared_hands.size());
    depths_.reserve(prepared_hands.size());
    for (const auto& [prepared, depth] : prepared_hands) {
        HandGpu hand;
        hand.joints = std::make_unique<GpuMesh>(prepared.joints);
        hand.hand_solid = std::make_unique<GpuMesh>(prepared.hand_solid);
        hand.hand_translucent = std::make_unique<GpuMesh>(prepared.hand_translucent);
        hands_.push_back(std::move(hand));
        depths_.push_back(depth);
    }
}

void FrameGpu::apply_hand_matrix(const Transform* transform, float scale) const {
    glPushMatrix();
    if (transform != nullptr) {
        glScalef(transform->scale, transform->scale, transform->scale);
        glTranslatef(transform->translate.x, transform->translate.y, transform->translate.z);
    }
    if (scale != 1.0f) {
        glScalef(scale, scale, scale);
    }
}

void FrameGpu::draw(bool translucent, const Transform* transform, std::optional<float> reference_depth) const {
    // Stabilise depth: snap every hand to the common reference plane.
    std::vector<float> scales;
    scales.reserve(depths_.size());
    for (float depth : depths_) {
        scales.push_back((reference_depth && depth != 0.0f) ? (*reference_depth / depth) : 1.0f);
    }

    for (std::size_t index = 0; index < hands_.size(); ++index) {
        apply_hand_matrix(transform, scales[index]);
        hands_[index].joints->draw();
        glPopMatrix();
    }

    if (translucent) {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glDepthMask(GL_FALSE);
    }
    for (std::size_t index = 0; index < hands_.size(); ++index) {
        apply_hand_matrix(transform, scales[index]);
        (translucent ? hands_[index].hand_translucent : hands_[index].hand_solid)->draw();
        glPopMatrix();
    }
    if (translucent) {
        glDepthMask(GL_TRUE);
        glDisable(GL_BLEND);
    }
}

// ── FrameCache ─────────────────────────────────

FrameCache::FrameCache(SequenceLoader& loader, int gpu_capacity) : loader_(loader), gpu_capacity_(std::max(1, std::min(gpu_capacity, loader.frame_count()))) {
    fully_resident_ = gpu_capacity_ >= loader.frame_count();
}

FrameCache::~FrameCache() {
    prefetch_pool_.shutdown();
    // FrameGpu destructors free GL buffers; this runs on the main (GL) thread.
    gpu_.clear();
}

void FrameCache::touch(int index) {
    auto found = gpu_.find(index);
    if (found != gpu_.end()) {
        lru_.splice(lru_.end(), lru_, found->second.second);
    }
}

FrameGpu* FrameCache::insert_gpu(int index, std::unique_ptr<FrameGpu> frame) {
    lru_.push_back(index);
    auto iterator = std::prev(lru_.end());
    FrameGpu* raw = frame.get();
    gpu_[index] = {std::move(frame), iterator};
    while (static_cast<int>(gpu_.size()) > gpu_capacity_) {
        const int oldest = lru_.front();
        lru_.pop_front();
        gpu_.erase(oldest);
    }
    return raw;
}

FrameGpu* FrameCache::ensure(int index) {
    loader_.prioritize(index);

    FrameGpu* cached = nullptr;
    auto found = gpu_.find(index);
    if (found != gpu_.end()) {
        touch(index);
        cached = found->second.first.get();
    } else {
        cached = build_now(index);
    }

    if (fully_resident_) {
        warm();
    } else {
        prefetch_neighbors(index);
    }
    return cached;
}

FrameGpu* FrameCache::build_now(int index) {
    std::optional<PreparedFrame> prepared;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        auto found = prepared_.find(index);
        if (found != prepared_.end()) {
            prepared = std::move(found->second);
            prepared_.erase(found);
        }
    }
    if (!prepared) {
        std::shared_ptr<const Frame> hands = loader_.get(index);
        if (hands == nullptr) {
            return nullptr;
        }
        prepared = prepare_frame(*hands);
    }
    return insert_gpu(index, std::make_unique<FrameGpu>(*prepared));
}

void FrameCache::warm() {
    int in_flight = 0;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        in_flight = static_cast<int>(prefetching_.size());
    }
    while (in_flight < WARM_PREP_AHEAD && warm_cursor_ < loader_.frame_count()) {
        const int index = warm_cursor_;
        if (gpu_.find(index) != gpu_.end()) {
            ++warm_cursor_;
            continue;
        }
        if (!loader_.frame_ready(index)) {
            break; // frames parse in order; wait for this one
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (prepared_.count(index) != 0 || prefetching_.count(index) != 0) {
                ++warm_cursor_;
                continue;
            }
            prefetching_.insert(index);
            ++in_flight;
        }
        prefetch_pool_.submit([this, index] { build_prepared(index); });
        ++warm_cursor_;
    }

    std::vector<int> ready;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        ready.reserve(prepared_.size());
        for (const auto& [index, frame] : prepared_) {
            ready.push_back(index);
        }
    }
    std::sort(ready.begin(), ready.end());

    int uploaded = 0;
    for (int index : ready) {
        if (uploaded >= WARM_UPLOAD_BUDGET) {
            break;
        }
        if (gpu_.find(index) != gpu_.end()) {
            continue;
        }
        std::optional<PreparedFrame> prepared;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            auto found = prepared_.find(index);
            if (found == prepared_.end()) {
                continue;
            }
            prepared = std::move(found->second);
            prepared_.erase(found);
        }
        insert_gpu(index, std::make_unique<FrameGpu>(*prepared));
        ++uploaded;
    }
}

void FrameCache::prefetch_neighbors(int index) {
    for (int neighbor : {index + 1, index - 1}) {
        if (neighbor < 0 || neighbor >= loader_.frame_count() || gpu_.find(neighbor) != gpu_.end()) {
            continue;
        }
        {
            std::lock_guard<std::mutex> guard(mutex_);
            if (prepared_.count(neighbor) != 0 || prefetching_.count(neighbor) != 0) {
                continue;
            }
            prefetching_.insert(neighbor);
        }
        prefetch_pool_.submit([this, neighbor] { build_prepared(neighbor); });
    }
}

void FrameCache::build_prepared(int index) {
    std::shared_ptr<const Frame> hands = loader_.get(index);
    if (hands != nullptr) {
        PreparedFrame prepared = prepare_frame(*hands);
        std::lock_guard<std::mutex> guard(mutex_);
        prepared_[index] = std::move(prepared);
    }
    std::lock_guard<std::mutex> guard(mutex_);
    prefetching_.erase(index);
}

// ── Background single-mesh loading ─────────────

PreparedMesh prepare_mesh(const std::string& path) {
    auto [hand, joints] = load_mesh(path);
    // Both parts share the same vertex list, so centre it once and reuse.
    center_model(hand.verts);
    joints.verts = hand.verts;

    PreparedMesh prepared;
    prepared.joints = build_arrays(joints);
    prepared.hand_solid = build_arrays(hand);
    prepared.hand_translucent = build_arrays(hand, 0.30f);
    prepared.hand_triangle_count = static_cast<int>(hand.faces.size());
    prepared.joint_triangle_count = static_cast<int>(joints.faces.size());
    return prepared;
}

MeshLoadJob::MeshLoadJob(const std::string& path) : path_(path), future_(std::async(std::launch::async, prepare_mesh, path).share()) {}

bool MeshLoadJob::done() const { return future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready; }

PreparedMesh MeshLoadJob::result() { return future_.get(); }

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
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, nullptr);
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

void OrbitCamera::apply() const {
    const float az = glm::radians(azimuth_);
    const float el = glm::radians(elevation_);
    const glm::vec3 target(target_[0], target_[1], target_[2]);
    const glm::vec3 eye(
        target.x + distance_ * std::cos(el) * std::sin(az), target.y + distance_ * std::sin(el), target.z + distance_ * std::cos(el) * std::cos(az)
    );
    const glm::mat4 view = glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    glLoadMatrixf(glm::value_ptr(view));
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

void FreeCamera::apply() const {
    const glm::vec3 dir = forward();
    const glm::vec3 eye(position_[0], position_[1], position_[2]);
    const glm::mat4 view = glm::lookAt(eye, eye + dir, glm::vec3(0.0f, 1.0f, 0.0f));
    glLoadMatrixf(glm::value_ptr(view));
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

// ── Lighting + scene render ────────────────────

void setup_lighting() {
    glEnable(GL_LIGHTING);
    glEnable(GL_LIGHT0);
    glEnable(GL_LIGHT1);
    const float light0_position[4] = {5.0f, 10.0f, 5.0f, 1.0f};
    const float light0_diffuse[4] = {0.85f, 0.85f, 0.85f, 1.0f};
    const float light0_ambient[4] = {0.15f, 0.15f, 0.15f, 1.0f};
    glLightfv(GL_LIGHT0, GL_POSITION, light0_position);
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light0_diffuse);
    glLightfv(GL_LIGHT0, GL_AMBIENT, light0_ambient);
    const float light1_position[4] = {-5.0f, 6.0f, -8.0f, 1.0f};
    const float light1_diffuse[4] = {0.3f, 0.35f, 0.5f, 1.0f};
    glLightfv(GL_LIGHT1, GL_POSITION, light1_position);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, light1_diffuse);
    // Per-vertex glColor drives ambient+diffuse so mesh colours show; keep a
    // fixed specular highlight.
    glEnable(GL_COLOR_MATERIAL);
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
    const float specular[4] = {0.4f, 0.4f, 0.4f, 1.0f};
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 32.0f);
}

void render_scene(
    const Framebuffer& framebuffer,
    const Camera& camera,
    const SceneMesh* scene,
    const FrameGpu* frame,
    bool translucent,
    const Transform* transform,
    std::optional<float> reference_depth,
    bool show_camera_marker
) {
    glx::BindFramebuffer(GL_FRAMEBUFFER, framebuffer.fbo());
    glViewport(0, 0, framebuffer.width(), framebuffer.height());
    glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
    glClear(static_cast<GLbitfield>(GL_COLOR_BUFFER_BIT) | static_cast<GLbitfield>(GL_DEPTH_BUFFER_BIT));

    const float aspect = framebuffer.height() != 0 ? static_cast<float>(framebuffer.width()) / static_cast<float>(framebuffer.height()) : 1.0f;
    glMatrixMode(GL_PROJECTION);
    const glm::mat4 projection = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 500.0f);
    glLoadMatrixf(glm::value_ptr(projection));
    glMatrixMode(GL_MODELVIEW);
    camera.apply();

    // Grid (lighting off so it stays flat-coloured).
    glDisable(GL_LIGHTING);
    draw_grid(10, 1.0f, 0.0f);
    glEnable(GL_LIGHTING);

    if (frame != nullptr) {
        frame->draw(translucent, transform, reference_depth);
    } else if (scene != nullptr && scene->has_mesh()) {
        glPushMatrix();
        if (transform != nullptr) {
            glScalef(transform->scale, transform->scale, transform->scale);
            glTranslatef(transform->translate.x, transform->translate.y, transform->translate.z);
        }
        scene->draw(translucent);
        glPopMatrix();
    }

    // Marker for the orbit camera's look-at point; drawn last so its
    // translucency blends over the scene.
    if (show_camera_marker && camera.draws_marker()) {
        draw_camera_marker(camera.marker_target());
    }

    glx::BindFramebuffer(GL_FRAMEBUFFER, 0);
}

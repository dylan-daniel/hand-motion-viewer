#include "graphics/image.h"

#include <cstdio>
#include <utility>

#include <SDL_opengl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

ImageTexture::~ImageTexture() {
    worker_.shutdown(); // stop the decode thread before tearing down GL/state
    release();
}

void ImageTexture::release() {
    if (texture_ != 0) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    width_ = 0;
    height_ = 0;
    tex_capacity_w_ = 0;
    tex_capacity_h_ = 0;
}

void ImageTexture::clear() {
    {
        std::lock_guard<std::mutex> guard(mutex_);
        desired_path_.clear(); // a late decode for this empty target is dropped on upload
        has_pending_ = false;
        pending_ = Decoded{};
    }
    release();
    path_.clear();
}

bool ImageTexture::load(const std::string& path) {
    // Upload a finished decode (if any) on this render thread.
    upload_ready();

    bool submit = false;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (path != desired_path_) {
            desired_path_ = path;
            // Only kick a new decode chain if one is not already running; a running
            // chain picks up the new desired path itself.
            if (!worker_busy_) {
                worker_busy_ = true;
                submit = true;
            }
        }
    }
    if (submit) {
        worker_.submit([this] { decode_loop(); });
    }
    return texture_ != 0;
}

void ImageTexture::decode_loop() {
    // Decode the latest requested path, collapsing any backlog: if the request
    // changed while we were decoding, decode the newer one instead of the stale
    // queue. Exits once the decoded path matches the still-current request.
    while (true) {
        std::string target;
        {
            std::lock_guard<std::mutex> guard(mutex_);
            target = desired_path_;
        }

        int width = 0;
        int height = 0;
        int channels = 0;
        unsigned char* pixels = target.empty() ? nullptr : stbi_load(target.c_str(), &width, &height, &channels, 4);

        std::lock_guard<std::mutex> guard(mutex_);
        pending_.path = target;
        pending_.width = width;
        pending_.height = height;
        pending_.pixels = std::unique_ptr<unsigned char, void (*)(void*)>(pixels, stbi_image_free);
        has_pending_ = true;
        if (desired_path_ == target) {
            worker_busy_ = false;
            return;
        }
    }
}

void ImageTexture::upload_ready() {
    Decoded ready;
    bool is_current_target = false;
    {
        std::lock_guard<std::mutex> guard(mutex_);
        if (!has_pending_) {
            return;
        }
        ready = std::move(pending_);
        has_pending_ = false;
        // Cleared since this was requested — drop it (ready frees the pixels).
        if (desired_path_.empty()) {
            return;
        }
        is_current_target = (ready.path == desired_path_);
    }

    if (ready.pixels == nullptr) {
        std::printf("Failed to load image %s: %s\n", ready.path.c_str(), stbi_failure_reason());
        // A missing or unreadable image must not leave the previous frame's
        // texture on screen. Drop it so the pane reads as having no image. Skip
        // stale failures: a newer decode is in flight and will set the texture.
        if (is_current_target) {
            release();
            path_ = ready.path;
        }
        return;
    }

    if (texture_ == 0) {
        glGenTextures(1, &texture_);
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    if (ready.width != tex_capacity_w_ || ready.height != tex_capacity_h_) {
        // First upload or a size change: (re)allocate the texture storage.
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, ready.width, ready.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, ready.pixels.get());
        tex_capacity_w_ = ready.width;
        tex_capacity_h_ = ready.height;
    } else {
        // Same size as the last frame: overwrite the existing storage in place.
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, ready.width, ready.height, GL_RGBA, GL_UNSIGNED_BYTE, ready.pixels.get());
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    width_ = ready.width;
    height_ = ready.height;
    path_ = ready.path;
}

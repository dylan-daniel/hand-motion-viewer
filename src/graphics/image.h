#pragma once

// Decodes an image file (the per-frame ``frame_NNNN_all_keypoints.jpg``) into an
// OpenGL texture for display in the image viewport. Decoding runs on a background
// worker thread so a 1080p JPEG decode never stalls the render thread during
// playback; the render thread only does the cheap GL upload of finished pixels,
// reusing the texture in place. While a new frame decodes, the previously
// uploaded texture keeps showing, so the image trails the meshes by at most a
// frame or two instead of blocking the frame.

#include <memory>
#include <mutex>
#include <string>

#include "util/worker_queue.h"

/// A single GL texture holding a decoded image, refreshed on demand by path.
class ImageTexture {
public:
    ImageTexture() = default;
    ~ImageTexture();

    ImageTexture(const ImageTexture&) = delete;
    ImageTexture& operator=(const ImageTexture&) = delete;

    /// Request ``path`` for display. Non-blocking: the decode happens on a worker
    /// thread and the finished pixels are uploaded by a later call (also from the
    /// render thread). Requesting the path already shown is a cheap no-op. Returns
    /// true if a valid texture is currently available.
    bool load(const std::string& path);

    /// Drop the texture and forget the loaded path (cancels any pending decode).
    void clear();

    bool valid() const { return texture_ != 0; }
    unsigned int texture() const { return texture_; }
    int width() const { return width_; }
    int height() const { return height_; }
    const std::string& path() const { return path_; }

private:
    /// A decoded image handed from the worker thread to the render thread. ``pixels``
    /// owns the stb_image buffer and frees it with stbi_image_free.
    struct Decoded {
        std::string path;
        int width = 0;
        int height = 0;
        std::unique_ptr<unsigned char, void (*)(void*)> pixels{nullptr, nullptr};
    };

    void release();
    void upload_ready(); // render/GL thread: upload a finished decode, if any
    void decode_loop();  // worker thread: decode the latest requested path

    // GL texture state — render thread only.
    std::string path_; // path currently shown
    unsigned int texture_ = 0;
    int width_ = 0;
    int height_ = 0;
    int tex_capacity_w_ = 0; // allocated texture dims, so a same-size frame reuses it
    int tex_capacity_h_ = 0;

    // Background-decode handoff — guarded by mutex_.
    std::mutex mutex_;
    std::string desired_path_; // most recently requested path
    bool worker_busy_ = false; // a decode chain is running
    bool has_pending_ = false; // pending_ holds an un-uploaded decode
    Decoded pending_;

    WorkerQueue worker_; // declared last so it joins before the state above is destroyed
};

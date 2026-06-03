#pragma once

// Decodes an image file (the per-frame ``frame_NNNN_all_keypoints.jpg``) into an
// OpenGL texture for display in the image viewport. The last-loaded path is
// remembered so re-requesting the same frame's image is a no-op and playback
// never re-decodes on the hot path.

#include <string>

/// A single GL texture holding a decoded image, reloaded on demand by path.
class ImageTexture {
public:
    ImageTexture() = default;
    ~ImageTexture();

    ImageTexture(const ImageTexture&) = delete;
    ImageTexture& operator=(const ImageTexture&) = delete;

    /// Decode ``path`` into the texture, replacing whatever was loaded before.
    /// Reloading the path already shown is a no-op. Returns true if a valid
    /// texture is available afterwards.
    bool load(const std::string& path);

    /// Drop the texture and forget the loaded path.
    void clear();

    bool valid() const { return texture_ != 0; }
    unsigned int texture() const { return texture_; }
    int width() const { return width_; }
    int height() const { return height_; }
    const std::string& path() const { return path_; }

private:
    void release();

    std::string path_;
    unsigned int texture_ = 0;
    int width_ = 0;
    int height_ = 0;
};

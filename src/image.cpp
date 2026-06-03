#include "image.h"

#include <cstdio>

#include <SDL_opengl.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

ImageTexture::~ImageTexture() { release(); }

void ImageTexture::release() {
    if (texture_ != 0) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    width_ = 0;
    height_ = 0;
}

void ImageTexture::clear() {
    release();
    path_.clear();
}

bool ImageTexture::load(const std::string& path) {
    // Already showing this image — nothing to do.
    if (path == path_ && texture_ != 0) {
        return true;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    // Force RGBA so the upload format is fixed regardless of the source.
    stbi_uc* pixels = stbi_load(path.c_str(), &width, &height, &channels, 4);
    if (pixels == nullptr) {
        std::printf("Failed to load image %s: %s\n", path.c_str(), stbi_failure_reason());
        release();
        path_ = path;
        return false;
    }

    if (texture_ == 0) {
        glGenTextures(1, &texture_);
    }
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    stbi_image_free(pixels);

    width_ = width;
    height_ = height;
    path_ = path;
    return true;
}

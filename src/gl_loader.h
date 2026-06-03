#pragma once

// Modern OpenGL entry points the viewer needs beyond the GL 1.1 core that
// opengl32 exports on Windows: vertex buffer objects (for the mesh GpuMesh) and
// framebuffer objects (the offscreen render target). They are resolved at
// runtime with SDL_GL_GetProcAddress, so the only dependency is SDL2 — no GLEW
// or GLAD. SDL_opengl.h supplies the function-pointer typedefs and the GL_*
// enums (it bundles a copy of glext) without declaring prototypes, so these
// names never clash with the platform GL headers.

#include <SDL_opengl.h>

namespace glx {
    // Vertex buffer objects.
    extern PFNGLGENBUFFERSPROC GenBuffers;
    extern PFNGLBINDBUFFERPROC BindBuffer;
    extern PFNGLBUFFERDATAPROC BufferData;
    extern PFNGLDELETEBUFFERSPROC DeleteBuffers;

    // Framebuffer objects.
    extern PFNGLGENFRAMEBUFFERSPROC GenFramebuffers;
    extern PFNGLBINDFRAMEBUFFERPROC BindFramebuffer;
    extern PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D;
    extern PFNGLFRAMEBUFFERRENDERBUFFERPROC FramebufferRenderbuffer;
    extern PFNGLDELETEFRAMEBUFFERSPROC DeleteFramebuffers;
    extern PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus;

    // Renderbuffer objects (the framebuffer's depth attachment).
    extern PFNGLGENRENDERBUFFERSPROC GenRenderbuffers;
    extern PFNGLBINDRENDERBUFFERPROC BindRenderbuffer;
    extern PFNGLRENDERBUFFERSTORAGEPROC RenderbufferStorage;
    extern PFNGLDELETERENDERBUFFERSPROC DeleteRenderbuffers;

    /// Resolve every entry point above; returns false if any could not be loaded.
    bool load();
} // namespace glx

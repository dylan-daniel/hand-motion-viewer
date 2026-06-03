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

    // Vertex array objects (required for every draw in a core profile).
    extern PFNGLGENVERTEXARRAYSPROC GenVertexArrays;
    extern PFNGLBINDVERTEXARRAYPROC BindVertexArray;
    extern PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays;

    // Generic vertex attributes (replace the fixed-function client arrays).
    extern PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray;
    extern PFNGLDISABLEVERTEXATTRIBARRAYPROC DisableVertexAttribArray;
    extern PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer;
    extern PFNGLVERTEXATTRIB3FPROC VertexAttrib3f;

    // Shader programs (replace the fixed-function transform + lighting).
    extern PFNGLCREATESHADERPROC CreateShader;
    extern PFNGLSHADERSOURCEPROC ShaderSource;
    extern PFNGLCOMPILESHADERPROC CompileShader;
    extern PFNGLGETSHADERIVPROC GetShaderiv;
    extern PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog;
    extern PFNGLDELETESHADERPROC DeleteShader;
    extern PFNGLCREATEPROGRAMPROC CreateProgram;
    extern PFNGLATTACHSHADERPROC AttachShader;
    extern PFNGLLINKPROGRAMPROC LinkProgram;
    extern PFNGLGETPROGRAMIVPROC GetProgramiv;
    extern PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog;
    extern PFNGLUSEPROGRAMPROC UseProgram;
    extern PFNGLDELETEPROGRAMPROC DeleteProgram;
    extern PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation;
    extern PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv;
    extern PFNGLUNIFORM1IPROC Uniform1i;

    /// Resolve every entry point above; returns false if any could not be loaded.
    bool load();
} // namespace glx

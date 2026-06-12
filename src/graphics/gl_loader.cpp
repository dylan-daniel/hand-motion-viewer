#include "graphics/gl_loader.h"

#include <SDL3/SDL.h>

namespace glx {
    PFNGLGENBUFFERSPROC GenBuffers = nullptr;
    PFNGLBINDBUFFERPROC BindBuffer = nullptr;
    PFNGLBUFFERDATAPROC BufferData = nullptr;
    PFNGLDELETEBUFFERSPROC DeleteBuffers = nullptr;

    PFNGLGENFRAMEBUFFERSPROC GenFramebuffers = nullptr;
    PFNGLBINDFRAMEBUFFERPROC BindFramebuffer = nullptr;
    PFNGLFRAMEBUFFERTEXTURE2DPROC FramebufferTexture2D = nullptr;
    PFNGLFRAMEBUFFERRENDERBUFFERPROC FramebufferRenderbuffer = nullptr;
    PFNGLDELETEFRAMEBUFFERSPROC DeleteFramebuffers = nullptr;
    PFNGLCHECKFRAMEBUFFERSTATUSPROC CheckFramebufferStatus = nullptr;

    PFNGLGENRENDERBUFFERSPROC GenRenderbuffers = nullptr;
    PFNGLBINDRENDERBUFFERPROC BindRenderbuffer = nullptr;
    PFNGLRENDERBUFFERSTORAGEPROC RenderbufferStorage = nullptr;
    PFNGLDELETERENDERBUFFERSPROC DeleteRenderbuffers = nullptr;

    PFNGLGENVERTEXARRAYSPROC GenVertexArrays = nullptr;
    PFNGLBINDVERTEXARRAYPROC BindVertexArray = nullptr;
    PFNGLDELETEVERTEXARRAYSPROC DeleteVertexArrays = nullptr;

    PFNGLENABLEVERTEXATTRIBARRAYPROC EnableVertexAttribArray = nullptr;
    PFNGLDISABLEVERTEXATTRIBARRAYPROC DisableVertexAttribArray = nullptr;
    PFNGLVERTEXATTRIBPOINTERPROC VertexAttribPointer = nullptr;
    PFNGLVERTEXATTRIB3FPROC VertexAttrib3f = nullptr;

    PFNGLCREATESHADERPROC CreateShader = nullptr;
    PFNGLSHADERSOURCEPROC ShaderSource = nullptr;
    PFNGLCOMPILESHADERPROC CompileShader = nullptr;
    PFNGLGETSHADERIVPROC GetShaderiv = nullptr;
    PFNGLGETSHADERINFOLOGPROC GetShaderInfoLog = nullptr;
    PFNGLDELETESHADERPROC DeleteShader = nullptr;
    PFNGLCREATEPROGRAMPROC CreateProgram = nullptr;
    PFNGLATTACHSHADERPROC AttachShader = nullptr;
    PFNGLLINKPROGRAMPROC LinkProgram = nullptr;
    PFNGLGETPROGRAMIVPROC GetProgramiv = nullptr;
    PFNGLGETPROGRAMINFOLOGPROC GetProgramInfoLog = nullptr;
    PFNGLUSEPROGRAMPROC UseProgram = nullptr;
    PFNGLDELETEPROGRAMPROC DeleteProgram = nullptr;
    PFNGLGETUNIFORMLOCATIONPROC GetUniformLocation = nullptr;
    PFNGLUNIFORMMATRIX4FVPROC UniformMatrix4fv = nullptr;
    PFNGLUNIFORM1IPROC Uniform1i = nullptr;
    PFNGLUNIFORM1FPROC Uniform1f = nullptr;

    namespace {
        /// Resolve one entry point, recording failure in ``ok``.
        template <typename Fn>
        void resolve(Fn& slot, const char* name, bool& ok) {
            slot = reinterpret_cast<Fn>(SDL_GL_GetProcAddress(name));
            if (slot == nullptr) {
                ok = false;
            }
        }
    } // namespace

    bool load() {
        bool ok = true;
        resolve(GenBuffers, "glGenBuffers", ok);
        resolve(BindBuffer, "glBindBuffer", ok);
        resolve(BufferData, "glBufferData", ok);
        resolve(DeleteBuffers, "glDeleteBuffers", ok);

        resolve(GenFramebuffers, "glGenFramebuffers", ok);
        resolve(BindFramebuffer, "glBindFramebuffer", ok);
        resolve(FramebufferTexture2D, "glFramebufferTexture2D", ok);
        resolve(FramebufferRenderbuffer, "glFramebufferRenderbuffer", ok);
        resolve(DeleteFramebuffers, "glDeleteFramebuffers", ok);
        resolve(CheckFramebufferStatus, "glCheckFramebufferStatus", ok);

        resolve(GenRenderbuffers, "glGenRenderbuffers", ok);
        resolve(BindRenderbuffer, "glBindRenderbuffer", ok);
        resolve(RenderbufferStorage, "glRenderbufferStorage", ok);
        resolve(DeleteRenderbuffers, "glDeleteRenderbuffers", ok);

        resolve(GenVertexArrays, "glGenVertexArrays", ok);
        resolve(BindVertexArray, "glBindVertexArray", ok);
        resolve(DeleteVertexArrays, "glDeleteVertexArrays", ok);

        resolve(EnableVertexAttribArray, "glEnableVertexAttribArray", ok);
        resolve(DisableVertexAttribArray, "glDisableVertexAttribArray", ok);
        resolve(VertexAttribPointer, "glVertexAttribPointer", ok);
        resolve(VertexAttrib3f, "glVertexAttrib3f", ok);

        resolve(CreateShader, "glCreateShader", ok);
        resolve(ShaderSource, "glShaderSource", ok);
        resolve(CompileShader, "glCompileShader", ok);
        resolve(GetShaderiv, "glGetShaderiv", ok);
        resolve(GetShaderInfoLog, "glGetShaderInfoLog", ok);
        resolve(DeleteShader, "glDeleteShader", ok);
        resolve(CreateProgram, "glCreateProgram", ok);
        resolve(AttachShader, "glAttachShader", ok);
        resolve(LinkProgram, "glLinkProgram", ok);
        resolve(GetProgramiv, "glGetProgramiv", ok);
        resolve(GetProgramInfoLog, "glGetProgramInfoLog", ok);
        resolve(UseProgram, "glUseProgram", ok);
        resolve(DeleteProgram, "glDeleteProgram", ok);
        resolve(GetUniformLocation, "glGetUniformLocation", ok);
        resolve(UniformMatrix4fv, "glUniformMatrix4fv", ok);
        resolve(Uniform1i, "glUniform1i", ok);
        resolve(Uniform1f, "glUniform1f", ok);
        return ok;
    }
} // namespace glx

#include "gl_loader.h"

#include <SDL.h>

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
        return ok;
    }
} // namespace glx

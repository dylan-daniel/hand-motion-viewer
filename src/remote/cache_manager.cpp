#include "remote/cache_manager.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

#include <SDL3/SDL.h>

namespace fs = std::filesystem;

namespace {
    std::string sanitize_identifier(const std::string& input) {
        std::string result;
        for (char c : input) {
            if (std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_') {
                result.push_back(c);
            } else {
                result.push_back('_');
            }
        }
        if (result.empty()) {
            result = "default";
        }
        return result;
    }
} // namespace

void CacheManager::init() {
    const std::string root = get_cache_root();
    std::error_code ec;
    fs::create_directories(root, ec);
}

std::string CacheManager::get_cache_root() {
    char* pref = SDL_GetPrefPath("hand_motion_viewer", "hand_motion_viewer");
    std::string root;
    if (pref != nullptr) {
        root = std::string(pref) + "remote_cache";
        SDL_free(pref);
    } else {
        root = "remote_cache";
    }
    return root;
}

std::string CacheManager::get_local_export_path(const std::string& host, const std::string& remote_export_path) {
    const std::string safe_host = sanitize_identifier(host);
    const fs::path remote_p(remote_export_path);
    const std::string filename = remote_p.filename().string();
    const std::string parent_name = remote_p.parent_path().filename().string();

    fs::path local_p = fs::path(get_cache_root()) / safe_host;
    if (!parent_name.empty()) {
        local_p /= parent_name;
    }
    local_p /= filename;
    return local_p.string();
}

std::string CacheManager::get_local_frames_dir(const std::string& local_export_path) {
    const fs::path p(local_export_path);
    return (p.parent_path() / "frames" / p.stem()).string();
}

bool CacheManager::is_export_cached(const std::string& local_export_path) {
    std::error_code ec;
    return fs::is_regular_file(local_export_path, ec) && fs::file_size(local_export_path, ec) > 0;
}

bool CacheManager::are_frames_cached(const std::string& local_frames_dir) {
    std::error_code ec;
    if (!fs::is_directory(local_frames_dir, ec)) {
        return false;
    }
    // Check if at least one frame image exists in the directory
    for (const auto& entry : fs::directory_iterator(local_frames_dir, ec)) {
        if (entry.is_regular_file()) {
            const std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png" || ext == ".jpeg") {
                return true;
            }
        }
    }
    return false;
}

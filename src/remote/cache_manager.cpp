#include "remote/cache_manager.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <system_error>

#include <SDL3/SDL.h>

namespace fs = std::filesystem;

namespace {
    std::string s_custom_cache_root;

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

std::string CacheManager::get_default_cache_root() {
    std::error_code ec;
    fs::path temp_dir = fs::temp_directory_path(ec);
    if (ec || temp_dir.empty()) {
        temp_dir = fs::path("/tmp");
    }
    return (temp_dir / "hand_motion_viewer_cache").string();
}

void CacheManager::set_custom_cache_root(const std::string& custom_root) {
    s_custom_cache_root = custom_root;
    init();
}

const std::string& CacheManager::get_custom_cache_root() { return s_custom_cache_root; }

std::string CacheManager::get_cache_root() {
    if (!s_custom_cache_root.empty()) {
        return s_custom_cache_root;
    }
    return get_default_cache_root();
}

std::uintmax_t CacheManager::calculate_cache_size_bytes() {
    const std::string root = get_cache_root();
    std::error_code ec;
    if (!fs::is_directory(root, ec)) {
        return 0;
    }
    std::uintmax_t total_size = 0;
    for (const auto& entry : fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec)) {
        if (entry.is_regular_file(ec)) {
            total_size += entry.file_size(ec);
        }
    }
    return total_size;
}

void CacheManager::clear_cache() {
    const std::string root = get_cache_root();
    std::error_code ec;
    if (fs::exists(root, ec)) {
        fs::remove_all(root, ec);
    }
    init();
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
    const std::string stem = p.stem().string();
    std::string frames_key = stem;
    const auto pos1 = stem.find("__");
    if (pos1 != std::string::npos) {
        const auto pos2 = stem.find("__", pos1 + 2);
        if (pos2 != std::string::npos) {
            frames_key = stem.substr(0, pos2);
        }
    }
    return (p.parent_path() / "frames" / frames_key).string();
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

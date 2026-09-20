#pragma once

#include <cstdint>
#include <string>

class CacheManager {
public:
    /// Initialize cache manager (ensures cache directories exist).
    static void init();

    /// Get default system temporary directory for cache:
    /// e.g. <temp_dir>/hand_motion_viewer_cache
    static std::string get_default_cache_root();

    /// Set custom cache root directory (if empty, defaults to get_default_cache_root()).
    static void set_custom_cache_root(const std::string& custom_root);

    /// Get custom cache root (empty if using default).
    static const std::string& get_custom_cache_root();

    /// Base cache directory on local machine.
    static std::string get_cache_root();

    /// Convert a remote host and export file path to a local cached .hexport path
    static std::string get_local_export_path(const std::string& host, const std::string& remote_export_path);

    /// Get local frames directory for a cached export path:
    /// returns <parent_dir>/frames/<hash>__<fps>
    static std::string get_local_frames_dir(const std::string& local_export_path);

    /// Check if the .hexport file is already cached locally and valid
    static bool is_export_cached(const std::string& local_export_path);

    /// Check if frames folder is already populated locally
    static bool are_frames_cached(const std::string& local_frames_dir);

    /// Calculate total disk space currently used by the cache in bytes.
    static std::uintmax_t calculate_cache_size_bytes();

    /// Clear all files in the current cache directory.
    static void clear_cache();
};

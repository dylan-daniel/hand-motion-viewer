#pragma once

#include <string>

class CacheManager {
public:
    /// Initialize cache manager (ensures cache directories exist).
    static void init();

    /// Base cache directory on local machine:
    /// <pref_path>/remote_cache
    static std::string get_cache_root();

    /// Convert a remote host and export file path to a local cached .hexport path
    static std::string get_local_export_path(const std::string& host, const std::string& remote_export_path);

    /// Get local frames directory for a cached export path:
    /// returns <parent_dir>/frames/<export_stem>
    static std::string get_local_frames_dir(const std::string& local_export_path);

    /// Check if the .hexport file is already cached locally and valid
    static bool is_export_cached(const std::string& local_export_path);

    /// Check if frames folder is already populated locally
    static bool are_frames_cached(const std::string& local_frames_dir);
};

#pragma once

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <nlohmann/json.hpp>

#include "ui/explorer.h"
#include "util/worker_queue.h"

struct RemoteConfig {
    std::string host = ""; // e.g. "tus43540@cis-a6000" or ssh alias, or "localhost"
    int port = 22;
    std::string python_bin = "python3";
    std::string script_path = "~/workspace/hand_motion_viewer/scripts/viewer_daemon.py";
    std::string root_folder = "/mnt/nvme1tb/infant_grasp_pipeline_cache";
};

enum class ConnectionState { Disconnected, Connecting, Connected, Error };

class RemoteClient {
public:
    RemoteClient();
    ~RemoteClient();

    RemoteClient(const RemoteClient&) = delete;
    RemoteClient& operator=(const RemoteClient&) = delete;

    ConnectionState state() const { return state_.load(); }
    bool is_connected() const { return state_.load() == ConnectionState::Connected; }
    bool is_connecting() const { return state_.load() == ConnectionState::Connecting; }
    std::string last_error() const;
    RemoteConfig config() const;

    /// Initiate connection to remote host asynchronously on a background worker.
    /// Never blocks the caller thread.
    void connect_async(const RemoteConfig& config);

    /// Synchronous connect (called on background worker thread).
    bool connect_sync(const RemoteConfig& config, std::string& error_out);

    /// Disconnect from remote host and terminate daemon process.
    void disconnect();

    /// Called on the UI thread every frame to monitor child process health
    /// and transition state if connection is lost.
    void poll();

    /// Returns true if a background connection attempt just succeeded this frame.
    bool consume_just_connected();

    /// Ping the daemon to ensure connection is live.
    bool ping(std::string& error_out);

    /// Scan directory on remote server and build a pruned tree for the explorer pane.
    bool scan_tree(const std::string& remote_root, FileExplorer::Node& out_node, std::string& error_out);

    /// Fetch a remote file (such as a .hexport file) and save directly to local_dest_path.
    bool fetch_file(const std::string& remote_path, const std::string& local_dest_path, std::string& error_out);

    /// Fetch a single frame image and save to local_file_dest.
    bool fetch_single_frame(const std::string& remote_export_path, int frame_number, const std::string& local_file_dest, std::string& error_out);

    /// Fetch all frames for a sequence as a zip bundle and extract to local_frames_dir.
    bool fetch_and_extract_bundle(const std::string& remote_export_path, const std::string& local_frames_dir, int& frame_count_out, std::string& error_out);

private:
    void disconnect_locked();
    bool send_command_locked(const nlohmann::json& req, nlohmann::json& resp_out, std::string& error_out);
    bool send_command_binary_locked(const nlohmann::json& req, nlohmann::json& resp_hdr_out, std::vector<uint8_t>& data_out, std::string& error_out);

    mutable std::mutex mutex_;
    RemoteConfig config_;
    std::atomic<ConnectionState> state_{ConnectionState::Disconnected};
    std::string last_error_;
    std::atomic<bool> just_connected_{false};

    SDL_Process* process_ = nullptr;
    SDL_IOStream* stdin_ = nullptr;
    SDL_IOStream* stdout_ = nullptr;
    int next_id_ = 1;

    WorkerQueue async_worker_;
};

#pragma once

#include "remote/remote_client.h"

#include <condition_variable>
#include <filesystem>
#include <mutex>
#include <string>
#include <thread>
#include <utility>

class ScrubWorker {
public:
    explicit ScrubWorker(RemoteClient& client) : client_(client), worker_([this] { run(); }) {}

    ~ScrubWorker() { shutdown(); }

    ScrubWorker(const ScrubWorker&) = delete;
    ScrubWorker& operator=(const ScrubWorker&) = delete;

    /// Request on-demand download of a single frame if not already present on disk.
    /// Debounces/collapses rapid requests: if a newer request arrives before the
    /// previous one starts, only the latest requested frame will be fetched.
    void request(const std::string& remote_path, int frame_number, const std::string& local_dest) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            remote_path_ = remote_path;
            frame_number_ = frame_number;
            local_dest_ = local_dest;
            has_request_ = true;
        }
        cv_.notify_one();
    }

    /// Cancel any pending unstarted request (e.g. on sequence change).
    void cancel() {
        std::lock_guard<std::mutex> lock(mutex_);
        has_request_ = false;
        frame_number_ = -1;
    }

    /// Shutdown the worker thread.
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) {
                return;
            }
            stop_ = true;
            has_request_ = false;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

private:
    void run() {
        namespace fs = std::filesystem;
        while (true) {
            std::string remote_path;
            int frame_number = -1;
            std::string local_dest;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || has_request_; });
                if (stop_) {
                    return;
                }
                remote_path = std::move(remote_path_);
                frame_number = frame_number_;
                local_dest = std::move(local_dest_);
                has_request_ = false;
            }

            if (frame_number > 0 && !remote_path.empty() && client_.is_connected()) {
                std::error_code ec;
                if (!fs::exists(local_dest, ec)) {
                    std::string err;
                    client_.fetch_single_frame(remote_path, frame_number, local_dest, err);
                }
            }
        }
    }

    RemoteClient& client_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
    bool has_request_ = false;
    std::string remote_path_;
    int frame_number_ = -1;
    std::string local_dest_;
    std::thread worker_;
};

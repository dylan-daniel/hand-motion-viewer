#include "remote/remote_client.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

#ifndef _WIN32
    #include <csignal>
namespace {
    struct SigPipeIgnorer {
        SigPipeIgnorer() { std::signal(SIGPIPE, SIG_IGN); }
    } s_sigpipe_ignorer;
} // namespace
#endif

#include <miniz.h>

namespace fs = std::filesystem;
using nlohmann::json;

namespace {
    constexpr int DEFAULT_TIMEOUT_MS = 15000;
    constexpr int BUNDLE_TIMEOUT_MS = 60000;

    bool read_exact(SDL_IOStream* stream, void* dest, size_t total_bytes, int timeout_ms) {
        uint8_t* ptr = static_cast<uint8_t*>(dest);
        size_t remaining = total_bytes;
        Uint64 start_time = SDL_GetTicks();
        while (remaining > 0) {
            size_t n = SDL_ReadIO(stream, ptr, remaining);
            if (n > 0) {
                ptr += n;
                remaining -= n;
                start_time = SDL_GetTicks();
            } else {
                SDL_IOStatus status = SDL_GetIOStatus(stream);
                if (status == SDL_IO_STATUS_ERROR || status == SDL_IO_STATUS_EOF) {
                    return false;
                }
                if (SDL_GetTicks() - start_time > static_cast<Uint64>(timeout_ms)) {
                    return false;
                }
                SDL_Delay(2);
            }
        }
        return true;
    }

    bool read_line(SDL_IOStream* stream, std::string& line_out, int timeout_ms) {
        line_out.clear();
        Uint64 start_time = SDL_GetTicks();
        char ch = 0;
        while (true) {
            size_t n = SDL_ReadIO(stream, &ch, 1);
            if (n == 1) {
                if (ch == '\n') {
                    return true;
                }
                if (ch != '\r') {
                    line_out.push_back(ch);
                }
                start_time = SDL_GetTicks();
            } else {
                SDL_IOStatus status = SDL_GetIOStatus(stream);
                if (status == SDL_IO_STATUS_ERROR || status == SDL_IO_STATUS_EOF) {
                    return !line_out.empty();
                }
                if (SDL_GetTicks() - start_time > static_cast<Uint64>(timeout_ms)) {
                    return false;
                }
                SDL_Delay(2);
            }
        }
    }

    bool write_all(SDL_IOStream* stream, const void* data, size_t total_bytes) {
        const uint8_t* ptr = static_cast<const uint8_t*>(data);
        size_t remaining = total_bytes;
        while (remaining > 0) {
            size_t n = SDL_WriteIO(stream, ptr, remaining);
            if (n == 0) {
                return false;
            }
            ptr += n;
            remaining -= n;
        }
        return SDL_FlushIO(stream);
    }

    FileExplorer::Node parse_json_node(const json& j) {
        FileExplorer::Node node;
        node.name = j.value("name", "");
        node.path = j.value("path", "");
        node.is_file = j.value("is_file", false);
        if (j.contains("children") && j["children"].is_array()) {
            for (const auto& c : j["children"]) {
                node.children.push_back(parse_json_node(c));
            }
        }
        return node;
    }
} // namespace

RemoteClient::RemoteClient() = default;

RemoteClient::~RemoteClient() {
    disconnect();
    async_worker_.shutdown();
}

std::string RemoteClient::last_error() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return last_error_;
}

RemoteConfig RemoteClient::config() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return config_;
}

void RemoteClient::connect_async(const RemoteConfig& config) {
    if (state_.load() == ConnectionState::Connecting) {
        return;
    }
    state_.store(ConnectionState::Connecting);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        config_ = config;
        last_error_.clear();
    }
    async_worker_.submit([this, config]() {
        std::string err;
        if (connect_internal(config, err)) {
            state_.store(ConnectionState::Connected);
            just_connected_.store(true);
        } else {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                last_error_ = err;
            }
            state_.store(ConnectionState::Error);
        }
    });
}

bool RemoteClient::connect_sync(const RemoteConfig& config, std::string& error_out) {
    state_.store(ConnectionState::Connecting);
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        config_ = config;
        last_error_.clear();
    }
    if (connect_internal(config, error_out)) {
        state_.store(ConnectionState::Connected);
        return true;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        last_error_ = error_out;
    }
    state_.store(ConnectionState::Error);
    return false;
}

bool RemoteClient::connect_internal(const RemoteConfig& config, std::string& error_out) {
    std::vector<std::string> args_str;
    const bool is_local = config.host.empty() || config.host == "localhost" || config.host == "127.0.0.1";

    if (is_local) {
        args_str.push_back(config.python_bin.empty() ? "python3" : config.python_bin);
        args_str.push_back(config.script_path.empty() ? "scripts/viewer_daemon.py" : config.script_path);
    } else {
        args_str.push_back("ssh");
        args_str.push_back("-T"); // Disable pseudo-tty allocation
        args_str.push_back("-q"); // Quiet mode (suppress warnings/banners)
        args_str.push_back("-o");
        args_str.push_back("BatchMode=yes"); // Never prompt for passwords interactively
        args_str.push_back("-o");
        args_str.push_back("ConnectTimeout=10");
        if (config.port != 22 && config.port > 0) {
            args_str.push_back("-p");
            args_str.push_back(std::to_string(config.port));
        }
        args_str.push_back(config.host);

        std::string remote_cmd = config.python_bin.empty() ? "python3" : config.python_bin;
        if (!config.script_path.empty()) {
            remote_cmd += " " + config.script_path;
        } else {
            remote_cmd +=
                " -c \"import os, sys; candidates = [os.path.expanduser('~/workspace/hand_motion_viewer/scripts/viewer_daemon.py'), "
                "os.path.expanduser('~/.hand_motion_viewer/viewer_daemon.py'), 'viewer_daemon.py', 'scripts/viewer_daemon.py']; target = next((p for p in "
                "candidates if os.path.isfile(p)), None); sys.exit('Daemon script not found. Please specify daemon path in Advanced SSH Settings.') if not "
                "target else None; exec(open(target).read())\"";
        }
        args_str.push_back(remote_cmd);
    }

    std::vector<const char*> args;
    args.reserve(args_str.size() + 1);
    for (const auto& s : args_str) {
        args.push_back(s.c_str());
    }
    args.push_back(nullptr);

    SDL_PropertiesID props = SDL_CreateProperties();
    SDL_SetPointerProperty(props, SDL_PROP_PROCESS_CREATE_ARGS_POINTER, args.data());
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDIN_NUMBER, SDL_PROCESS_STDIO_APP);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDOUT_NUMBER, SDL_PROCESS_STDIO_APP);
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_NULL);
    // Background boolean ensures CREATE_NO_WINDOW on Windows so zero console window is displayed
    SDL_SetBooleanProperty(props, SDL_PROP_PROCESS_CREATE_BACKGROUND_BOOLEAN, true);

    SDL_Process* new_process = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);

    if (new_process == nullptr) {
        error_out = std::string("Failed to spawn process: ") + SDL_GetError();
        return false;
    }

    SDL_IOStream* new_stdin = SDL_GetProcessInput(new_process);
    SDL_IOStream* new_stdout = SDL_GetProcessOutput(new_process);

    if (new_stdin == nullptr || new_stdout == nullptr) {
        error_out = "Failed to obtain process standard I/O streams.";
        SDL_KillProcess(new_process, true);
        SDL_DestroyProcess(new_process);
        return false;
    }

    // Ping daemon to verify connection and protocol without holding mutex_
    json ping_req = {{"id", 1}, {"cmd", "ping"}};
    const std::string line = ping_req.dump() + "\n";
    if (!write_all(new_stdin, line.data(), line.size())) {
        error_out = "Connection failed: unable to send handshake to daemon";
        SDL_KillProcess(new_process, true);
        SDL_DestroyProcess(new_process);
        return false;
    }

    std::string resp_line;
    if (!read_line(new_stdout, resp_line, DEFAULT_TIMEOUT_MS)) {
        error_out = "Connection failed: timed out waiting for daemon response";
        SDL_KillProcess(new_process, true);
        SDL_DestroyProcess(new_process);
        return false;
    }

    try {
        json ping_resp = json::parse(resp_line);
        if (ping_resp.value("status", "") != "ok") {
            error_out = ping_resp.value("message", "Daemon returned error status on handshake");
            SDL_KillProcess(new_process, true);
            SDL_DestroyProcess(new_process);
            return false;
        }
    } catch (const json::exception& e) {
        error_out = std::string("Malformed JSON during handshake: ") + e.what();
        SDL_KillProcess(new_process, true);
        SDL_DestroyProcess(new_process);
        return false;
    }

    // Abort if canceled while connecting
    if (state_.load() != ConnectionState::Connecting) {
        error_out = "Connection canceled";
        SDL_KillProcess(new_process, true);
        SDL_DestroyProcess(new_process);
        return false;
    }

    // Swap in active connection under lock
    {
        std::lock_guard<std::mutex> guard(mutex_);
        disconnect_locked();
        process_ = new_process;
        stdin_ = new_stdin;
        stdout_ = new_stdout;
        next_id_ = 2;
    }

    return true;
}

void RemoteClient::disconnect() {
    state_.store(ConnectionState::Disconnected);
    std::lock_guard<std::mutex> guard(mutex_);
    disconnect_locked();
}

void RemoteClient::disconnect_locked() {
    if (process_ != nullptr) {
        int exitcode = 0;
        bool already_exited = SDL_WaitProcess(process_, false, &exitcode);
        if (!already_exited && stdin_ != nullptr) {
            json quit_req = {{"id", next_id_++}, {"cmd", "quit"}};
            std::string line = quit_req.dump() + "\n";
            write_all(stdin_, line.data(), line.size());
        }
        if (!already_exited) {
            SDL_KillProcess(process_, false);
            Uint64 start = SDL_GetTicks();
            while (!SDL_WaitProcess(process_, false, &exitcode) && (SDL_GetTicks() - start < 100)) {
                SDL_Delay(5);
            }
            if (!SDL_WaitProcess(process_, false, &exitcode)) {
                SDL_KillProcess(process_, true);
            }
        }
        SDL_DestroyProcess(process_);
        process_ = nullptr;
    }
    stdin_ = nullptr;
    stdout_ = nullptr;
}

void RemoteClient::poll() {
    if (state_.load() != ConnectionState::Connected) {
        return;
    }
    std::unique_lock<std::mutex> guard(mutex_, std::try_to_lock);
    if (!guard.owns_lock()) {
        return;
    }
    if (process_ != nullptr) {
        int exitcode = 0;
        if (SDL_WaitProcess(process_, false, &exitcode)) {
            disconnect_locked();
            {
                std::lock_guard<std::mutex> state_guard(state_mutex_);
                last_error_ = "Remote server connection closed unexpectedly.";
            }
            state_.store(ConnectionState::Disconnected);
        }
    }
}

bool RemoteClient::consume_just_connected() { return just_connected_.exchange(false); }

bool RemoteClient::ping(std::string& error_out) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }
    json req = {{"id", next_id_++}, {"cmd", "ping"}};
    json resp;
    return send_command_locked(req, resp, error_out);
}

bool RemoteClient::scan_tree(const std::string& remote_root, FileExplorer::Node& out_node, std::string& error_out) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {{"id", next_id_++}, {"cmd", "scan_tree"}, {"root", remote_root}};
    json resp;
    if (!send_command_locked(req, resp, error_out)) {
        return false;
    }

    if (resp.value("status", "") != "ok") {
        error_out = resp.value("message", "Scan failed on remote server");
        return false;
    }

    if (!resp.contains("tree") || !resp["tree"].is_object()) {
        error_out = "Invalid tree response from remote server";
        return false;
    }

    out_node = parse_json_node(resp["tree"]);
    out_node.default_open = true;
    return true;
}

bool RemoteClient::fetch_file(const std::string& remote_path, const std::string& local_dest_path, std::string& error_out) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {{"id", next_id_++}, {"cmd", "get_file"}, {"path", remote_path}};
    json hdr;
    std::vector<uint8_t> data;
    if (!send_command_binary_locked(req, hdr, data, error_out)) {
        return false;
    }

    if (hdr.value("status", "") != "ok") {
        error_out = hdr.value("message", "Failed to fetch remote file");
        return false;
    }

    std::error_code ec;
    fs::path local_p(local_dest_path);
    fs::create_directories(local_p.parent_path(), ec);

    std::ofstream out(local_dest_path, std::ios::binary);
    if (!out) {
        error_out = "Failed to open local destination for writing: " + local_dest_path;
        return false;
    }

    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        if (!out) {
            error_out = "Failed writing to local file: " + local_dest_path;
            return false;
        }
    }

    return true;
}

bool RemoteClient::fetch_single_frame(const std::string& remote_export_path, int frame_number, const std::string& local_file_dest, std::string& error_out) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {
        {"id", next_id_++},
        {"cmd", "get_frame"},
        {"path", remote_export_path},
        {"export_path", remote_export_path},
        {"frame", frame_number},
        {"frame_number", frame_number}
    };
    json hdr;
    std::vector<uint8_t> data;
    if (!send_command_binary_locked(req, hdr, data, error_out)) {
        return false;
    }

    if (hdr.value("status", "") != "ok") {
        error_out = hdr.value("message", "Failed to fetch remote frame");
        return false;
    }

    std::error_code ec;
    fs::path local_p(local_file_dest);
    fs::create_directories(local_p.parent_path(), ec);

    std::ofstream out(local_file_dest, std::ios::binary);
    if (!out) {
        error_out = "Failed to open local frame destination: " + local_file_dest;
        return false;
    }

    if (!data.empty()) {
        out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    }

    const std::string original_filename = hdr.value("filename", "");
    if (!original_filename.empty() && original_filename != local_p.filename().string() && !data.empty()) {
        const fs::path alt_path = local_p.parent_path() / original_filename;
        std::ofstream alt_out(alt_path, std::ios::binary);
        if (alt_out) {
            alt_out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
        }
    }

    return true;
}

bool RemoteClient::fetch_and_extract_bundle(
    const std::string& remote_export_path, const std::string& local_frames_dir, int start_frame, int count, int& frame_count_out, std::string& error_out
) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {
        {"id", next_id_++},
        {"cmd", "bundle_frames"},
        {"path", remote_export_path},
        {"export_path", remote_export_path},
    };
    if (start_frame > 0) {
        req["start_frame"] = start_frame;
    }
    if (count > 0) {
        req["count"] = count;
    }

    json hdr;
    std::vector<uint8_t> zip_data;
    if (!send_command_binary_locked(req, hdr, zip_data, error_out)) {
        return false;
    }

    if (hdr.value("status", "") != "ok") {
        error_out = hdr.value("message", "Bundle generation failed on remote server");
        return false;
    }

    frame_count_out = hdr.value("frame_count", 0);
    if (zip_data.empty()) {
        return true;
    }

    // Extract ZIP archive in-memory using miniz
    mz_zip_archive zip_archive;
    std::memset(&zip_archive, 0, sizeof(zip_archive));

    if (!mz_zip_reader_init_mem(&zip_archive, zip_data.data(), zip_data.size(), 0)) {
        error_out = "Failed to initialize ZIP reader for extracted bundle";
        return false;
    }

    std::error_code ec;
    fs::create_directories(local_frames_dir, ec);

    mz_uint num_files = mz_zip_reader_get_num_files(&zip_archive);
    for (mz_uint i = 0; i < num_files; ++i) {
        mz_zip_archive_file_stat file_stat;
        if (!mz_zip_reader_file_stat(&zip_archive, i, &file_stat)) {
            continue;
        }

        if (!mz_zip_reader_is_file_a_directory(&zip_archive, i)) {
            fs::path out_file = fs::path(local_frames_dir) / file_stat.m_filename;
            mz_zip_reader_extract_to_file(&zip_archive, i, out_file.string().c_str(), 0);
        }
    }
    mz_zip_reader_end(&zip_archive);
    return true;
}

bool RemoteClient::fetch_and_extract_bundle(
    const std::string& remote_export_path, const std::string& local_frames_dir, int& frame_count_out, std::string& error_out
) {
    return fetch_and_extract_bundle(remote_export_path, local_frames_dir, 1, 0, frame_count_out, error_out);
}

bool RemoteClient::send_command_locked(const json& req, json& resp_out, std::string& error_out) {
    if (stdin_ == nullptr || stdout_ == nullptr) {
        error_out = "Streams not available";
        return false;
    }

    const std::string line = req.dump() + "\n";
    if (!write_all(stdin_, line.data(), line.size())) {
        error_out = "Failed to send request to remote daemon (connection broken)";
        disconnect_locked();
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            last_error_ = error_out;
        }
        state_.store(ConnectionState::Disconnected);
        return false;
    }

    std::string resp_line;
    if (!read_line(stdout_, resp_line, DEFAULT_TIMEOUT_MS)) {
        error_out = "Timed out or connection dropped waiting for remote response";
        disconnect_locked();
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            last_error_ = error_out;
        }
        state_.store(ConnectionState::Disconnected);
        return false;
    }

    try {
        resp_out = json::parse(resp_line);
    } catch (const json::exception& e) {
        error_out = std::string("Malformed JSON from remote server: ") + e.what();
        return false;
    }

    return true;
}

bool RemoteClient::send_command_binary_locked(const json& req, json& resp_hdr_out, std::vector<uint8_t>& data_out, std::string& error_out) {
    if (stdin_ == nullptr || stdout_ == nullptr) {
        error_out = "Streams not available";
        return false;
    }

    const std::string line = req.dump() + "\n";
    if (!write_all(stdin_, line.data(), line.size())) {
        error_out = "Failed to send request to remote daemon (connection broken)";
        disconnect_locked();
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            last_error_ = error_out;
        }
        state_.store(ConnectionState::Disconnected);
        return false;
    }

    std::string hdr_line;
    if (!read_line(stdout_, hdr_line, DEFAULT_TIMEOUT_MS)) {
        error_out = "Timed out waiting for binary header from remote server";
        disconnect_locked();
        {
            std::lock_guard<std::mutex> state_guard(state_mutex_);
            last_error_ = error_out;
        }
        state_.store(ConnectionState::Disconnected);
        return false;
    }

    try {
        resp_hdr_out = json::parse(hdr_line);
    } catch (const json::exception& e) {
        error_out = std::string("Malformed header JSON: ") + e.what();
        return false;
    }

    if (resp_hdr_out.value("status", "") != "ok") {
        return true;
    }

    const size_t byte_size = resp_hdr_out.value("size", static_cast<size_t>(0));
    data_out.resize(byte_size);

    if (byte_size > 0) {
        if (!read_exact(stdout_, data_out.data(), byte_size, BUNDLE_TIMEOUT_MS)) {
            error_out = "Timed out or dropped connection reading binary payload";
            disconnect_locked();
            {
                std::lock_guard<std::mutex> state_guard(state_mutex_);
                last_error_ = error_out;
            }
            state_.store(ConnectionState::Disconnected);
            return false;
        }
    }

    return true;
}

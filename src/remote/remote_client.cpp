#include "remote/remote_client.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <utility>

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

RemoteClient::~RemoteClient() { disconnect(); }

bool RemoteClient::connect(const RemoteConfig& config, std::string& error_out) {
    disconnect();

    std::lock_guard<std::mutex> guard(mutex_);
    config_ = config;
    state_.store(ConnectionState::Connecting);

    std::vector<std::string> args_str;
    const bool is_local = config.host.empty() || config.host == "localhost" || config.host == "127.0.0.1";

    if (is_local) {
        args_str.push_back(config.python_bin.empty() ? "python3" : config.python_bin);
        args_str.push_back(config.script_path);
    } else {
        args_str.push_back("ssh");
        args_str.push_back("-T");
        if (config.port != 22 && config.port > 0) {
            args_str.push_back("-p");
            args_str.push_back(std::to_string(config.port));
        }
        args_str.push_back(config.host);

        std::string remote_cmd = config.python_bin.empty() ? "python3" : config.python_bin;
        remote_cmd += " " + config.script_path;
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
    SDL_SetNumberProperty(props, SDL_PROP_PROCESS_CREATE_STDERR_NUMBER, SDL_PROCESS_STDIO_INHERITED);

    process_ = SDL_CreateProcessWithProperties(props);
    SDL_DestroyProperties(props);

    if (process_ == nullptr) {
        error_out = std::string("Failed to spawn process: ") + SDL_GetError();
        last_error_ = error_out;
        state_.store(ConnectionState::Error);
        return false;
    }

    stdin_ = SDL_GetProcessInput(process_);
    stdout_ = SDL_GetProcessOutput(process_);

    if (stdin_ == nullptr || stdout_ == nullptr) {
        error_out = "Failed to obtain process standard I/O streams.";
        last_error_ = error_out;
        disconnect();
        state_.store(ConnectionState::Error);
        return false;
    }

    // Ping daemon to verify connection and protocol
    json ping_req = {{"id", next_id_++}, {"cmd", "ping"}};
    json ping_resp;
    if (!send_command_locked(ping_req, ping_resp, error_out)) {
        last_error_ = "Connection failed during handshake: " + error_out;
        error_out = last_error_;
        disconnect();
        state_.store(ConnectionState::Error);
        return false;
    }

    state_.store(ConnectionState::Connected);
    last_error_.clear();
    return true;
}

void RemoteClient::disconnect() {
    std::lock_guard<std::mutex> guard(mutex_);
    if (process_ != nullptr) {
        if (stdin_ != nullptr) {
            json quit_req = {{"id", next_id_++}, {"cmd", "quit"}};
            std::string line = quit_req.dump() + "\n";
            write_all(stdin_, line.data(), line.size());
        }
        SDL_DestroyProcess(process_);
        process_ = nullptr;
    }
    stdin_ = nullptr;
    stdout_ = nullptr;
    state_.store(ConnectionState::Disconnected);
}

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
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

bool RemoteClient::fetch_single_frame(const std::string& remote_export_path, int frame_number, const std::string& local_file_dest, std::string& error_out) {
    std::lock_guard<std::mutex> guard(mutex_);
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {{"id", next_id_++}, {"cmd", "get_frame"}, {"export_path", remote_export_path}, {"frame_number", frame_number}};
    json hdr;
    std::vector<uint8_t> data;
    if (!send_command_binary_locked(req, hdr, data, error_out)) {
        return false;
    }

    if (hdr.value("status", "") != "ok") {
        error_out = hdr.value("message", "Frame not found");
        return false;
    }

    std::error_code ec;
    fs::path local_p(local_file_dest);
    fs::create_directories(local_p.parent_path(), ec);

    std::ofstream out(local_file_dest, std::ios::binary);
    if (!out) {
        error_out = "Failed to open local destination for writing frame: " + local_file_dest;
        return false;
    }
    out.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
    return true;
}

bool RemoteClient::fetch_and_extract_bundle(
    const std::string& remote_export_path, const std::string& local_frames_dir, int& frame_count_out, std::string& error_out
) {
    std::lock_guard<std::mutex> guard(mutex_);
    frame_count_out = 0;
    if (!is_connected()) {
        error_out = "Not connected";
        return false;
    }

    json req = {{"id", next_id_++}, {"cmd", "bundle_frames"}, {"export_path", remote_export_path}};
    json hdr;
    std::vector<uint8_t> data;
    if (!send_command_binary_locked(req, hdr, data, error_out)) {
        return false;
    }

    if (hdr.value("status", "") != "ok") {
        error_out = hdr.value("message", "Failed to bundle frames on remote server");
        return false;
    }

    frame_count_out = hdr.value("frame_count", 0);
    if (data.empty() || frame_count_out == 0) {
        return true; // no frames on server
    }

    std::error_code ec;
    fs::create_directories(local_frames_dir, ec);

    // Unzip in-memory archive to local_frames_dir
    mz_zip_archive zip_archive;
    std::memset(&zip_archive, 0, sizeof(zip_archive));
    if (!mz_zip_reader_init_mem(&zip_archive, data.data(), data.size(), 0)) {
        error_out = "Failed to parse ZIP frame bundle from server";
        return false;
    }

    const mz_uint num_files = mz_zip_reader_get_num_files(&zip_archive);
    for (mz_uint i = 0; i < num_files; ++i) {
        mz_zip_archive_file_stat file_stat;
        if (mz_zip_reader_file_stat(&zip_archive, i, &file_stat)) {
            fs::path out_file = fs::path(local_frames_dir) / file_stat.m_filename;
            mz_zip_reader_extract_to_file(&zip_archive, i, out_file.string().c_str(), 0);
        }
    }
    mz_zip_reader_end(&zip_archive);
    return true;
}

bool RemoteClient::send_command_locked(const json& req, json& resp_out, std::string& error_out) {
    if (stdin_ == nullptr || stdout_ == nullptr) {
        error_out = "Streams not available";
        return false;
    }

    const std::string line = req.dump() + "\n";
    if (!write_all(stdin_, line.data(), line.size())) {
        error_out = "Failed to send request to remote daemon";
        disconnect();
        return false;
    }

    std::string resp_line;
    if (!read_line(stdout_, resp_line, DEFAULT_TIMEOUT_MS)) {
        error_out = "Timed out or connection dropped waiting for remote response";
        disconnect();
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
        error_out = "Failed to send request to remote daemon";
        disconnect();
        return false;
    }

    std::string hdr_line;
    if (!read_line(stdout_, hdr_line, DEFAULT_TIMEOUT_MS)) {
        error_out = "Timed out waiting for binary header from remote server";
        disconnect();
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
            disconnect();
            return false;
        }
    }

    return true;
}

#include "config.h"

#include <SDL.h>

#include <fstream>
#include <nlohmann/json.hpp>

using nlohmann::json;

namespace {
    /// Assign ``out`` from ``data[key]`` only when the key exists and has the right
    /// type, so old/new config files and missing keys keep their defaults.
    template <typename T>
    void read_field(const json& data, const char* key, T& out) {
        auto found = data.find(key);
        if (found != data.end() && !found->is_null()) {
            try {
                out = found->get<T>();
            } catch (const json::exception&) {
                // Wrong type for this key — leave the default in place.
            }
        }
    }

    /// Optional<string> variant: a present, non-null string sets the value; an
    /// explicit null clears it.
    void read_optional_string(const json& data, const char* key, std::optional<std::string>& out) {
        auto found = data.find(key);
        if (found == data.end()) {
            return;
        }
        if (found->is_null()) {
            out = std::nullopt;
        } else if (found->is_string()) {
            out = found->get<std::string>();
        }
    }

    void read_optional_int(const json& data, const char* key, std::optional<int>& out) {
        auto found = data.find(key);
        if (found == data.end()) {
            return;
        }
        if (found->is_null()) {
            out = std::nullopt;
        } else if (found->is_number_integer()) {
            out = found->get<int>();
        }
    }
} // namespace

Config load_config(const std::string& path) {
    Config config;
    std::ifstream config_file(path);
    if (!config_file) {
        return config;
    }
    json data;
    try {
        config_file >> data;
    } catch (const json::exception&) {
        return config;
    }
    if (!data.is_object()) {
        return config;
    }

    read_optional_string(data, "last_file", config.last_file);
    read_optional_string(data, "last_folder", config.last_folder);
    read_field(data, "last_frame", config.last_frame);
    read_field(data, "hand_translucent", config.hand_translucent);
    read_field(data, "show_overlay_hands", config.show_overlay_hands);
    read_field(data, "show_controls", config.show_controls);
    read_field(data, "show_camera_marker", config.show_camera_marker);
    read_field(data, "free_camera", config.free_camera);

    read_optional_int(data, "window_x", config.window_x);
    read_optional_int(data, "window_y", config.window_y);
    read_field(data, "window_width", config.window_width);
    read_field(data, "window_height", config.window_height);
    read_field(data, "window_fullscreen", config.window_fullscreen);
    read_field(data, "window_display", config.window_display);

    read_field(data, "camera_azimuth", config.camera_azimuth);
    read_field(data, "camera_elevation", config.camera_elevation);
    read_field(data, "camera_distance", config.camera_distance);
    read_field(data, "camera_target", config.camera_target);
    return config;
}

void save_config(const std::string& path, const Config& config) {
    json data;
    data["last_file"] = config.last_file ? json(*config.last_file) : json(nullptr);
    data["last_folder"] = config.last_folder ? json(*config.last_folder) : json(nullptr);
    data["last_frame"] = config.last_frame;
    data["hand_translucent"] = config.hand_translucent;
    data["show_overlay_hands"] = config.show_overlay_hands;
    data["show_controls"] = config.show_controls;
    data["show_camera_marker"] = config.show_camera_marker;
    data["free_camera"] = config.free_camera;

    data["window_x"] = config.window_x ? json(*config.window_x) : json(nullptr);
    data["window_y"] = config.window_y ? json(*config.window_y) : json(nullptr);
    data["window_width"] = config.window_width;
    data["window_height"] = config.window_height;
    data["window_fullscreen"] = config.window_fullscreen;
    data["window_display"] = config.window_display;

    data["camera_azimuth"] = config.camera_azimuth;
    data["camera_elevation"] = config.camera_elevation;
    data["camera_distance"] = config.camera_distance;
    data["camera_target"] = config.camera_target;

    std::ofstream config_file(path);
    if (config_file) {
        config_file << data.dump(4);
    }
}

std::string default_config_path() {
    std::string result = "config.json";
    char* base = SDL_GetBasePath();
    if (base != nullptr) {
        result = std::string(base) + "config.json";
        SDL_free(base);
    }
    return result;
}

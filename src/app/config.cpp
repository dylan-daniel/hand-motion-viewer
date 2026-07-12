#include "app/config.h"

#include <SDL3/SDL.h>

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

    read_optional_string(data, "last_folder", config.last_folder);
    read_optional_string(data, "images_folder", config.images_folder);
    read_field(data, "last_frame", config.last_frame);
    read_field(data, "playback_speed", config.playback_speed);
    read_field(data, "hand_translucent", config.hand_translucent);
    read_field(data, "show_controls", config.show_controls);
    read_field(data, "show_camera_marker", config.show_camera_marker);
    read_field(data, "free_camera", config.free_camera);
    read_field(data, "active_pane", config.active_pane);
    read_field(data, "hide_duplicates", config.hide_duplicates);
    read_field(data, "hide_adults", config.hide_adults);

    read_optional_string(data, "data_folder", config.data_folder);
    read_field(data, "expanded_folders", config.expanded_folders);

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

    read_field(data, "intrinsics_fx", config.intrinsics_fx);
    read_field(data, "intrinsics_fy", config.intrinsics_fy);
    read_field(data, "intrinsics_cx", config.intrinsics_cx);
    read_field(data, "intrinsics_cy", config.intrinsics_cy);
    read_field(data, "intrinsics_k_metric", config.intrinsics_k_metric);
    read_field(data, "hand_depth_source", config.hand_depth_source);
    read_field(data, "smoothing_enabled", config.smoothing_enabled);

    read_optional_string(data, "sam3_folder", config.sam3_folder);
    read_optional_string(data, "da3_folder", config.da3_folder);
    read_field(data, "object_label", config.object_label);
    read_field(data, "object_shape", config.object_shape);
    read_field(data, "object_size_m", config.object_size_m);

    read_optional_string(data, "tracking_folder", config.tracking_folder);
    read_optional_string(data, "baby_hand_idx_folder", config.baby_hand_idx_folder);
    read_optional_string(data, "k_metric_csv", config.k_metric_csv);
    read_optional_string(data, "production_csv", config.production_csv);
    return config;
}

void save_config(const std::string& path, const Config& config) {
    json data;
    data["last_folder"] = config.last_folder ? json(*config.last_folder) : json(nullptr);
    data["images_folder"] = config.images_folder ? json(*config.images_folder) : json(nullptr);
    data["last_frame"] = config.last_frame;
    data["playback_speed"] = config.playback_speed;
    data["hand_translucent"] = config.hand_translucent;
    data["show_controls"] = config.show_controls;
    data["show_camera_marker"] = config.show_camera_marker;
    data["free_camera"] = config.free_camera;
    data["active_pane"] = config.active_pane;
    data["hide_duplicates"] = config.hide_duplicates;
    data["hide_adults"] = config.hide_adults;

    data["data_folder"] = config.data_folder ? json(*config.data_folder) : json(nullptr);
    data["expanded_folders"] = config.expanded_folders;

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

    data["intrinsics_fx"] = config.intrinsics_fx;
    data["intrinsics_fy"] = config.intrinsics_fy;
    data["intrinsics_cx"] = config.intrinsics_cx;
    data["intrinsics_cy"] = config.intrinsics_cy;
    data["intrinsics_k_metric"] = config.intrinsics_k_metric;
    data["hand_depth_source"] = config.hand_depth_source;
    data["smoothing_enabled"] = config.smoothing_enabled;

    data["sam3_folder"] = config.sam3_folder ? json(*config.sam3_folder) : json(nullptr);
    data["da3_folder"] = config.da3_folder ? json(*config.da3_folder) : json(nullptr);
    data["object_label"] = config.object_label;
    data["object_shape"] = config.object_shape;
    data["object_size_m"] = config.object_size_m;

    data["tracking_folder"] = config.tracking_folder ? json(*config.tracking_folder) : json(nullptr);
    data["baby_hand_idx_folder"] = config.baby_hand_idx_folder ? json(*config.baby_hand_idx_folder) : json(nullptr);
    data["k_metric_csv"] = config.k_metric_csv ? json(*config.k_metric_csv) : json(nullptr);
    data["production_csv"] = config.production_csv ? json(*config.production_csv) : json(nullptr);

    std::ofstream config_file(path);
    if (config_file) {
        config_file << data.dump(4);
    }
}

std::string default_config_path() {
    std::string result = "config.json";
    const char* base = SDL_GetBasePath();
    if (base != nullptr) {
        result = std::string(base) + "config.json";
    }
    return result;
}

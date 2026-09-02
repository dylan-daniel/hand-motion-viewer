#include "data/hand_export.h"

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>

#include <miniz.h>

namespace {
    constexpr char MAGIC[4] = {'H', 'E', 'X', 'P'};
    constexpr std::uint32_t VERSION = 1;
    constexpr std::size_t STRING_FIELD_WIDTH = 32;
    constexpr std::size_t HEADER_SIZE = 16; // magic(4) + version(4) + payload_size(4) + compressed_size(4)

    enum DType : std::uint8_t { kFloat32 = 0, kInt32 = 1, kInt8 = 2, kString = 3 };

    struct ColumnInfo {
        DType dtype;
        std::size_t offset; // byte offset into the decompressed payload where this column's data starts
    };

    /// Axis-angle -> row-major flattened 3x3 rotation matrix (Rodrigues'
    /// formula). Exact away from the trivial zero-rotation case (handled
    /// below by falling back to identity); matches what the Python side
    /// (scipy.spatial.transform.Rotation.from_rotvec) produces to within
    /// float round-off.
    std::array<float, 9> axis_angle_to_matrix(float x, float y, float z) {
        const float theta = std::sqrt(x * x + y * y + z * z);
        if (theta < 1e-12f) {
            return {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f};
        }
        const float kx = x / theta, ky = y / theta, kz = z / theta;
        const float s = std::sin(theta), c = std::cos(theta), one_c = 1.0f - c;

        // R = I + sin(theta) K + (1 - cos(theta)) K^2, K the skew-symmetric
        // cross-product matrix of the unit axis (kx, ky, kz).
        return {
            c + kx * kx * one_c,
            kx * ky * one_c - kz * s,
            kx * kz * one_c + ky * s,
            ky * kx * one_c + kz * s,
            c + ky * ky * one_c,
            ky * kz * one_c - kx * s,
            kz * kx * one_c - ky * s,
            kz * ky * one_c + kx * s,
            c + kz * kz * one_c,
        };
    }

    std::vector<char> read_file(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            throw std::runtime_error("failed to open hand export file: " + path);
        }
        const std::streamsize size = file.tellg();
        file.seekg(0);
        std::vector<char> data(static_cast<std::size_t>(size));
        if (size > 0 && !file.read(data.data(), size)) {
            throw std::runtime_error("failed to read hand export file: " + path);
        }
        return data;
    }

    std::uint32_t read_u32(const char* data) {
        std::uint32_t value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }

    std::size_t item_size(DType dtype) {
        switch (dtype) {
            case kFloat32:
            case kInt32:
                return 4;
            case kInt8:
                return 1;
            case kString:
                return STRING_FIELD_WIDTH;
        }
        throw std::runtime_error("unreachable dtype tag");
    }
} // namespace

std::vector<HandExportRow> load_hand_export(const std::string& export_path) {
    const std::vector<char> raw = read_file(export_path);
    if (raw.size() < HEADER_SIZE || std::memcmp(raw.data(), MAGIC, 4) != 0) {
        throw std::runtime_error("not a hand export binary file (bad magic): " + export_path);
    }

    const std::uint32_t version = read_u32(raw.data() + 4);
    const std::uint32_t payload_size = read_u32(raw.data() + 8);
    const std::uint32_t compressed_size = read_u32(raw.data() + 12);
    if (version != VERSION) {
        throw std::runtime_error(
            "unsupported hand export binary version " + std::to_string(version) + " (expected " + std::to_string(VERSION) + "): " + export_path
        );
    }
    if (raw.size() < HEADER_SIZE + compressed_size) {
        throw std::runtime_error("truncated hand export file: " + export_path);
    }

    std::vector<unsigned char> payload(payload_size);
    mz_ulong actual_payload_size = payload_size;
    const int rc = mz_uncompress(payload.data(), &actual_payload_size, reinterpret_cast<const unsigned char*>(raw.data() + HEADER_SIZE), compressed_size);
    if (rc != MZ_OK || actual_payload_size != payload_size) {
        throw std::runtime_error("failed to decompress hand export payload (miniz error " + std::to_string(rc) + "): " + export_path);
    }

    const char* p = reinterpret_cast<const char*>(payload.data());
    std::size_t offset = 0;
    const std::uint32_t row_count = read_u32(p + offset);
    offset += 4;
    const std::uint32_t column_count = read_u32(p + offset);
    offset += 4;

    struct PendingColumn {
        std::string name;
        DType dtype;
    };
    std::vector<PendingColumn> ordered;
    ordered.reserve(column_count);
    for (std::uint32_t i = 0; i < column_count; ++i) {
        const auto name_len = static_cast<std::uint8_t>(p[offset]);
        offset += 1;
        const auto dtype = static_cast<DType>(static_cast<std::uint8_t>(p[offset]));
        offset += 1;
        std::string name(p + offset, name_len);
        offset += name_len;
        ordered.push_back({std::move(name), dtype});
    }

    std::map<std::string, ColumnInfo> columns;
    for (const auto& col : ordered) {
        columns[col.name] = ColumnInfo{col.dtype, offset};
        offset += item_size(col.dtype) * row_count;
    }

    auto require = [&](const std::string& name) -> const ColumnInfo& {
        const auto it = columns.find(name);
        if (it == columns.end()) {
            throw std::runtime_error("hand export file is missing expected column '" + name + "': " + export_path);
        }
        return it->second;
    };
    auto get_float = [&](const ColumnInfo& col, std::uint32_t row) -> float {
        float value;
        std::memcpy(&value, p + col.offset + static_cast<std::size_t>(row) * 4, 4);
        return value;
    };
    auto get_int32 = [&](const ColumnInfo& col, std::uint32_t row) -> std::int32_t {
        std::int32_t value;
        std::memcpy(&value, p + col.offset + static_cast<std::size_t>(row) * 4, 4);
        return value;
    };
    auto get_int8 = [&](const ColumnInfo& col, std::uint32_t row) -> std::int8_t { return static_cast<std::int8_t>(p[col.offset + row]); };
    auto get_string = [&](const ColumnInfo& col, std::uint32_t row) -> std::string {
        const char* s = p + col.offset + static_cast<std::size_t>(row) * STRING_FIELD_WIDTH;
        std::size_t len = 0;
        while (len < STRING_FIELD_WIDTH && s[len] != '\0') {
            ++len;
        }
        return std::string(s, len);
    };

    const ColumnInfo& subject_col = require("subject");
    const ColumnInfo& trial_col = require("trial");
    const ColumnInfo& frame_col = require("frame");
    const ColumnInfo& is_right_col = require("is_right");
    const ColumnInfo& hand_track_id_col = require("hand_track_id");
    const ColumnInfo& label_col = require("label");
    const ColumnInfo& scaled_focal_length_col = require("scaled_focal_length");
    const ColumnInfo& img_w_col = require("img_w");
    const ColumnInfo& img_h_col = require("img_h");
    const ColumnInfo& cam_t_x_col = require("cam_t_x");
    const ColumnInfo& cam_t_y_col = require("cam_t_y");
    const ColumnInfo& cam_t_z_col = require("cam_t_z");

    std::array<const ColumnInfo*, 10> beta_cols{};
    for (int i = 0; i < 10; ++i) {
        beta_cols[static_cast<std::size_t>(i)] = &require("beta_" + std::to_string(i));
    }
    std::array<const ColumnInfo*, 3> gorient_rotvec_cols{};
    for (int i = 0; i < 3; ++i) {
        gorient_rotvec_cols[static_cast<std::size_t>(i)] = &require("gorient_rotvec_" + std::to_string(i));
    }
    std::array<const ColumnInfo*, 45> pose_rotvec_cols{};
    for (int i = 0; i < 45; ++i) {
        pose_rotvec_cols[static_cast<std::size_t>(i)] = &require("pose_rotvec_" + std::to_string(i));
    }

    std::vector<HandExportRow> rows;
    rows.reserve(row_count);
    for (std::uint32_t r = 0; r < row_count; ++r) {
        HandExportRow row;
        row.subject = get_string(subject_col, r);
        row.trial = get_string(trial_col, r);
        row.frame = get_int32(frame_col, r);
        row.is_right = get_int8(is_right_col, r);
        row.hand_track_id = get_int32(hand_track_id_col, r);
        row.label = get_string(label_col, r);

        for (int i = 0; i < 10; ++i) {
            row.params.betas[static_cast<std::size_t>(i)] = get_float(*beta_cols[static_cast<std::size_t>(i)], r);
        }
        row.params.global_orient =
            axis_angle_to_matrix(get_float(*gorient_rotvec_cols[0], r), get_float(*gorient_rotvec_cols[1], r), get_float(*gorient_rotvec_cols[2], r));
        for (int j = 0; j < 15; ++j) {
            const auto base = static_cast<std::size_t>(j * 3);
            const auto mat = axis_angle_to_matrix(
                get_float(*pose_rotvec_cols[base], r), get_float(*pose_rotvec_cols[base + 1], r), get_float(*pose_rotvec_cols[base + 2], r)
            );
            const auto matrix_base = static_cast<std::size_t>(j * 9);
            for (int k = 0; k < 9; ++k) {
                row.params.hand_pose[matrix_base + static_cast<std::size_t>(k)] = mat[static_cast<std::size_t>(k)];
            }
        }
        row.params.is_right = row.is_right != 0;
        row.params.cam_t = glm::vec3(get_float(cam_t_x_col, r), get_float(cam_t_y_col, r), get_float(cam_t_z_col, r));

        row.scaled_focal_length = get_float(scaled_focal_length_col, r);
        row.img_w = get_int32(img_w_col, r);
        row.img_h = get_int32(img_h_col, r);

        rows.push_back(std::move(row));
    }

    return rows;
}

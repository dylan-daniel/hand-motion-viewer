#include "data/hand_export.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

namespace {
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

    std::shared_ptr<arrow::ChunkedArray> require_column(const std::shared_ptr<arrow::Table>& table, const std::string& name) {
        auto column = table->GetColumnByName(name);
        if (!column) {
            throw std::runtime_error("hand export Parquet file is missing expected column: " + name);
        }
        return column;
    }

    template <typename ArrowArrayType, typename ValueType>
    std::vector<ValueType> read_numeric_column(const std::shared_ptr<arrow::Table>& table, const std::string& name) {
        auto column = require_column(table, name);
        std::vector<ValueType> values;
        values.reserve(static_cast<std::size_t>(column->length()));
        for (const auto& chunk : column->chunks()) {
            auto typed = std::static_pointer_cast<ArrowArrayType>(chunk);
            for (std::int64_t i = 0; i < typed->length(); ++i) {
                values.push_back(static_cast<ValueType>(typed->Value(i)));
            }
        }
        return values;
    }

    std::vector<std::string> read_string_column(const std::shared_ptr<arrow::Table>& table, const std::string& name) {
        auto column = require_column(table, name);
        std::vector<std::string> values;
        values.reserve(static_cast<std::size_t>(column->length()));
        for (const auto& chunk : column->chunks()) {
            auto typed = std::static_pointer_cast<arrow::StringArray>(chunk);
            for (std::int64_t i = 0; i < typed->length(); ++i) {
                values.emplace_back(typed->GetString(i));
            }
        }
        return values;
    }
} // namespace

std::vector<HandExportRow> load_hand_export(const std::string& parquet_path) {
    auto maybe_infile = arrow::io::ReadableFile::Open(parquet_path);
    if (!maybe_infile.ok()) {
        throw std::runtime_error("failed to open hand export file '" + parquet_path + "': " + maybe_infile.status().ToString());
    }
    std::shared_ptr<arrow::io::RandomAccessFile> infile = *maybe_infile;

    std::unique_ptr<parquet::arrow::FileReader> reader;
    auto open_status = parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader);
    if (!open_status.ok()) {
        throw std::runtime_error("failed to open Parquet reader for '" + parquet_path + "': " + open_status.ToString());
    }

    std::shared_ptr<arrow::Table> table;
    auto read_status = reader->ReadTable(&table);
    if (!read_status.ok()) {
        throw std::runtime_error("failed to read table from '" + parquet_path + "': " + read_status.ToString());
    }

    const auto subject = read_string_column(table, "subject");
    const auto trial = read_string_column(table, "trial");
    const auto frame = read_numeric_column<arrow::Int32Array, std::int32_t>(table, "frame");
    const auto is_right = read_numeric_column<arrow::Int8Array, std::int8_t>(table, "is_right");
    const auto hand_track_id = read_numeric_column<arrow::Int32Array, std::int32_t>(table, "hand_track_id");
    const auto label = read_string_column(table, "label");

    std::vector<std::vector<float>> beta(10), pose_rotvec(45);
    std::vector<float> gorient_rotvec[3];
    for (int i = 0; i < 10; ++i) {
        beta[static_cast<std::size_t>(i)] = read_numeric_column<arrow::FloatArray, float>(table, "beta_" + std::to_string(i));
    }
    for (int i = 0; i < 3; ++i) {
        gorient_rotvec[i] = read_numeric_column<arrow::FloatArray, float>(table, "gorient_rotvec_" + std::to_string(i));
    }
    for (int i = 0; i < 45; ++i) {
        pose_rotvec[static_cast<std::size_t>(i)] = read_numeric_column<arrow::FloatArray, float>(table, "pose_rotvec_" + std::to_string(i));
    }
    const auto cam_t_x = read_numeric_column<arrow::FloatArray, float>(table, "cam_t_x");
    const auto cam_t_y = read_numeric_column<arrow::FloatArray, float>(table, "cam_t_y");
    const auto cam_t_z = read_numeric_column<arrow::FloatArray, float>(table, "cam_t_z");
    const auto scaled_focal_length = read_numeric_column<arrow::FloatArray, float>(table, "scaled_focal_length");
    const auto img_w = read_numeric_column<arrow::Int32Array, std::int32_t>(table, "img_w");
    const auto img_h = read_numeric_column<arrow::Int32Array, std::int32_t>(table, "img_h");

    const auto row_count = static_cast<std::size_t>(table->num_rows());
    std::vector<HandExportRow> rows;
    rows.reserve(row_count);

    for (std::size_t i = 0; i < row_count; ++i) {
        HandExportRow row;
        row.subject = subject[i];
        row.trial = trial[i];
        row.frame = frame[i];
        row.is_right = is_right[i];
        row.hand_track_id = hand_track_id[i];
        row.label = label[i];

        for (int b = 0; b < 10; ++b) {
            row.params.betas[static_cast<std::size_t>(b)] = beta[static_cast<std::size_t>(b)][i];
        }
        row.params.global_orient = axis_angle_to_matrix(gorient_rotvec[0][i], gorient_rotvec[1][i], gorient_rotvec[2][i]);
        for (int j = 0; j < 15; ++j) {
            const auto rotvec_base = static_cast<std::size_t>(j * 3);
            const auto mat = axis_angle_to_matrix(pose_rotvec[rotvec_base][i], pose_rotvec[rotvec_base + 1][i], pose_rotvec[rotvec_base + 2][i]);
            const auto matrix_base = static_cast<std::size_t>(j * 9);
            for (int k = 0; k < 9; ++k) {
                row.params.hand_pose[matrix_base + static_cast<std::size_t>(k)] = mat[static_cast<std::size_t>(k)];
            }
        }
        row.params.is_right = row.is_right != 0;
        row.params.cam_t = glm::vec3(cam_t_x[i], cam_t_y[i], cam_t_z[i]);

        row.scaled_focal_length = scaled_focal_length[i];
        row.img_w = img_w[i];
        row.img_h = img_h[i];

        rows.push_back(std::move(row));
    }

    return rows;
}

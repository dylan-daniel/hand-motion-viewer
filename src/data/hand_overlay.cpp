#include "data/hand_overlay.h"

#include "data/npy.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <unordered_map>

namespace {
    namespace fs = std::filesystem;
    using nlohmann::json;

    // HaMeR's assumed camera model: a fixed focal length derived from its
    // training-time crop size (256) and assumed focal length (5000), scaled by
    // the real image's longer side. Independent of the actual camera.
    constexpr float HAMER_FOCAL_BASE = 5000.0f;
    constexpr float HAMER_CROP_SIZE = 256.0f;

    float hamer_focal_length(int img_width, int img_height) { return HAMER_FOCAL_BASE / HAMER_CROP_SIZE * static_cast<float>(std::max(img_width, img_height)); }

    std::vector<std::string> split_csv_line(const std::string& line) {
        std::vector<std::string> fields;
        std::size_t start = 0;
        while (true) {
            const std::size_t comma = line.find(',', start);
            if (comma == std::string::npos) {
                fields.push_back(line.substr(start));
                break;
            }
            fields.push_back(line.substr(start, comma - start));
            start = comma + 1;
        }
        return fields;
    }

    struct MaskRow {
        int instance_id = 0;
        float score = 0.0f;
        std::string masks_npy; // path relative to the trial's sam3 directory
    };

    // Pick the highest-scoring detection row for ``frame_number`` from a SAM3
    // index CSV (index_left_hand.csv / index_right_hand.csv). Returns nullopt if
    // the file is missing or has no row for that frame.
    std::optional<MaskRow> best_row_for_frame(const std::string& index_csv_path, int frame_number) {
        std::ifstream file(index_csv_path);
        if (!file) {
            return std::nullopt;
        }
        std::string line;
        if (!std::getline(file, line)) {
            return std::nullopt;
        }
        std::unordered_map<std::string, int> column;
        {
            const std::vector<std::string> headers = split_csv_line(line);
            for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
                column[headers[index]] = index;
            }
        }
        const auto frame_col = column.find("frame");
        const auto score_col = column.find("score");
        const auto instance_col = column.find("instance_id");
        const auto masks_col = column.find("masks_npy");
        const auto status_col = column.find("status");
        if (frame_col == column.end() || score_col == column.end() || instance_col == column.end() || masks_col == column.end()) {
            return std::nullopt;
        }

        std::optional<MaskRow> best;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            const std::vector<std::string> fields = split_csv_line(line);
            const int max_col = std::max({frame_col->second, score_col->second, instance_col->second, masks_col->second});
            if (static_cast<int>(fields.size()) <= max_col) {
                continue;
            }
            if (status_col != column.end() && static_cast<int>(fields.size()) > status_col->second && fields[status_col->second] != "detected") {
                continue;
            }
            if (std::stoi(fields[frame_col->second]) != frame_number) {
                continue;
            }
            MaskRow row;
            row.score = fields[score_col->second].empty() ? 0.0f : std::stof(fields[score_col->second]);
            row.instance_id = fields[instance_col->second].empty() ? 0 : std::stoi(fields[instance_col->second]);
            row.masks_npy = fields[masks_col->second];
            if (!best || row.score > best->score) {
                best = row;
            }
        }
        return best;
    }

    struct Da3Meta {
        float fx = 0.0f;
        float fy = 0.0f;
        float cx = 0.0f;
        float cy = 0.0f;
    };

    std::optional<Da3Meta> load_da3_meta(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            return std::nullopt;
        }
        json data;
        try {
            file >> data;
        } catch (const json::exception&) {
            return std::nullopt;
        }
        Da3Meta meta;
        try {
            meta.fx = data.at("fx").get<float>();
            meta.fy = data.at("fy").get<float>();
            meta.cx = data.at("cx").get<float>();
            meta.cy = data.at("cy").get<float>();
        } catch (const json::exception&) {
            return std::nullopt;
        }
        return meta;
    }

    std::string frame_stem(int frame_number) {
        char name[32];
        std::snprintf(name, sizeof(name), "frame_%04d", frame_number);
        return name;
    }
} // namespace

CacheRoots resolve_cache_roots(const std::string& hand_cache_root) {
    if (hand_cache_root.empty()) {
        return {};
    }
    const fs::path root(hand_cache_root);
    return CacheRoots{(root / "sam3_cache").string(), (root / "da3_cache").string()};
}

bool cache_available_for(const CacheRoots& roots, const std::string& subject, const std::string& trial) {
    if (roots.sam3_root.empty() || roots.da3_root.empty()) {
        return false;
    }
    return fs::is_directory(fs::path(roots.sam3_root) / subject / trial) && fs::is_directory(fs::path(roots.da3_root) / subject / trial);
}

std::vector<glm::vec3> build_hand_overlay_points(
    const CacheRoots& roots,
    const std::string& subject,
    const std::string& trial,
    int frame_number,
    const HandData& hand,
    int img_width,
    int img_height,
    std::optional<float> reference_depth
) {
    std::vector<glm::vec3> result;
    if (hand.verts.empty() || img_width <= 0 || img_height <= 0) {
        return result;
    }

    const fs::path sam3_dir = fs::path(roots.sam3_root) / subject / trial;
    const fs::path da3_dir = fs::path(roots.da3_root) / subject / trial;
    const std::string stem = frame_stem(frame_number);

    // Step 1-2: undo the (x, -y, -z) viewer flip to recover raw full-frame
    // HaMeR camera-space vertices, then project the hand's mean vertex with
    // HaMeR's own (assumed) focal length to find its anchor pixel.
    glm::dvec3 mean_hamer(0.0);
    for (const glm::vec3& vertex : hand.verts) {
        mean_hamer += glm::dvec3(vertex.x, -vertex.y, -vertex.z);
    }
    mean_hamer /= static_cast<double>(hand.verts.size());
    if (mean_hamer.z <= 0.0) {
        return result;
    }

    const float f_hamer = hamer_focal_length(img_width, img_height);
    const float cx_img = static_cast<float>(img_width) / 2.0f;
    const float cy_img = static_cast<float>(img_height) / 2.0f;
    const float anchor_u = f_hamer * static_cast<float>(mean_hamer.x / mean_hamer.z) + cx_img;
    const float anchor_v = f_hamer * static_cast<float>(mean_hamer.y / mean_hamer.z) + cy_img;
    const int anchor_px = static_cast<int>(std::lround(anchor_u));
    const int anchor_py = static_cast<int>(std::lround(anchor_v));
    if (anchor_px < 0 || anchor_px >= img_width || anchor_py < 0 || anchor_py >= img_height) {
        return result;
    }

    const auto meta = load_da3_meta((da3_dir / "arrays" / (stem + "_meta.json")).string());
    if (!meta) {
        return result;
    }
    NpyArray depth;
    try {
        depth = load_npy_f32((da3_dir / "arrays" / (stem + "_depth.npy")).string());
    } catch (const std::runtime_error&) {
        return result;
    }
    if (depth.shape.size() != 2 || depth.shape[0] != img_height || depth.shape[1] != img_width) {
        return result;
    }
    const float* depth_pixels = reinterpret_cast<const float*>(depth.data.data());
    auto depth_at = [&](int px, int py) { return depth_pixels[static_cast<std::size_t>(py) * img_width + px]; };

    // Step 2-3: sample DA3's depth at the anchor pixel, purely as a sanity
    // check that this pixel has valid depth data. MANO's vertex *positions*
    // (cam_t) are in HaMeR's own inflated, uncalibrated units (mean_hamer.z
    // is typically tens of units for a hand a fraction of a metre away), but
    // its vertex *shape* (each vertex's offset from the mesh's own mean) is
    // already genuine real-world metric scale — confirmed against the CSV's
    // placed_bbox extents, which are ~0.1-0.2 units regardless of depth. So
    // DA3's real metric offsets from its own anchor point need no rescaling
    // to match: they're added directly to the mesh's mean vertex.
    const float z_anchor = depth_at(anchor_px, anchor_py);
    if (z_anchor <= 0.0f) {
        return result;
    }
    // The anchor pixel's own real 3D position (DA3 space), used to re-centre
    // every masked point relative to the mesh's mean vertex below.
    const glm::vec3 anchor_da3(
        (static_cast<float>(anchor_px) - meta->cx) * z_anchor / meta->fx, (static_cast<float>(anchor_py) - meta->cy) * z_anchor / meta->fy, z_anchor
    );

    // Step 4: locate this hand's SAM3 mask for this frame.
    const std::string index_name = hand.is_right ? "index_right_hand.csv" : "index_left_hand.csv";
    const auto mask_row = best_row_for_frame((sam3_dir / index_name).string(), frame_number);
    if (!mask_row) {
        return result;
    }
    NpyArray masks;
    try {
        masks = load_npy_bool((sam3_dir / mask_row->masks_npy).string());
    } catch (const std::runtime_error&) {
        return result;
    }
    if (masks.shape.size() != 3 || masks.shape[1] != img_height || masks.shape[2] != img_width) {
        return result;
    }
    const std::int64_t instance = std::clamp<std::int64_t>(mask_row->instance_id, 0, masks.shape[0] - 1);
    const std::uint8_t* mask_pixels = masks.data.data() + static_cast<std::size_t>(instance) * img_height * img_width;
    auto mask_at = [&](int px, int py) { return mask_pixels[static_cast<std::size_t>(py) * img_width + px] != 0; };

    // Gather masked pixels' DA3 depth first so the silhouette-edge cleanup
    // (drop points whose depth is a median-absolute-deviation outlier) can run
    // before the per-point unprojection.
    struct MaskedPixel {
        int px, py;
        float z_da3;
    };
    std::vector<MaskedPixel> pixels;
    for (int py = 0; py < img_height; ++py) {
        for (int px = 0; px < img_width; ++px) {
            if (!mask_at(px, py)) {
                continue;
            }
            const float z_da3 = depth_at(px, py);
            if (z_da3 > 0.0f) {
                pixels.push_back({px, py, z_da3});
            }
        }
    }
    if (pixels.empty()) {
        return result;
    }

    std::vector<float> depths;
    depths.reserve(pixels.size());
    for (const MaskedPixel& pixel : pixels) {
        depths.push_back(pixel.z_da3);
    }
    std::nth_element(depths.begin(), depths.begin() + static_cast<std::ptrdiff_t>(depths.size() / 2), depths.end());
    const float median_z = depths[depths.size() / 2];
    std::vector<float> abs_dev;
    abs_dev.reserve(pixels.size());
    for (const MaskedPixel& pixel : pixels) {
        abs_dev.push_back(std::fabs(pixel.z_da3 - median_z));
    }
    std::nth_element(abs_dev.begin(), abs_dev.begin() + static_cast<std::ptrdiff_t>(abs_dev.size() / 2), abs_dev.end());
    const float mad = abs_dev[abs_dev.size() / 2];
    const float depth_threshold = std::max(6.0f * mad, 0.03f);

    // Per-hand depth stabilisation, matching FrameGpu::hand_matrix: the
    // renderer scales the hand about the origin by reference_depth / its mean
    // viewer-space depth. Applying the same factor here keeps the overlay
    // aligned with the drawn (stabilised) hand.
    float stabilization = 1.0f;
    if (reference_depth) {
        double sum_view_z = 0.0;
        for (const glm::vec3& vertex : hand.verts) {
            sum_view_z += vertex.z;
        }
        const float hand_depth = static_cast<float>(sum_view_z / static_cast<double>(hand.verts.size()));
        if (hand_depth != 0.0f) {
            stabilization = *reference_depth / hand_depth;
        }
    }

    result.reserve(pixels.size());
    for (const MaskedPixel& pixel : pixels) {
        if (std::fabs(pixel.z_da3 - median_z) > depth_threshold) {
            continue;
        }
        // Unproject with DA3's real intrinsics (already-metric, no correction
        // needed on this side), then bring it into HaMeR space: its real
        // metric offset from the anchor point, re-centred on the mesh's own
        // mean vertex (no rescaling — see the comment above on why).
        const glm::vec3 point_da3(
            (static_cast<float>(pixel.px) - meta->cx) * pixel.z_da3 / meta->fx, (static_cast<float>(pixel.py) - meta->cy) * pixel.z_da3 / meta->fy, pixel.z_da3
        );
        const glm::vec3 point_hamer = glm::vec3(mean_hamer) + (point_da3 - anchor_da3);

        // Flip to viewer convention (x, -y, -z), matching HandData::verts.
        result.emplace_back(point_hamer.x * stabilization, -point_hamer.y * stabilization, -point_hamer.z * stabilization);
    }
    return result;
}

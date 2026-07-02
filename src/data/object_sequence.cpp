#include "data/object_sequence.h"

#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>

namespace {
    /// Split a CSV line on commas. The cube CSV has no quoted fields, so a plain
    /// split suffices; empty cells are preserved as empty strings.
    std::vector<std::string> split_csv(const std::string& line) {
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

    float to_float(const std::string& text) { return text.empty() ? 0.0f : std::stof(text); }
} // namespace

CubeSequence::CubeSequence(const std::string& csv_path) {
    std::ifstream file(csv_path);
    if (!file) {
        throw std::runtime_error("Could not open cube CSV " + csv_path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        return; // empty file: no poses
    }
    // Map column name -> index so the parse is order-independent.
    std::unordered_map<std::string, int> column;
    {
        const std::vector<std::string> headers = split_csv(line);
        for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
            column[headers[index]] = index;
        }
    }
    auto column_index = [&](const std::string& name) -> int {
        const auto found = column.find(name);
        if (found == column.end()) {
            throw std::runtime_error("Cube CSV missing column '" + name + "' in " + csv_path);
        }
        return found->second;
    };
    const int frame_column = column_index("frame");
    const int center_x = column_index("placed_center_x");
    const int center_y = column_index("placed_center_y");
    const int center_z = column_index("placed_center_z");
    const int half_x = column_index("obb_half_x");
    const int half_y = column_index("obb_half_y");
    const int half_z = column_index("obb_half_z");
    const int orient_base = column_index("orient_0"); // 9 contiguous, row-major
    const int mean_z = column_index("placed_mean_z");

    // Every row we keep must reach the orientation block's last column.
    const int last_needed = orient_base + 8;

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= last_needed || fields[frame_column].empty()) {
            continue; // malformed / short / blank-frame row
        }

        CubePose pose;
        pose.center = glm::vec3(to_float(fields[center_x]), to_float(fields[center_y]), to_float(fields[center_z]));
        pose.half_extent = glm::vec3(to_float(fields[half_x]), to_float(fields[half_y]), to_float(fields[half_z]));
        pose.placed_mean_z = to_float(fields[mean_z]);

        // orient_0..8 is a row-major 3x3 whose rows are the cube's OBB axes in
        // HaMeR *camera* space. The cube's local→camera rotation is therefore the
        // transpose (its columns are those axes). Composing with the camera→placed
        // flip (which negates y and z; see the place transform) gives local→placed:
        // each axis keeps its x and has its y/z negated. glm is column-major, so the
        // three placed axes become the matrix's three columns directly.
        const float orient[9] = {
            to_float(fields[orient_base + 0]),
            to_float(fields[orient_base + 1]),
            to_float(fields[orient_base + 2]),
            to_float(fields[orient_base + 3]),
            to_float(fields[orient_base + 4]),
            to_float(fields[orient_base + 5]),
            to_float(fields[orient_base + 6]),
            to_float(fields[orient_base + 7]),
            to_float(fields[orient_base + 8]),
        };
        pose.basis[0] = glm::vec3(orient[0], -orient[1], -orient[2]); // axis 0 (row 0), flipped to placed space
        pose.basis[1] = glm::vec3(orient[3], -orient[4], -orient[5]); // axis 1 (row 1)
        pose.basis[2] = glm::vec3(orient[6], -orient[7], -orient[8]); // axis 2 (row 2, face normal)

        poses_[std::stoi(fields[frame_column])] = pose;
    }
}

std::optional<CubePose> CubeSequence::pose_for_frame(int frame_number) const {
    const auto found = poses_.find(frame_number);
    if (found == poses_.end()) {
        return std::nullopt;
    }
    return found->second;
}

glm::mat4 cube_model_matrix(const CubePose& pose, const Transform* transform, std::optional<float> reference_depth) {
    // Scene fit (matches FrameGpu::hand_matrix), then the about-origin depth snap
    // the hands also get, then the cube's own placed-space pose.
    glm::mat4 model(1.0f);
    if (transform != nullptr) {
        model = glm::scale(model, glm::vec3(transform->scale));
        model = glm::translate(model, transform->translate);
    }
    const float stabilize = (reference_depth && pose.placed_mean_z != 0.0f) ? (*reference_depth / pose.placed_mean_z) : 1.0f;
    model = glm::scale(model, glm::vec3(stabilize));
    model = glm::translate(model, pose.center);
    model = model * glm::mat4(pose.basis);
    model = glm::scale(model, pose.half_extent);
    return model;
}

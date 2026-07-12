#include "data/production_log.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <regex>
#include <vector>

namespace {
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

    // Strip a leading UTF-8 BOM (EF BB BF), which this study's Excel-exported
    // CSV carries on its header line (Python's csv reader is opened with
    // encoding="utf-8-sig" specifically to swallow it).
    std::string strip_bom(const std::string& line) {
        if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF && static_cast<unsigned char>(line[1]) == 0xBB &&
            static_cast<unsigned char>(line[2]) == 0xBF) {
            return line.substr(3);
        }
        return line;
    }

    // Same physical size for both shapes (a sphere's diameter, a cube's edge
    // length) — mirrors scene_placement.py's SIZE_CM exactly.
    std::optional<float> size_cm_for_label(const std::string& size_label) {
        if (size_label == "Large") return 4.0f;
        if (size_label == "Medium") return 2.5f;
        if (size_label == "Small") return 1.0f;
        return std::nullopt;
    }

    std::string key_for(const std::string& subject, const std::string& trial_number) { return subject + "/" + trial_number; }
} // namespace

ProductionLog::ProductionLog(const std::string& csv_path) {
    if (csv_path.empty()) {
        return;
    }
    std::ifstream file(csv_path);
    std::string line;
    if (!file || !std::getline(file, line)) {
        return;
    }
    line = strip_bom(line);
    std::unordered_map<std::string, int> column;
    {
        const std::vector<std::string> headers = split_csv(line);
        for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
            column[headers[index]] = index;
        }
    }
    const auto subject_column = column.find("Baby ID");
    const auto trial_column = column.find("Trial");
    const auto size_column = column.find("Size");
    const auto shape_column = column.find("Shape");
    if (subject_column == column.end() || trial_column == column.end() || size_column == column.end() || shape_column == column.end()) {
        return;
    }
    const int last_needed = std::max({subject_column->second, trial_column->second, size_column->second, shape_column->second});

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= last_needed) {
            continue;
        }
        const std::optional<float> size_cm = size_cm_for_label(fields[size_column->second]);
        const std::string& shape_label = fields[shape_column->second];
        if (!size_cm || (shape_label != "Cube" && shape_label != "Sphere")) {
            continue;
        }
        ObjectSizeInfo info;
        info.size_m = *size_cm / 100.0f;
        info.shape = shape_label == "Cube" ? 1 : 2;
        info.object_label = shape_label == "Cube" ? "small_cube" : "small_ball";
        info_by_subject_trial_[key_for(fields[subject_column->second], fields[trial_column->second])] = info;
    }
}

std::optional<ObjectSizeInfo> ProductionLog::lookup(const std::string& subject, const std::string& trial_view) const {
    // trial_view is e.g. "T1_BabyView"; only the leading trial number matters
    // (mirrors scene_placement.object_size_meters's own regex match).
    std::smatch match;
    static const std::regex trial_number_pattern(R"(^T(\d+))");
    if (!std::regex_search(trial_view, match, trial_number_pattern)) {
        return std::nullopt;
    }
    const auto found = info_by_subject_trial_.find(key_for(subject, match[1].str()));
    return found == info_by_subject_trial_.end() ? std::nullopt : std::optional<ObjectSizeInfo>(found->second);
}

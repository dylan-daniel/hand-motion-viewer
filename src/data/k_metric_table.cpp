#include "data/k_metric_table.h"

#include <algorithm>
#include <fstream>
#include <unordered_map>
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

    std::string key_for(const std::string& subject, const std::string& trial) { return subject + "/" + trial; }
} // namespace

KMetricTable::KMetricTable(const std::string& csv_path) {
    if (csv_path.empty()) {
        return;
    }
    std::ifstream file(csv_path);
    std::string line;
    if (!file || !std::getline(file, line)) {
        return;
    }
    std::unordered_map<std::string, int> column;
    {
        const std::vector<std::string> headers = split_csv(line);
        for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
            column[headers[index]] = index;
        }
    }
    const auto subject_column = column.find("subject");
    const auto trial_column = column.find("trial");
    const auto k_metric_column = column.find("k_metric_median");
    if (subject_column == column.end() || trial_column == column.end() || k_metric_column == column.end()) {
        return;
    }
    const int last_needed = std::max({subject_column->second, trial_column->second, k_metric_column->second});

    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= last_needed) {
            continue;
        }
        const std::string subject = fields[subject_column->second];
        const std::string trial = fields[trial_column->second];
        const float k_metric = std::stof(fields[k_metric_column->second]);
        k_metric_by_subject_trial_[key_for(subject, trial)] = k_metric;
    }
}

std::optional<float> KMetricTable::lookup(const std::string& subject, const std::string& trial) const {
    const auto found = k_metric_by_subject_trial_.find(key_for(subject, trial));
    return found == k_metric_by_subject_trial_.end() ? std::nullopt : std::optional<float>(found->second);
}

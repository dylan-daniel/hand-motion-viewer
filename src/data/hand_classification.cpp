#include "data/hand_classification.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace {
    namespace fs = std::filesystem;

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

    long long frame_hand_key(int frame_number, int hand_idx) { return static_cast<long long>(frame_number) * 100000LL + hand_idx; }

    /// Read a CSV's header into a name->column map, or return empty if the file
    /// can't be opened (caller treats that as "classification unavailable").
    bool read_header(std::ifstream& file, std::unordered_map<std::string, int>& column) {
        std::string line;
        if (!std::getline(file, line)) {
            return false;
        }
        const std::vector<std::string> headers = split_csv(line);
        for (int index = 0; index < static_cast<int>(headers.size()); ++index) {
            column[headers[index]] = index;
        }
        return true;
    }
} // namespace

HandClassification::HandClassification(
    const std::string& tracking_dir, const std::string& baby_hand_idx_dir, const std::string& subject, const std::string& trial
) {
    if (tracking_dir.empty() || baby_hand_idx_dir.empty()) {
        return;
    }
    const fs::path tracker_path = fs::path(tracking_dir) / ("tracks3_" + subject + "_" + trial + ".csv");
    const fs::path baby_path = fs::path(baby_hand_idx_dir) / (subject + "_" + trial + "_hand_idx.csv");

    // Pass 1: the baby-hand classification CSV, keyed by track_id.
    std::unordered_map<int, bool> is_baby_by_track;
    {
        std::ifstream file(baby_path);
        std::unordered_map<std::string, int> column;
        if (!file || !read_header(file, column)) {
            return;
        }
        const auto track_column = column.find("track_id");
        const auto baby_column = column.find("is_baby");
        if (track_column == column.end() || baby_column == column.end()) {
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            const std::vector<std::string> fields = split_csv(line);
            if (static_cast<int>(fields.size()) <= std::max(track_column->second, baby_column->second)) {
                continue;
            }
            const int track_id = std::stoi(fields[track_column->second]);
            is_baby_by_track[track_id] = std::stoi(fields[baby_column->second]) != 0;
        }
    }

    // Pass 2: the tracker CSV, keyed by (frame, idx == hamer_cache's hand_idx).
    // is_duplicate==1 rows are kept too (rather than filtered out, as the
    // Python reference does before building its per-frame idx set) so a
    // duplicate detection still resolves a real track_id/is_duplicate here;
    // HandFilter is what actually hides it.
    {
        std::ifstream file(tracker_path);
        std::unordered_map<std::string, int> column;
        if (!file || !read_header(file, column)) {
            loaded_ = false;
            by_frame_hand_.clear();
            return;
        }
        const auto frame_column = column.find("frame");
        const auto idx_column = column.find("idx");
        const auto track_column = column.find("track_id");
        const auto duplicate_column = column.find("is_duplicate");
        if (frame_column == column.end() || idx_column == column.end() || track_column == column.end() || duplicate_column == column.end()) {
            return;
        }
        const int last_needed = std::max({frame_column->second, idx_column->second, track_column->second, duplicate_column->second});
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) {
                continue;
            }
            const std::vector<std::string> fields = split_csv(line);
            if (static_cast<int>(fields.size()) <= last_needed) {
                continue;
            }
            const int frame_number = std::stoi(fields[frame_column->second]);
            const int hand_idx = std::stoi(fields[idx_column->second]);
            const int track_id = std::stoi(fields[track_column->second]);
            const bool is_duplicate = std::stoi(fields[duplicate_column->second]) != 0;

            HandClassificationEntry entry;
            entry.track_id = track_id;
            entry.is_duplicate = is_duplicate;
            // Mirrors the Python reference join exactly: a track only counts as
            // the baby's when it has an explicit is_baby==1 row in the
            // classification CSV. A track absent from that CSV (e.g. an
            // adult/caregiver track the classifier never voted on) is NOT a
            // baby hand — it must not default to shown.
            const auto found_baby = is_baby_by_track.find(track_id);
            entry.is_baby = found_baby != is_baby_by_track.end() && found_baby->second;
            by_frame_hand_[frame_hand_key(frame_number, hand_idx)] = entry;
        }
    }

    loaded_ = true;
}

const HandClassificationEntry* HandClassification::lookup(int frame_number, int hand_idx) const {
    const auto found = by_frame_hand_.find(frame_hand_key(frame_number, hand_idx));
    return found == by_frame_hand_.end() ? nullptr : &found->second;
}

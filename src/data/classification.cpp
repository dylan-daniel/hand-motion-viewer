#include "data/classification.h"

#include <cstdio>
#include <exception>
#include <fstream>
#include <sstream>
#include <string>

HandClassification HandClassification::load(const std::string& csv_path) {
    HandClassification classification;

    std::ifstream file(csv_path);
    if (!file) {
        return classification;
    }

    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        // Skip the header row (subject,trial,view,frame,hand_id,is_baby,...).
        if (first) {
            first = false;
            if (line.rfind("subject", 0) == 0) {
                continue;
            }
        }

        // Columns are comma-separated. We need frame (3), hand_id (4) and is_baby
        // (5); the remaining columns (match_iou, gemma_side) are ignored.
        std::stringstream stream(line);
        std::string subject;
        std::string trial;
        std::string view;
        std::string frame;
        std::string hand_id;
        std::string is_baby;
        if (!std::getline(stream, subject, ',') || !std::getline(stream, trial, ',') || !std::getline(stream, view, ',') || !std::getline(stream, frame, ',') ||
            !std::getline(stream, hand_id, ',') || !std::getline(stream, is_baby, ',')) {
            continue;
        }

        // The CSV frame is 0-based and the hmesh files are 1-based, so the matching
        // file is frame_<frame+1>_<hand_id>.hmesh (zero-padded to 4 like on disk). A
        // malformed numeric cell skips the row rather than mislabelling a hand.
        int frame_number = 0;
        int hand_index = 0;
        bool baby = true;
        try {
            frame_number = std::stoi(frame);
            hand_index = std::stoi(hand_id);
            baby = std::stoi(is_baby) != 0;
        } catch (const std::exception&) {
            continue;
        }
        char filename[40];
        std::snprintf(filename, sizeof(filename), "frame_%04d_%d.hmesh", frame_number + 1, hand_index);
        classification.is_baby_[filename] = baby;
    }

    return classification;
}

bool HandClassification::is_baby(const std::string& hmesh_filename) const {
    const auto found = is_baby_.find(hmesh_filename);
    return found == is_baby_.end() ? true : found->second;
}

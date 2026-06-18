#include "data/classification.h"

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
        // Skip the header row
        // (subject,trial,view,track_id,n_votes,n_baby,n_adult,frac_baby,label).
        if (first) {
            first = false;
            if (line.rfind("subject", 0) == 0) {
                continue;
            }
        }

        // Columns are comma-separated. We need track_id (column 3), n_votes
        // (column 4) and the final label column; the remaining columns are ignored.
        std::stringstream stream(line);
        std::string subject;
        std::string trial;
        std::string view;
        std::string track_id;
        std::string n_votes;
        if (!std::getline(stream, subject, ',') || !std::getline(stream, trial, ',') || !std::getline(stream, view, ',') ||
            !std::getline(stream, track_id, ',') || !std::getline(stream, n_votes, ',')) {
            continue;
        }

        // The label is the last column on the row.
        std::string label;
        std::string cell;
        while (std::getline(stream, cell, ',')) {
            label = cell;
        }
        if (label.empty()) {
            continue;
        }

        // A malformed track_id or vote count skips the row rather than mislabelling
        // a hand.
        int track_number = 0;
        int vote_count = 0;
        try {
            track_number = std::stoi(track_id);
            vote_count = std::stoi(n_votes);
        } catch (const std::exception&) {
            continue;
        }
        classification.is_baby_[track_number] = (label == "baby");
        classification.votes_[track_number] = vote_count;
    }

    return classification;
}

bool HandClassification::is_baby(int track_id) const {
    if (track_id < 0) {
        return true;
    }
    const auto found = is_baby_.find(track_id);
    return found == is_baby_.end() ? true : found->second;
}

int HandClassification::track_votes(int track_id) const {
    if (track_id < 0) {
        return -1;
    }
    const auto found = votes_.find(track_id);
    return found == votes_.end() ? -1 : found->second;
}

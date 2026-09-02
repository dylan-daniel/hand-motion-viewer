#pragma once

// Loader for the compact binary export written by infant_grasp_pipeline's
// run_export_hands (src/pipeline/export/binary_format.py in that repo): a
// small header (magic, version, decompressed payload size) followed by one
// zlib-compressed payload holding a name+dtype schema table and the raw
// column arrays -- see that file for the exact byte layout, which this
// loader mirrors. One row per tracked hand per frame; rotations are stored
// there as axis-angle to save space and get converted back to the row-major
// 3x3 matrices ManoParams expects here at load time (see
// axis_angle_to_matrix in the .cpp) -- mathematically exact, with only float
// round-off error (~1e-7), not a meaningful precision loss.

#include <cstdint>
#include <string>
#include <vector>

#include "data/mano_model.h"

/// One row of the export: a tracked hand's MANO params for one frame, plus the
/// pipeline's own identity/classification metadata. hand_track_id is only
/// unique within (subject, trial, is_right) -- see
/// .claude/docs/TRACKING_CSV_SCHEMA.md in infant_grasp_pipeline.
struct HandExportRow {
    std::string subject;
    std::string trial;
    std::int32_t frame = 0;
    std::int8_t is_right = 0;
    std::int32_t hand_track_id = -1;
    std::string label; // "infant" | "adult" | "unknown" | "unclassified"
    ManoParams params;
    float scaled_focal_length = 0.0f;
    std::int32_t img_w = 0;
    std::int32_t img_h = 0;
};

/// Read every row of a hand-motion export binary file. Throws
/// std::runtime_error on any read failure: a missing file, a bad magic
/// number, an unsupported version, a decompression failure, or a missing
/// expected column.
std::vector<HandExportRow> load_hand_export(const std::string& export_path);

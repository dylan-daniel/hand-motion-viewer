#pragma once

// Minimal reader for the two .npy layouts the tracked-object pipeline needs:
// SAM3's boolean instance-mask stacks (frame_XXXX_<label>_masks.npy, shape
// (N, H, W), dtype bool) and DA3's float32 depth maps (frame_XXXX_depth.npy,
// shape (H, W), dtype float32). Both are written by plain ``np.save`` — no
// compression, no pickled objects, no fancy dtypes — so a full numpy format
// implementation isn't needed, just the fixed v1.0/v2.0 header and a raw copy
// of the C-contiguous data that follows it.

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

/// A DA3 depth map for one frame: ``height`` x ``width`` float32 meters,
/// row-major (``depth[v * width + u]``).
struct NpyDepthMap {
    std::vector<float> depth;
    int height = 0;
    int width = 0;
};

/// Read a (H, W) float32 .npy file. Returns nullopt if the file is missing,
/// malformed, or not a 2D float32 C-contiguous array.
std::optional<NpyDepthMap> read_npy_depth(const std::string& path);

/// A SAM3 instance-mask stack for one frame/label: ``count`` instances, each
/// ``height`` x ``width`` bytes (0 or 1), row-major
/// (``masks[instance * height * width + v * width + u]``). ``count`` is 0 for
/// a "not detected this frame" cache entry (numpy shape (0, H, W)).
struct NpyMaskStack {
    std::vector<std::uint8_t> masks;
    int count = 0;
    int height = 0;
    int width = 0;
};

/// Read a (N, H, W) bool .npy file. Returns nullopt if the file is missing,
/// malformed, or not a 3D bool/uint8 C-contiguous array.
std::optional<NpyMaskStack> read_npy_mask_stack(const std::string& path);

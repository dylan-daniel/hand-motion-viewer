#pragma once

// Minimal ``.npy`` reader for the two array kinds the SAM3/DA3 cache actually
// writes: C-order float32 depth/confidence maps and C-order bool mask stacks.
// Not a general numpy reader — no Fortran order, no other dtypes, no pickled
// object arrays.

#include <cstdint>
#include <string>
#include <vector>

/// A loaded ``.npy`` array: raw bytes plus its shape, one entry per axis in
/// numpy's row-major order (e.g. mask stacks are ``{instances, height, width}``).
struct NpyArray {
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> data; // raw element bytes, tightly packed, C-order

    std::int64_t element_count() const;
};

/// Load ``path`` as float32 data (descr ``<f4``). Throws std::runtime_error if
/// the file is missing, malformed, or not float32.
NpyArray load_npy_f32(const std::string& path);

/// Load ``path`` as bool/uint8 data (descr ``|b1`` or ``|u1``). Throws
/// std::runtime_error if the file is missing, malformed, or not that dtype.
NpyArray load_npy_bool(const std::string& path);

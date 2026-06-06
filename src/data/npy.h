#pragma once

// Minimal reader for NumPy ``.npy`` arrays of little-endian float32 (``<f4``).
// HaMeR exports one array per hand per frame (cam translation, 3D joints, MANO
// shape/pose) beside every ``.obj``; the cross-view aligner reads those to
// recover the transform between two camera views. Only the single dtype the
// pipeline emits is supported, which keeps the parser tiny and dependency free.

#include <cstddef>
#include <string>
#include <vector>

/// A loaded float32 array: flat row-major data plus its shape.
struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<float> data;

    bool empty() const { return data.empty(); }
    std::size_t size() const { return data.size(); }
};

/// Load a little-endian float32 ``.npy`` file. Returns an empty array if the
/// file is missing, malformed, or not ``<f4`` typed.
NpyArray load_npy_f32(const std::string& path);

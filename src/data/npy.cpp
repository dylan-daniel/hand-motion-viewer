#include "data/npy.h"

#include <cstdint>
#include <cstring>
#include <fstream>

namespace {
    /// Read a whole file into a byte buffer; empty on failure.
    std::vector<char> read_bytes(const std::string& path) {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        if (!file) {
            return {};
        }
        const std::streamsize size = file.tellg();
        if (size <= 0) {
            return {};
        }
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        file.read(bytes.data(), size);
        return bytes;
    }

    /// Parse the shape tuple out of an ``.npy`` header dict, e.g. ``(21, 3)`` or
    /// ``(10,)``. Returns the dimension sizes in order.
    std::vector<std::size_t> parse_shape(const std::string& header) {
        std::vector<std::size_t> shape;
        const std::size_t key = header.find("'shape'");
        if (key == std::string::npos) {
            return shape;
        }
        const std::size_t open = header.find('(', key);
        const std::size_t close = header.find(')', open);
        if (open == std::string::npos || close == std::string::npos) {
            return shape;
        }
        std::size_t value = 0;
        bool in_number = false;
        for (std::size_t pos = open + 1; pos < close; ++pos) {
            const char character = header[pos];
            if (character >= '0' && character <= '9') {
                value = value * 10 + static_cast<std::size_t>(character - '0');
                in_number = true;
            } else if (in_number) {
                shape.push_back(value);
                value = 0;
                in_number = false;
            }
        }
        if (in_number) {
            shape.push_back(value);
        }
        return shape;
    }
} // namespace

NpyArray load_npy_f32(const std::string& path) {
    const std::vector<char> bytes = read_bytes(path);
    NpyArray array;
    // Magic (6) + version (2) + header-length field (2 for v1.0) = 10 byte prefix.
    if (bytes.size() < 10 || std::memcmp(bytes.data(), "\x93NUMPY", 6) != 0) {
        return array;
    }
    const std::uint8_t major = static_cast<std::uint8_t>(bytes[6]);
    std::size_t header_len = 0;
    std::size_t header_start = 0;
    if (major == 1) {
        header_len = static_cast<std::uint8_t>(bytes[8]) | (static_cast<std::uint8_t>(bytes[9]) << 8);
        header_start = 10;
    } else {
        // v2.0+ uses a 4-byte header length.
        if (bytes.size() < 12) {
            return array;
        }
        header_len = static_cast<std::uint8_t>(bytes[8]) | (static_cast<std::uint8_t>(bytes[9]) << 8) | (static_cast<std::uint8_t>(bytes[10]) << 16) |
            (static_cast<std::uint8_t>(bytes[11]) << 24);
        header_start = 12;
    }
    const std::size_t data_start = header_start + header_len;
    if (data_start > bytes.size()) {
        return array;
    }
    const std::string header(bytes.data() + header_start, header_len);
    // Only little-endian float32 is emitted by the export pipeline.
    if (header.find("'<f4'") == std::string::npos && header.find("\"<f4\"") == std::string::npos) {
        return array;
    }
    array.shape = parse_shape(header);
    const std::size_t byte_count = bytes.size() - data_start;
    const std::size_t float_count = byte_count / sizeof(float);
    array.data.resize(float_count);
    std::memcpy(array.data.data(), bytes.data() + data_start, float_count * sizeof(float));
    return array;
}

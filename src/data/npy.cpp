#include "data/npy.h"

#include <cstring>
#include <fstream>
#include <stdexcept>

namespace {
    // Pull the shape tuple's integers out of the header dict text, e.g.
    // ``'shape': (1, 1080, 1920), `` -> {1, 1080, 1920}. Good enough for the
    // fixed layout numpy always emits; not a general Python-literal parser.
    std::vector<std::int64_t> parse_shape(const std::string& header) {
        const std::size_t key = header.find("'shape'");
        if (key == std::string::npos) {
            throw std::runtime_error("npy header missing 'shape'");
        }
        const std::size_t open = header.find('(', key);
        const std::size_t close = header.find(')', open);
        if (open == std::string::npos || close == std::string::npos) {
            throw std::runtime_error("npy header malformed shape tuple");
        }
        std::vector<std::int64_t> shape;
        std::size_t pos = open + 1;
        while (pos < close) {
            while (pos < close && (header[pos] == ' ' || header[pos] == ',')) {
                ++pos;
            }
            if (pos >= close) {
                break;
            }
            std::size_t end = pos;
            while (end < close && header[end] != ',' && header[end] != ' ') {
                ++end;
            }
            shape.push_back(std::stoll(header.substr(pos, end - pos)));
            pos = end;
        }
        return shape;
    }

    std::string find_descr(const std::string& header) {
        const std::size_t key = header.find("'descr'");
        if (key == std::string::npos) {
            throw std::runtime_error("npy header missing 'descr'");
        }
        const std::size_t value_start = header.find('\'', key + 7);
        const std::size_t value_end = header.find('\'', value_start + 1);
        if (value_start == std::string::npos || value_end == std::string::npos) {
            throw std::runtime_error("npy header malformed descr");
        }
        return header.substr(value_start + 1, value_end - value_start - 1);
    }

    NpyArray load_npy(const std::string& path, const char* expected_descr) {
        std::ifstream file(path, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Could not open npy file " + path);
        }
        char magic[6];
        file.read(magic, 6);
        if (!file || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
            throw std::runtime_error("Not a npy file: " + path);
        }
        std::uint8_t major = 0;
        std::uint8_t minor = 0;
        file.read(reinterpret_cast<char*>(&major), 1);
        file.read(reinterpret_cast<char*>(&minor), 1);

        std::uint32_t header_len = 0;
        if (major == 1) {
            std::uint16_t header_len16 = 0;
            file.read(reinterpret_cast<char*>(&header_len16), 2);
            header_len = header_len16;
        } else {
            file.read(reinterpret_cast<char*>(&header_len), 4);
        }
        std::string header(header_len, '\0');
        file.read(header.data(), static_cast<std::streamsize>(header_len));
        if (!file) {
            throw std::runtime_error("Truncated npy header: " + path);
        }

        const std::string descr = find_descr(header);
        if (descr != expected_descr) {
            throw std::runtime_error("Unexpected npy dtype '" + descr + "' (wanted '" + expected_descr + "') in " + path);
        }

        NpyArray array;
        array.shape = parse_shape(header);
        const std::int64_t element_size = (descr == "<f4") ? 4 : 1;
        const std::int64_t byte_count = array.element_count() * element_size;
        array.data.resize(static_cast<std::size_t>(byte_count));
        file.read(reinterpret_cast<char*>(array.data.data()), byte_count);
        if (!file) {
            throw std::runtime_error("Truncated npy data: " + path);
        }
        return array;
    }
} // namespace

std::int64_t NpyArray::element_count() const {
    std::int64_t count = 1;
    for (std::int64_t dim : shape) {
        count *= dim;
    }
    return count;
}

NpyArray load_npy_f32(const std::string& path) { return load_npy(path, "<f4"); }

NpyArray load_npy_bool(const std::string& path) {
    // Some writers emit '|u1' for what's semantically a bool mask; both are one
    // byte per element and either 0 or non-zero, so accept both under one loader.
    std::ifstream probe(path, std::ios::binary);
    if (!probe) {
        throw std::runtime_error("Could not open npy file " + path);
    }
    probe.close();
    try {
        return load_npy(path, "|b1");
    } catch (const std::runtime_error&) {
        return load_npy(path, "|u1");
    }
}

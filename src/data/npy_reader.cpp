#include "data/npy_reader.h"

#include <array>
#include <cstring>
#include <fstream>

namespace {
    struct NpyHeader {
        std::string descr;    // e.g. "<f4", "|b1", "|u1"
        bool fortran_order = false;
        std::vector<std::int64_t> shape;
    };

    /// Parse the '{'descr': ..., 'fortran_order': ..., 'shape': (...), }' dict
    /// literal numpy writes as the header's ASCII text. Deliberately not a
    /// general Python literal parser — just enough string search for the
    /// specific, consistent layout np.save always produces.
    bool parse_header_dict(const std::string& text, NpyHeader& out) {
        const std::size_t descr_key = text.find("'descr'");
        if (descr_key == std::string::npos) {
            return false;
        }
        const std::size_t descr_quote0 = text.find('\'', descr_key + 7);
        if (descr_quote0 == std::string::npos) {
            return false;
        }
        const std::size_t descr_quote1 = text.find('\'', descr_quote0 + 1);
        if (descr_quote1 == std::string::npos) {
            return false;
        }
        out.descr = text.substr(descr_quote0 + 1, descr_quote1 - descr_quote0 - 1);

        const std::size_t fortran_key = text.find("'fortran_order'");
        if (fortran_key == std::string::npos) {
            return false;
        }
        out.fortran_order = text.find("True", fortran_key) != std::string::npos && (text.find("True", fortran_key) < text.find("False", fortran_key));

        const std::size_t shape_key = text.find("'shape'");
        if (shape_key == std::string::npos) {
            return false;
        }
        const std::size_t paren0 = text.find('(', shape_key);
        const std::size_t paren1 = text.find(')', shape_key);
        if (paren0 == std::string::npos || paren1 == std::string::npos || paren1 < paren0) {
            return false;
        }
        const std::string inside = text.substr(paren0 + 1, paren1 - paren0 - 1);
        std::size_t start = 0;
        while (start < inside.size()) {
            const std::size_t comma = inside.find(',', start);
            const std::string token = inside.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            // Trim whitespace; a trailing comma before ')' (numpy always writes
            // one, even for a 1-tuple) leaves an empty final token — skip it.
            std::size_t left = token.find_first_not_of(" \t");
            std::size_t right = token.find_last_not_of(" \t");
            if (left != std::string::npos) {
                out.shape.push_back(std::stoll(token.substr(left, right - left + 1)));
            }
            if (comma == std::string::npos) {
                break;
            }
            start = comma + 1;
        }
        return true;
    }

    /// Read and parse a .npy file's header, leaving ``file``'s position at the
    /// start of the raw data. Returns nullopt on any format mismatch.
    std::optional<NpyHeader> read_npy_header(std::ifstream& file) {
        std::array<char, 6> magic{};
        file.read(magic.data(), 6);
        if (!file || std::memcmp(magic.data(), "\x93NUMPY", 6) != 0) {
            return std::nullopt;
        }
        std::uint8_t version[2];
        file.read(reinterpret_cast<char*>(version), 2);
        if (!file) {
            return std::nullopt;
        }
        std::uint32_t header_len = 0;
        if (version[0] == 1) {
            std::uint16_t header_len16 = 0;
            file.read(reinterpret_cast<char*>(&header_len16), 2);
            header_len = header_len16;
        } else {
            file.read(reinterpret_cast<char*>(&header_len), 4);
        }
        if (!file) {
            return std::nullopt;
        }
        std::string header_text(header_len, '\0');
        file.read(header_text.data(), static_cast<std::streamsize>(header_len));
        if (!file) {
            return std::nullopt;
        }
        NpyHeader header;
        if (!parse_header_dict(header_text, header) || header.fortran_order) {
            return std::nullopt; // Fortran-order arrays aren't produced by this pipeline
        }
        return header;
    }
} // namespace

std::optional<NpyDepthMap> read_npy_depth(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    const std::optional<NpyHeader> header = read_npy_header(file);
    if (!header || header->descr != "<f4" || header->shape.size() != 2) {
        return std::nullopt;
    }
    NpyDepthMap result;
    result.height = static_cast<int>(header->shape[0]);
    result.width = static_cast<int>(header->shape[1]);
    const std::size_t count = static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width);
    result.depth.resize(count);
    file.read(reinterpret_cast<char*>(result.depth.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!file) {
        return std::nullopt;
    }
    return result;
}

std::optional<NpyMaskStack> read_npy_mask_stack(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }
    const std::optional<NpyHeader> header = read_npy_header(file);
    if (!header || header->shape.size() != 3) {
        return std::nullopt;
    }
    // Bool arrays are one byte per element (values 0/1); some caches may store
    // masks as plain uint8 instead — both read identically here.
    if (header->descr != "|b1" && header->descr != "|u1" && header->descr != "b1") {
        return std::nullopt;
    }
    NpyMaskStack result;
    result.count = static_cast<int>(header->shape[0]);
    result.height = static_cast<int>(header->shape[1]);
    result.width = static_cast<int>(header->shape[2]);
    const std::size_t count =
        static_cast<std::size_t>(result.count) * static_cast<std::size_t>(result.height) * static_cast<std::size_t>(result.width);
    result.masks.resize(count);
    if (count > 0) {
        file.read(reinterpret_cast<char*>(result.masks.data()), static_cast<std::streamsize>(count));
        if (!file) {
            return std::nullopt;
        }
    }
    return result;
}

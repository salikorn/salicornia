// salikorn / st — mmap safetensors reader (Windows).
// Format: [u64 LE header_len][JSON header][data...]; header maps tensor name ->
// {"dtype","shape","data_offsets":[begin,end)} with offsets relative to the data region.
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <Windows.h>

#include "nlohmann/json.hpp"

namespace sal::st {

struct TensorInfo {
    std::string dtype;            // "BF16","F16","F32","F8_E4M3","I64",...
    std::vector<uint64_t> shape;
    uint64_t offset = 0;          // relative to data region
    uint64_t size_bytes = 0;
    const uint8_t* data = nullptr;  // pointer into the mmap

    uint64_t numel() const {
        uint64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

class SafeTensorsFile {
  public:
    explicit SafeTensorsFile(const std::filesystem::path& path) {
        HANDLE file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file == INVALID_HANDLE_VALUE) throw std::runtime_error("open failed: " + path.string());
        LARGE_INTEGER sz{};
        GetFileSizeEx(file, &sz);
        file_size_ = uint64_t(sz.QuadPart);
        HANDLE map = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!map) {
            CloseHandle(file);
            throw std::runtime_error("CreateFileMapping failed");
        }
        base_ = static_cast<const uint8_t*>(MapViewOfFile(map, FILE_MAP_READ, 0, 0, 0));
        CloseHandle(file);  // view keeps a reference
        if (!base_) {
            CloseHandle(map);
            throw std::runtime_error("MapViewOfFile failed");
        }
        map_ = map;

        uint64_t header_len;
        std::memcpy(&header_len, base_, 8);
        if (8 + header_len > file_size_) throw std::runtime_error("header larger than file");
        auto header = nlohmann::json::parse(base_ + 8, base_ + 8 + header_len);
        data_begin_ = 8 + header_len;
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") {
                if (it.value().is_object())
                    for (auto m = it.value().begin(); m != it.value().end(); ++m)
                        metadata_[m.key()] = m.value().get<std::string>();
                continue;
            }
            TensorInfo t;
            t.dtype = it.value().at("dtype").get<std::string>();
            for (auto& s : it.value().at("shape")) t.shape.push_back(s.get<uint64_t>());
            auto off = it.value().at("data_offsets");
            t.offset = off[0].get<uint64_t>();
            t.size_bytes = off[1].get<uint64_t>() - t.offset;
            t.data = base_ + data_begin_ + t.offset;
            tensors_[it.key()] = std::move(t);
        }
    }

    ~SafeTensorsFile() {
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle(map_);
    }

    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    uint64_t file_size() const { return file_size_; }
    uint64_t data_begin() const { return data_begin_; }

    const TensorInfo* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

  private:
    const uint8_t* base_ = nullptr;
    HANDLE map_ = nullptr;
    uint64_t file_size_ = 0;
    uint64_t data_begin_ = 0;
    std::map<std::string, TensorInfo> tensors_;
    std::map<std::string, std::string> metadata_;
};

}  // namespace sal::st

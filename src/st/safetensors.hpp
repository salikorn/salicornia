/**
 * \file safetensors.hpp
 * \brief mmap safetensors reader (Windows).
 *
 * Format: `[u64 LE header_len][JSON header][data...]`; the header maps tensor
 * name -> {"dtype","shape","data_offsets":[begin,end)} with offsets relative
 * to the data region (i.e. relative to 8 + header_len in the file).
 *
 * Declarations here; the constructor/destructor (Windows API + JSON parsing)
 * live in safetensors.cpp. Windows.h is included because of the HANDLE member.
 */
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <Windows.h>

namespace sal::st {

/// \brief Metadata and mmap view of one tensor stored in a safetensors file.
struct TensorInfo {
    std::string dtype;            ///< Element type as written in the JSON header: "BF16","F16","F32","F8_E4M3","I64",...
    std::vector<uint64_t> shape;  ///< Tensor dimensions, row-major order.
    uint64_t offset = 0;          ///< Byte offset of the tensor payload, relative to the data region.
    uint64_t size_bytes = 0;      ///< Payload size in bytes (end - begin from data_offsets).
    const uint8_t* data = nullptr; ///< Pointer into the mmap; valid only while the owning SafeTensorsFile is alive.

    /// Total element count: product of the shape extents (1 for a scalar).
    uint64_t numel() const {
        uint64_t n = 1;
        for (auto s : shape) n *= s;
        return n;
    }
};

/// \brief Read-only memory-mapped safetensors file (Windows).
///
/// The constructor opens the file, creates a PAGE_READONLY file mapping, maps
/// a read-only view, parses the JSON header, and points every tensor's
/// TensorInfo::data into the view. The file HANDLE is closed right after
/// mapping (the mapping object keeps the file referenced). All tensor data
/// pointers handed out remain valid for the lifetime of this object; there is
/// no bounds re-checking of tensor offsets against the data region after the
/// "header larger than file" guard.
class SafeTensorsFile {
  public:
    /// Opens \p path, maps it read-only, and parses the safetensors header.
    /// \throws std::runtime_error on open/mapping failure or when the header
    ///         length field claims more bytes than the file contains.
    explicit SafeTensorsFile(const std::filesystem::path& path);

    /// Unmaps the view and closes the file mapping.
    ~SafeTensorsFile();

    /// Not copyable: the object owns exactly one mmap view.
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    /// Not copyable: the object owns exactly one mmap view.
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    /// All tensors from the header, keyed by tensor name.
    const std::map<std::string, TensorInfo>& tensors() const { return tensors_; }
    /// Contents of the header's "__metadata__" object (empty when absent).
    const std::map<std::string, std::string>& metadata() const { return metadata_; }
    /// Total file size in bytes.
    uint64_t file_size() const { return file_size_; }
    /// Byte offset in the file where the tensor data region starts (8 + header_len).
    uint64_t data_begin() const { return data_begin_; }

    /// Looks up a tensor by name.
    /// \param name Tensor name as it appears in the JSON header.
    /// \return Pointer to its TensorInfo, or nullptr when the name is unknown.
    const TensorInfo* find(const std::string& name) const {
        auto it = tensors_.find(name);
        return it == tensors_.end() ? nullptr : &it->second;
    }

  private:
    const uint8_t* base_ = nullptr;   ///< Start of the mapped view (nullptr until mapping succeeds).
    HANDLE map_ = nullptr;            ///< File-mapping handle; the file HANDLE itself is closed after MapViewOfFile.
    uint64_t file_size_ = 0;          ///< File size captured via GetFileSizeEx.
    uint64_t data_begin_ = 0;         ///< 8 + header_len: start of the tensor data region.
    std::map<std::string, TensorInfo> tensors_;  ///< Parsed per-tensor entries.
    std::map<std::string, std::string> metadata_; ///< Parsed "__metadata__" key/value pairs.
};

}  // namespace sal::st

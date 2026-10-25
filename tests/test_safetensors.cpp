// unit: safetensors mmap reader (synthetic fixture + real shard header when present)
/// \file
/// \brief Tests for st/safetensors.hpp's mmap reader.
///
/// Two tiers: (1) a synthetic fixture written by write_fixture() with an odd-length
/// header (exercises the header-size parsing) and mixed F32/BF16 tensors with exact
/// expected bytes; (2) the real model shard-62 header, checked only when the file is
/// present on disk (SKIP_IF otherwise) — pins the tensor count, file size, and that
/// every mmap offset stays inside the data region.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "st/safetensors.hpp"
#include "xlab/test.hpp"

/// \brief Writes a minimal two-tensor safetensors fixture to the temp directory:
///        F32 [2,3] at offsets [0,24) and BF16 [4,2] at [24,40), with an odd-length
///        JSON header (one trailing space) to exercise header-size handling.
/// \return Path of the written fixture; the caller removes it after the test.
static std::filesystem::path write_fixture() {
    auto path = std::filesystem::temp_directory_path() / "salikorn_test_fixture.safetensors";
    // header: two tensors — F32 [2,3] then BF16 [4,2] — with 16-byte-alignment padding offsets
    const std::string header =
        R"({"__metadata__":{"format":"salikorn-test"},"t_f32":{"dtype":"F32","shape":[2,3],"data_offsets":[0,24]},)"
        R"("t_bf16":{"dtype":"BF16","shape":[4,2],"data_offsets":[24,40]}})";
    std::string json(header);
    json.push_back(' ');  // odd length to exercise the parser
    uint64_t hlen = json.size();
    std::vector<uint8_t> data(40);
    float f32vals[6] = {0.f, 1.f, -2.f, 3.5f, -4.25f, 100.f};
    std::memcpy(data.data(), f32vals, 24);
    uint16_t bf16vals[8] = {0x3F80, 0x4000, 0xBF80, 0x4200, 0x0000, 0x8000, 0x1234, 0xABCD};
    std::memcpy(data.data() + 24, bf16vals, 16);
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(&hlen), 8);
    out.write(json.data(), json.size());
    out.write(reinterpret_cast<const char*>(data.data()), data.size());
    return path;
}

XLAB_TEST("synthetic fixture parses") {
    auto path = write_fixture();
    sal::st::SafeTensorsFile f{path};
    CHECK_EQ(f.tensors().size(), size_t(2));
    CHECK(f.metadata().count("format") && f.metadata().at("format") == "salikorn-test");
    const auto* a = f.find("t_f32");
    const auto* b = f.find("t_bf16");
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);
    CHECK(a->dtype == "F32");
    CHECK_EQ(a->shape.size(), size_t(2));
    CHECK_EQ(a->shape[0], uint64_t(2));
    CHECK_EQ(a->shape[1], uint64_t(3));
    CHECK_EQ(a->size_bytes, uint64_t(24));
    const float* fa = reinterpret_cast<const float*>(a->data);
    CHECK_EQ(fa[0], 0.0f);
    CHECK_EQ(fa[2], -2.0f);
    CHECK_EQ(fa[5], 100.0f);
    CHECK(b->dtype == "BF16");
    CHECK_EQ(b->numel(), uint64_t(8));
    const uint16_t* fb = reinterpret_cast<const uint16_t*>(b->data);
    CHECK_EQ(fb[0], uint16_t(0x3F80));
    CHECK_EQ(fb[6], uint16_t(0x1234));
    CHECK(f.find("does_not_exist") == nullptr);
    std::filesystem::remove(path);
}

XLAB_TEST("real shard-62 header validates") {
    const char* kPath = "models/GLM-5.3-Flash/model-00062-of-00062.safetensors";
    SKIP_IF(!std::filesystem::exists(kPath));
    sal::st::SafeTensorsFile f{kPath};
    CHECK_EQ(f.tensors().size(), size_t(351));
    CHECK_EQ(f.file_size(), uint64_t(1261584968));
    const auto* merger = f.find("model.visual.merger.down_proj.weight");
    REQUIRE(merger != nullptr);
    CHECK(merger->dtype == "BF16");
    CHECK_EQ(merger->shape[0], uint64_t(4096));
    CHECK_EQ(merger->shape[1], uint64_t(10240));
    CHECK_EQ(merger->size_bytes, uint64_t(4096 * 10240 * 2));
    // mmap offsets stay inside the data region
    uint64_t data_bytes = f.file_size() - f.data_begin();
    for (auto& [name, t] : f.tensors()) CHECK(t.offset + t.size_bytes <= data_bytes);
}

/// \brief Entry point: runs the "safetensors" suite (JSONL events under runs/).
/// \return 0 iff all tests passed (CTest-compatible).
int main() { return xlab::test::run_all("safetensors"); }

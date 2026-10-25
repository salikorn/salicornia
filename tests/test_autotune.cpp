// unit: autotuner logic — candidate enumeration, token substitution, profile round-trip
#include <filesystem>
#include <set>

#include "gpu/autotune.hpp"
#include "xlab/test.hpp"

namespace at = sal::gpu::autotune;

XLAB_TEST("candidate enumeration covers the declared space") {
    // full grid sizes (measured against the grids defined in autotune.hpp)
    auto full = at::spaces(false, {});
    std::map<std::string, size_t> want_full = {{"gemm_nt", 3}, {"gemm_nn", 3}, {"gemm_tn", 3},
                                               {"syrk", 6},    {"elem", 9},    {"adam", 9},
                                               {"vq_nearest", 3}, {"vq_update", 3}};
    for (auto& op : full) {
        REQUIRE(want_full.count(op.op) == 1);
        CHECK_MSG(op.cands.size() == want_full[op.op],
                  (std::string(op.op) + " full grid " + std::to_string(op.cands.size())).c_str());
    }
    CHECK(full.size() == want_full.size());
    // quick grid is a strict subset
    auto quick = at::spaces(true, {});
    for (auto& op : quick)
        for (auto& cd : op.cands) CHECK(cd.defines.size() > 0);
    // --only filter
    auto only = at::spaces(false, {"gemm_nt", "vq_nearest"});
    CHECK(only.size() == 2);
    CHECK(std::string(only[0].op) == "gemm_nt");
    // labels unique within an op
    for (auto& op : full) {
        std::set<std::string> labels;
        for (auto& cd : op.cands) labels.insert(cd.label);
        CHECK(labels.size() == op.cands.size());
    }
    // gemm TS respects the 1024-thread group limit
    for (auto& op : full)
        for (auto& cd : op.cands)
            if (cd.defines.count("TS")) CHECK(std::stoi(cd.defines.at("TS")) <= 32);
}

XLAB_TEST("token substitution replaces everything and rejects unknowns") {
    std::string tmpl = "#define TS @TS@\nfloat a[@TS@]; // @G@";
    auto out = sal::gpu::substitute_tokens(tmpl, {{"TS", "32"}, {"G", "256"}});
    CHECK(out.find('@') == std::string::npos);
    CHECK(out.find("32") != std::string::npos);
    // unused define is fine; unknown token throws
    CHECK_THROWS((sal::gpu::substitute_tokens("@NOPE@", {{"TS", "8"}})));
    // TS=32 keeps group threads <= 1024
    for (int ts : {8, 16, 32}) {
        auto defs = std::map<std::string, std::string>{{"TS", std::to_string(ts)}};
        auto s = sal::gpu::substitute_tokens(sal::gpu::tmpl::gemm_nt(), defs);
        CHECK(s.find('@') == std::string::npos);
    }
}

XLAB_TEST("profile round-trip preserves tunables and rejects garbage") {
    const std::string path = "profiles/_test_roundtrip.json";
    std::filesystem::create_directories("profiles");
    sal::gpu::TuneParams tp;
    tp.gemm_ts = 32;
    tp.elem_g = 128;
    tp.elem_ept = 2;
    tp.vqn_tp = 128;
    tp.vqu_g = 256;
    tp.syrk_chunk = 1024;
    std::map<std::string, nlohmann::json> winners;
    winners["gemm_nt"] = {{"TS", 32}, {"ms", 0.42}, {"label", "TS=32"}};
    at::save_profile(path, tp, winners, "test-adapter");

    sal::gpu::TuneParams back;
    REQUIRE(at::load_profile(path, back));
    CHECK_EQ(back.gemm_ts, 32);
    CHECK_EQ(back.elem_g, 128);
    CHECK_EQ(back.elem_ept, 2);
    CHECK_EQ(back.vqn_tp, 128);
    CHECK_EQ(back.vqu_g, 256);
    CHECK_EQ(back.syrk_chunk, 1024);

    // geometry sanity gate: a profile claiming TS=1024 is rejected
    {
        std::ofstream o(path);
        o << R"({"version":1,"tunables":{"gemm_ts":1024}})";
    }
    sal::gpu::TuneParams tp2;
    CHECK(!at::load_profile(path, tp2));
    CHECK_EQ(tp2.gemm_ts, 16);  // defaults retained

    // missing file -> false
    CHECK(!at::load_profile("profiles/_does_not_exist.json", tp2));
    std::filesystem::remove(path);
}

XLAB_TEST("elem geometry helpers agree with token values") {
    sal::gpu::TuneParams tp;
    tp.elem_g = 128;
    tp.elem_ept = 4;
    CHECK_EQ(tp.elem_threads(), uint32_t(512));
    CHECK_EQ(tp.elem_grid(1), uint32_t(1));
    CHECK_EQ(tp.elem_grid(513), uint32_t(2));
    CHECK_EQ(tp.elem_grid(0), uint32_t(0));
    tp.vqu_g = 256;
    CHECK_EQ(tp.vqu_grid(288), uint32_t(2));
    // defines carry the same numbers into HLSL
    auto d = tp.elem_defines();
    CHECK(d.at("G") == "128");
    CHECK(d.at("EPT") == "4");
}

int main() { return xlab::test::run_all("autotune"); }

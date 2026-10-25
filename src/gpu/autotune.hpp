// salikorn / gpu / autotune — tuning-parameter space, candidate enumeration, and
// profiles/best.json I/O shared by tools/autotune.cpp and production kernel sets.
#pragma once

#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gpu/kernels.hpp"

namespace sal::gpu::autotune {

// One launch-geometry configuration for one op.
struct Cand {
    std::map<std::string, std::string> defines;  // HLSL @TOKEN@ substitutions
    std::string label;                           // e.g. "TS=32"
    // dispatch-level params (not compile-time):
    int syrk_chunk = 2048;
    int ept = 1;  // elements per thread for the elem family (also in defines as @EPT@)
};

struct OpSpace {
    const char* op;
    std::vector<Cand> cands;
};

// Candidate grids. Full = exhaustive over the declared space; quick = pruned to the
// historically competitive half. Keep grids in one place — test_autotune checks counts.
inline std::vector<int> ts_grid(bool quick) { return quick ? std::vector<int>{16, 32}
                                                            : std::vector<int>{8, 16, 32}; }
inline std::vector<int> g_grid(bool quick) { return quick ? std::vector<int>{128, 256}
                                                           : std::vector<int>{64, 128, 256}; }
inline std::vector<int> ept_grid(bool quick) { return quick ? std::vector<int>{1, 2}
                                                             : std::vector<int>{1, 2, 4}; }
inline std::vector<int> tp_grid(bool quick) { return quick ? std::vector<int>{64, 128}
                                                            : std::vector<int>{32, 64, 128}; }
inline std::vector<int> chunk_grid(bool quick) { return quick ? std::vector<int>{2048}
                                                               : std::vector<int>{1024, 2048}; }

inline std::vector<OpSpace> spaces(bool quick, const std::vector<std::string>& only) {
    auto wanted = [&](const char* op) {
        if (only.empty()) return true;
        return std::find(only.begin(), only.end(), std::string(op)) != only.end();
    };
    std::vector<OpSpace> out;
    auto gemm_cands = [&] {
        std::vector<Cand> v;
        for (int ts : ts_grid(quick)) v.push_back({{{"TS", std::to_string(ts)}},
                                                   "TS=" + std::to_string(ts)});
        return v;
    };
    auto elem_cands = [&] {
        std::vector<Cand> v;
        for (int g : g_grid(quick))
            for (int e : ept_grid(quick))
                v.push_back({{{"G", std::to_string(g)}, {"EPT", std::to_string(e)}},
                             "G=" + std::to_string(g) + ",EPT=" + std::to_string(e), 2048, e});
        return v;
    };
    if (wanted("gemm_nt")) out.push_back({"gemm_nt", gemm_cands()});
    if (wanted("gemm_nn")) out.push_back({"gemm_nn", gemm_cands()});
    if (wanted("gemm_tn")) out.push_back({"gemm_tn", gemm_cands()});
    if (wanted("syrk")) {
        std::vector<Cand> v;
        for (int ts : ts_grid(quick))
            for (int ch : chunk_grid(quick))
                v.push_back({{{"TS", std::to_string(ts)}},
                             "TS=" + std::to_string(ts) + ",chunk=" + std::to_string(ch), ch});
        out.push_back({"syrk", v});
    }
    if (wanted("elem")) out.push_back({"elem", elem_cands()});
    if (wanted("adam")) out.push_back({"adam", elem_cands()});
    if (wanted("vq_nearest")) {
        std::vector<Cand> v;
        for (int tp : tp_grid(quick))
            v.push_back({{{"TP", std::to_string(tp)}}, "TP=" + std::to_string(tp)});
        out.push_back({"vq_nearest", v});
    }
    if (wanted("vq_update")) {
        std::vector<Cand> v;
        for (int g : g_grid(quick))
            v.push_back({{{"G", std::to_string(g)}}, "G=" + std::to_string(g)});
        out.push_back({"vq_update", v});
    }
    return out;
}

// Apply one candidate to a TuneParams (dispatch-level fields only; defines are per-kernel).
inline void apply_dispatch(TuneParams& tp, const Cand& c) {
    tp.syrk_chunk = c.syrk_chunk;
    tp.elem_ept = c.ept;
}

// Serialize. winners: op -> {param key -> value, "ms": best median, "label": ...}.
inline void save_profile(const std::string& path, const TuneParams& tp,
                         const std::map<std::string, nlohmann::json>& winners,
                         const std::string& adapter) {
    nlohmann::json j;
    j["version"] = 1;
    j["adapter"] = adapter;
    j["tunables"] = {{"gemm_ts", tp.gemm_ts},
                     {"elem_g", tp.elem_g},
                     {"elem_ept", tp.elem_ept},
                     {"vqn_tp", tp.vqn_tp},
                     {"vqu_g", tp.vqu_g},
                     {"syrk_chunk", tp.syrk_chunk}};
    j["winners"] = winners;
    std::ofstream out(path);
    out << j.dump(2) << "\n";
}

// Returns false when the file is missing or unparsable (caller keeps defaults).
inline bool load_profile(const std::string& path, TuneParams& tp) {
    if (!std::filesystem::exists(path)) return false;
    try {
        nlohmann::json j = nlohmann::json::parse(std::ifstream(path));
        auto t = j.at("tunables");
        TuneParams cand;  // parse into a temporary; commit only when fully valid
        cand.gemm_ts = t.value("gemm_ts", tp.gemm_ts);
        cand.elem_g = t.value("elem_g", tp.elem_g);
        cand.elem_ept = t.value("elem_ept", tp.elem_ept);
        cand.vqn_tp = t.value("vqn_tp", tp.vqn_tp);
        cand.vqu_g = t.value("vqu_g", tp.vqu_g);
        cand.syrk_chunk = t.value("syrk_chunk", tp.syrk_chunk);
        if (cand.gemm_ts < 4 || cand.gemm_ts > 32 || cand.elem_g < 32 || cand.elem_g > 1024 ||
            cand.elem_ept < 1 || cand.elem_ept > 8 || cand.vqn_tp < 8 || cand.vqn_tp > 256 ||
            cand.vqu_g < 8 || cand.vqu_g > 1024 || cand.syrk_chunk < 64 ||
            cand.syrk_chunk > 65536)
            return false;
        tp = cand;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace sal::gpu::autotune

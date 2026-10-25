/** \file
 *  \brief Tuning-parameter space, candidate enumeration, and profiles/best.json
 *         I/O shared by tools/autotune.cpp and production kernel sets.
 *
 * Candidate grids/spaces stay inline here; the profile I/O implementations live in
 * autotune.cpp. <nlohmann/json.hpp> remains because save_profile's signature
 * exposes nlohmann::json.
 */
#pragma once

#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "gpu/kernels.hpp"

namespace sal::gpu::autotune {

/// \brief One launch-geometry configuration for one op: the compile-time
///        defines to substitute into the shader template plus the dispatch-level
///        parameters the tuner varies alongside them.
struct Cand {
    std::map<std::string, std::string> defines;  ///< HLSL &#64;TOKEN&#64; substitutions
    std::string label;                           ///< human-readable id, e.g. "TS=32"
    int syrk_chunk = 2048; ///< dispatch-level params (not compile-time): syrk output-column chunk
    int ept = 1;  ///< elements per thread for the elem family (also in defines as &#64;EPT&#64;)
};

/// \brief The candidate list for one op, keyed by the op name used in profiles
///        and in spaces()' `only` filter.
struct OpSpace {
    const char* op;
    std::vector<Cand> cands;
};

// Candidate grids. Full = exhaustive over the declared space; quick = pruned to the
// historically competitive half. Keep grids in one place — test_autotune checks counts.

/// \brief GEMM/syrk tile-edge candidates: {8,16,32} full, {16,32} quick.
inline std::vector<int> ts_grid(bool quick) { return quick ? std::vector<int>{16, 32}
                                                            : std::vector<int>{8, 16, 32}; }
/// \brief Elementwise group-size candidates: {64,128,256} full, {128,256} quick.
inline std::vector<int> g_grid(bool quick) { return quick ? std::vector<int>{128, 256}
                                                           : std::vector<int>{64, 128, 256}; }
/// \brief Elements-per-thread candidates: {1,2,4} full, {1,2} quick.
inline std::vector<int> ept_grid(bool quick) { return quick ? std::vector<int>{1, 2}
                                                             : std::vector<int>{1, 2, 4}; }
/// \brief vq_nearest threads-per-group candidates: {32,64,128} full, {64,128} quick.
inline std::vector<int> tp_grid(bool quick) { return quick ? std::vector<int>{64, 128}
                                                            : std::vector<int>{32, 64, 128}; }
/// \brief syrk dispatch-chunk candidates: {1024,2048} full, {2048} quick.
inline std::vector<int> chunk_grid(bool quick) { return quick ? std::vector<int>{2048}
                                                               : std::vector<int>{1024, 2048}; }

/** \brief Enumerate the tuning space as one OpSpace per requested op.
 *
 * Candidate-gating protocol, producer side: each op gets only the candidate
 * family its shader templates accept (gemm_nt/nn/tn share the TS grid; syrk
 * pairs TS with a dispatch-level chunk; elem/adam sweep the G x EPT cross
 * product; vq_nearest sweeps TP; vq_update sweeps G). `quick` prunes each grid
 * to the historically competitive half so a sweep fits in interactive time.
 *
 * \param quick  prune to the reduced grids.
 * \param only   op names to include; empty means all ops.
 */
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

/// \brief Apply one candidate's dispatch-level fields to a TuneParams (syrk_chunk,
///        elem_ept). The compile-time defines are NOT applied here — they are
///        per-kernel and already baked into the compiled Kernel objects the
///        caller rebuilds per candidate.
inline void apply_dispatch(TuneParams& tp, const Cand& c) {
    tp.syrk_chunk = c.syrk_chunk;
    tp.elem_ept = c.ept;
}

/// \brief Write a tuning profile as JSON: schema version (1), the adapter name,
///        the full TuneParams snapshot under "tunables", and the per-op winners
///        map. Caller's job to pick a writable `path` (convention: profiles/best.json).
void save_profile(const std::string& path, const TuneParams& tp,
                  const std::map<std::string, nlohmann::json>& winners,
                  const std::string& adapter);

/** \brief Read a tuning profile into `tp` — the consumer side of the
 *         candidate-gating protocol.
 *
 * Gate: the "tunables" block is parsed into a temporary TuneParams and committed
 * to `tp` only when every field is inside the hard validity ranges
 * (gemm_ts 4..32 — the 1024-thread group limit makes 32 the ceiling for the
 * TS x TS group; elem_g 32..1024; elem_ept 1..8; vqn_tp 8..256; vqu_g 8..1024;
 * syrk_chunk 64..65536). Missing keys fall back to `tp`'s current values.
 *
 * \return True on a fully valid parse; false when the file is missing,
 *         unparsable, or any value is out of range (caller keeps defaults).
 */
bool load_profile(const std::string& path, TuneParams& tp);

}  // namespace sal::gpu::autotune

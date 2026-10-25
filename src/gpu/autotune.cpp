/** \file
 *  \brief profiles/best.json I/O implementations declared in autotune.hpp.
 *
 * Bodies moved verbatim from the former header-only definitions; the documented
 * contracts (winners-map shape, validation ranges, all-or-nothing commit) live
 * on the declarations in autotune.hpp. load_profile() catches every exception
 * from parsing and returns false rather than throwing, so production startup
 * can treat a bad profile as "no profile".
 */
#include "gpu/autotune.hpp"

#include <filesystem>
#include <fstream>

namespace sal::gpu::autotune {

void save_profile(const std::string& path, const TuneParams& tp,
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

bool load_profile(const std::string& path, TuneParams& tp) {
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

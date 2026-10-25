// salikorn / compress — bake-off result parsing shared by spikes, tests, and tools.
#pragma once

#include <string>
#include <vector>

namespace sal {

struct CodecPoint {
    std::string codec;
    double bpw = 0.0;
    double rel_mse = 0.0;
};

// Parses spike_vqvae --run output lines of the form
//   [gpu_codec_point] "codec":"ae_L64","bpw":4.117,"rel_mse":7.48e-01,...
// NaN/Inf rel_mse values are preserved as non-finite doubles (callers check isfinite).
inline std::vector<CodecPoint> parse_codec_points(const std::string& out) {
    std::vector<CodecPoint> pts;
    for (auto it = out.cbegin(); it != out.cend();) {
        static const std::string tag = "[gpu_codec_point] \"codec\":\"";
        auto pos = out.find(tag, size_t(it - out.cbegin()));
        if (pos == std::string::npos) break;
        auto name_b = pos + tag.size();
        auto name_e = out.find('"', name_b);
        if (name_e == std::string::npos) break;
        CodecPoint p;
        p.codec = out.substr(name_b, name_e - name_b);
        auto bpw_k = out.find("\"bpw\":", name_e);
        auto mse_k = out.find("\"rel_mse\":", name_e);
        if (bpw_k == std::string::npos || mse_k == std::string::npos) break;
        p.bpw = std::stod(out.substr(bpw_k + 6));
        p.rel_mse = std::stod(out.substr(mse_k + 10));
        pts.push_back(std::move(p));
        it = out.cbegin() + mse_k + 10;
    }
    return pts;
}

}  // namespace sal

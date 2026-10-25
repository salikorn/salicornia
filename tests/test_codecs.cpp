// unit: codec result parsing + int8 RTN + bpw arithmetic
#include <cmath>
#include <cstdint>
#include <string>

#include "compress/codec_points.hpp"
#include "xlab/test.hpp"

using sal::parse_codec_points;

XLAB_TEST("codec point parser: regular, NaN, and truncated output") {
    const std::string sample =
        "[env] \"adapter\":\"Intel(R) UHD Graphics\"\n"
        "[gpu_codec_point] \"codec\":\"ae_L64\",\"bpw\":4.117,\"rel_mse\":7.483640e-01,\"train_s\":25.4\n"
        "[gpu_codec_point] \"codec\":\"vae_L64_b1e-4\",\"bpw\":8.117,\"rel_mse\":9.957349e-01\n"
        "[gpu_codec_point] \"codec\":\"vqvae_K256_L64\",\"bpw\":0.156,\"rel_mse\":-nan(ind)\n"
        "[gpu_codec_point] \"codec\":\"vqvae_K4096_L64\",\"bpw\":0.289,\"rel_mse\":9.665478e-01\n"
        "[done] \"status\":\"ok\"";
    auto pts = parse_codec_points(sample);
    REQUIRE(pts.size() == 4);
    CHECK(pts[0].codec == "ae_L64");
    CHECK_CLOSE(pts[0].bpw, 4.117, 1e-9);
    CHECK_CLOSE(pts[0].rel_mse, 0.7483640, 1e-9);
    CHECK(!std::isfinite(pts[2].rel_mse));  // NaN preserved, detectable
    CHECK(std::isfinite(pts[3].rel_mse));
    // truncated output mid-value parses what completed
    auto partial = parse_codec_points(sample.substr(0, sample.find("vqvae_K4096")));
    CHECK(partial.size() == 3);
    // no points from unrelated output
    CHECK(parse_codec_points("[env] \"adapter\":\"x\"\n[done]").empty());
}

XLAB_TEST("int8 RTN quantization math") {
    auto q = [](float x, float scale) {
        float clamped = std::max(-scale, std::min(scale, x));
        return std::round(clamped / scale * 127.0f) / 127.0f * scale;
    };
    CHECK_CLOSE(q(0.5f, 1.0f), 0.5f, 1.0f / 127.0f);
    CHECK_CLOSE(q(5.0f, 1.0f), 1.0f, 1e-9);    // clamps at the scale
    CHECK_CLOSE(q(-5.0f, 1.0f), -1.0f, 1e-9);
    CHECK_EQ(q(0.0f, 1.0f), 0.0f);
    double max_err = 0;
    for (int i = -1000; i <= 1000; ++i) {
        float x = float(i) / 1000.0f;  // within [-1, 1] = the scale
        max_err = std::max(max_err, double(std::fabs(q(x, 1.0f) - x)));
    }
    CHECK(max_err <= 0.5 / 127.0 + 1e-6);  // half-step bound
}

XLAB_TEST("bpw arithmetic") {
    // AE with fp16 latents: L*16 bits per 256-elem patch + shared decoder amortized
    const double npatch = 131072.0;
    const double shared =
        (384.0 * 256 + 384.0 * 64 + 64.0 * 384 + 256.0 * 384) * 16.0 / npatch / 256.0;
    const double bpw_L64 = 64.0 * 16.0 / 256.0 + shared;
    CHECK(bpw_L64 > 4.0 && bpw_L64 < 4.3);  // matches the measured 4.117
    // VQ-VAE: log2(K) index bits per patch + codebook + shared
    const double bpw_K256 = std::log2(256.0) / 256.0 + 256.0 * 64 * 16.0 / npatch / 256.0 + shared;
    CHECK(bpw_K256 > 0.1 && bpw_K256 < 0.2);  // matches the measured 0.156
}

int main() { return xlab::test::run_all("codecs"); }

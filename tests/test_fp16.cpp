// unit: fp16 / bf16 / fp8-e4m3 conversions (exhaustive round-trips + known values + 10M random)
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>

#include "xlab/fp16.hpp"
#include "xlab/test.hpp"

using sal::bf16_to_f32;
using sal::f16_to_f32;
using sal::f32_to_bf16;
using sal::f32_to_f16;
using sal::f8e4m3_to_f32;

XLAB_TEST("fp16 known bit patterns") {
    CHECK_EQ(f32_to_f16(0.0f), uint16_t(0x0000));
    CHECK_EQ(f32_to_f16(-0.0f), uint16_t(0x8000));
    CHECK_EQ(f32_to_f16(1.0f), uint16_t(0x3C00));
    CHECK_EQ(f32_to_f16(0.5f), uint16_t(0x3800));
    CHECK_EQ(f32_to_f16(2.0f), uint16_t(0x4000));
    CHECK_EQ(f32_to_f16(65504.0f), uint16_t(0x7BFF));      // max normal
    CHECK_EQ(f32_to_f16(6.0975552e-5f), uint16_t(0x03FF));  // max subnormal
    CHECK_EQ(f32_to_f16(5.9604645e-8f), uint16_t(0x0001));  // min subnormal
    const uint16_t inf16 = f32_to_f16(std::numeric_limits<float>::infinity());
    CHECK_EQ(inf16, uint16_t(0x7C00));
    CHECK(std::isnan(f16_to_f32(f32_to_f16(std::numeric_limits<float>::quiet_NaN()))));
    CHECK_EQ(f16_to_f32(0x3C00), 1.0f);
    CHECK_EQ(f16_to_f32(0x7BFF), 65504.0f);
    CHECK_EQ(f16_to_f32(0x0001), 5.9604644775390625e-08f);
}

XLAB_TEST("fp16 exhaustive round-trip f16->f32->f16") {
    int bad = 0, nans = 0;
    for (uint32_t h = 0; h <= 0xFFFF; ++h) {
        uint16_t v = uint16_t(h);
        float f = f16_to_f32(v);
        uint16_t back = f32_to_f16(f);
        if (std::isnan(f)) {
            ++nans;
            continue;  // NaN payloads may canonicalize; only NaN-ness is guaranteed
        }
        if (back != v) ++bad;
    }
    CHECK_MSG(bad == 0, ("identity failed for " + std::to_string(bad) + " of 65536 values").c_str());
    CHECK(nans == 2046);  // exponent=31 with 1023 non-zero mantissas per sign
}

XLAB_TEST("fp16 random f32->f16->f32 error bound") {
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-60000.f, 60000.f);
    double max_rel = 0;
    int overflow = 0;
    for (int i = 0; i < 10'000'000; ++i) {
        float x = u(rng);
        float y = f16_to_f32(f32_to_f16(x));
        if (x == 0) continue;
        if (!std::isfinite(y)) {
            ++overflow;  // |x| beyond 65504 rounds to Inf
            continue;
        }
        double rel = std::fabs(double(y) - double(x)) / std::fabs(double(x));
        if (std::fabs(x) > 6.1e-5) max_rel = std::max(max_rel, rel);  // normals: <= 2^-11
    }
    CHECK(max_rel <= 4.883e-4 + 1e-9);  // 2^-11 half-ulp bound
    (void)overflow;  // range is within fp16 max, so nothing overflows
}

XLAB_TEST("bf16 exhaustive round-trip") {
    int bad = 0;
    for (uint32_t h = 0; h <= 0xFFFF; ++h) {
        uint16_t v = uint16_t(h);
        uint16_t back = f32_to_bf16(bf16_to_f32(v));
        if (back != v) ++bad;  // bf16 -> f32 is exact; rounding back is identity
    }
    CHECK_MSG(bad == 0, ("bf16 identity failed for " + std::to_string(bad) + " values").c_str());
}

XLAB_TEST("bf16 known values") {
    CHECK_EQ(bf16_to_f32(0x3F80), 1.0f);
    CHECK_EQ(bf16_to_f32(0xBF80), -1.0f);
    CHECK_EQ(bf16_to_f32(0x0000), 0.0f);
    CHECK_EQ(f32_to_bf16(1.0f), uint16_t(0x3F80));
    CHECK_EQ(f32_to_bf16(-2.0f), uint16_t(0xC000));
    CHECK_EQ(f32_to_bf16(1.0078125f), uint16_t(0x3F81));       // exactly representable (2^-7 step)
    CHECK_EQ(f32_to_bf16(1.00390625f), uint16_t(0x3F80));      // tie -> even (down)
    CHECK_EQ(f32_to_bf16(1.01171875f), uint16_t(0x3F82));      // tie -> even (up: mantissa 2)
    CHECK_EQ(f32_to_bf16(1.005859375f), uint16_t(0x3F81));     // 0.75 step -> nearest
}

XLAB_TEST("fp8 e4m3 exhaustive range + known values") {
    CHECK_EQ(f8e4m3_to_f32(0), 0.0f);
    CHECK_EQ(f8e4m3_to_f32(0x38), 1.0f);              // e=7, m=0, bias 7 -> 2^0
    CHECK_EQ(f8e4m3_to_f32(0x38 ^ 0x80), -1.0f);
    CHECK_CLOSE(f8e4m3_to_f32(0x39), 1.125, 1e-9);    // e=7, m=1
    CHECK_CLOSE(f8e4m3_to_f32(0x08), 0.015625, 1e-12);  // e=1, m=0 -> 2^-6
    CHECK_CLOSE(f8e4m3_to_f32(0x01), 0.001953125, 1e-12);  // subnormal m=1 -> 2^-9
    int nans = 0;
    for (uint32_t v = 0; v <= 0xFF; ++v) {
        float f = f8e4m3_to_f32(uint8_t(v));
        if (std::isnan(f)) {
            ++nans;
            continue;
        }
        CHECK_MSG(std::fabs(f) <= 448.0 + 1e-6,
                  ("v=0x" + std::to_string(v) + " -> " + std::to_string(f)).c_str());
    }
    CHECK_MSG(nans == 2, ("nan count " + std::to_string(nans)).c_str());  // 0x7F, 0xFF
}

int main() { return xlab::test::run_all("fp16"); }

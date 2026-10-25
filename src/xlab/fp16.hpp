// salikorn / xlab — IEEE 754 half + bfloat16 conversions (shared across the project).
#pragma once

#include <cstdint>
#include <cstring>

namespace sal {

inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t e = int32_t((x >> 23) & 0xFFu);
    uint32_t m = x & 0x7FFFFFu;
    if (e == 255) return uint16_t(sign | 0x7C00u | (m ? 0x200u : 0u));  // Inf/NaN
    e = e - 127 + 15;
    if (e >= 31) return uint16_t(sign | 0x7C00u);                       // overflow -> Inf
    if (e <= 0) {
        if (e < -10) return uint16_t(sign);                             // underflow -> 0
        m |= 0x800000u;                                                 // 24-bit significand
        uint32_t shift = 14u - uint32_t(e);
        uint32_t kept = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t half = 1u << (shift - 1u);
        if (rem > half || (rem == half && (kept & 1u))) kept += 1;
        return uint16_t(sign | kept);
    }
    uint32_t kept = m >> 13;
    uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (kept & 1u))) {
        kept += 1;
        if (kept == 0x400u) {
            kept = 0;
            e += 1;
            if (e >= 31) return uint16_t(sign | 0x7C00u);
        }
    }
    return uint16_t(sign | (uint32_t(e) << 10) | kept);
}

inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h >> 15) & 1u;
    uint32_t e = uint32_t(h >> 10) & 0x1Fu;
    uint32_t m = uint32_t(h) & 0x3FFu;
    uint32_t x;
    if (e == 0) {
        if (m == 0) {
            x = sign << 31;
        } else {  // subnormal: normalize
            int k = -1;
            while (!(m & 0x400u)) {
                m <<= 1;
                k += 1;
            }
            m &= 0x3FFu;
            x = (sign << 31) | (uint32_t(127 - 15 - k) << 23) | (m << 13);
        }
    } else if (e == 31) {
        x = (sign << 31) | 0x7F800000u | (m << 13);
    } else {
        x = (sign << 31) | ((e - 15 + 127) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

inline float bf16_to_f32(uint16_t h) {
    uint32_t x = uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &x, 4);
    return f;
}

inline uint16_t f32_to_bf16(float f) {  // round-to-nearest-even, no denormal care (bf16 range == f32)
    uint32_t x;
    std::memcpy(&x, &f, 4);
    uint32_t lsb = (x >> 16) & 1u;
    x += 0x7FFFu + lsb;
    return uint16_t(x >> 16);
}

// FP8 e4m3 (safetensors "F8_E4M3") -> float. Layout: s eeee mmm.
inline float f8e4m3_to_f32(uint8_t v) {
    uint32_t sign = uint32_t(v >> 7) & 1u;
    uint32_t e = uint32_t(v >> 3) & 0xFu;
    uint32_t m = uint32_t(v) & 0x7u;
    // e4m3: no inf; 0b1111.111 is NaN; bias 7
    float out;
    if (e == 0) {
        if (m == 0) {
            out = sign ? -0.0f : 0.0f;
        } else {  // subnormal: m * 2^-9
            out = float(m) * (1.0f / 512.0f);
            if (sign) out = -out;
        }
    } else if (e == 15 && m == 7) {
        uint32_t bits = (sign << 31) | 0x7FC00000u;  // quiet NaN (mantissa bit required!)
        std::memcpy(&out, &bits, 4);
    } else {
        int32_t be = int32_t(e) - 7 + 127;
        uint32_t bits = (sign << 31) | (uint32_t(be) << 23) | (m << 20);
        std::memcpy(&out, &bits, 4);
    }
    return out;
}

}  // namespace sal

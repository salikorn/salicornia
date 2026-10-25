// salikorn / gpu / kernels — compute-shader library for weight-codec training
// (AE / VAE / VQ-VAE) and GPU linear algebra, with TUNABLE launch geometry.
//
// Every kernel's HLSL lives in sal::gpu::tmpl as a @TOKEN@-parameterized template;
// sal::gpu::TuneParams picks the values and the kernel structs compile them via
// Kernel(Device&, template, defines). autotune.exe sweeps the same templates and
// writes profiles/best.json; production code loads it (defaults when absent).
//
// Root signature (all kernels): [0]=SRV t0, [1]=SRV t1, [2]=UAV u0, [3]=eight 32-bit
// constants visible in-shader as cbuffer KC at register(b0) with members p (words 0-3)
// and q (words 4-7). Multi-output kernels pack outputs into the single UAV at element
// offsets; floats stored via asuint().
#pragma once

#include <map>
#include <string>

#include "gpu/cs.hpp"

namespace sal::gpu {

// ---------------------------------------------------------------------------
// Tunable parameters (one per launch-geometry axis). Defaults reproduce the
// hand-tuned kernels; profiles/best.json (written by autotune.exe) overrides.
// ---------------------------------------------------------------------------
struct TuneParams {
    int gemm_ts = 16;      // GEMM/syrk tile edge; group = TS x TS threads (max 32: 1024 limit)
    int elem_g = 256;      // elementwise-family group size
    int elem_ept = 1;      // elementwise elements per thread (contiguous run)
    int vqn_tp = 64;       // vq_nearest threads per patch-group (codes strided, group reduction)
    int vqu_g = 64;        // vq_update group size (one thread per code)
    int syrk_chunk = 2048; // syrk output-column chunk per dispatch packet (dispatch-level)

    uint32_t elem_threads() const { return uint32_t(elem_g * elem_ept); }
    uint32_t elem_grid(uint32_t count) const {
        return (count + elem_threads() - 1) / elem_threads();
    }
    uint32_t vqu_grid(uint32_t codes) const {
        return (codes + uint32_t(vqu_g) - 1) / uint32_t(vqu_g);
    }

    std::map<std::string, std::string> gemm_defines() const {
        return {{"TS", std::to_string(gemm_ts)}};
    }
    std::map<std::string, std::string> elem_defines() const {
        return {{"G", std::to_string(elem_g)},
                {"EPT", std::to_string(elem_ept)},
                {"BOUND", "p.x"}};
    }
    std::map<std::string, std::string> adam_defines() const {
        auto d = elem_defines();
        d["BOUND"] = "p.w";  // adam's element count lives in q-word 3
        return d;
    }
    std::map<std::string, std::string> pair_bound_defines() const {
        auto d = elem_defines();
        d["BOUND"] = "p.x * p.y";  // [B, IO]/[B, L] shaped elementwise kernels
        return d;
    }
    std::map<std::string, std::string> vqn_defines() const {
        return {{"TP", std::to_string(vqn_tp)}};
    }
    std::map<std::string, std::string> vqu_defines() const {
        return {{"G", std::to_string(vqu_g)}};
    }
};

// ---------------------------------------------------------------------------
// Tokenized shader templates. Bodies match the validated kernels; geometry
// (tile sizes, group sizes, elements-per-thread) is compile-time here.
// ---------------------------------------------------------------------------
namespace tmpl {

inline const char* gemm_nt() {
    return R"(#define TS @TS@
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // M, N, K | A off, B off, C off
groupshared float Ta[TS][TS];
groupshared float Tb[TS][TS];
[numthreads(TS,TS,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint M = p.x, N = p.y, K = p.z;
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < K; k0 += TS) {
        Ta[tid.y][tid.x] = (row < M && k0 + tid.x < K) ? A[q.x + row * K + k0 + tid.x] : 0.0;
        Tb[tid.x][tid.y] = (col < N && k0 + tid.y < K) ? B[q.y + col * K + k0 + tid.y] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < TS; ++t) acc += Ta[tid.y][t] * Tb[tid.x][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) C[q.z + row * N + col] = acc;
}
)";
}

inline const char* gemm_nn() {
    return R"(#define TS @TS@
StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
groupshared float Ta[TS][TS];
groupshared float Tb[TS][TS];
[numthreads(TS,TS,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint M = p.x, N = p.y, K = p.z;
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < K; k0 += TS) {
        Ta[tid.y][tid.x] = (row < M && k0 + tid.x < K) ? A[q.x + row * K + k0 + tid.x] : 0.0;
        Tb[tid.x][tid.y] = (col < N && k0 + tid.y < K) ? B[q.y + (k0 + tid.y) * N + col] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < TS; ++t) acc += Ta[tid.y][t] * Tb[tid.x][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) C[q.z + row * N + col] = acc;
}
)";
}

inline const char* gemm_tn() {
    return R"(#define TS @TS@
StructuredBuffer<float> A : register(t0);   // [K, M] row-major (read transposed)
StructuredBuffer<float> B : register(t1);   // [K, N] row-major
RWStructuredBuffer<float> C : register(u0); // [M, N]
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
groupshared float Ta[TS][TS];
groupshared float Tb[TS][TS];
[numthreads(TS,TS,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint M = p.x, N = p.y, K = p.z;
    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < K; k0 += TS) {
        Ta[tid.y][tid.x] = (row < M && k0 + tid.x < K) ? A[q.x + (k0 + tid.x) * M + row] : 0.0;
        Tb[tid.x][tid.y] = (col < N && k0 + tid.y < K) ? B[q.y + (k0 + tid.y) * N + col] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < TS; ++t) acc += Ta[tid.y][t] * Tb[tid.x][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) C[q.z + row * N + col] = acc;
}
)";
}

inline const char* syrk() {
    return R"(#define TS @TS@
StructuredBuffer<float> X : register(t0);
RWStructuredBuffer<float> G : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // m, n, rowbase (chunk over output rows)
groupshared float T1[TS][TS];
groupshared float T2[TS][TS];
[numthreads(TS,TS,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint m = p.x, n = p.y;
    uint c1 = p.z + gid.x * TS + tid.x;
    uint c2 = gid.y * TS + tid.y;
    float acc = 0.0;
    for (uint i0 = 0; i0 < m; i0 += TS) {
        T1[tid.x][tid.y] = (c1 < n && i0 + tid.y < m) ? X[(i0 + tid.y) * n + c1] : 0.0;
        T2[tid.y][tid.x] = (c2 < n && i0 + tid.x < m) ? X[(i0 + tid.x) * n + c2] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < TS; ++t) acc += T1[tid.x][t] * T2[tid.y][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (c1 < n && c2 < n) G[c1 * n + c2] = acc;
}
)";
}

// Elementwise family: thread t owns the EPT contiguous elements [t*EPT, (t+1)*EPT);
// grid = ceil(count / (G*EPT)) gives exact single coverage for any count.
#define SAL_ELEM_PRELUDE                                            \
    "[numthreads(@G@,1,1)]\n"                                       \
    "void cs_main(uint3 tid : SV_DispatchThreadID) {\n"             \
    "        [unroll] for (uint e = 0; e < @EPT@; ++e) {\n"         \
    "            uint i = tid.x * @EPT@ + e;\n"                     \
    "            if (i < @BOUND@) {\n"

inline const char* gelu() {
    return R"(StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> O : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
)" SAL_ELEM_PRELUDE R"(
    float x = A[i];
    const float k0 = 0.7978845608;
    O[i] = 0.5 * x * (1.0 + tanh(k0 * (x + 0.044715 * x * x * x)));
        }
    }
}
)";
}

inline const char* gelu_bwd() {
    return R"(StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> D : register(t1);
RWStructuredBuffer<float> G : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
)" SAL_ELEM_PRELUDE R"(
    float x = A[i];
    const float k0 = 0.7978845608;
    float x3 = x + 0.044715 * x * x * x;
    float t = tanh(k0 * x3);
    float dt = (1 - t * t) * k0 * (1 + 3 * 0.044715 * x * x);
    G[i] = D[i] * (0.5 * (1 + t) + 0.5 * x * dt);
        }
    }
}
)";
}

inline const char* mse_diff() {
    return R"(StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> D : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
)" SAL_ELEM_PRELUDE R"(
    D[i] = A[i] - B[i];
        }
    }
}
)";
}

inline const char* mse_grad() {
    return R"(StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> D : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // q.x = asfloat(scale)
)" SAL_ELEM_PRELUDE R"(
    D[i] = asfloat(q.x) * (A[i] - B[i]);
        }
    }
}
)";
}

inline const char* elem_add() {
    return R"(StructuredBuffer<float> A : register(t0);
StructuredBuffer<float> B : register(t1);
RWStructuredBuffer<float> C : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
)" SAL_ELEM_PRELUDE R"(
    C[i] = A[i] + B[i];
        }
    }
}
)";
}

inline const char* sum_sq() {  // single group; O[0] = sum(A^2)
    return R"(StructuredBuffer<float> A : register(t0);
RWStructuredBuffer<float> O : register(u0);
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
groupshared float acc[@G@];
[numthreads(@G@,1,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    float s = 0.0;
    for (uint i = tid.x; i < p.x; i += @G@) s += A[i] * A[i];
    acc[tid.x] = s;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        float t = 0.0;
        [unroll] for (uint i = 0; i < @G@; ++i) t += acc[i];
        O[0] = t;
    }
}
)";
}

inline const char* adam() {
    return R"(StructuredBuffer<float> G : register(t0);
RWStructuredBuffer<uint> S : register(u0);   // packed [P | M | V], floats as uint
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // poff, moff, voff, count | lr, step, eps, goff
)" SAL_ELEM_PRELUDE R"(
    const float b1 = 0.9, b2 = 0.999;
    float lr = asfloat(q.x), step = asfloat(q.y), eps = asfloat(q.z);
    float g = G[q.w + i];
    uint pi = p.x + i, mi = p.y + i, vi = p.z + i;
    float m = asfloat(S[mi]), v = asfloat(S[vi]);
    m = b1 * m + (1.0 - b1) * g;
    v = b2 * v + (1.0 - b2) * g * g;
    S[mi] = asuint(m);
    S[vi] = asuint(v);
    // update-norm clamp + non-finite grad guard: one step moves |dP| <= lr per element
    if (isfinite(g)) {
        float u = (m / (1.0 - pow(b1, step))) / (sqrt(v / (1.0 - pow(b2, step))) + eps);
        u = clamp(u, -1.0, 1.0);
        S[pi] = asuint(asfloat(S[pi]) - lr * u);
    }
        }
    }
}
)";
}

inline const char* vq_nearest() {
    return R"(StructuredBuffer<float> Z : register(t0);     // [B, L]
StructuredBuffer<float> CB : register(t1);    // [K, L]
RWStructuredBuffer<uint> OUT : register(u0);  // Q floats at +0, IDX uints at q.x
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
// Tiled: one @TP@-thread GROUP per patch; threads stride the K codes in parallel, then a
// groupshared reduction picks the argmin. Dispatch grid = B (one group per patch).
// NaN distances lose every `<` comparison, so non-finite inputs cannot poison the argmin.
groupshared float bdist[@TP@];
groupshared uint bidx[@TP@];
[numthreads(@TP@,1,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint b = gid.x;
    uint L = p.y, K = p.z;
    float best = 3.4e38; uint bk = 0;
    for (uint k = tid.x; k < K; k += @TP@) {
        float s = 0.0;
        for (uint l = 0; l < L; ++l) {
            float d = Z[b * L + l] - CB[k * L + l];
            s += d * d;
        }
        if (s < best) { best = s; bk = k; }
    }
    bdist[tid.x] = best;
    bidx[tid.x] = bk;
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0) {
        float bb = bdist[0]; uint wi = 0;
        for (uint i = 1; i < @TP@; ++i)
            if (bdist[i] < bb) { bb = bdist[i]; wi = i; }
        uint kbest = bidx[wi];
        for (uint l = 0; l < L; ++l)
            OUT[b * L + l] = asuint(CB[kbest * L + l]);
        OUT[q.x + b] = kbest;
    }
}
)";
}

inline const char* vq_update() {
    return R"(StructuredBuffer<float> Z : register(t0);     // [B, L]
StructuredBuffer<uint> IDX : register(t1);    // [B] (at element offset q.x)
RWStructuredBuffer<float> CB : register(u0);  // [K, L]
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};
// One thread per code; EMA update CB += gamma*(batch_mean - CB) for assigned codes
// (gamma in q.y; hard replacement oscillates against the moving encoder).
// NOTE acc[] caps L at 64 (all current configs use L = 64); empty codes unchanged.
[numthreads(@G@,1,1)]
void cs_main(uint3 tid : SV_DispatchThreadID) {
    uint k = tid.x;
    if (k >= p.z) return;
    uint B = p.x, L = p.y;
    float gamma = asfloat(q.y);
    float acc[64];
    [unroll] for (uint l = 0; l < 64; ++l) acc[l] = 0.0;
    uint cnt = 0;
    for (uint b = 0; b < B; ++b) {
        if (IDX[q.x + b] != k) continue;
        ++cnt;
        for (uint l = 0; l < L; ++l) acc[l] += Z[b * L + l];
    }
    if (cnt > 0) {
        for (uint l = 0; l < L; ++l) {
            float ema = CB[k * L + l] + gamma * (acc[l] / float(cnt) - CB[k * L + l]);
            if (isfinite(ema)) CB[k * L + l] = ema;
        }
    }
}
)";
}

inline const char* vq_commit() {
    return R"(StructuredBuffer<float> Z : register(t0);      // encoder output z [B, L]
StructuredBuffer<float> Q : register(t1);     // quantized codes (vqout Q region) [B, L]
RWStructuredBuffer<float> GZ : register(u0);  // grad wrt z (STE already in), committed in place
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // B*L | q.x = beta_eff
)" SAL_ELEM_PRELUDE R"(
    GZ[i] += asfloat(q.x) * (Z[i] - Q[i]);
        }
    }
}
)";
}

inline const char* vae_sample() {
    return R"(StructuredBuffer<float> PACK : register(t0);  // [mu(BL) | lam(BL) | eps(BL)]
RWStructuredBuffer<float> Z : register(u0);   // [B, L]
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // B, L, lam_off, eps_off
)" SAL_ELEM_PRELUDE R"(
    float lam = PACK[p.z + i];
    float eps = PACK[p.w + i];
    Z[i] = PACK[i] + eps * exp(0.5 * lam);
        }
    }
}
)";
}

inline const char* vae_grads() {
    return R"(StructuredBuffer<float> PACK : register(t0);  // [mu | lam | eps]
StructuredBuffer<float> GZ : register(t1);    // [B,L] grad wrt z
RWStructuredBuffer<uint> OUT : register(u0);  // g_mu at +0, g_lam at q.z (element offsets)
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // B, L, lam_off, eps_off | beta, invB, gl_off
)" SAL_ELEM_PRELUDE R"(
    float beta = asfloat(q.x), invB = asfloat(q.y);
    float mu = PACK[i];
    float lam = PACK[p.z + i];
    float eps = PACK[p.w + i];
    float sigma = exp(0.5 * lam);
    float gz = GZ[i];
    float g_mu = gz + beta * invB * mu;
    float g_lam = gz * eps * sigma * 0.5 + beta * invB * (exp(lam) - 1.0) * 0.5;
    OUT[i] = asuint(g_mu);
    OUT[q.z + i] = asuint(g_lam);
        }
    }
}
)";
}

inline const char* gather_rows() {
    return R"(StructuredBuffer<float> T : register(t0);   // train rows [N, IO]
StructuredBuffer<uint> IX : register(t1);    // selected rows [B]
RWStructuredBuffer<float> O : register(u0);  // batch [B, IO]
cbuffer KC : register(b0)
{
    uint4 p;
    uint4 q;
};   // B, IO
)" SAL_ELEM_PRELUDE R"(
    uint b = i / p.y;
    O[i] = T[q.x + uint(IX[b]) * p.y + (i % p.y)];
        }
    }
}
)";
}

}  // namespace tmpl

// ---------------------------------------------------------------------------
// Kernel sets (production entry points). Pass nullptr for defaults; load
// profiles/best.json into TuneParams to adopt autotuned geometry.
// ---------------------------------------------------------------------------
struct LinAlgKernels {
    Kernel gemm_nt;
    Kernel gemm_nn;
    Kernel gemm_tn;
    Kernel syrk;
    TuneParams p;

    explicit LinAlgKernels(Device& d, const TuneParams* tp = nullptr)
        : gemm_nt(d, tmpl::gemm_nt(), (tp ? *tp : TuneParams()).gemm_defines()),
          gemm_nn(d, tmpl::gemm_nn(), (tp ? *tp : TuneParams()).gemm_defines()),
          gemm_tn(d, tmpl::gemm_tn(), (tp ? *tp : TuneParams()).gemm_defines()),
          syrk(d, tmpl::syrk(), (tp ? *tp : TuneParams()).gemm_defines()),
          p(tp ? *tp : TuneParams()) {}

    // C[M,N] = A[M,K] * B[N,K]^T, with element offsets into packed buffers.
    void gemm_nt_rec(Device& d, Buffer& a, Buffer& b, Buffer& o, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t aoff = 0, uint32_t boff = 0, uint32_t coff = 0) {
        uint32_t c[8] = {M, N, K, 0, aoff, boff, coff, 0};
        d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gemm_nt.dispatch(d, a, b, o, c, (N + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts),
                         (M + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts), 1);
        d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON);
    }
    void gemm_nn_rec(Device& d, Buffer& a, Buffer& b, Buffer& o, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t aoff = 0, uint32_t boff = 0, uint32_t coff = 0) {
        uint32_t c[8] = {M, N, K, 0, aoff, boff, coff, 0};
        d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gemm_nn.dispatch(d, a, b, o, c, (N + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts),
                         (M + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts), 1);
        d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON);
    }
    void gemm_tn_rec(Device& d, Buffer& a, Buffer& b, Buffer& o, uint32_t M, uint32_t N,
                     uint32_t K, uint32_t aoff = 0, uint32_t boff = 0, uint32_t coff = 0) {
        uint32_t c[8] = {M, N, K, 0, aoff, boff, coff, 0};
        d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        gemm_tn.dispatch(d, a, b, o, c, (N + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts),
                         (M + uint32_t(p.gemm_ts) - 1) / uint32_t(p.gemm_ts), 1);
        d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON);
    }
};

struct TrainKernels {
    Kernel gelu, gelu_bwd, mse_diff, mse_grad, sum_sq, adam, elem_add;
    Kernel vq_nearest, vq_update, vq_commit, vae_sample, vae_grads, gather_rows;
    TuneParams p;

    explicit TrainKernels(Device& d, const TuneParams* tp = nullptr)
        : gelu(d, tmpl::gelu(), (tp ? *tp : TuneParams()).elem_defines()),
          gelu_bwd(d, tmpl::gelu_bwd(), (tp ? *tp : TuneParams()).elem_defines()),
          mse_diff(d, tmpl::mse_diff(), (tp ? *tp : TuneParams()).elem_defines()),
          mse_grad(d, tmpl::mse_grad(), (tp ? *tp : TuneParams()).elem_defines()),
          sum_sq(d, tmpl::sum_sq(), (tp ? *tp : TuneParams()).elem_defines()),
          adam(d, tmpl::adam(), (tp ? *tp : TuneParams()).adam_defines()),
          elem_add(d, tmpl::elem_add(), (tp ? *tp : TuneParams()).elem_defines()),
          vq_nearest(d, tmpl::vq_nearest(), (tp ? *tp : TuneParams()).vqn_defines()),
          vq_update(d, tmpl::vq_update(), (tp ? *tp : TuneParams()).vqu_defines()),
          vq_commit(d, tmpl::vq_commit(), (tp ? *tp : TuneParams()).elem_defines()),
          vae_sample(d, tmpl::vae_sample(), (tp ? *tp : TuneParams()).pair_bound_defines()),
          vae_grads(d, tmpl::vae_grads(), (tp ? *tp : TuneParams()).pair_bound_defines()),
          gather_rows(d, tmpl::gather_rows(), (tp ? *tp : TuneParams()).pair_bound_defines()),
          p(tp ? *tp : TuneParams()) {}

    // elementwise helper: transitions + dispatch (a,b inputs; o output).
    // Guards duplicate transitions when the same resource fills two slots.
    void elem(Device& d, Kernel& k, Buffer& a, Buffer& b, Buffer& o, const uint32_t c[8],
              uint32_t count) {
        bool b_unique = b.res() && b.res() != a.res();
        bool o_unique = o.res() != a.res() && (!b.res() || o.res() != b.res());
        d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (b_unique)
            d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                         D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (o_unique)
            d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        else if (o.res() == a.res())
            d.transition(o.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_UNORDERED_ACCESS);  // in-place same-index
        k.dispatch(d, a, b, o, c, p.elem_grid(count), 1, 1);
        d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        if (b_unique)
            d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                         D3D12_RESOURCE_STATE_COMMON);
        if (o_unique)
            d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON);
        else if (o.res() == a.res())
            d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                         D3D12_RESOURCE_STATE_COMMON);
    }
};

}  // namespace sal::gpu

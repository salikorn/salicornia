/** \file
 *  \brief Tokenized HLSL shader templates.
 *
 * Included by kernels.hpp before the kernel-set structs (their inline constructors
 * call these). Not compiled standalone. Bodies match the validated kernels;
 * geometry (tile sizes, group sizes, elements-per-thread) is compile-time here
 * via &#64;TOKEN&#64; substitution (see sal::gpu::substitute_tokens).
 *
 * Shared constant-block convention (all templates): cbuffer KC at register(b0)
 * holds `uint4 p` (words 0-3) and `uint4 q` (words 4-7); per-kernel word meanings
 * are noted in each template and mirrored by the *_rec()/elem() launchers in
 * kernels.hpp. Buffers: t0/t1 are StructuredBuffer<float> inputs, u0 is the
 * single RWStructuredBuffer output (multi-output kernels pack at element
 * offsets; floats are bit-cast with asuint()/asfloat()).
 */
#pragma once

// ---------------------------------------------------------------------------
// Tokenized shader templates. Bodies match the validated kernels; geometry
// (tile sizes, group sizes, elements-per-thread) is compile-time here.
// ---------------------------------------------------------------------------
namespace tmpl {

/// \brief GEMM C[M,N] = A[M,K] * B[N,K]^T. Tiled TS x TS: 2-D group per output
///        tile, A/B tiles staged through groupshared Ta/Tb with a K-strided
///        outer loop; threads map row=tid.y, col=tid.x. Offsets in q.x/q.y/q.z.
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

/// \brief GEMM C[M,N] = A[M,K] * B[N,K], both row-major; same tiling as gemm_nt().
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

/// \brief GEMM C[M,N] = A[K,M]^T * B[K,N] with A stored [K, M] row-major (read
///        transposed); same tiling as gemm_nt().
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

/// \brief Gramian chunk: G[c1*n + c2] = sum_i X[i*n + c1] * X[i*n + c2] for a
///        chunk of output rows c1 = p.z + gid.x*TS + tid.x (c1 is a row of G /
///        a column of X [m, n]; p.z "rowbase" is where this dispatch packet's
///        chunk starts — callers chunk the G rows / X columns across packets to
///        respect the TDR budget, see TuneParams::syrk_chunk), c2 =
///        gid.y*TS + tid.y; the result is the symmetric X^T * X over the full
///        [n, n] index space.
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
//
// Single-pass EPT coverage (the ElemGrid contract, TuneParams::elem_grid):
//   - The dispatch launches ceil(count/(G*EPT)) groups of &#64;G&#64; threads, so the
//     flat thread id space spans exactly [0, ceil(count/EPT)) threads...
//   - ...and each thread t claims the contiguous run i = t*EPT + e for e in
//     [0, EPT), guarded by `i < &#64;BOUND&#64;`.
//   - The map (t, e) -> t*EPT + e is injective and its image is exactly
//     [0, G*EPT*grid), so every element of [0, count) is processed exactly once
//     — no strided re-visits, no overlap between threads, no tail group that
//     could double-write — for ANY count and ANY (G, EPT) pairing. The BOUND
//     guard absorbs both the last-group tail and any count not divisible by EPT.
// &#64;BOUND&#64; is the element-count expression: "p.x" for plain elementwise kernels,
// "p.w" for adam (packed [P|M|V] layout), "p.x * p.y" for [B, IO]/[B, L]-shaped
// kernels (see TuneParams::elem_defines/adam_defines/pair_bound_defines).
#define SAL_ELEM_PRELUDE                                            \
    "[numthreads(@G@,1,1)]\n"                                       \
    "void cs_main(uint3 tid : SV_DispatchThreadID) {\n"             \
    "        [unroll] for (uint e = 0; e < @EPT@; ++e) {\n"         \
    "            uint i = tid.x * @EPT@ + e;\n"                     \
    "            if (i < @BOUND@) {\n"

/// \brief GELU forward via the tanh approximation; A -> O elementwise.
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

/// \brief GELU backward: G[i] = D[i] * dGELU/dx(x) with x = A[i] (tanh
///        approximation derivative); A = activations, D = incoming gradient.
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

/// \brief Prediction residual D[i] = A[i] - B[i].
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

/// \brief Scaled residual D[i] = scale * (A[i] - B[i]); scale is a float
///        bit-cast into q.x (asfloat(q.x)).
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

/// \brief C[i] = A[i] + B[i].
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

/// \brief Sum of squares, O[0] = sum(A[i]^2) for i < p.x. NOT an EPT kernel:
///        single group of &#64;G&#64; threads, grid-stride accumulation, groupshared
///        reduction, thread 0 writes the scalar. Dispatch grid must be (1,1,1).
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

/// \brief Adam optimizer step. G = gradient buffer (offset q.w); S = one UAV
///        packing [params P | moments M | moments V] as uint bit-cast floats, at
///        element offsets p.x/p.y/p.z; count in p.w (hence BOUND = p.w).
///        Constants: lr, step, eps as floats in q.x/q.y/q.z. Update-norm clamp
///        keeps |dP| <= lr per element; non-finite gradients leave P untouched.
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

/// \brief VQ nearest-code lookup. Z = patches [B, L], CB = codebook [K, L]; one
///        &#64;TP&#64;-thread GROUP per patch (dispatch grid = B, not elem_grid): threads
///        stride the K codes in parallel, a groupshared reduction picks the
///        argmin, and thread 0 writes the quantized patch (asuint floats at
///        OUT[b*L + l]) and the winning index (OUT[q.x + b]). NaN distances lose
///        every `<` comparison, so non-finite inputs cannot poison the argmin.
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

/// \brief VQ codebook EMA update: one thread per code k (grid = vqu_grid(K)),
///        CB[k] += gamma * (batch_mean_of_assigned_z - CB[k]) with gamma in q.y
///        (EMA, because hard replacement oscillates against the moving encoder).
///        EMA writes guarded by isfinite; codes with no assigned patches are
///        unchanged. Dispatch-level indices come from IDX at element offset q.x.
///        acc[] caps L at 64 (all current configs use L = 64).
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

/// \brief VQ commitment-loss gradient: GZ[i] += beta_eff * (Z[i] - Q[i]), in
///        place on GZ (the straight-through estimator term is already in GZ).
///        beta_eff bit-cast in q.x; BOUND is p.x = B*L.
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

/// \brief VAE reparameterization from a packed [mu | lam | eps] buffer: Z[i] =
///        mu + eps * exp(0.5*lam). Offsets: mu at +0, lam at p.z, eps at p.w;
///        BOUND is p.x * p.y = B*L.
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

/// \brief VAE backward: g_mu = gz + beta*invB*mu, g_lam = gz*eps*sigma/2 +
///        beta*invB*(exp(lam)-1)/2 (sigma = exp(0.5*lam)); the KL weight beta
///        and 1/B come as floats in q.x/q.y. Two outputs packed into one UAV as
///        uint: g_mu at +0, g_lam at element offset q.z.
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

/// \brief Batch gather from a row-indexed training table: O[i] =
///        T[IX[i/IO]*IO + (i % IO)], T = [N, IO] rows, IX = selected row ids [B]
///        (uint, at element offset q.x); BOUND is p.x * p.y = B*IO.
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

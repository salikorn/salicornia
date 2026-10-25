/** \file
 *  \brief Compute-shader library for weight-codec training (AE / VAE / VQ-VAE)
 *         and GPU linear algebra, with TUNABLE launch geometry.
 *
 * Every kernel's HLSL lives in sal::gpu::tmpl (kernels.inl) as a &#64;TOKEN&#64;-parameterized
 * template; sal::gpu::TuneParams picks the values and the kernel structs compile them via
 * Kernel(Device&, template, defines). autotune.exe sweeps the same templates and
 * writes profiles/best.json; production code loads it (defaults when absent).
 *
 * Root signature (all kernels): [0]=SRV t0, [1]=SRV t1, [2]=UAV u0, [3]=eight 32-bit
 * constants visible in-shader as cbuffer KC at register(b0) with members p (words 0-3)
 * and q (words 4-7). Multi-output kernels pack outputs into the single UAV at element
 * offsets; floats stored via asuint().
 */
#pragma once

#include <map>
#include <string>

#include "gpu/cs.hpp"

namespace sal::gpu {

/** \brief Tunable launch-geometry parameters (one per launch-geometry axis).
 *
 * Defaults reproduce the hand-tuned kernels; profiles/best.json (written by
 * autotune.exe) overrides them via load_profile(). Compile-time axes surface to
 * the shaders as &#64;TOKEN&#64; defines (see the *_defines() helpers); dispatch-level
 * axes (syrk_chunk, elem_ept-as-grid-multiplier) are read at launch time.
 */
struct TuneParams {
    int gemm_ts = 16;      ///< GEMM/syrk tile edge; group = TS x TS threads (max 32: 1024-thread group limit)
    int elem_g = 256;      ///< elementwise-family group size (threads per group)
    int elem_ept = 1;      ///< elementwise elements per thread (a contiguous run per thread)
    int vqn_tp = 64;       ///< vq_nearest threads per patch-group (codes strided, group reduction)
    int vqu_g = 64;        ///< vq_update group size (one thread per code)
    int syrk_chunk = 2048; ///< syrk output-column chunk per dispatch packet (dispatch-level, not a define)

    /// \brief Total elementwise elements covered per group: G * EPT.
    uint32_t elem_threads() const { return uint32_t(elem_g * elem_ept); }
    /** \brief Elementwise dispatch grid (groups along x) covering `count` elements.
     *
     * ElemGrid contract: grid = ceil(count / (G*EPT)); combined with the element
     * prelude in kernels.inl (thread t owns the contiguous elements
     * [t*EPT, (t+1)*EPT), each guarded by `i < BOUND`) this yields exact
     * single-pass coverage of [0, count) — every element touched by exactly one
     * (thread, e) lane, none twice, none skipped, for any count and any (G, EPT).
     */
    uint32_t elem_grid(uint32_t count) const {
        return (count + elem_threads() - 1) / elem_threads();
    }
    /// \brief Dispatch grid for vq_update: one group of vqu_g threads covers `codes` codes.
    uint32_t vqu_grid(uint32_t codes) const {
        return (codes + uint32_t(vqu_g) - 1) / uint32_t(vqu_g);
    }

    /// \brief Defines for the GEMM/syrk family: &#64;TS&#64; = gemm_ts.
    std::map<std::string, std::string> gemm_defines() const {
        return {{"TS", std::to_string(gemm_ts)}};
    }
    /** \brief Defines for the elementwise family: &#64;G&#64;, &#64;EPT&#64;, and &#64;BOUND&#64; = "p.x".
     *         p.x (KC constant word 0) is the element count for plain
     *         elementwise kernels — see the prelude macro in kernels.inl.
     */
    std::map<std::string, std::string> elem_defines() const {
        return {{"G", std::to_string(elem_g)},
                {"EPT", std::to_string(elem_ept)},
                {"BOUND", "p.x"}};
    }
    /// \brief Like elem_defines() but &#64;BOUND&#64; = "p.w": adam's element count lives
    ///        in q-word 3 (p holds the packed [P|M|V] offsets instead).
    std::map<std::string, std::string> adam_defines() const {
        auto d = elem_defines();
        d["BOUND"] = "p.w";  // adam's element count lives in q-word 3
        return d;
    }
    /// \brief Like elem_defines() but &#64;BOUND&#64; = "p.x * p.y": elementwise kernels
    ///        over a [B, IO]/[B, L] shaped 2-D buffer (total = rows * cols).
    std::map<std::string, std::string> pair_bound_defines() const {
        auto d = elem_defines();
        d["BOUND"] = "p.x * p.y";  // [B, IO]/[B, L] shaped elementwise kernels
        return d;
    }
    /// \brief Defines for vq_nearest: &#64;TP&#64; = vqn_tp (threads per patch-group).
    std::map<std::string, std::string> vqn_defines() const {
        return {{"TP", std::to_string(vqn_tp)}};
    }
    /// \brief Defines for vq_update: &#64;G&#64; = vqu_g (one thread per code).
    std::map<std::string, std::string> vqu_defines() const {
        return {{"G", std::to_string(vqu_g)}};
    }
};

#include "gpu/kernels.inl"  // shader templates: must precede the kernel-set
                            // structs (their inline ctors call tmpl::...)

// ---------------------------------------------------------------------------
// Kernel sets (production entry points). Pass nullptr for defaults; load
// profiles/best.json into TuneParams to adopt autotuned geometry.
// ---------------------------------------------------------------------------
/** \brief GEMM/syrk kernel set: gemm_nt, gemm_nn, gemm_tn and syrk, all tiled
 *         TS x TS and compiled with gemm_defines().
 *
 * The *_rec() wrappers own the resource-state discipline for one dispatch:
 * inputs COMMON -> NON_PIXEL_SHADER_RESOURCE, output COMMON ->
 * UNORDERED_ACCESS, dispatch, then everything back to COMMON. Buffers must be
 * at rest in COMMON on entry and are left in COMMON.
 */
struct LinAlgKernels {
    /// \brief C[M,N] = A[M,K] * B[N,K]^T (B row-major, K-major stride).
    Kernel gemm_nt;
    /// \brief C[M,N] = A[M,K] * B[N,K], both row-major.
    Kernel gemm_nn;
    /// \brief C[M,N] = A[K,M]^T * B[K,N] (A read transposed: [K, M] row-major).
    Kernel gemm_tn;
    /// \brief G[n,n] = X_chunk^T * X_chunk over a row-chunk of X[m,n] (gramian).
    Kernel syrk;
    /// \brief Effective parameters (defaults, or the caller's TuneParams).
    TuneParams p;

    /// \brief Compile all four kernels; `tp == nullptr` uses TuneParams defaults.
    explicit LinAlgKernels(Device& d, const TuneParams* tp = nullptr)
        : gemm_nt(d, tmpl::gemm_nt(), (tp ? *tp : TuneParams()).gemm_defines()),
          gemm_nn(d, tmpl::gemm_nn(), (tp ? *tp : TuneParams()).gemm_defines()),
          gemm_tn(d, tmpl::gemm_tn(), (tp ? *tp : TuneParams()).gemm_defines()),
          syrk(d, tmpl::syrk(), (tp ? *tp : TuneParams()).gemm_defines()),
          p(tp ? *tp : TuneParams()) {}

    // C[M,N] = A[M,K] * B[N,K]^T, with element offsets into packed buffers.
    /** \brief C[M,N] = A[M,K] * B[N,K]^T, with element offsets into packed buffers.
     *
     * Grid: ceil(N/TS) groups along x (columns), ceil(M/TS) along y (rows).
     * \param d     device whose command list the dispatch is recorded into.
     * \param a     A operand buffer ([M, K] floats at aoff), bound as SRV t0.
     * \param b     B operand buffer ([N, K] floats at boff), bound as SRV t1.
     * \param o     output buffer ([M, N] floats at coff), bound as UAV u0.
     * \param M     rows of A / C.
     * \param N     columns of B / C.
     * \param K     inner (reduction) dimension.
     * \param aoff  float-element offset of A inside its (possibly packed) buffer.
     * \param boff  float-element offset of B inside its buffer.
     * \param coff  float-element offset of C inside its buffer.
     */
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
    /// \brief Same contract as gemm_nt_rec(), with B row-major: C = A * B.
    /// \param d     device whose command list the dispatch is recorded into.
    /// \param a     A operand buffer ([M, K] floats at aoff), bound as SRV t0.
    /// \param b     B operand buffer ([K, N] floats at boff), bound as SRV t1.
    /// \param o     output buffer ([M, N] floats at coff), bound as UAV u0.
    /// \param M     rows of A / C.
    /// \param N     columns of B / C.
    /// \param K     inner (reduction) dimension.
    /// \param aoff  float-element offset of A inside its buffer.
    /// \param boff  float-element offset of B inside its buffer.
    /// \param coff  float-element offset of C inside its buffer.
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
    /// \brief Same contract as gemm_nt_rec(), with A stored [K, M] row-major:
    ///        C = A^T * B.
    /// \param d     device whose command list the dispatch is recorded into.
    /// \param a     A operand buffer ([K, M] floats at aoff, read transposed), SRV t0.
    /// \param b     B operand buffer ([K, N] floats at boff), bound as SRV t1.
    /// \param o     output buffer ([M, N] floats at coff), bound as UAV u0.
    /// \param M     rows of C (columns of A).
    /// \param N     columns of B / C.
    /// \param K     inner (reduction) dimension.
    /// \param aoff  float-element offset of A inside its buffer.
    /// \param boff  float-element offset of B inside its buffer.
    /// \param coff  float-element offset of C inside its buffer.
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

/** \brief Training kernel set: elementwise family (elem_defines), adam
 *         (adam_defines), VQ / VAE / gather kernels — plus a shared
 *         transition+dispatch helper (elem()).
 *
 * All elementwise members are launched via elem_grid(count) and rely on the
 * ElemGrid contract (exact single-pass coverage; see TuneParams::elem_grid).
 * Members like vae_sample/vae_grads/gather_rows are [B, L]-shaped and use
 * pair_bound_defines() (BOUND = p.x * p.y).
 */
struct TrainKernels {
    /// \brief GELU forward: O = 0.5*x*(1+tanh(0.7979*(x+0.044715x^3))).
    Kernel gelu;
    /// \brief GELU backward: G = D * dGELU/dx (tanh approximation).
    Kernel gelu_bwd;
    /// \brief D = A - B (prediction residual).
    Kernel mse_diff;
    /// \brief D = scale * (A - B), scale as float in q.x.
    Kernel mse_grad;
    /// \brief O[0] = sum(A[i]^2); single group, groupshared reduction.
    Kernel sum_sq;
    /// \brief Adam step on parameters packed [P|M|V] in one UAV; update-norm
    ///        clamped to [-1,1] (|dP| <= lr per element), non-finite grads skipped.
    Kernel adam;
    /// \brief C = A + B.
    Kernel elem_add;
    /// \brief VQ nearest-code lookup: one TP-thread group per patch, groupshared
    ///        argmin over K codes; writes quantized patch + code index.
    Kernel vq_nearest;
    /// \brief VQ codebook EMA update: one thread per code, gamma from q.y.
    Kernel vq_update;
    /// \brief VQ commitment loss gradient (straight-through estimator already
    ///        applied): GZ += beta_eff * (Z - Q), in place on GZ.
    Kernel vq_commit;
    /// \brief VAE reparameterization: Z = mu + eps * exp(0.5*lam) from packed
    ///        [mu | lam | eps].
    Kernel vae_sample;
    /// \brief VAE gradients wrt mu and lam from dL/dz; two outputs packed into
    ///        one UAV (g_mu at +0, g_lam at q.z).
    Kernel vae_grads;
    /// \brief Batch gather: O[b*IO+j] = T[IX[b]*IO+j] from a row-indexed source.
    Kernel gather_rows;
    /// \brief Effective parameters (defaults, or the caller's TuneParams).
    TuneParams p;

    /// \brief Compile all thirteen kernels; `tp == nullptr` uses TuneParams defaults.
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

    /** \brief Elementwise helper: records the resource-state transitions for one
     *         elementwise dispatch (a, b inputs; o output) and launches it.
     *
     * \param d      device whose command list the dispatch is recorded into.
     * \param k      already-compiled kernel (typically one of this set's members).
     * \param a      first input, bound as SRV t0 (may double as the output).
     * \param b      second input, bound as SRV t1 (may be empty or alias `a`).
     * \param o      output, bound as UAV u0 (may alias `a`/`b` for in-place ops).
     * \param c      the eight KC constants; word 0 (or the kernel's BOUND source)
     *               carries the element count the shader guards against.
     * \param count  element count; the grid is p.elem_grid(count).
     *
     * State discipline with aliasing guard: transitions are deduplicated when the
     * same resource fills two slots (b_unique / o_unique checks) — a resource
     * needed as both SRV and UAV is transitioned SRV->UAV once (valid for
     * same-index in-place updates, which is the only supported aliasing), and
     * transitions back at the end. Each resource is left in COMMON.
     */
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

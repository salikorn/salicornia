// salikorn / VQ-VAE + VAE trained ON GPU — neural weight codecs (user direction).
//   --gate  : CPU-vs-GPU single-step Adam-update parity check (runs first, must pass)
//   --run   : AE / VAE(beta) / VQ-VAE(K) bake-off on real weight patches
// Models: 256 -> 384(GELU) -> L -> 384(GELU) -> 256, no biases.
// Storage bits (per weight element): AE/VAE = 16L/256 + shared; VQ-VAE = log2(K)/256 + codebook +
// shared. All training (forward, backward, Adam) executes as D3D12 compute shaders.
/// \file
/// \brief VQ-VAE / VAE / AE codecs trained entirely ON GPU via D3D12 compute shaders
///        (--run), with a mandatory CPU-vs-GPU one-step parity gate (--gate).
///
/// Model: 256 -> 384(GELU) -> L -> 384(GELU) -> 256, no biases. Mode 0 = AE, mode 1 =
/// VAE (mu/logvar heads, reparameterization, KL beta), mode 2 = VQ-VAE (nearest-code
/// quantizer, straight-through + commitment loss, EMA codebook update). Weights live
/// in one packed state buffer [P | M | V] (params, Adam m, Adam v) addressed by tensor
/// byte offsets through the 8-word constant block; GEMMs, elementwise kernels, Adam,
/// and the VQ ops all run as kernels from gpu/kernels.hpp — nothing falls back to CPU
/// during training.
///
/// Resource-state discipline: every record_* stage emits explicit
/// COMMON <-> NON_PIXEL_SHADER_RESOURCE / UNORDERED_ACCESS / COPY_DEST transitions for
/// exactly the resources it touches (the same contract the production Device wrapper
/// and tests/test_gpu.cpp document); buffers are created in COMMON. gather_rows feeds
/// dataset-resident training: the full train matrix stays on GPU (allocated EARLY —
/// the iGPU driver has shown root-SRV instability with late-allocated large buffers,
/// see main) and each step gathers one batch by index. --gate must pass (AE 1e-4,
/// VQ 1e-3 Frobenius-relative — looser because a single fp32 near-tie can flip one
/// nearest-neighbor index of 512; broken wiring lands at O(1)) before --run is trusted.
/// TuneParams come from profiles/best.json when present.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "gpu/autotune.hpp"
#include <d3d12sdklayers.h>

using Microsoft::WRL::ComPtr;
#include "st/safetensors.hpp"
#include "xlab/fp16.hpp"
#include "xlab/log.hpp"

namespace gpu = sal::gpu;
using Eigen::MatrixXf;

/// \brief Monotonic steady-clock timestamp in seconds (step/codec timing).
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
/// \brief Bit-casts a float into the 8-word kernel constant block (see test_gpu.cpp).
/// \param f Value to encode. \return Its uint32 bit pattern.
static uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}
/// \brief Inverse of f2u().
/// \param u uint32 bit pattern. \return Reinterpreted float.
static float u2f(uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
/// \brief tanh-approximation GELU forward — CPU twin of the GPU kernel formula.
/// \param x Input. \return Activated value.
static float gelu(float x) {
    const float k0 = 0.7978845608f;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + 0.044715f * x * x * x)));
}
/// \brief Analytic GELU derivative (tanh form) — CPU reference for the gate.
/// \param x Pre-activation input. \return d(gelu)/dx at \p x.
static float dgelu(float x) {
    const float k0 = 0.7978845608f;
    float x3 = x + 0.044715f * x * x * x;
    float t = std::tanh(k0 * x3);
    float dt = (1 - t * t) * k0 * (1 + 3 * 0.044715f * x * x);
    return 0.5f * (1 + t) + 0.5f * x * dt;
}

static const int IO = 256, HID = 384;

/// \brief Named-tensor parameter table: keeps each matrix (Eigen, for CPU reference
///        and init) plus its offset in the packed row-major GPU state buffer
///        [P | M | V] (offsets into P; M and V repeat the layout at +total, +2*total).
struct Params {
    /// \brief One named weight tensor with its state-buffer offset.
    struct T {
        std::string name;   ///< Tensor name ("W1", "W2", ...).
        uint32_t rows, cols;///< Shape.
        uint64_t off;       ///< Element offset within the P region.
        MatrixXf w;         ///< Host copy (init + CPU reference).
    };
    std::vector<T> ts;      ///< Tensors in insertion order.
    uint64_t total = 0;     ///< Total element count (== size of each of P/M/V).

    /// \brief Adds a He-initialized tensor (scale sqrt(2/cols)) at the next offset.
    /// \param name Tensor name.
    /// \param rows,cols Shape.
    /// \param rng Seeded generator.
    void add(const std::string& name, uint32_t rows, uint32_t cols, std::mt19937& rng) {
        T t{name, rows, cols, total, MatrixXf(rows, cols)};
        std::normal_distribution<float> g(0, 1);
        float s = std::sqrt(2.0f / float(cols));
        for (auto& v : t.w.reshaped()) v = g(rng) * s;
        ts.push_back(std::move(t));
        total += uint64_t(rows) * cols;
    }
    /// \brief Looks up a tensor by name.
    /// \param n Tensor name.
    /// \return Pointer to the entry, or nullptr if absent.
    const T* find(const std::string& n) const {
        for (auto& t : ts)
            if (t.name == n) return &t;
        return nullptr;
    }
    /// \brief Packs all tensors ROW-MAJOR into one flat vector — the layout the
    ///        shaders expect (Eigen's own storage is column-major, hence the loop).
    /// \return Flat [P]-region image of all tensors, in insertion order.
    std::vector<float> flat() const {  // ROW-MAJOR packing (what the shaders expect)
        std::vector<float> v;
        v.reserve(total);
        for (auto& t : ts)
            for (uint32_t r = 0; r < t.rows; ++r)
                for (uint32_t c = 0; c < t.cols; ++c) v.push_back(t.w(r, c));
        return v;
    }
};

/// \brief GPU-resident codec trainer: owns all buffers and records one full training
///        step (batch gather -> forward -> backward -> Adam) into an open command
///        list as pure kernel dispatches. Modes: 0 = AE, 1 = VAE (mu/logvar heads +
///        reparameterization, KL weight \p beta), 2 = VQ-VAE (nearest-code
///        quantizer, commitment loss, EMA codebook).
///
/// The ElemGrid contract shows up in the dispatch sizes: elementwise/adam kernels
/// use tk.p.elem_grid(count) (ceil(count / (G*EPT)) so the shader prelude's EPT runs
/// cover every element exactly once); vq_nearest is dispatched DIRECTLY with one
/// group per patch (grid B, matching its numthreads(64) — NOT via elem(), whose
/// (count+255)/256 grid would cover only a quarter of the batch rows); vq_update
/// uses tk.p.vqu_grid(K). Adam walks the per-tensor [P|M|V] regions with the
/// tensor's offset triple in the constant block.
struct GpuCodec {
    gpu::Device& dev;               ///< Device (owns queue/list/fence).
    gpu::LinAlgKernels& lk;         ///< GEMM kernels (autotuned geometry).
    gpu::TrainKernels& tk;          ///< Elementwise/adam/vq kernels.
    int L, mode, K;                 ///< Latent width, codec mode (0/1/2), VQ codebook size.
    float beta;                     ///< VAE KL weight (mode 1 only).
    uint32_t B;                     ///< Batch size (patches per step).
    // VQ-VAE stabilization (standard kit): commitment beta (encoder-side only), codebook
    // EMA gamma (< 1; hard batch-mean replacement oscillates against the moving encoder).
    float vq_beta = 0.25f;          ///< Commitment-loss weight (encoder side, mode 2).
    float vq_gamma = 0.25f;         ///< Codebook EMA rate (hard batch-mean replacement oscillates against the moving encoder).
    Params par;                     ///< Parameter table + packed state layout.
    std::unique_ptr<gpu::Buffer> state, gbuf;     ///< [P|M|V] state and gradient buffers.
    std::unique_ptr<gpu::Buffer> bX, a1, h1, z, a3, h3, xh, dq, gz, gh3, gh1, gh1b; ///< Activation/gradient scratch (names match the math).
    std::unique_ptr<gpu::Buffer> pack, vgrads, vqout, cb, lossbuf;  ///< VAE mu/lam/eps pack, VAE grads, VQ quantized+idx, codebook.
    // dataset-resident training: full train matrix on GPU + per-step index gather
    std::unique_ptr<gpu::Buffer> idxBuf, bxCpu, epsCpu;  ///< Per-step batch index staging (+ eps staging for VAE).
    std::unique_ptr<gpu::Buffer> trainBufOwned;          ///< Owned train matrix buffer (when not borrowed).
    gpu::Buffer* trainBuf = nullptr;  ///< Owned by trainBufOwned, or borrowed (external_train)
    gpu::Buffer* external_train = nullptr;  ///< Borrowed dataset buffer (set by main; never freed here).
    const float* train_rows = nullptr;   ///< Row-major [ntr, IO], set by bind_dataset
    uint32_t ntrain = 0;                 ///< Dataset row count.
    uint64_t BL = 0;                     ///< B * L, the latent-region length.

    /// \brief Allocates all buffers, He-initializes parameters (seeded deterministically
    ///        from L/mode/K/beta so --gate CPU and GPU share identical init), uploads
    ///        [P] and zeroes [M|V]; for mode 2 also seeds the codebook N(0, 0.3)
    ///        (main later re-seeds it data-aligned for the real runs).
    /// \param d,l,t Device + kernel sets.
    /// \param L_,mode_,B_ Latent width, codec mode, batch size.
    /// \param K_ Codebook size (mode 2 only; 0 otherwise).
    /// \param beta_ VAE KL weight (mode 1 only).
    GpuCodec(gpu::Device& d, gpu::LinAlgKernels& l, gpu::TrainKernels& t, int L_, int mode_,
             uint32_t B_, int K_ = 0, float beta_ = 0.f)
        : dev(d), lk(l), tk(t), L(L_), mode(mode_), K(K_), beta(beta_), B(B_) {
        std::mt19937 rng(1234 + L_ + mode_ * 7 + K_ + int(beta_ * 1e6f));
        par.add("W1", HID, IO, rng);
        if (mode == 1) {
            par.add("Wmu", uint32_t(L), HID, rng);
            par.add("Wlam", uint32_t(L), HID, rng);
        } else {
            par.add("W2", uint32_t(L), HID, rng);
        }
        par.add("W3", HID, uint32_t(L), rng);
        par.add("W4", IO, HID, rng);
        BL = uint64_t(B) * L;

        state = std::make_unique<gpu::Buffer>(dev, par.total * 4 * 3, gpu::BufferKind::Uav);
        gbuf = std::make_unique<gpu::Buffer>(dev, par.total * 4, gpu::BufferKind::Uav);
        state->upload(par.flat().data(), par.total * 4, 0);
        std::vector<float> zero(par.total * 2, 0.f);
        state->upload(zero.data(), par.total * 4 * 2, par.total * 4);

        auto mk = [&](uint64_t floats, gpu::BufferKind kind = gpu::BufferKind::Uav) {
            return std::make_unique<gpu::Buffer>(dev, floats * 4, kind);
        };
        bX = mk(B * IO);
        a1 = mk(B * HID);
        h1 = mk(B * HID);
        z = mk(BL);
        a3 = mk(B * HID);
        h3 = mk(B * HID);
        xh = mk(B * IO);
        dq = mk(B * IO);
        gz = mk(BL);
        gh3 = mk(B * HID);
        gh1 = mk(B * HID);
        gh1b = mk(B * HID);
        pack = mk(3 * BL);
        vgrads = mk(2 * BL);
        vqout = mk(BL + B);
        if (K > 0) {
            cb = mk(uint64_t(K) * L);
            MatrixXf init(K, L);
            std::mt19937 kr(9);
            std::normal_distribution<float> g(0, 1);
            for (auto& v : init.reshaped()) v = g(kr) * 0.3f;
            std::vector<float> cbrow(uint64_t(K) * L);
            for (int r = 0; r < K; ++r)
                for (int c = 0; c < L; ++c) cbrow[size_t(r) * L + c] = init(r, c);
            cb->upload(cbrow.data(), uint64_t(K) * L * 4, 0);
        }
        lossbuf = mk(1);
    }

    // weights are read from the packed state's P region at tensor offsets
    /// \brief Byte/element offset of a named tensor inside the packed P region.
    /// \param n Tensor name. \return Element offset (throws via find()->off if absent).
    uint32_t woff(const char* n) { return uint32_t(par.find(n)->off); }
    /// \brief Resolves the buffer + shape holding a named tensor (always the packed
    ///        state buffer; weights are read from its P region at the tensor offset).
    /// \param[out] buf State buffer.
    /// \param n Tensor name to resolve.
    /// \param[out] rows Tensor row count.
    /// \param[out] cols Tensor column count.
    void welem(gpu::Buffer*& buf, const char* n, uint32_t& rows, uint32_t& cols) {
        auto* t = par.find(n);
        buf = state.get();
        rows = t->rows;
        cols = t->cols;
    }

    /// \brief Binds the training dataset: borrows an external buffer when
    ///        \c external_train is set (the shared early-allocated buffer), else
    ///        uploads \p rows_rowmajor into an owned one; also allocates the per-step
    ///        index/index-staging buffers and, for mode 1, the eps staging buffer.
    /// \param rows_rowmajor Row-major [n_rows, IO] normalized patch data.
    /// \param n_rows Dataset row count.
    void bind_dataset(const float* rows_rowmajor, uint32_t n_rows) {
        train_rows = rows_rowmajor;
        ntrain = n_rows;
        if (external_train) {
            trainBuf = external_train;  // borrowed, never freed here
        } else {
            trainBufOwned = std::make_unique<gpu::Buffer>(dev, uint64_t(n_rows) * IO * 4,
                                                          gpu::BufferKind::Uav);
            trainBufOwned->upload(rows_rowmajor, uint64_t(n_rows) * IO * 4);
            trainBuf = trainBufOwned.get();
        }
        idxBuf = std::make_unique<gpu::Buffer>(dev, B * 4, gpu::BufferKind::Uav);
        bxCpu = std::make_unique<gpu::Buffer>(dev, B * 4, gpu::BufferKind::CpuWrite);
        if (mode == 1)
            epsCpu = std::make_unique<gpu::Buffer>(dev, uint64_t(B) * L * 4,
                                                   gpu::BufferKind::CpuWrite);
    }

    // Records the batch into the OPEN step list: idx+eps staged via persistent upload
    // buffers, then one gather dispatch. Zero per-step allocations, zero extra submits.
    /// \brief Records the batch for one step into the OPEN command list: memcpys the
    ///        host indices (and eps for mode 1) into persistent CpuWrite staging
    ///        buffers, copies them into GPU-visible buffers under COPY_DEST
    ///        transitions, then runs one gather_rows dispatch (elem_grid geometry)
    ///        to materialize batch X. Zero per-step allocations, zero extra submits —
    ///        the whole step stays one packet.
    /// \param idx_host Batch of dataset row indices, length B.
    /// \param eps_host Reparameterization noise [B, L] (mode 1 only; else nullptr).
    void record_batch(const uint32_t* idx_host, const float* eps_host) {
        {
            void* p = nullptr;
            D3D12_RANGE none{0, 0};
            gpu::check(bxCpu->res()->Map(0, &none, &p), "Map(idx stage)");
            std::memcpy(p, idx_host, size_t(B) * 4);
            bxCpu->res()->Unmap(0, nullptr);
        }
        if (mode == 1 && eps_host) {
            void* p = nullptr;
            D3D12_RANGE none{0, 0};
            gpu::check(epsCpu->res()->Map(0, &none, &p), "Map(eps stage)");
            std::memcpy(p, eps_host, size_t(B) * L * 4);
            epsCpu->res()->Unmap(0, nullptr);
        }
        dev.transition(idxBuf->res(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_COPY_DEST);
        dev.list()->CopyBufferRegion(idxBuf->res(), 0, bxCpu->res(), 0, uint64_t(B) * 4);
        if (mode == 1 && eps_host) {
            dev.transition(pack->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_COPY_DEST);
            dev.list()->CopyBufferRegion(pack->res(), 2 * BL * 4, epsCpu->res(), 0,
                                         uint64_t(B) * L * 4);
            dev.transition(pack->res(), D3D12_RESOURCE_STATE_COPY_DEST,
                           D3D12_RESOURCE_STATE_COMMON);
        }
        dev.transition(idxBuf->res(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dev.transition(trainBuf->res(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dev.transition(bX->res(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        uint32_t c[8] = {B, uint32_t(IO), 0, 0, 0, 0, 0, 0};
        tk.gather_rows.dispatch(dev, *trainBuf, *idxBuf, *bX, c,
                                tk.p.elem_grid(uint32_t(B) * uint32_t(IO)), 1, 1);
        dev.transition(trainBuf->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_COMMON);
        dev.transition(idxBuf->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                       D3D12_RESOURCE_STATE_COMMON);
        dev.transition(bX->res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
    }

    /// \brief Records the forward pass into the open list: GEMM/GELU stack to h3/xh;
    ///        mode 1 adds the mu/logvar heads (packed into `pack`) and the
    ///        reparameterization z = mu + eps*exp(lam/2) via vae_sample; mode 2
    ///        quantizes z via vq_nearest dispatched DIRECTLY — grid B, one group per
    ///        patch, matching that kernel's numthreads(64); elem()'s (count+255)/256
    ///        grid would cover only a quarter of the batch rows. Every touched buffer
    ///        gets its explicit transitions here.
    /// \param with_eps Unused placeholder flag (eps always rides in `pack` at 2*BL).
    void record_forward(bool with_eps) {
        gpu::Buffer* w; uint32_t wr, wc;
        welem(w, "W1", wr, wc);
        lk.gemm_nt_rec(dev, *bX, *state, *a1, B, HID, IO, 0, woff("W1"));
        {
            uint32_t c[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
            tk.elem(dev, tk.gelu, *a1, *a1, *h1, c, B * HID);
        }
        if (mode == 1) {
            // heads
            lk.gemm_nt_rec(dev, *h1, *state, *pack, B, uint32_t(L), HID, 0, woff("Wmu"));
            lk.gemm_nt_rec(dev, *h1, *state, *pack, B, uint32_t(L), HID, 0, woff("Wlam"), BL);
            // z = mu + eps*exp(lam/2); eps already uploaded at 2*BL
            uint32_t c[8] = {B, uint32_t(L), uint32_t(BL), uint32_t(2 * BL), 0, 0, 0, 0};
            tk.elem(dev, tk.vae_sample, *pack, *pack, *z, c, uint32_t(BL));
        } else {
            lk.gemm_nt_rec(dev, *h1, *state, *z, B, uint32_t(L), HID, 0, woff("W2"));
        }
        if (mode == 2) {
            // quantize: OUT packs Q (floats) then IDX (uints). Dispatched directly — grid
            // (B+63)/64 matches vq_nearest's numthreads(64); elem()'s (count+255)/256 grid
            // would cover only a quarter of the batch rows.
            uint32_t c[8] = {B, uint32_t(L), uint32_t(K), 0, uint32_t(BL), 0, 0, 0};
            dev.transition(z->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            dev.transition(cb->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            dev.transition(vqout->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            tk.vq_nearest.dispatch(dev, *z, *cb, *vqout, c, B, 1, 1);  // one group per patch
            dev.transition(z->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.transition(cb->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.transition(vqout->res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON);
        }
        gpu::Buffer& decin = mode == 2 ? *vqout : *z;
        uint32_t doff = 0;
        lk.gemm_nt_rec(dev, decin, *state, *a3, B, HID, uint32_t(L), doff, woff("W3"));
        {
            uint32_t c[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
            tk.elem(dev, tk.gelu, *a3, *a3, *h3, c, B * HID);
        }
        lk.gemm_nt_rec(dev, *h3, *state, *xh, B, IO, HID, 0, woff("W4"));
    }

    /// \brief Records the backward pass into the open list: MSE gradient
    ///        (2/(B*IO) scale), decoder backward, GELU backward, encoder backward;
    ///        mode 2 adds the commitment gradient vq_commit (encoder side only,
    ///        beta-scaled and matched to the mean-squared recon loss scale 2/(B*IO),
    ///        decoder/codebook see nothing) and the EMA codebook update vq_update
    ///        (vqu_grid geometry); mode 1 accumulates mu/lam gradients via vae_grads
    ///        and sums them elementwise in place. Scratch buffers are reused by
    ///        liveness (e.g. gh1 doubles as the GELU'd GH3 output). The host batch
    ///        pointer parameter is intentionally unnamed/unused: X is dataset-resident
    ///        since record_batch.
    void record_backward(const float* /*host_x*/) {  // host batch no longer needed (resident)
        // D = 2/(B*IO) * (Xh - X)
        uint32_t g[8] = {B * IO, 0, 0, 0, f2u(2.0f / float(B * IO)), 0, 0, 0};
        tk.elem(dev, tk.mse_grad, *xh, *bX, *dq, g, B * IO);
        // decoder backward; gelu'd GH3 goes into gh1 scratch ([B,HID]; free until encoder bwd)
        lk.gemm_nn_rec(dev, *dq, *state, *gh3, B, HID, IO, 0, woff("W4"));
        lk.gemm_tn_rec(dev, *dq, *h3, *gbuf, IO, HID, B, 0, 0, woff("W4"));   // gW4
        {
            uint32_t c[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
            tk.elem(dev, tk.gelu_bwd, *a3, *gh3, *gh1, c, B * HID);  // gh1 = GH3*gelu'
        }
        gpu::Buffer& decin = mode == 2 ? *vqout : *z;
        lk.gemm_nn_rec(dev, *gh1, *state, *gz, B, uint32_t(L), HID, 0, woff("W3"));
        lk.gemm_tn_rec(dev, *gh1, decin, *gbuf, HID, uint32_t(L), B, 0, 0, woff("W3"));  // gW3
        if (mode == 2) {
            // commitment loss: encoder-side grad += beta * 2/(B*L) * (z - sg(q)), matched to
            // the mean-squared recon loss scale (2/(B*IO)); decoder/codebook see nothing here.
            float beta_eff = vq_beta * 2.0f / (float(B) * float(L));
            uint32_t c[8] = {uint32_t(BL), 0, 0, 0, f2u(beta_eff), 0, 0, 0};
            tk.elem(dev, tk.vq_commit, *z, *vqout, *gz, c, uint32_t(BL));
        }
        // encoder path
        if (mode == 1) {
            uint32_t c[8] = {B, uint32_t(L), uint32_t(BL), uint32_t(2 * BL), f2u(beta),
                             f2u(1.0f / float(B)), uint32_t(BL), 0};
            tk.elem(dev, tk.vae_grads, *pack, *gz, *vgrads, c, uint32_t(BL));
            lk.gemm_nn_rec(dev, *vgrads, *state, *gh1, B, HID, uint32_t(L), 0, woff("Wmu"));
            lk.gemm_nn_rec(dev, *vgrads, *state, *gh1b, B, HID, uint32_t(L), uint32_t(BL),
                           woff("Wlam"));
            lk.gemm_tn_rec(dev, *vgrads, *h1, *gbuf, uint32_t(L), HID, B, 0, 0, woff("Wmu"));
            lk.gemm_tn_rec(dev, *vgrads, *h1, *gbuf, uint32_t(L), HID, B, uint32_t(BL), 0,
                           woff("Wlam"));
            {
                uint32_t a[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
                tk.elem(dev, tk.elem_add, *gh1, *gh1b, *gh1, a, B * HID);  // same-index in-place
            }
            // gh3 is dead now — use as gelu_bwd output scratch
            uint32_t a[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
            tk.elem(dev, tk.gelu_bwd, *a1, *gh1, *gh3, a, B * HID);
            lk.gemm_tn_rec(dev, *gh3, *bX, *gbuf, HID, IO, B, 0, 0, woff("W1"));
        } else {
            lk.gemm_nn_rec(dev, *gz, *state, *gh1, B, HID, uint32_t(L), 0, woff("W2"));
            lk.gemm_tn_rec(dev, *gz, *h1, *gbuf, uint32_t(L), HID, B, 0, 0, woff("W2"));
            uint32_t a[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
            tk.elem(dev, tk.gelu_bwd, *a1, *gh1, *gh1b, a, B * HID);
            lk.gemm_tn_rec(dev, *gh1b, *bX, *gbuf, HID, IO, B, 0, 0, woff("W1"));
        }
        if (mode == 2) {
            dev.transition(z->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            dev.transition(vqout->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            dev.transition(cb->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            uint32_t c[8] = {B, uint32_t(L), uint32_t(K), 0, uint32_t(BL), f2u(vq_gamma), 0,
                             0};
            tk.vq_update.dispatch(dev, *z, *vqout, *cb, c, tk.p.vqu_grid(uint32_t(K)), 1, 1);
            dev.transition(z->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.transition(vqout->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.transition(cb->res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON);
        }
    }

    /// \brief Records the Adam step for every tensor into the open list: for each
    ///        tensor, transitions gbuf/state, dispatches the adam kernel with the
    ///        tensor's [P|M|V] offset triple, its count, the bias-corrected lr and
    ///        step, and elem_grid geometry, then transitions back.
    /// \param lr   Bias-corrected learning rate for this step.
    /// \param step 1-based step counter (drives Adam bias correction).
    void record_adam(float lr, uint32_t step) {
        for (auto& t : par.ts) {
            dev.transition(gbuf->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            dev.transition(state->res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            uint32_t c[8] = {uint32_t(t.off), uint32_t(par.total + t.off),
                             uint32_t(2 * par.total + t.off), uint32_t(t.rows) * t.cols, f2u(lr),
                             f2u(float(step)), f2u(1e-8f), uint32_t(t.off)};
            uint32_t count = t.rows * t.cols;
            tk.adam.dispatch(dev, *gbuf, *gbuf, *state, c, tk.p.elem_grid(count), 1, 1);
            dev.transition(gbuf->res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.transition(state->res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON);
        }
    }
};

// ---------------- CPU reference (gate) ----------------
/// \brief Minimal CPU replica of the AE forward pass used by the parity gate —
///        copies the four weights from a Params table and forwards with the same
///        GELU formula, keeping pre-activations for the CPU backward.
struct CpuRef {
    MatrixXf W1, W2, W3, W4;  ///< Copied from the GPU codec's init (identical weights).
    /// \brief Copies the named weights out of \p p.
    explicit CpuRef(const Params& p) {
        W1 = p.find("W1")->w;
        W2 = p.find("W2")->w;
        W3 = p.find("W3")->w;
        W4 = p.find("W4")->w;
    }
    /// \brief Forward pass; fills pre-activations A1/A3 and activations H1/H3/Z.
    /// \param X Input batch. \param[out] A1,H1,Z,A3,H3 Intermediates.
    /// \return Reconstruction.
    MatrixXf fwd(const MatrixXf& X, MatrixXf& A1, MatrixXf& H1, MatrixXf& Z, MatrixXf& A3,
                 MatrixXf& H3) const {
        A1 = X * W1.transpose();          // pre-activation (kept for dgelu in backward)
        H1 = A1;
        for (auto& v : H1.reshaped()) v = gelu(v);
        Z = H1 * W2.transpose();
        A3 = Z * W3.transpose();          // pre-activation
        H3 = A3;
        for (auto& v : H3.reshaped()) v = gelu(v);
        return H3 * W4.transpose();
    }
};

/// \brief The --gate parity check: runs one full GPU training step (forward,
///        backward, Adam) for AE (mode 0) and VQ-VAE (mode 2) on a fixed batch and
///        compares every intermediate, gradient and post-update parameter against a
///        CPU reference computed from the same (seeded, identical) init. Includes
///        mini GEMM self-tests (gemm_nt, gemm_tn), a state-buffer round-trip probe,
///        an isolated first-GEMM probe, and per-layer/parameter events
///        (gate_fwd_layer / gate_bwd_layer / gate_param). AE gate: max Frobenius
///        relative error 1e-4; VQ gate: 1e-3 — looser because a single fp32 near-tie
///        in nearest-neighbor can flip one index of 512, while broken wiring lands
///        at O(1) instead of O(1e-4).
/// \param run Event sink.
/// \param dev,lk,tk Device + kernel sets (already initialized).
/// \param Xb Input batch (first `batch` training rows).
/// \return 0 iff both gates passed, 1 otherwise (a \c fatal event is logged).
static int run_gate(xlab::Run& run, gpu::Device& dev, gpu::LinAlgKernels& lk, gpu::TrainKernels& tk,
                    const MatrixXf& Xb) {
    const uint32_t B = uint32_t(Xb.rows());
    // row-major batch packing (Eigen .data() is column-major!) — shared by AE + VQ sections
    std::vector<float> xf(size_t(Xb.rows()) * Xb.cols());
    for (int r = 0; r < Xb.rows(); ++r)
        for (int c = 0; c < Xb.cols(); ++c) xf[size_t(r) * Xb.cols() + c] = Xb(r, c);
    // Adam step 1, lr 3e-3 (shared CPU reference)
    auto adam1 = [](MatrixXf& P, const MatrixXf& G) {
        float lr = 3e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float bc1 = 1 - b1, bc2 = 1 - b2;
        for (int i = 0; i < P.size(); ++i) {
            float g = G.data()[i];
            float m = (1 - b1) * g, v = (1 - b2) * g * g;
            P.data()[i] -= lr * (m / bc1) / (std::sqrt(v / bc2) + eps);
        }
    };
    {
        // mini self-test of the kernels.hpp gemms before anything else
        const uint32_t M = 64, N = 48, K = 32;
        std::mt19937 gr(4);
        std::uniform_real_distribution<float> gu(-1, 1);
        std::vector<float> a(M * K), b(N * K), cexp(size_t(M) * N, 0.f);
        for (auto& v : a) v = gu(gr);
        for (auto& v : b) v = gu(gr);
        for (uint32_t i = 0; i < M; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                float s = 0;
                for (uint32_t k = 0; k < K; ++k) s += a[i * K + k] * b[j * K + k];
                cexp[i * N + j] = s;
            }
        gpu::Buffer ba(dev, a.size() * 4, gpu::BufferKind::Uav);
        gpu::Buffer bb(dev, b.size() * 4, gpu::BufferKind::Uav);
        gpu::Buffer bc(dev, cexp.size() * 4, gpu::BufferKind::Uav);
        ba.upload(a.data(), a.size() * 4);
        bb.upload(b.data(), b.size() * 4);
        dev.begin();
        lk.gemm_nt_rec(dev, ba, bb, bc, M, N, K);
        dev.submit_and_wait();
        std::vector<float> got(cexp.size());
        bc.download(got.data(), got.size() * 4);
        double md = 0;
        for (size_t i = 0; i < cexp.size(); ++i) md = std::max(md, double(std::fabs(got[i] - cexp[i])));
        char bb2[128];
        std::snprintf(bb2, sizeof bb2, "\"gemm_nt_selftest_max_diff\":%.3e", md);
        run.event("gate_gemm_selftest", bb2);
    }

    GpuCodec gc(dev, lk, tk, 64, 0, B);
    gc.bX->upload(xf.data(), uint64_t(xf.size()) * 4);
    {
        // probe: state P region must equal the packed row-major weights
        std::vector<float> pf = gc.par.flat();
        std::vector<float> back(gc.par.total);
        gc.state->download(back.data(), gc.par.total * 4, 0);
        double md = 0;
        for (size_t i = 0; i < pf.size(); ++i) md = std::max(md, double(std::fabs(pf[i] - back[i])));
        char bb[128];
        std::snprintf(bb, sizeof bb, "\"state_roundtrip_max_diff\":%.3e", md);
        run.event("gate_state_probe", bb);
    }
    // isolated first-gemm probe with the model's exact buffers
    {
        MatrixXf rA1 = Xb * gc.par.find("W1")->w.transpose();
        dev.begin();
        lk.gemm_nt_rec(dev, *gc.bX, *gc.state, *gc.a1, B, HID, IO, 0, gc.woff("W1"));
        dev.submit_and_wait();
        std::vector<float> got(size_t(B) * HID);
        gc.a1->download(got.data(), got.size() * 4);
        double md = 0;
        for (uint32_t r = 0; r < B; ++r)
            for (uint32_t c = 0; c < uint32_t(HID); ++c)
                md = std::max(md, double(std::fabs(got[size_t(r) * HID + c] - rA1(int(r), int(c)))));
        char bb2[128];
        std::snprintf(bb2, sizeof bb2, "\"isolated_gemm1_max_diff\":%.3e,\"ref_scale\":%.3e", md,
                      double(rA1.cwiseAbs().maxCoeff()));
        run.event("gate_isolated_gemm1", bb2);
    }
    dev.begin();
    gc.record_forward(false);
    dev.submit_and_wait();

    // layer-by-layer forward comparison vs CPU
    CpuRef cpu0(gc.par);
    MatrixXf rA1, rH1, rZ, rA3, rH3;
    MatrixXf rXh = cpu0.fwd(Xb, rA1, rH1, rZ, rA3, rH3);
    auto cmp = [&](const char* what, gpu::Buffer& buf, const MatrixXf& ref) {
        std::vector<float> got(ref.size());
        buf.download(got.data(), uint64_t(got.size()) * 4);
        double maxd = 0;
        for (int r = 0; r < ref.rows(); ++r)
            for (int c = 0; c < ref.cols(); ++c)
                maxd = std::max(maxd, double(std::fabs(got[size_t(r) * ref.cols() + c] - ref(r, c))));
        char b[160];
        std::snprintf(b, sizeof b, "\"layer\":\"%s\",\"max_abs_diff\":%.4e,\"scale\":%.3e", what,
                      maxd, double(ref.cwiseAbs().maxCoeff()));
        run.event("gate_fwd_layer", b);
    };
    cmp("a1", *gc.a1, rA1);
    cmp("h1", *gc.h1, rH1);
    cmp("z", *gc.z, rZ);
    cmp("a3", *gc.a3, rA3);
    cmp("h3", *gc.h3, rH3);
    cmp("xh", *gc.xh, rXh);

    {
        // mini self-test for gemm_tn: C[M,N] = A[K,M]^T B[K,N]
        const uint32_t M = 48, N = 32, K = 64;
        std::mt19937 gr(5);
        std::uniform_real_distribution<float> gu(-1, 1);
        std::vector<float> a(size_t(K) * M), b(size_t(K) * N), cexp(size_t(M) * N, 0.f);
        for (auto& v : a) v = gu(gr);
        for (auto& v : b) v = gu(gr);
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t n = 0; n < N; ++n) {
                float s = 0;
                for (uint32_t k = 0; k < K; ++k) s += a[k * M + m] * b[k * N + n];
                cexp[m * N + n] = s;
            }
        gpu::Buffer ba(dev, a.size() * 4, gpu::BufferKind::Uav);
        gpu::Buffer bb(dev, b.size() * 4, gpu::BufferKind::Uav);
        gpu::Buffer bc(dev, cexp.size() * 4, gpu::BufferKind::Uav);
        ba.upload(a.data(), a.size() * 4);
        bb.upload(b.data(), b.size() * 4);
        dev.begin();
        lk.gemm_tn_rec(dev, ba, bb, bc, M, N, K);
        dev.submit_and_wait();
        std::vector<float> got(cexp.size());
        bc.download(got.data(), got.size() * 4);
        double md = 0;
        for (size_t i = 0; i < cexp.size(); ++i) md = std::max(md, double(std::fabs(got[i] - cexp[i])));
        char bb2[128];
        std::snprintf(bb2, sizeof bb2, "\"gemm_tn_selftest_max_diff\":%.3e", md);
        run.event("gate_gemm_tn_selftest", bb2);
    }

    dev.begin();
    gc.record_backward(xf.data());
    gc.record_adam(3e-3f, 1);
    dev.submit_and_wait();

    // gradient comparisons (recompute CPU backward with correct pre-activations)
    {
        CpuRef cpuB(gc.par);
        MatrixXf A1, H1, Z, A3, H3;
        MatrixXf Xh = cpuB.fwd(Xb, A1, H1, Z, A3, H3);
        MatrixXf D = 2.0f / float(B * IO) * (Xh - Xb);
        MatrixXf GH3 = D * cpuB.W4;
        for (int r = 0; r < GH3.rows(); ++r)
            for (int c = 0; c < GH3.cols(); ++c) GH3(r, c) *= dgelu(A3(r, c));
        MatrixXf GZ = GH3 * cpuB.W3;
        MatrixXf gW4 = D.transpose() * H3;
        MatrixXf gW3 = GH3.transpose() * Z;
        MatrixXf GH1 = GZ * cpuB.W2;
        MatrixXf gW2 = GZ.transpose() * H1;
        for (int r = 0; r < GH1.rows(); ++r)
            for (int c = 0; c < GH1.cols(); ++c) GH1(r, c) *= dgelu(A1(r, c));
        MatrixXf gW1 = GH1.transpose() * Xb;
        auto cmpg = [&](const char* what, gpu::Buffer* buf, uint64_t off, const MatrixXf& ref) {
            std::vector<float> got(ref.size());
            buf->download(got.data(), uint64_t(ref.size()) * 4, off * 4);
            double md = 0;
            for (int r = 0; r < ref.rows(); ++r)
                for (int c = 0; c < ref.cols(); ++c)
                    md = std::max(md, double(std::fabs(got[size_t(r) * ref.cols() + c] - ref(r, c))));
            char bb2[160];
            std::snprintf(bb2, sizeof bb2, "\"grad\":\"%s\",\"max_abs_diff\":%.4e,\"scale\":%.3e",
                          what, md, double(ref.cwiseAbs().maxCoeff()));
            run.event("gate_bwd_layer", bb2);
        };
        // gh1 now holds GH3*gelu' (post-gelu-bwd) — compare against GH3
        cmpg("gh3", gc.gh1.get(), 0, GH3);
        cmpg("gz", gc.gz.get(), 0, GZ);
        cmpg("gh1_gelu", gc.gh1b.get(), 0, GH1);
        cmpg("gW4", gc.gbuf.get(), gc.par.find("W4")->off, gW4);
        cmpg("gW3", gc.gbuf.get(), gc.par.find("W3")->off, gW3);
        cmpg("gW2", gc.gbuf.get(), gc.par.find("W2")->off, gW2);
        cmpg("gW1", gc.gbuf.get(), gc.par.find("W1")->off, gW1);
    }

    // CPU replica with identical init & math
    CpuRef cpu(gc.par);
    MatrixXf A1, H1, Z, A3, H3;
    MatrixXf Xh = cpu.fwd(Xb, A1, H1, Z, A3, H3);
    MatrixXf D = 2.0f / float(B * IO) * (Xh - Xb);
    MatrixXf GH3 = D * cpu.W4;
    for (int r = 0; r < GH3.rows(); ++r)
        for (int c = 0; c < GH3.cols(); ++c) GH3(r, c) *= dgelu(A3(r, c));
    MatrixXf GZ = GH3 * cpu.W3;
    MatrixXf gW4 = D.transpose() * H3;
    MatrixXf gW3 = GH3.transpose() * Z;
    MatrixXf GH1 = GZ * cpu.W2;
    MatrixXf gW2 = GZ.transpose() * H1;
    for (int r = 0; r < GH1.rows(); ++r)
        for (int c = 0; c < GH1.cols(); ++c) GH1(r, c) *= dgelu(A1(r, c));
    MatrixXf gW1 = GH1.transpose() * Xb;
    // (adam1 hoisted above — shared with the VQ section)
    adam1(cpu.W1, gW1);
    adam1(cpu.W2, gW2);
    adam1(cpu.W3, gW3);
    adam1(cpu.W4, gW4);

    // compare GPU-updated params vs CPU
    std::vector<float> flat(gc.par.total);
    gc.state->download(flat.data(), gc.par.total * 4, 0);
    double max_rel = 0;
    for (auto& nm : {"W1", "W2", "W3", "W4"}) {
        auto* t = gc.par.find(nm);
        const MatrixXf& cpuW = std::string(nm) == "W1"   ? cpu.W1
                               : std::string(nm) == "W2" ? cpu.W2
                               : std::string(nm) == "W3" ? cpu.W3
                                                          : cpu.W4;
        double dsq = 0, nsq = 0;
        for (uint32_t r = 0; r < t->rows; ++r)
            for (uint32_t c = 0; c < t->cols; ++c) {
                double g = flat[t->off + uint64_t(r) * t->cols + c];
                double cc = cpuW(int(r), int(c));
                dsq += (g - cc) * (g - cc);
                nsq += cc * cc;
            }
        double frel = std::sqrt(dsq) / std::max(1e-30, std::sqrt(nsq));
        char tb[128];
        std::snprintf(tb, sizeof tb, "\"param\":\"%s\",\"frob_rel\":%.3e", nm, frel);
        run.event("gate_param", tb);
        max_rel = std::max(max_rel, frel);
    }
    char b[128];
    std::snprintf(b, sizeof b, "\"max_param_frob_rel\":%.3e", max_rel);
    run.event("gpu_trainer_gate", b);
    if (max_rel > 1e-4) {
        run.event("fatal", "\"error\":\"trainer gate FAILED\"");
        return 1;
    }

    // ---- VQ-VAE (mode 2) single-step parity: quantizer, STE + commitment, Adam, EMA cb ----
    {
        const int Lq = 64, Kq = 256;
        GpuCodec gv(dev, lk, tk, Lq, 2, B, Kq);
        gv.bX->upload(xf.data(), uint64_t(xf.size()) * 4);
        // CPU reference (same init weights; codebook read back from the GPU buffer)
        std::vector<float> cbh(size_t(Kq) * Lq);
        gv.cb->download(cbh.data(), uint64_t(cbh.size()) * 4, 0);
        MatrixXf CBm(Kq, Lq);
        for (int k = 0; k < Kq; ++k)
            for (int l = 0; l < Lq; ++l) CBm(k, l) = cbh[size_t(k) * Lq + l];
        MatrixXf W1 = gv.par.find("W1")->w, W2 = gv.par.find("W2")->w, W3 = gv.par.find("W3")->w,
                   W4 = gv.par.find("W4")->w;
        MatrixXf qA1 = Xb * W1.transpose(), qH1 = qA1;
        for (auto& v : qH1.reshaped()) v = gelu(v);
        MatrixXf qZ = qH1 * W2.transpose();
        MatrixXf qQm(uint32_t(B), Lq);
        std::vector<uint32_t> qIdx(size_t(B), 0u);  // count ctor — {B} would be a 1-elem list!
        for (uint32_t b2 = 0; b2 < B; ++b2) {
            float best = 3.4e38f;
            int bk = 0;
            for (int k = 0; k < Kq; ++k) {
                float s = 0;
                for (int l = 0; l < Lq; ++l) {
                    float d = qZ(int(b2), l) - CBm(k, l);
                    s += d * d;
                }
                if (s < best) {
                    best = s;
                    bk = k;
                }
            }
            qQm.row(int(b2)) = CBm.row(bk);
            qIdx[b2] = uint32_t(bk);
        }
        MatrixXf qA3 = qQm * W3.transpose(), qH3 = qA3;
        for (auto& v : qH3.reshaped()) v = gelu(v);
        MatrixXf qXh = qH3 * W4.transpose();

        dev.begin();
        gv.record_forward(false);
        dev.submit_and_wait();
        std::vector<float> vq(uint64_t(B) * Lq + B);
        gv.vqout->download(vq.data(), uint64_t(vq.size()) * 4);
        double qmd = 0;
        uint32_t idxmis = 0;
        for (uint32_t b2 = 0; b2 < B; ++b2) {
            if (f2u(vq[uint64_t(B) * Lq + b2]) != qIdx[b2]) ++idxmis;
            for (int l = 0; l < Lq; ++l)
                qmd = std::max(qmd,
                               double(std::fabs(vq[uint64_t(b2) * Lq + l] - qQm(int(b2), l))));
        }
        {
            char bb[160];
            std::snprintf(bb, sizeof bb, "\"q_max_abs_diff\":%.4e,\"idx_mismatch\":%u,\"B\":%u",
                          qmd, idxmis, B);
            run.event("gate_vq_forward", bb);
        }

        // GPU backward + Adam + EMA codebook update, then the same on CPU
        dev.begin();
        gv.record_backward(xf.data());
        gv.record_adam(3e-3f, 1);
        dev.submit_and_wait();

        MatrixXf qD = 2.0f / float(B * IO) * (qXh - Xb);
        MatrixXf qGH3 = qD * W4;
        for (int r = 0; r < qGH3.rows(); ++r)
            for (int c = 0; c < qGH3.cols(); ++c) qGH3(r, c) *= dgelu(qA3(r, c));
        MatrixXf qGZ = qGH3 * W3;
        MatrixXf qgW4 = qD.transpose() * qH3;
        MatrixXf qgW3 = qGH3.transpose() * qQm;
        float beta_eff = gv.vq_beta * 2.0f / (float(B) * float(Lq));  // = vq_commit constant
        qGZ.array() += beta_eff * (qZ.array() - qQm.array());
        MatrixXf qGH1 = qGZ * W2;
        MatrixXf qgW2 = qGZ.transpose() * qH1;
        for (int r = 0; r < qGH1.rows(); ++r)
            for (int c = 0; c < qGH1.cols(); ++c) qGH1(r, c) *= dgelu(qA1(r, c));
        MatrixXf qgW1 = qGH1.transpose() * Xb;
        adam1(W1, qgW1);
        adam1(W2, qgW2);
        adam1(W3, qgW3);
        adam1(W4, qgW4);
        for (int k = 0; k < Kq; ++k) {
            MatrixXf acc = MatrixXf::Zero(1, Lq);
            int cnt = 0;
            for (uint32_t b2 = 0; b2 < B; ++b2)
                if (qIdx[b2] == uint32_t(k)) {
                    acc += qZ.row(int(b2));
                    ++cnt;
                }
            if (cnt > 0) CBm.row(k) += gv.vq_gamma * (acc / float(cnt) - CBm.row(k));
        }

        std::vector<float> qflat(gv.par.total);
        gv.state->download(qflat.data(), gv.par.total * 4, 0);
        double vmax = 0;
        for (auto& nm : {"W1", "W2", "W3", "W4"}) {
            auto* t = gv.par.find(nm);
            const MatrixXf& cpuW = std::string(nm) == "W1"   ? W1
                                   : std::string(nm) == "W2" ? W2
                                   : std::string(nm) == "W3" ? W3
                                                              : W4;
            double dsq = 0, nsq = 0;
            for (uint32_t r = 0; r < t->rows; ++r)
                for (uint32_t c = 0; c < t->cols; ++c) {
                    double g = qflat[t->off + uint64_t(r) * t->cols + c];
                    double cc = cpuW(int(r), int(c));
                    dsq += (g - cc) * (g - cc);
                    nsq += cc * cc;
                }
            double frel = std::sqrt(dsq) / std::max(1e-30, std::sqrt(nsq));
            char tb[128];
            std::snprintf(tb, sizeof tb, "\"param\":\"vq_%s\",\"frob_rel\":%.3e", nm, frel);
            run.event("gate_param", tb);
            vmax = std::max(vmax, frel);
        }
        std::vector<float> cb2(size_t(Kq) * Lq);
        gv.cb->download(cb2.data(), uint64_t(cb2.size()) * 4, 0);
        double cdsq = 0, cnsq = 0;
        for (int k = 0; k < Kq; ++k)      // cb2 is row-major; CBm must be indexed (k, l)
            for (int l = 0; l < Lq; ++l) {
                double g = cb2[size_t(k) * Lq + l], cc = CBm(k, l);
                cdsq += (g - cc) * (g - cc);
                cnsq += cc * cc;
            }
        double cfrel = std::sqrt(cdsq) / std::max(1e-30, std::sqrt(cnsq));
        {
            char tb[128];
            std::snprintf(tb, sizeof tb, "\"param\":\"vq_codebook\",\"frob_rel\":%.3e", cfrel);
            run.event("gate_param", tb);
        }
        vmax = std::max(vmax, cfrel);
        char b3[128];
        std::snprintf(b3, sizeof b3, "\"vq_max_param_frob_rel\":%.3e", vmax);
        run.event("gpu_trainer_gate_vq", b3);
        // looser than the AE gate (1e-4): a single fp32 near-tie in nearest-neighbor can
        // flip one index of 512; broken wiring lands at O(1) instead of O(1e-4).
        if (vmax > 1e-3) {
            run.event("fatal", "\"error\":\"VQ trainer gate FAILED\"");
            return 1;
        }
    }
    return 0;
}

// ---------------- training loop ----------------
/// \brief Evaluates a trained codec on \p batches batches of the validation matrix:
///        uploads each batch, records forward only, and accumulates relative MSE.
///        For mode 1 (VAE) eps is zeroed so eval is deterministic (z = mu).
/// \param gc      Trained codec.
/// \param dev,lk,tk Device + kernel sets.
/// \param Va      Validation matrix, rows divisible by gc.B.
/// \param batches Number of batches to evaluate.
/// \return Relative MSE over the evaluated rows.
static float eval_rel_mse(GpuCodec& gc, gpu::Device& dev, gpu::LinAlgKernels& lk, gpu::TrainKernels& tk,
                          const MatrixXf& Va, int batches) {
    double err = 0, tot = 0;
    uint32_t B = gc.B;
    std::vector<float> xf(size_t(B) * IO), out(size_t(B) * IO);
    if (gc.mode == 1) {  // deterministic eval: eps = 0 -> z = mu
        std::vector<float> z(size_t(B) * gc.L, 0.f);
        gc.pack->upload(z.data(), z.size() * 4, 2 * size_t(B) * gc.L * 4);
    }
    for (int bi = 0; bi < batches; ++bi) {
        for (uint32_t r = 0; r < B; ++r)
            for (int c = 0; c < IO; ++c)
                xf[size_t(r) * IO + c] = Va(int(bi) * int(B) + int(r), c);
        gc.bX->upload(xf.data(), xf.size() * 4);
        dev.begin();
        gc.record_forward(false);
        dev.submit_and_wait();
        gc.xh->download(out.data(), out.size() * 4);
        for (size_t i = 0; i < xf.size(); ++i) {
            double d = double(out[i]) - double(xf[i]);
            err += d * d;
            tot += double(xf[i]) * double(xf[i]);
        }
    }
    return float(err / tot);
}

/// \brief Entry point: parses --gate | --run (one is required) plus --steps,
///        --batch, --shard, --tensor, --bench; optionally enables the D3D12 debug
///        layer (SAL_DML_DEBUG); loads profiles/best.json when present; builds the
///        patch dataset (same pipeline as the CPU bake-off in spike_nonlinear.cpp);
///        --gate runs the parity gate; --run trains each codec config (ae_L64,
///        vae_L64 at two betas, vqvae K=256/4096) for \p steps with cosine LR and a
///        100-step VQ warmup ramp, data-aligned codebook init (seeded from real z's
///        + 0.1*std jitter — random N(0,0.3) codes start far from the z-cloud, which
///        caused dead codes and a violent first-contact EMA), one packet per step,
///        then evaluates relative MSE and logs \c gpu_codec_point events
///        (or \c gpu_bench_point with --bench). Fatal errors dump the info queue.
/// \return 0 on success, 1 on gate/fatal failure, 2 on usage error.
int main(int argc, char** argv) {
    std::string shard = "models/GLM-5.3-Flash/model-00062-of-00062.safetensors";
    std::string tensor = "model.language_model.layers.9.self_attn.q_proj.weight";
    std::string do_run = "";
    bool bench = false;
    int steps = 2000;
    uint32_t batch = 512;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bench") bench = true;
        else if (a == "--gate") do_run = "gate";
        else if (a == "--run") do_run = "run";
        else if (a == "--steps") steps = std::stoi(argv[++i]);
        else if (a == "--batch") batch = uint32_t(std::stoul(argv[++i]));
        else if (a == "--shard") shard = argv[++i];
        else if (a == "--tensor") tensor = argv[++i];
    }
    if (do_run.empty()) {
        std::fprintf(stderr, "usage: spike_vqvae --gate | --run [--steps N]\n");
        return 2;
    }
    xlab::Run run("vqvae-gpu-" + do_run);
    gpu::Device dev;
    ComPtr<ID3D12InfoQueue> iq;
    try {
        if (getenv("SAL_DML_DEBUG")) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) dbg->EnableDebugLayer();
        }
        dev.init(false);
        dev.dev()->QueryInterface(IID_PPV_ARGS(&iq));
        gpu::TuneParams tp;
        if (gpu::autotune::load_profile("profiles/best.json", tp))
            run.event("tune_profile", "\"loaded\":true,\"gemm_ts\":" +
                                          std::to_string(tp.gemm_ts) + ",\"elem_g\":" +
                                          std::to_string(tp.elem_g) + ",\"vqn_tp\":" +
                                          std::to_string(tp.vqn_tp));
        else
            run.event("tune_profile", "\"loaded\":false");
        gpu::LinAlgKernels lk(dev, &tp);
        gpu::TrainKernels tk(dev, &tp);
        run.event("env", "\"adapter\":\"" + dev.adapter_name() + "\"");

        // data (same pipeline as the CPU bake-off)
        sal::st::SafeTensorsFile f{std::filesystem::path(shard)};
        const sal::st::TensorInfo* ti = f.find(tensor);
        const size_t m = size_t(ti->shape[0]), n = size_t(ti->shape[1]);
        const uint16_t* X = reinterpret_cast<const uint16_t*>(ti->data);
        const int P16 = 16;
        const size_t pr = m / P16, pc = n / P16, npatch = pr * pc;
        std::mt19937 rng(11);
        std::vector<size_t> order(npatch);
        for (size_t i = 0; i < npatch; ++i) order[i] = i;
        std::shuffle(order.begin(), order.end(), rng);
        size_t ntr = std::min<size_t>(40000, npatch), nva = std::min<size_t>(8000, npatch - ntr);
        MatrixXf Tr(ntr, IO), Va(nva, IO);
        auto patch = [&](size_t idx, MatrixXf& out, int row) {
            size_t r = (idx / pc) * P16, c = (idx % pc) * P16;
            for (int i = 0; i < P16; ++i)
                for (int j = 0; j < P16; ++j)
                    out(row, i * P16 + j) = sal::bf16_to_f32(X[(r + i) * n + c + j]);
        };
        for (size_t i = 0; i < ntr; ++i) patch(order[i], Tr, int(i));
        for (size_t i = 0; i < nva; ++i) patch(order[ntr + i], Va, int(i));
        float scale = 0;
        for (int i = 0; i < Tr.size(); ++i) scale = std::max(scale, std::fabs(Tr.data()[i]));
        Tr.array() /= scale;
        Va.array() /= scale;

        if (do_run == "gate") return run_gate(run, dev, lk, tk, Tr.topRows(batch));

        struct Cfg {
            const char* name;
            int mode, L, K;
            float beta;
        };
        std::vector<Cfg> cfgs = {
            {"ae_L64", 0, 64, 0, 0.f},
            {"vae_L64_b1e-4", 1, 64, 0, 1e-4f},
            {"vae_L64_b1e-2", 1, 64, 0, 1e-2f},
            {"vqvae_K256_L64", 2, 64, 256, 0.f},
            {"vqvae_K4096_L64", 2, 64, 4096, 0.f},
        };
        std::mt19937 brng(77);
        std::vector<float> trainRows(size_t(ntr) * IO);  // row-major, uploaded once per codec
        for (size_t r = 0; r < ntr; ++r)
            for (int c = 0; c < IO; ++c) trainRows[size_t(r) * IO + c] = Tr(int(r), c);
        // dataset buffer allocated FIRST (lowest GPU addresses) — the iGPU driver has shown
        // root-SRV instability with late-allocated large buffers in this process shape
        gpu::Buffer earlyTrain(dev, uint64_t(ntr) * IO * 4, gpu::BufferKind::Uav);
        earlyTrain.upload(trainRows.data(), uint64_t(ntr) * IO * 4);
        for (auto& cfg : cfgs) {
            double t0 = now_s();
            GpuCodec gc(dev, lk, tk, cfg.L, cfg.mode, batch, cfg.K, cfg.beta);
            gc.external_train = &earlyTrain;
            if (cfg.mode == 2) {
                // data-aligned codebook init: one forward pass on the first training rows,
                // then seed each of the K codes from a batch z (+ 0.1*std jitter to break
                // ties). Random N(0,0.3) codes start far from the z-cloud -> dead codes and
                // a violent first-contact EMA; seeding from real z's removes that transient.
                std::vector<float> x0(size_t(batch) * IO);
                for (uint32_t b = 0; b < batch; ++b)
                    for (int j = 0; j < IO; ++j) x0[size_t(b) * IO + j] = Tr(int(b), j);
                gc.bX->upload(x0.data(), x0.size() * 4);
                dev.begin();
                gc.record_forward(false);
                dev.submit_and_wait();
                std::vector<float> z0(size_t(batch) * cfg.L);
                gc.z->download(z0.data(), z0.size() * 4);
                double zmu = 0;
                for (float v : z0) zmu += double(v);
                zmu /= double(z0.size());
                double zvar = 0;
                for (float v : z0) zvar += (double(v) - zmu) * (double(v) - zmu);
                float zstd = float(std::sqrt(zvar / double(z0.size())));
                std::mt19937 kr(9);
                std::normal_distribution<float> gn(0, 1);
                std::vector<float> cbrow(size_t(cfg.K) * cfg.L);
                for (int k = 0; k < cfg.K; ++k) {
                    int src = int((uint64_t(k) * 7919u) % batch);  // coprime stride: covers B
                    for (int l = 0; l < cfg.L; ++l)
                        cbrow[size_t(k) * cfg.L + l] =
                            z0[size_t(src) * cfg.L + l] + 0.1f * zstd * gn(kr);
                }
                gc.cb->upload(cbrow.data(), cbrow.size() * 4, 0);
            }
            std::uniform_int_distribution<int> pick(0, int(ntr) - 1);
            std::vector<uint32_t> chosen;
            chosen.assign(size_t(batch), 0u);
            std::vector<float> eps(cfg.mode == 1 ? size_t(batch) * cfg.L : 0);
            std::normal_distribution<float> gn(0, 1);
            gc.bind_dataset(trainRows.data(), uint32_t(ntr));
            double train_s = 0;
            for (int s = 1; s <= steps; ++s) {
                for (auto& i2 : chosen) i2 = uint32_t(pick(brng));
                if (cfg.mode == 1)
                    for (auto& e : eps) e = gn(brng);
                double st = now_s();
                float lr = 3e-3f * 0.5f * (1.f + std::cos(float(3.14159265) * s / steps));
                if (cfg.mode == 2)
                    lr *= std::min(1.0f, float(s) / 100.0f);  // VQ warmup: 100-step ramp
                const bool trace = getenv("SAL_TRACE") != nullptr;
                if (trace) std::fprintf(stderr, "[step %d] begin\n", s);
                dev.begin();
                gc.record_batch(chosen.data(), cfg.mode == 1 ? eps.data() : nullptr);
                if (trace) std::fprintf(stderr, "[step %d] batch recorded\n", s);
                gc.record_forward(cfg.mode == 1);
                if (trace) std::fprintf(stderr, "[step %d] fwd recorded\n", s);
                gc.record_backward(nullptr);
                if (trace) std::fprintf(stderr, "[step %d] bwd recorded\n", s);
                gc.record_adam(lr, uint32_t(s));
                if (trace) std::fprintf(stderr, "[step %d] adam recorded, submitting\n", s);
                dev.submit_and_wait();
                if (trace) std::fprintf(stderr, "[step %d] submitted ok\n", s);
                train_s += now_s() - st;
            }
            if (bench) {
                char bb[160];
                std::snprintf(bb, sizeof bb, "\"codec\":\"%s\",\"steps\":%d,\"ms_per_step\":%.3f",
                              cfg.name, steps, 1000.0 * train_s / steps);
                run.event("gpu_bench_point", bb);
                continue;
            }
            float rmse = eval_rel_mse(gc, dev, lk, tk, Va, int(nva / batch));
            double shared = (384.0 * 256 + 384.0 * cfg.L + cfg.L * 384 + 256.0 * 384) * 16.0 /
                            double(npatch) / 256.0;
            double bpw = 0;
            if (cfg.mode == 2)
                bpw = std::log2(double(cfg.K)) / 256.0 +
                      double(cfg.K) * cfg.L * 16.0 / double(npatch) / 256.0 + shared;
            else if (cfg.mode == 1)
                bpw = double(cfg.L) * 32.0 / 256.0 + shared;  // mu+logvar both stored
            else
                bpw = double(cfg.L) * 16.0 / 256.0 + shared;
            char b[256];
            std::snprintf(b, sizeof b,
                          "\"codec\":\"%s\",\"bpw\":%.3f,\"rel_mse\":%.6e,\"train_s\":%.1f,"
                          "\"steps\":%d,\"ms_per_step\":%.2f",
                          cfg.name, bpw, rmse, now_s() - t0, steps,
                          1000.0 * train_s / steps);
            run.event("gpu_codec_point", b);
        }
        run.event("done", "\"status\":\"ok\"");
        return 0;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        if (iq) {
            UINT64 n = iq->GetNumStoredMessages();
            for (UINT64 i = 0; i < n && i < 20; ++i) {
                SIZE_T sz = 0;
                if (FAILED(iq->GetMessage(i, nullptr, &sz)) || sz == 0) continue;
                std::vector<char> buf(sz);
                D3D12_MESSAGE* msg = (D3D12_MESSAGE*)buf.data();
                if (SUCCEEDED(iq->GetMessage(i, msg, &sz)))
                    std::fprintf(stderr, "[d3d12] %s\n",
                                 msg->pDescription ? msg->pDescription : "");
            }
        }
        return 1;
    }
}

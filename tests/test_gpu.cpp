// gpu (SKIP without a working adapter): kernels + one-step trainer parity vs CPU fp64
/// \file
/// \brief GPU kernel and one-step-trainer parity tests (all SKIP without a working
///        D3D12 adapter — SKIP_IF(!env().ok) in every test).
///
/// Methodology: every GPU path is compared against a straight-line CPU reference
/// computed from the SAME inputs (fixed seeds), asserting max-abs or Frobenius-relative
/// error far above fp32 noise but far below any wiring bug (1e-4 Frobenius on
/// gradients, 2e-4 on GEMM elements). One suite-level GpuEnv lazily initializes the
/// Device once, on first use, so the whole suite degrades to SKIP on adapter-less
/// machines.
///
/// Resource-state discipline (the contract every dispatch here obeys, matching the
/// production wrappers in kernels.hpp): buffers are created in
/// D3D12_RESOURCE_STATE_COMMON; before a dispatch each input is transitioned
/// COMMON -> NON_PIXEL_SHADER_RESOURCE and each output COMMON -> UNORDERED_ACCESS;
/// after the dispatch the transitions are reversed back to COMMON. Nothing relies on
/// implicit state promotion/decay — the 2022 Intel iGPU driver under test mishandles
/// it (the explicit pattern is also what spike_dml.cpp documents). The
/// "gather_rows production pattern" test additionally exercises the upload path:
/// a CpuWrite staging buffer copied into place via CopyBufferRegion with
/// COPY_DEST transitions, then the gather output consumed immediately by a GEMM in
/// the next packet.
///
/// ElemGrid contract: elementwise/adam dispatches size their grid with
/// TuneParams::elem_grid(count) = ceil(count / (G*EPT)) so the shader prelude's
/// per-thread EPT run [t*EPT,(t+1)*EPT) covers every element exactly once
/// (kernels.inl SAL_ELEM_PRELUDE). The Adam tests also show the packed-state layout:
/// one buffer holds [P | M | V] regions at offsets [0, total), [total, 2*total),
/// [2*total, 3*total), passed via the 8-word constant block.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "gpu/kernels.hpp"
#include "xlab/test.hpp"

namespace gpu = sal::gpu;
using Eigen::MatrixXf;

/// \brief Reinterprets a float's bits as uint32 for packing into the 8-word kernel
///        constant block (floats ride in the same words as integers, e.g. lr/eps).
/// \param f Value to bit-cast.
/// \return Bit pattern of \p f.
static uint32_t f2u(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

/// \brief Lazily-initialized shared GPU environment for the whole suite.
///
/// On first env() call: Device::init(false) runs FIRST (it must precede the kernel
/// sets — they compile shaders against the live device), then LinAlgKernels and
/// TrainKernels are constructed. Any exception marks ok=false; every test then
/// SKIP_IFs, so a missing/broken adapter is reported, not fatal.
struct GpuEnv {
    gpu::Device dev;                             ///< D3D12 device + queue/list/fence.
    std::unique_ptr<gpu::LinAlgKernels> lk;      ///< GEMM/syrk kernel set.
    std::unique_ptr<gpu::TrainKernels> tk;       ///< Elementwise/adam/vq kernel set.
    bool ok = false;                             ///< True iff init fully succeeded.
};
/// \brief Returns the process-wide GpuEnv, initializing it on the first call.
/// \return Reference to the singleton; check \c .ok before using the device.
static GpuEnv& env() {
    static GpuEnv* e = []() -> GpuEnv* {
        auto p = new GpuEnv();
        try {
            p->dev.init(false);                       // device FIRST...
            p->lk = std::make_unique<gpu::LinAlgKernels>(p->dev);
            p->tk = std::make_unique<gpu::TrainKernels>(p->dev);
            p->ok = true;
        } catch (const std::exception&) {
            p->ok = false;
        }
        return p;
    }();
    return *e;
}

/// \brief tanh-approximation GELU forward, matching the HLSL kernel's formula
///        (0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))); CPU reference for the elem tests.
/// \param x Input value.
/// \return Activated value.
static float gelu(float x) {
    const float k0 = 0.7978845608f;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + 0.044715f * x * x * x)));
}
/// \brief Analytic GELU derivative (tanh form), CPU reference for gelu_bwd tests.
/// \param x Input value (pre-activation).
/// \return d(gelu)/dx at \p x.
static float dgelu(float x) {
    const float k0 = 0.7978845608f;
    float x3 = x + 0.044715f * x * x * x;
    float t = std::tanh(k0 * x3);
    float dt = (1 - t * t) * k0 * (1 + 3 * 0.044715f * x * x);
    return 0.5f * (1 + t) + 0.5f * x * dt;
}

/// \brief CPU reference for the NT GEMM kernel: C[M,N] = A[M,K] * B[N,K]^T with all
///        matrices row-major (B stored transposed, as the shader consumes it).
/// \param a Row-major A, M*K floats.
/// \param b Row-major B, N*K floats.
/// \param c Output, resized to M*N.
/// \param M Rows of A / C. \param N Cols of C / rows of B. \param K Inner dimension.
static void gemm_nt_ref(const std::vector<float>& a, const std::vector<float>& b,
                        std::vector<float>& c, uint32_t M, uint32_t N, uint32_t K) {
    c.assign(size_t(M) * N, 0.f);
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            float av = a[size_t(i) * K + k];
            for (uint32_t j = 0; j < N; ++j) c[size_t(i) * N + j] += av * b[size_t(j) * K + k];
        }
}

XLAB_TEST("gemm nt/nn/tn vs CPU incl. packed-buffer offsets") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& lk = *env().lk;
    const uint32_t M = 96, N = 80, K = 64, PAD = 32;  // PAD exercises the offset addressing
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float> a(size_t(M) * K), b(size_t(N) * K), b2(size_t(K) * N);
    for (auto& v : a) v = u(rng);
    for (auto& v : b) v = u(rng);
    for (auto& v : b2) v = u(rng);

    // pack: [pad | A | pad | B | pad | B2 | pad | A2] — full layout, one buffer, one upload
    std::vector<float> a2(size_t(K) * M);
    for (auto& v : a2) v = u(rng);
    const uint32_t aoff = PAD;
    const uint32_t boff = aoff + uint32_t(M) * K + PAD;
    const uint32_t b2off = boff + uint32_t(N) * K + PAD;
    const uint32_t a2off = b2off + uint32_t(K) * N + PAD;
    std::vector<float> pack(a2off + uint32_t(K) * M + PAD);
    for (auto& v : pack) v = u(rng);
    std::memcpy(pack.data() + aoff, a.data(), a.size() * 4);
    std::memcpy(pack.data() + boff, b.data(), b.size() * 4);
    std::memcpy(pack.data() + b2off, b2.data(), b2.size() * 4);
    std::memcpy(pack.data() + a2off, a2.data(), a2.size() * 4);
    gpu::Buffer bp(d, pack.size() * 4, gpu::BufferKind::Uav);
    bp.upload(pack.data(), pack.size() * 4);
    gpu::Buffer out(d, size_t(M) * N * 4, gpu::BufferKind::Uav);

    std::vector<float> ref;
    // NT with offsets: C = A * B^T
    d.begin();
    lk.gemm_nt_rec(d, bp, bp, out, M, N, K, aoff, boff, 0);
    d.submit_and_wait();
    std::vector<float> got(size_t(M) * N);
    out.download(got.data(), got.size() * 4);
    gemm_nt_ref(a, b, ref, M, N, K);
    double md = 0;
    for (size_t i = 0; i < ref.size(); ++i) md = std::max(md, double(std::fabs(got[i] - ref[i])));
    CHECK_MSG(md < 2e-4, ("gemm_nt offsets max diff " + std::to_string(md)).c_str());

    // NN with offsets: C = A * B2
    d.begin();
    lk.gemm_nn_rec(d, bp, bp, out, M, N, K, aoff, b2off, 0);
    d.submit_and_wait();
    out.download(got.data(), got.size() * 4);
    ref.assign(size_t(M) * N, 0.f);
    for (uint32_t i = 0; i < M; ++i)
        for (uint32_t k = 0; k < K; ++k) {
            float av = a[size_t(i) * K + k];
            for (uint32_t j = 0; j < N; ++j) ref[size_t(i) * N + j] += av * b2[size_t(k) * N + j];
        }
    md = 0;
    for (size_t i = 0; i < ref.size(); ++i) md = std::max(md, double(std::fabs(got[i] - ref[i])));
    CHECK_MSG(md < 2e-4, ("gemm_nn offsets max diff " + std::to_string(md)).c_str());

    // TN with offsets: C = A2^T * B2 where A2 is [K, M] stored at a2off (already in bp)
    d.begin();
    lk.gemm_tn_rec(d, bp, bp, out, M, N, K, a2off, b2off, 0);
    d.submit_and_wait();
    out.download(got.data(), got.size() * 4);
    ref.assign(size_t(M) * N, 0.f);
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t j = 0; j < N; ++j) {
            float s = 0;
            for (uint32_t i2 = 0; i2 < M; ++i2) s += a2[size_t(k) * M + i2] * b2[size_t(k) * N + j];
            (void)s;
        }
    for (uint32_t k = 0; k < K; ++k)
        for (uint32_t i2 = 0; i2 < M; ++i2) {
            float av = a2[size_t(k) * M + i2];
            for (uint32_t j = 0; j < N; ++j) ref[size_t(i2) * N + j] += av * b2[size_t(k) * N + j];
        }
    md = 0;
    for (size_t i = 0; i < ref.size(); ++i) md = std::max(md, double(std::fabs(got[i] - ref[i])));
    CHECK_MSG(md < 2e-4, ("gemm_tn offsets max diff " + std::to_string(md)).c_str());
}

XLAB_TEST("elem kernels: gelu and mse_grad vs CPU") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& tk = *env().tk;
    const uint32_t n = 4096;
    std::mt19937 rng(4);
    std::uniform_real_distribution<float> u(-3, 3);
    std::vector<float> a(n), b(n);
    for (auto& v : a) v = u(rng);
    for (auto& v : b) v = u(rng);
    gpu::Buffer ba(d, n * 4, gpu::BufferKind::Uav);
    gpu::Buffer bb(d, n * 4, gpu::BufferKind::Uav);
    gpu::Buffer bo(d, n * 4, gpu::BufferKind::Uav);
    ba.upload(a.data(), n * 4);
    bb.upload(b.data(), n * 4);
    uint32_t c[8] = {n, 0, 0, 0, 0, 0, 0, 0};
    d.begin();
    tk.elem(d, tk.gelu, ba, ba, bo, c, n);
    d.submit_and_wait();
    std::vector<float> got(n);
    bo.download(got.data(), n * 4);
    double md = 0;
    for (uint32_t i = 0; i < n; ++i)
        md = std::max(md, double(std::fabs(got[i] - gelu(a[i]))));
    CHECK_MSG(md < 1e-5, ("gelu max diff " + std::to_string(md)).c_str());

    const float scale = 2.0f / 777.0f;
    uint32_t g[8] = {n, 0, 0, 0, f2u(scale), 0, 0, 0};
    d.begin();
    tk.elem(d, tk.mse_grad, ba, bb, bo, g, n);
    d.submit_and_wait();
    bo.download(got.data(), n * 4);
    md = 0;
    for (uint32_t i = 0; i < n; ++i)
        md = std::max(md, double(std::fabs(got[i] - scale * (a[i] - b[i]))));
    CHECK_MSG(md < 1e-6, ("mse_grad max diff " + std::to_string(md)).c_str());
}

XLAB_TEST("packed-state Adam single step vs CPU") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& tk = *env().tk;
    const uint32_t count = 1024, region = 1024;
    std::mt19937 rng(5);
    std::normal_distribution<float> gn(0, 1);
    std::vector<float> p0(count), g(count);
    for (auto& v : p0) v = gn(rng);
    for (auto& v : g) v = gn(rng);
    std::vector<float> state(p0);
    state.insert(state.end(), region * 2, 0.f);  // [P | M | V]
    gpu::Buffer bs(d, state.size() * 4, gpu::BufferKind::Uav);
    gpu::Buffer bg(d, count * 4, gpu::BufferKind::Uav);
    bs.upload(state.data(), state.size() * 4);
    bg.upload(g.data(), count * 4);
    const float lr = 3e-3f, eps = 1e-8f;
    d.begin();
    d.transition(bg.res(), D3D12_RESOURCE_STATE_COMMON,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.transition(bs.res(), D3D12_RESOURCE_STATE_COMMON,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    uint32_t c[8] = {0, region, 2 * region, count, f2u(lr), f2u(1.0f), f2u(eps), 0};
    tk.adam.dispatch(d, bg, bg, bs, c, tk.p.elem_grid(count), 1, 1);
    d.transition(bg.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 D3D12_RESOURCE_STATE_COMMON);
    d.transition(bs.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    d.submit_and_wait();
    std::vector<float> back(state.size());
    bs.download(back.data(), back.size() * 4);
    double md = 0;
    uint32_t mdi = 0;
    for (uint32_t i = 0; i < count; ++i) {
        float m = 0.1f * g[i], v = 0.001f * g[i] * g[i];
        float upd = lr * (m / 0.1f) / (std::sqrt(v / 0.001f) + eps);
        double d = double(std::fabs(back[i] - (p0[i] - upd)));
        if (d > md) { md = d; mdi = i; }
    }
    CHECK_MSG(md < 1e-6, ("adam max diff " + std::to_string(md) + " at i=" + std::to_string(mdi) +
                          " g=" + std::to_string(g[mdi]) + " gpu=" + std::to_string(back[mdi]) +
                          " p0=" + std::to_string(p0[mdi]))
                             .c_str());
}

// ---- one-step trainer parity (AE), random data, no shard dependency ----
static const int IO = 256, HID = 384;

XLAB_TEST("trainer one-step parity (AE) vs CPU fp64") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& lk = *env().lk;
    auto& tk = *env().tk;
    const uint32_t B = 128, L = 32;
    std::mt19937 rng(1234);
    std::normal_distribution<float> gn(0, 1);
    auto mk = [&](uint32_t r, uint32_t c, uint32_t fanin) {
        MatrixXf m(r, c);
        float s = std::sqrt(2.0f / float(fanin));
        for (auto& v : m.reshaped()) v = gn(rng) * s;
        return m;
    };
    MatrixXf W1 = mk(HID, IO, IO), W2 = mk(L, HID, HID), W3 = mk(HID, L, L), W4 = mk(IO, HID, HID);
    MatrixXf X(B, IO);
    for (auto& v : X.reshaped()) v = gn(rng);

    // CPU reference (fp64 accumulate via double loops would be slow; float Eigen is the ref)
    auto fwd = [&](const MatrixXf& Xin, MatrixXf& A1, MatrixXf& H1, MatrixXf& Z, MatrixXf& A3,
                   MatrixXf& H3) {
        A1 = Xin * W1.transpose();
        H1 = A1;
        for (auto& v : H1.reshaped()) v = gelu(v);
        Z = H1 * W2.transpose();
        A3 = Z * W3.transpose();
        H3 = A3;
        for (auto& v : H3.reshaped()) v = gelu(v);
        return H3 * W4.transpose();
    };
    MatrixXf A1, H1, Z, A3, H3;
    MatrixXf Xh = fwd(X, A1, H1, Z, A3, H3);
    MatrixXf D = 2.0f / float(B * IO) * (Xh - X);
    MatrixXf GH3 = D * W4;
    for (int r = 0; r < GH3.rows(); ++r)
        for (int c = 0; c < GH3.cols(); ++c) GH3(r, c) *= dgelu(A3(r, c));
    MatrixXf GZ = GH3 * W3;
    MatrixXf gW4 = D.transpose() * H3, gW3 = GH3.transpose() * Z;
    MatrixXf GH1 = GZ * W2;
    MatrixXf gW2 = GZ.transpose() * H1;
    for (int r = 0; r < GH1.rows(); ++r)
        for (int c = 0; c < GH1.cols(); ++c) GH1(r, c) *= dgelu(A1(r, c));
    MatrixXf gW1 = GH1.transpose() * X;
    struct NW {
        MatrixXf *w, *g;
    } refs[4] = {{&W1, &gW1}, {&W2, &gW2}, {&W3, &gW3}, {&W4, &gW4}};

    // GPU buffers
    uint64_t total = uint64_t(W1.size()) + W2.size() + W3.size() + W4.size();
    std::vector<float> pflat;
    auto append = [](std::vector<float>& v, const MatrixXf& m) {
        for (int r = 0; r < m.rows(); ++r)
            for (int c = 0; c < m.cols(); ++c) v.push_back(m(r, c));
    };
    append(pflat, W1);
    append(pflat, W2);
    append(pflat, W3);
    append(pflat, W4);
    uint64_t o1 = 0, o2 = W1.size(), o3 = o2 + W2.size(), o4 = o3 + W3.size();
    gpu::Buffer state(d, total * 4 * 3, gpu::BufferKind::Uav);
    gpu::Buffer gbuf(d, total * 4, gpu::BufferKind::Uav);
    state.upload(pflat.data(), total * 4);
    std::vector<float> zero(total * 2, 0.f);
    state.upload(zero.data(), total * 8, total * 4);
    gpu::Buffer bX(d, uint64_t(B) * IO * 4, gpu::BufferKind::Uav);
    gpu::Buffer a1b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer h1b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer zb(d, uint64_t(B) * L * 4, gpu::BufferKind::Uav);
    gpu::Buffer a3b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer h3b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer xhb(d, uint64_t(B) * IO * 4, gpu::BufferKind::Uav);
    gpu::Buffer dqb(d, uint64_t(B) * IO * 4, gpu::BufferKind::Uav);
    gpu::Buffer gh3b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer gh1b(d, uint64_t(B) * HID * 4, gpu::BufferKind::Uav);
    gpu::Buffer gzb(d, uint64_t(B) * L * 4, gpu::BufferKind::Uav);
    std::vector<float> xf(size_t(B) * IO);
    for (int r = 0; r < int(B); ++r)
        for (int c = 0; c < IO; ++c) xf[size_t(r) * IO + c] = X(r, c);
    bX.upload(xf.data(), xf.size() * 4);

    auto woff = [&](uint64_t o) { return uint32_t(o); };
    d.begin();
    lk.gemm_nt_rec(d, bX, state, a1b, B, HID, IO, 0, woff(o1));
    uint32_t c1[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
    tk.elem(d, tk.gelu, a1b, a1b, h1b, c1, B * HID);
    lk.gemm_nt_rec(d, h1b, state, zb, B, L, HID, 0, woff(o2));
    lk.gemm_nt_rec(d, zb, state, a3b, B, HID, L, 0, woff(o3));
    tk.elem(d, tk.gelu, a3b, a3b, h3b, c1, B * HID);
    lk.gemm_nt_rec(d, h3b, state, xhb, B, IO, HID, 0, woff(o4));
    // backward
    uint32_t g8[8] = {B * IO, 0, 0, 0, f2u(2.0f / float(B * IO)), 0, 0, 0};
    tk.elem(d, tk.mse_grad, xhb, bX, dqb, g8, B * IO);
    lk.gemm_nn_rec(d, dqb, state, gh3b, B, HID, IO, 0, woff(o4));
    lk.gemm_tn_rec(d, dqb, h3b, gbuf, IO, HID, B, 0, 0, woff(o4));
    uint32_t cb[8] = {B * HID, 0, 0, 0, 0, 0, 0, 0};
    tk.elem(d, tk.gelu_bwd, a3b, gh3b, gh1b, cb, B * HID);        // gh1b = GH3*gelu'
    lk.gemm_nn_rec(d, gh1b, state, gzb, B, L, HID, 0, woff(o3));  // gzb = GZ (zb keeps Z)
    lk.gemm_tn_rec(d, gh1b, zb, gbuf, HID, L, B, 0, 0, woff(o3)); // gW3 = GH3'^T Z
    lk.gemm_nn_rec(d, gzb, state, gh3b, B, HID, L, 0, woff(o2));  // gh3b = GH1
    lk.gemm_tn_rec(d, gzb, h1b, gbuf, L, HID, B, 0, 0, woff(o2)); // gW2 = GZ^T H1
    tk.elem(d, tk.gelu_bwd, a1b, gh3b, gh1b, cb, B * HID);  // gh1b = GH1*gelu'
    lk.gemm_tn_rec(d, gh1b, bX, gbuf, HID, IO, B, 0, 0, woff(o1));
    // adam step 1 on each tensor
    const float lr = 3e-3f, eps = 1e-8f;
    for (auto [o, cnt] : std::initializer_list<std::pair<uint64_t, uint64_t>>{
             {o1, W1.size()}, {o2, W2.size()}, {o3, W3.size()}, {o4, W4.size()}}) {
        d.transition(gbuf.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        d.transition(state.res(), D3D12_RESOURCE_STATE_COMMON,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        uint32_t ac[8] = {uint32_t(o), uint32_t(total + o), uint32_t(2 * total + o),
                          uint32_t(cnt), f2u(lr), f2u(1.0f), f2u(eps), uint32_t(o)};
        tk.adam.dispatch(d, gbuf, gbuf, state, ac, tk.p.elem_grid(uint32_t(cnt)), 1, 1);
        d.transition(gbuf.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                     D3D12_RESOURCE_STATE_COMMON);
        d.transition(state.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COMMON);
    }
    d.submit_and_wait();

    // compare gradients first (from gbuf), then post-Adam params
    std::vector<float> gflat(total);
    gbuf.download(gflat.data(), total * 4);
    auto cmp_mat = [&](const char* what, uint64_t off, const MatrixXf& ref) {
        double dsq = 0, nsq = 0;
        for (int r = 0; r < ref.rows(); ++r)
            for (int c = 0; c < ref.cols(); ++c) {
                double gv = gflat[off + uint64_t(r) * ref.cols() + c];
                double rv = ref(r, c);
                dsq += (gv - rv) * (gv - rv);
                nsq += rv * rv;
            }
        double rel = std::sqrt(dsq) / std::max(1e-30, std::sqrt(nsq));
        CHECK_MSG(rel < 1e-4, (std::string(what) + " grad frob rel " + std::to_string(rel)).c_str());
    };
    cmp_mat("gW1", o1, gW1);
    cmp_mat("gW2", o2, gW2);
    cmp_mat("gW3", o3, gW3);
    cmp_mat("gW4", o4, gW4);

    std::vector<float> pback(total);
    state.download(pback.data(), total * 4);
    auto adam1 = [&](float p, float g) {
        float m = 0.1f * g, v = 0.001f * g * g;
        return p - lr * (m / 0.1f) / (std::sqrt(v / 0.001f) + eps);
    };
    auto cmp_post = [&](uint64_t o, const MatrixXf& p, const MatrixXf& g) {
        double dsq = 0, nsq = 0;
        for (int r = 0; r < p.rows(); ++r)
            for (int c = 0; c < p.cols(); ++c) {
                double gv = pback[o + uint64_t(r) * p.cols() + c];
                double rv = adam1(p(r, c), g(r, c));
                dsq += (gv - rv) * (gv - rv);
                nsq += rv * rv;
            }
        double rel = std::sqrt(dsq) / std::max(1e-30, std::sqrt(nsq));
        CHECK_MSG(rel < 1e-4, ("post-adam frob rel " + std::to_string(rel)).c_str());
    };
    cmp_post(o1, W1, gW1);
    cmp_post(o2, W2, gW2);
    cmp_post(o3, W3, gW3);
    cmp_post(o4, W4, gW4);
    (void)refs;
}

XLAB_TEST("gather_rows kernel vs CPU (tiny)") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& tk = *env().tk;
    const uint32_t N = 8, B = 4;
    std::vector<float> train(size_t(N) * IO);
    std::vector<uint32_t> idx{3u, 0u, 7u, 3u};
    for (uint32_t i = 0; i < train.size(); ++i) train[i] = float(i);
    gpu::Buffer bt(d, train.size() * 4, gpu::BufferKind::Uav);
    gpu::Buffer bi(d, B * 4, gpu::BufferKind::Uav);
    gpu::Buffer bo(d, size_t(B) * IO * 4, gpu::BufferKind::Uav);
    bt.upload(train.data(), train.size() * 4);
    bi.upload(idx.data(), B * 4);
    d.begin();
    d.transition(bt.res(), D3D12_RESOURCE_STATE_COMMON,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.transition(bi.res(), D3D12_RESOURCE_STATE_COMMON,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.transition(bo.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    uint32_t c[8] = {B, uint32_t(IO), 0, 0, 0, 0, 0, 0};
    tk.gather_rows.dispatch(d, bt, bi, bo, c, tk.p.elem_grid(B * uint32_t(IO)), 1, 1);
    d.transition(bt.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 D3D12_RESOURCE_STATE_COMMON);
    d.transition(bi.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 D3D12_RESOURCE_STATE_COMMON);
    d.transition(bo.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    d.submit_and_wait();
    std::vector<float> got(size_t(B) * IO);
    bo.download(got.data(), got.size() * 4);
    for (uint32_t b = 0; b < B; ++b)
        for (int j = 0; j < IO; ++j)
            CHECK(got[size_t(b) * IO + j] == train[size_t(idx[b]) * IO + j]);
}

XLAB_TEST("gather_rows production pattern (40MB, in-list idx copy)") {
    SKIP_IF(!env().ok);
    auto& d = env().dev;
    auto& tk = *env().tk;
    const uint32_t N = 40000, B = 512;
    std::vector<float> train(size_t(N) * IO, 0.5f);
    std::vector<uint32_t> idx(B);
    std::mt19937 rng(2);
    std::uniform_int_distribution<uint32_t> pick(0, N - 1);
    for (auto& i : idx) i = pick(rng);
    for (uint32_t b = 0; b < B; ++b)  // mark rows so we can verify
        for (int j = 0; j < IO; ++j) train[size_t(idx[b]) * IO + j] = float(b) + float(j) / 1024.f;
    gpu::Buffer bt(d, train.size() * 4, gpu::BufferKind::SrcRead);
    gpu::Buffer bi(d, B * 4, gpu::BufferKind::SrcRead);
    gpu::Buffer stage(d, B * 4, gpu::BufferKind::CpuWrite);
    gpu::Buffer bo(d, size_t(B) * IO * 4, gpu::BufferKind::Uav);
    bt.upload(train.data(), train.size() * 4);
    {
        void* p = nullptr;
        D3D12_RANGE none{0, 0};
        stage.res()->Map(0, &none, &p);
        std::memcpy(p, idx.data(), size_t(B) * 4);
        stage.res()->Unmap(0, nullptr);
    }
    d.begin();
    d.transition(bi.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
    d.list()->CopyBufferRegion(bi.res(), 0, stage.res(), 0, uint64_t(B) * 4);
    d.transition(bi.res(), D3D12_RESOURCE_STATE_COPY_DEST,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.transition(bt.res(), D3D12_RESOURCE_STATE_COMMON,
                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    d.transition(bo.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    uint32_t c[8] = {B, uint32_t(IO), 0, 0, 0, 0, 0, 0};
    tk.gather_rows.dispatch(d, bt, bi, bo, c, tk.p.elem_grid(B * uint32_t(IO)), 1, 1);
    d.transition(bt.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 D3D12_RESOURCE_STATE_COMMON);
    d.transition(bi.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                 D3D12_RESOURCE_STATE_COMMON);
    d.transition(bo.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COMMON);
    d.submit_and_wait();
    std::vector<float> got(size_t(B) * IO);
    bo.download(got.data(), got.size() * 4);
    double md = 0;  // compare against the row content AS MARKED (collisions overwrite)
    for (uint32_t b = 0; b < B; ++b)
        for (int j = 0; j < IO; ++j)
            md = std::max(md, double(std::fabs(
                                     got[size_t(b) * IO + j] -
                                     train[size_t(idx[b]) * IO + j])));
    CHECK_MSG(md < 1e-5, ("gather 40MB max diff " + std::to_string(md)).c_str());

    // replicate the trainer continuation: gemm reads the gather output immediately after
    {
        std::vector<float> w(size_t(HID) * IO);
        std::mt19937 wr(6);
        std::uniform_real_distribution<float> wu(-0.05f, 0.05f);
        for (auto& v : w) v = wu(wr);
        gpu::Buffer bw(d, w.size() * 4, gpu::BufferKind::SrcRead);
        bw.upload(w.data(), w.size() * 4);
        gpu::Buffer bout(d, size_t(B) * HID * 4, gpu::BufferKind::Uav);
        d.begin();
        env().lk->gemm_nt_rec(d, bo, bw, bout, B, uint32_t(HID), uint32_t(IO));
        d.submit_and_wait();
        std::vector<float> g2(size_t(B) * HID);
        bout.download(g2.data(), g2.size() * 4);
        double md2 = 0;
        for (uint32_t b = 0; b < B; ++b)
            for (int h = 0; h < HID; ++h) {
                float ref = 0;
                for (int j = 0; j < IO; ++j) ref += got[size_t(b) * IO + j] * w[size_t(h) * IO + j];
                md2 = std::max(md2, double(std::fabs(g2[size_t(b) * HID + h] - ref)));
            }
        CHECK_MSG(md2 < 5e-3, ("gather+gemm max diff " + std::to_string(md2)).c_str());
    }
}

/// \brief Entry point: runs the "gpu" suite. Requires a working D3D12 adapter;
///        every test SKIPs otherwise (suite still exits 0).
/// \return 0 iff all tests passed or skipped (CTest-compatible).
int main() { return xlab::test::run_all("gpu"); }

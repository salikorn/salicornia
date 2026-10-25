// salikorn / GPU-linalg spike — run analysis linear algebra on the iGPU (user direction),
// with an explicit precision gate before approving it ("make sure it is OK to do so").
//   --validate : fp32-GPU Gram + hybrid GPU subspace eigensolver vs CPU fp64 reference on a
//                real shard tensor; gates on trace/lambda/tail-energy deviations.
//   --bench    : wall-clock CPU-only vs hybrid pipeline.
// Hybrid = GPU does G*V multiplies (chunked <2s dispatches); CPU does Rayleigh+Cholesky QR.
/// \file
/// \brief GPU-linalg spike: fp32 Gram + hybrid subspace eigensolver on the iGPU, with
///        an explicit precision gate against a CPU fp64 reference.
///
/// Division of labor: the GPU performs the O(n^2 k) G*V products with tiled fp32
/// shaders (chunked column blocks so every submitted packet stays under the ~2 s WDDM
/// TDR budget — see gpu_gram's per-chunk submits); the CPU performs the fp64
/// Rayleigh quotients and Cholesky-QR orthonormalization between iterations.
/// \p --validate gates the GPU path against la::topk_eigen on fp64 data (trace within
/// 1e-5, lambda_1 within 1e-4, tail energies at {0.25,0.5,0.75,1.0} of k within 1e-4)
/// and exits non-zero if the gate fails; \p --bench just reports wall-clock. The
/// embedded shaders follow the same resource-state discipline as the production
/// Device wrapper (explicit COMMON <-> SRV/UAV transitions, no implicit promotion).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "compress/linalg.hpp"
#include "gpu/cs.hpp"
#include "st/safetensors.hpp"
#include "xlab/fp16.hpp"
#include "xlab/log.hpp"

namespace la = sal::la;
namespace gpu = sal::gpu;

static const char* kSyrkHlsl = R"(
StructuredBuffer<float> X : register(t0);   // [m, n] row-major
RWStructuredBuffer<float> G : register(u0); // [n, n] row-major, G = X^T X
uint4 p0 : register(b0);  // x=m, y=n, z=colbase (c1 chunk), w=unused
groupshared float T1[16][16];
groupshared float T2[16][16];
[numthreads(16,16,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint m = p0.x, n = p0.y;
    uint c1 = p0.z + gid.x * 16 + tid.x;   // chunked output columns
    uint c2 = gid.y * 16 + tid.y;
    float acc = 0.0;
    for (uint i0 = 0; i0 < m; i0 += 16) {
        T1[tid.x][tid.y] = (c1 < n && i0 + tid.y < m) ? X[(i0 + tid.y) * n + c1] : 0.0;
        T2[tid.y][tid.x] = (c2 < n && i0 + tid.x < m) ? X[(i0 + tid.x) * n + c2] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < 16; ++t) acc += T1[tid.x][t] * T2[tid.y][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (c1 < n && c2 < n) G[c1 * n + c2] = acc;   // symmetric by construction of the sweep
}
)";

static const char* kGemmNnHlsl = R"(
StructuredBuffer<float> A : register(t0);   // [M, K] row-major
StructuredBuffer<float> B : register(t1);   // [K, N] row-major
RWStructuredBuffer<float> C : register(u0); // [M, N] row-major, C = A * B
uint4 p0 : register(b0);  // x=M, y=N, z=K, w=colbase (N chunk)
groupshared float Ta[16][16];
groupshared float Tb[16][16];
[numthreads(16,16,1)]
void cs_main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    const uint M = p0.x, N = p0.y, K = p0.z;
    uint row = gid.y * 16 + tid.y;
    uint col = p0.w + gid.x * 16 + tid.x;
    float acc = 0.0;
    for (uint k0 = 0; k0 < K; k0 += 16) {
        Ta[tid.y][tid.x] = (row < M && k0 + tid.x < K) ? A[row * K + k0 + tid.x] : 0.0;
        Tb[tid.x][tid.y] = (col < N && k0 + tid.y < K) ? B[(k0 + tid.y) * N + col] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        [unroll] for (uint t = 0; t < 16; ++t) acc += Ta[tid.y][t] * Tb[tid.x][t];
        GroupMemoryBarrierWithGroupSync();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}
)";

/// \brief Monotonic steady-clock timestamp in seconds (stage timing).
static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

// Loads a real bf16 2-D tensor as row-major fp32 (m x n), plus an fp64 Gram on CPU.
/// \brief A real shard tensor loaded for GPU-vs-CPU comparison.
struct TensorData {
    size_t m = 0, n = 0;         ///< Tensor shape (rows x cols).
    std::vector<float> x32;      ///< Row-major [m, n] fp32 copy (bf16 widened).
    std::vector<double> gram64;  ///< [n, n] fp64 CPU Gram of x32, the reference.
    std::string name;            ///< Tensor name in the shard.
};

/// \brief Loads one 2-D BF16 tensor from a shard, widens it to fp32 row-major
///        (parallel over rows), and computes the fp64 CPU Gram as ground truth.
/// \param shard Safetensors file path.
/// \param name  Exact tensor name.
/// \exception std::runtime_error If the tensor is missing or not BF16.
/// \return Filled TensorData.
static TensorData load_tensor(const std::string& shard, const std::string& name) {
    sal::st::SafeTensorsFile f{std::filesystem::path(shard)};
    const sal::st::TensorInfo* t = f.find(name);
    if (!t) throw std::runtime_error("tensor not found: " + name);
    TensorData d;
    d.name = name;
    d.m = size_t(t->shape[0]);
    d.n = size_t(t->shape[1]);
    if (t->dtype != "BF16") throw std::runtime_error("expected BF16 tensor");
    const uint16_t* X = reinterpret_cast<const uint16_t*>(t->data);
    d.x32.resize(d.m * d.n);
    la::parallel_for(d.m, la::nthreads(), [&](size_t r) {
        float* dst = d.x32.data() + r * d.n;
        const uint16_t* src = X + r * d.n;
        for (size_t c = 0; c < d.n; ++c) dst[c] = sal::bf16_to_f32(src[c]);
    });
    d.gram64.assign(d.n * d.n, 0.0);
    la::gram_f32(d.x32.data(), d.m, d.n, d.gram64.data());
    return d;
}

// fp32 GPU Gram via chunked syrk dispatches (each < ~2 s).
/// \brief Computes G = X^T X with the tiled fp32 syrk shader, one chunk of output
///        columns per submitted packet.
/// \note TDR discipline: WDDM applies the ~2 s timeout to each submitted packet, so
///       a single ExecuteCommandLists containing several ~0.5 s dispatches can still
///       trip it — hence one chunk (default 512 columns) per begin/submit_and_wait.
///       X is transitioned to NON_PIXEL_SHADER_RESOURCE once up front; G goes
///       COMMON -> UNORDERED_ACCESS -> COMMON per chunk.
/// \param dev   Initialized device.
/// \param syrk  Compiled syrk kernel.
/// \param x     Row-major [m, n] fp32 input.
/// \param m,n   Shape of \p x.
/// \param secs  Optional out: total wall-clock seconds.
/// \param chunk Output-column block per packet.
/// \return Row-major [n, n] fp32 Gram, downloaded after the last chunk.
static std::vector<float> gpu_gram(gpu::Device& dev, gpu::Kernel& syrk, const std::vector<float>& x,
                                   size_t m, size_t n, double* secs = nullptr,
                                   uint32_t chunk = 512) {
    gpu::Buffer bx(dev, uint64_t(m) * n * 4, gpu::BufferKind::SrcRead);
    gpu::Buffer bg(dev, uint64_t(n) * n * 4, gpu::BufferKind::Uav);
    bx.upload(x.data(), uint64_t(m) * n * 4);
    double t0 = now_s();
    // One chunk per submit: WDDM TDR applies to whole submitted packets, so a single
    // ExecuteCommandLists containing several ~0.5 s dispatches can still trip the 2 s limit.
    dev.begin();
    dev.transition(bx.res(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dev.submit_and_wait();
    for (uint32_t c0 = 0; c0 < n; c0 += chunk) {
        dev.begin();
        dev.transition(bg.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        uint32_t consts[8] = {uint32_t(m), uint32_t(n), c0, 0, 0, 0, 0, 0};
        syrk.dispatch(dev, bx, bx, bg, consts, (chunk + 15) / 16, uint32_t((n + 15) / 16), 1);
        dev.transition(bg.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
        dev.submit_and_wait();
    }
    if (secs) *secs = now_s() - t0;
    std::vector<float> g(size_t(n) * n);
    bg.download(g.data(), uint64_t(n) * n * 4);
    return g;
}

// Small-scale syrk correctness check before any big dispatch.
/// \brief Small-scale syrk correctness gate: 1024x256 Gram vs a double-precision CPU
///        reference, max abs diff < 1e-2. Runs BEFORE any large dispatch; throws on
///        failure. Logs \c syrk_selftest.
/// \exception std::runtime_error If the tolerance is exceeded.
static void syrk_selftest(gpu::Device& dev, gpu::Kernel& syrk, xlab::Run& run) {
    const size_t m = 1024, n = 256;
    std::mt19937 rng(3);
    std::uniform_real_distribution<float> u(-1, 1);
    std::vector<float> x(m * n);
    for (auto& v : x) v = u(rng);
    std::vector<float> g = gpu_gram(dev, syrk, x, m, n, nullptr, 256);
    double maxd = 0;
    for (size_t c1 = 0; c1 < n; ++c1)
        for (size_t c2 = 0; c2 < n; ++c2) {
            double ref = 0;
            for (size_t i = 0; i < m; ++i) ref += double(x[i * n + c1]) * double(x[i * n + c2]);
            maxd = std::max(maxd, std::fabs(double(g[c1 * n + c2]) - ref));
        }
    char b[128];
    std::snprintf(b, sizeof b, "\"m\":%zu,\"n\":%zu,\"max_abs_diff\":%.3e", m, n, maxd);
    run.event("syrk_selftest", b);
    if (maxd > 1e-2) throw std::runtime_error("syrk selftest FAILED");
}

// Hybrid top-k eigensolver: GPU W=G*V (fp32), CPU Rayleigh + Cholesky QR (fp64 on fp32 data).
/// \brief Result of the hybrid subspace eigensolver.
struct Hybrid {
    std::vector<double> values;   ///< Eigenvalues, descending.
    std::vector<double> vectors;  ///< n x k row-major eigenvectors, ordered like values.
    int iters = 0;                ///< Subspace iterations performed (including the final one).
    double gpu_s = 0, cpu_s = 0;  ///< Cumulative seconds spent on GPU / CPU stages.
};

/// \brief Hybrid top-k eigensolver for a symmetric [n, n] fp32 matrix: the GPU
///        computes W = G*V with the tiled NN GEMM shader (512-column chunks, one
///        packet per chunk under the TDR budget); the CPU computes Rayleigh quotients
///        and re-orthonormalizes via la::orthonormalize each iteration, converging
///        when the max eigenvalue change < tol * max(1, lambda_1). V is uploaded
///        row-major per iteration (Eigen is column-major — converted by index).
/// \param dev        Initialized device.
/// \param gemm       Compiled NN GEMM kernel used as the G*V multiply.
/// \param gf         Row-major [n, n] fp32 matrix (the GPU Gram).
/// \param n          Matrix edge.
/// \param k          Subspace width (number of eigenpairs wanted).
/// \param max_iters  Iteration cap (default 20).
/// \param tol        Convergence tolerance (default 1e-11).
/// \return Hybrid result with values sorted descending.
static Hybrid gpu_topk_eigen(gpu::Device& dev, gpu::Kernel& gemm, const std::vector<float>& gf,
                             size_t n, size_t k, int max_iters = 20, double tol = 1e-11) {
    Hybrid out;
    gpu::Buffer bg(dev, uint64_t(n) * n * 4, gpu::BufferKind::SrcRead);
    bg.upload(gf.data(), uint64_t(n) * n * 4);
    gpu::Buffer bv(dev, uint64_t(n) * k * 4, gpu::BufferKind::SrcRead);
    gpu::Buffer bw(dev, uint64_t(n) * k * 4, gpu::BufferKind::Uav);

    std::mt19937_64 rng(7);
    std::normal_distribution<double> gauss(0, 1);
    Eigen::MatrixXd V(n, k);
    for (auto& v : V.reshaped()) v = gauss(rng);
    if (!la::orthonormalize(V.data(), n, k)) throw std::runtime_error("orth init failed");

    Eigen::MatrixXd W(n, k);
    Eigen::VectorXd lam(k), lamPrev = Eigen::VectorXd::Zero(k);
    std::vector<float> v32(size_t(n) * k), w32(size_t(n) * k);
    bool converged = false;
    int it = 0;
    for (; it < max_iters && !converged; ++it) {
        // upload V as ROW-MAJOR [n,k] fp32 (V is Eigen column-major — convert by index)
        for (size_t r = 0; r < n; ++r)
            for (size_t c = 0; c < k; ++c) v32[r * k + c] = float(V(r, Eigen::Index(c)));
        bv.upload(v32.data(), v32.size() * 4);
        double tg0 = now_s();
        const uint32_t CH = 512;
        dev.begin();
        dev.transition(bg.res(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dev.transition(bv.res(), D3D12_RESOURCE_STATE_COMMON,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        dev.submit_and_wait();
        for (uint32_t c0 = 0; c0 < k; c0 += CH) {
            dev.begin();
            dev.transition(bw.res(), D3D12_RESOURCE_STATE_COMMON,
                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            uint32_t consts[8] = {uint32_t(n), uint32_t(k), uint32_t(n), c0, 0, 0, 0, 0};
            gemm.dispatch(dev, bg, bv, bw, consts, (CH + 15) / 16, uint32_t((n + 15) / 16), 1);
            dev.transition(bw.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                           D3D12_RESOURCE_STATE_COMMON);
            dev.submit_and_wait();
        }
        out.gpu_s += now_s() - tg0;
        bw.download(w32.data(), w32.size() * 4);
        double tc0 = now_s();
        for (size_t r = 0; r < n; ++r)
            for (size_t c = 0; c < k; ++c) W(r, Eigen::Index(c)) = double(w32[r * k + c]);
        double max_change = 0;
        for (Eigen::Index c = 0; c < Eigen::Index(k); ++c) {
            lam(c) = V.col(c).dot(W.col(c));
            max_change = std::max(max_change, std::fabs(lam(c) - lamPrev(c)));
        }
        lamPrev = lam;
        if (it > 0 && max_change < tol * std::max(1.0, lam(0))) converged = true;
        if (!la::orthonormalize(W.data(), n, k)) throw std::runtime_error("orth failed");
        V.swap(W);
        out.cpu_s += now_s() - tc0;
    }
    out.iters = it;
    // final Rayleigh on converged basis (row-major upload again)
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < k; ++c) v32[r * k + c] = float(V(r, Eigen::Index(c)));
    bv.upload(v32.data(), v32.size() * 4);
    dev.begin();
    dev.transition(bg.res(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dev.transition(bv.res(), D3D12_RESOURCE_STATE_COMMON,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    dev.submit_and_wait();
    for (uint32_t c0 = 0; c0 < k; c0 += 512) {
        dev.begin();
        dev.transition(bw.res(), D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        uint32_t consts[8] = {uint32_t(n), uint32_t(k), uint32_t(n), c0, 0, 0, 0, 0};
        gemm.dispatch(dev, bg, bv, bw, consts, (512 + 15) / 16, uint32_t((n + 15) / 16), 1);
        dev.transition(bw.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COMMON);
        dev.submit_and_wait();
    }
    bw.download(w32.data(), w32.size() * 4);
    for (size_t r = 0; r < n; ++r)
        for (size_t c = 0; c < k; ++c) W(r, Eigen::Index(c)) = double(w32[r * k + c]);
    std::vector<size_t> order(k);
    for (size_t i = 0; i < k; ++i) order[i] = i;
    for (Eigen::Index c = 0; c < Eigen::Index(k); ++c) lam(c) = V.col(c).dot(W.col(c));
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return lam(a) > lam(b); });
    out.values.assign(k, 0.0);
    out.vectors.assign(n * k, 0.0);
    for (size_t c = 0; c < k; ++c) {
        out.values[c] = std::max(0.0, lam(order[c]));
        for (size_t r = 0; r < n; ++r) out.vectors[r * k + c] = V(r, order[c]);
    }
    return out;
}

/// \brief Entry point: parses --validate/--bench (default validate), --shard,
///        --tensor, --k, --warp; runs the syrk self-test, loads the tensor, computes
///        GPU + CPU Grams, the CPU fp64 and hybrid GPU top-k solvers, evaluates the
///        precision gates (trace, lambda_1, tail energies) and logs one
///        \c gpu_linalg_validate or \c gpu_linalg_bench event.
/// \return 0 iff gates passed (or in bench mode), 1 on gate failure or fatal error.
int main(int argc, char** argv) {
    std::string mode = "validate";
    std::string shard = "models/GLM-5.3-Flash/model-00062-of-00062.safetensors";
    std::string tensor = "model.language_model.layers.9.self_attn.q_proj.weight";
    size_t k = 2048;
    bool warp = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--validate") mode = "validate";
        else if (a == "--bench") mode = "bench";
        else if (a == "--shard") shard = argv[++i];
        else if (a == "--tensor") tensor = argv[++i];
        else if (a == "--k") k = size_t(std::stoull(argv[++i]));
        else if (a == "--warp") warp = true;
    }
    xlab::Run run("gpu-linalg-" + mode);
    try {
        gpu::Device dev;
        dev.init(warp);
        gpu::Kernel syrk(dev, kSyrkHlsl);
        gpu::Kernel gemm(dev, kGemmNnHlsl);
        run.event("env", "\"adapter\":\"" + dev.adapter_name() + "\",\"warp\":" +
                             (dev.warp() ? "true" : "false"));

        syrk_selftest(dev, syrk, run);

        run.event("load_start", "\"tensor\":\"" + tensor + "\"");
        TensorData d = load_tensor(shard, tensor);
        run.event("loaded",
                  "\"m\":" + std::to_string(d.m) + ",\"n\":" + std::to_string(d.n));

        // GPU fp32 Gram
        double gsec = 0;
        std::vector<float> g32 = gpu_gram(dev, syrk, d.x32, d.m, d.n, &gsec, 512);
        double trG = 0, trC = 0;
        for (size_t i = 0; i < d.n; ++i) {
            trG += g32[i * d.n + i];
            trC += d.gram64[i * d.n + i];
        }
        {
            char b[192];
            std::snprintf(b, sizeof b,
                          "\"gram_gpu_s\":%.2f,\"trace_gpu\":%.6e,\"trace_cpu\":%.6e,"
                          "\"trace_rel_err\":%.3e",
                          gsec, trG, trC, std::fabs(trG / trC - 1.0));
            run.event("gpu_gram", b);
        }

        // CPU reference top-k (fp64)
        double c0 = now_s();
        auto cpuEig = la::topk_eigen(d.gram64.data(), d.n, k, 25, 1e-11);
        double cpu_sec = now_s() - c0;

        // Hybrid GPU top-k (fp32 data)
        double h0 = now_s();
        Hybrid hyb = gpu_topk_eigen(dev, gemm, g32, d.n, k, 25, 1e-11);
        double hyb_total = now_s() - h0;

        // gates
        double lam1g = hyb.values[0], lam1c = cpuEig.values[0];
        double lam1_rel = std::fabs(lam1g / lam1c - 1.0);
        auto tail = [&](const std::vector<double>& vals, size_t r) {
            double acc = 0;
            for (size_t i = 0; i < r && i < vals.size(); ++i) acc += vals[i];
            return trC - acc;  // both against the fp64 trace for comparability
        };
        bool gates_ok = std::fabs(trG / trC - 1.0) < 1e-5 && lam1_rel < 1e-4;
        std::string rankcmp;
        for (double f : {0.25, 0.5, 0.75, 1.0}) {
            size_t r = size_t(f * k);
            double tg = tail(hyb.values, r), tc = tail(cpuEig.values, r);
            double dev_abs = std::fabs(tg - tc) / trC;
            gates_ok = gates_ok && dev_abs < 1e-4;
            char b[96];
            std::snprintf(b, sizeof b, "{\"r\":%zu,\"tail_gpu\":%.4e,\"tail_cpu\":%.4e}", r, tg,
                          tc);
            rankcmp += (rankcmp.empty() ? "" : ",") + std::string(b);
        }
        {
            char b[512];
            std::snprintf(b, sizeof b,
                          "\"k\":%zu,\"iters_gpu\":%d,\"iters_cpu\":%d,"
                          "\"hybrid_total_s\":%.1f,\"gpu_s\":%.1f,\"cpu_in_hybrid_s\":%.1f,"
                          "\"cpu_only_s\":%.1f,\"lambda1_rel_err\":%.3e,\"gates_ok\":%s,"
                          "\"ranks\":%s",
                          k, hyb.iters, cpuEig.iters_used, hyb_total, hyb.gpu_s, hyb.cpu_s,
                          cpu_sec, lam1_rel, gates_ok ? "true" : "false", rankcmp.c_str());
            run.event(mode == "bench" ? "gpu_linalg_bench" : "gpu_linalg_validate", b);
        }
        run.event("done", "\"status\":\"ok\"");
        return gates_ok || mode == "bench" ? 0 : 1;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}

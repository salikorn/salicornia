// salikorn / E-DECOMP spike — linear-algebraic lossless parameter reduction experiments.
//   --validate : solver ground-truth checks + square-Q exact-fold proof + exact-rank lossless demo
//   --synth    : known-intrinsic-rank activations -> PCA energy curves for slice targets
//   --st FILE  : real safetensors shard: per-tensor spectrum, energy ranks, L2-lossless ranks,
//                fp16-quantization comparison, parameter-savings table
// All events -> runs/<ts>-edecomp/events.jsonl (xlab).
/// \file
/// \brief E-DECOMP spike: CPU fp64 experiments establishing what linear decomposition
///        can do to model weights losslessly, in three modes.
///
/// \p --validate runs ground-truth checks on constructed matrices with known spectra
/// (Jacobi + top-k recovery, the SliceGPT square-Q exact fold, exact-rank truncation).
/// \p --synth builds rank-300 activations at d=2048 and measures where PCA energy
/// fractions land and whether the rank cliff is detected. \p --st FILE walks the largest
/// 2-D float tensors of a real shard: fp64 Gram -> top-k spectrum, tail energy at
/// {0.5, 0.75, 0.8125} rank fractions, an L2-lossless rank (tail <= fp16 round-trip
/// error energy), and a parameter-savings table. All results stream as xlab events.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "compress/linalg.hpp"
#include "st/safetensors.hpp"
#include "xlab/fp16.hpp"
#include "xlab/log.hpp"

namespace la = sal::la;
namespace fs = std::filesystem;

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---------------------------------------------------------------- helpers
/// \brief Euclidean (Frobenius) norm of a dense double vector.
/// \param v Values. \return sqrt(sum of squares).
static double fro_norm(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x * x;
    return std::sqrt(s);
}

// y = W (rmsnorm(x) .* gamma)  — the pre-norm linear block used in the fold test
/// \brief Computes y = W (rmsnorm(x) .* gamma) — the pre-norm linear block the
///        square-Q fold theorem is stated on.
/// \param W     Row-major [m, d] weight matrix.
/// \param m     Output rows. \param d Input width.
/// \param x     Input vector, length d. \param gamma Per-channel scale, length d.
/// \param y     Output vector, resized to m.
static void norm_linear(const std::vector<double>& W, size_t m, size_t d,
                        const std::vector<double>& x, const std::vector<double>& gamma,
                        std::vector<double>& y) {
    double ms = 0;
    for (size_t i = 0; i < d; ++i) ms += x[i] * x[i];
    double rms = std::sqrt(ms / double(d));
    y.assign(m, 0.0);
    for (size_t r = 0; r < m; ++r)
        for (size_t c = 0; c < d; ++c) y[r] += W[r * d + c] * (x[c] * gamma[c] / rms);
}

// ---------------------------------------------------------------- --validate
/// \brief --validate mode: solver ground-truth on constructed matrices with known
///        spectra. Builds W = U diag(sigma) V^T (216 of 256 singulars non-zero,
///        geometric decay), then checks (1) Jacobi eigen recovery of W^T W and the
///        top-k subspace solver against the same ground truth, (2) the SliceGPT
///        square-Q exact fold (RMS invariance + output equality to 1e-12), and
///        (3) that truncation to the exact intrinsic rank is lossless, reporting the
///        parameter saving. Each check logs one \c validate_* event.
/// \param run Event sink.
/// \return 0 (throws on solver failure).
static int mode_validate(xlab::Run& run) {
    std::mt19937_64 rng(99);
    std::normal_distribution<double> gauss(0, 1);

    // (1) known-spectrum recovery: W = U diag(sigma) V^T, n=256 cols, m=512 rows,
    //     sigma = geometric decay for 216 dims, then EXACT zeros for 40 dims.
    const size_t n = 256, m = 512, r = 216;
    std::vector<double> U(m * n), V(n * n);
    if (!la::random_orthogonal(n, V.data(), 11)) throw std::runtime_error("randorth V");
    {
        std::vector<double> Qm(m * m);
        if (!la::random_orthogonal(m, Qm.data(), 12)) throw std::runtime_error("randorth U");
        // take first n columns of Qm as U
        for (size_t i = 0; i < m; ++i)
            for (size_t j = 0; j < n; ++j) U[i * n + j] = Qm[i * m + j];
    }
    std::vector<double> sigma(n);
    for (size_t i = 0; i < r; ++i) sigma[i] = std::pow(0.5, double(i) / 40.0);  // slow-ish decay
    for (size_t i = r; i < n; ++i) sigma[i] = 0.0;
    std::vector<double> W(m * n, 0.0);
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < n; ++j) {
            double s = 0;
            for (size_t k = 0; k < n; ++k) s += U[i * n + k] * sigma[k] * V[j * n + k];
            W[i * n + j] = s;
        }
    std::vector<double> G(n * n);
    {
        la::parallel_for(n, la::nthreads(), [&](size_t c) {
            for (size_t rr = 0; rr <= c; ++rr) {
                double s = 0;
                for (size_t i = 0; i < m; ++i) s += W[i * n + c] * W[i * n + rr];
                G[c * n + rr] = G[rr * n + c] = s;
            }
        });
    }
    std::vector<double> jvals, jvecs;
    la::jacobi_eigen(G.data(), n, jvals, jvecs);
    double max_rel = 0;
    for (size_t i = 0; i < n; ++i) {
        double want = sigma[i] * sigma[i];
        double got = jvals[i];
        if (want > 1e-24) max_rel = std::max(max_rel, std::fabs(got - want) / want);
    }
    {
        char b[160];
        std::snprintf(b, sizeof b,
                      "\"n\":%zu,\"exact_zero_sv\":%zu,\"max_rel_err\":%.3e,\"zero_tail_max\":%.3e",
                      n, n - r, max_rel, jvals[r]);
        run.event("validate_jacobi_spectrum", b);
    }

    // top-k solver vs Jacobi on the same matrix
    auto topk = la::topk_eigen(G.data(), n, 128, 100, 1e-13);
    double max_rel2 = 0;
    for (size_t i = 0; i < 128; ++i) {
        double want = sigma[i] * sigma[i];
        if (want > 1e-24) max_rel2 = std::max(max_rel2, std::fabs(topk.values[i] - want) / want);
    }
    {
        char b[160];
        std::snprintf(b, sizeof b, "\"k\":128,\"max_rel_err\":%.3e,\"iters\":%d,\"resid\":%.3e",
                      max_rel2, topk.iters_used, topk.max_residual);
        run.event("validate_topk_spectrum", b);
    }

    // (2) square-Q exact-fold test (SliceGPT theorem in code)
    {
        const size_t d = 256, mm = 128;
        std::vector<double> x(d), g(d), W(mm * d);
        for (auto& v : x) v = gauss(rng);
        for (auto& v : g) v = 0.5 + 1.0 * std::fabs(gauss(rng));
        for (auto& v : W) v = gauss(rng);
        std::vector<double> y;
        norm_linear(W, mm, d, x, g, y);

        std::vector<double> Q(d * d);
        if (!la::random_orthogonal(d, Q.data(), 7)) throw std::runtime_error("randorth Q");
        // x~ = Q^T x ; rms invariance
        std::vector<double> xt(d, 0.0);
        for (size_t i = 0; i < d; ++i)
            for (size_t j = 0; j < d; ++j) xt[i] += Q[j * d + i] * x[j];
        double rms_x = 0, rms_xt = 0;
        for (size_t i = 0; i < d; ++i) {
            rms_x += x[i] * x[i];
            rms_xt += xt[i] * xt[i];
        }
        rms_x = std::sqrt(rms_x / d);
        rms_xt = std::sqrt(rms_xt / d);
        // W~ = W diag(gamma) Q ; y~ = W~ x~ / rms(x~)
        std::vector<double> Wf(mm * d, 0.0);
        for (size_t rrow = 0; rrow < mm; ++rrow)
            for (size_t c = 0; c < d; ++c) {
                double s = 0;
                for (size_t t = 0; t < d; ++t) s += W[rrow * d + t] * g[t] * Q[t * d + c];
                Wf[rrow * d + c] = s;
            }
        std::vector<double> yt(mm, 0.0);
        for (size_t rrow = 0; rrow < mm; ++rrow)
            for (size_t c = 0; c < d; ++c) yt[rrow] += Wf[rrow * d + c] * (xt[c] / rms_xt);

        double rel = std::sqrt(la::fro_sq(y.data(), yt.data(), mm)) / fro_norm(y);
        char b[192];
        std::snprintf(b, sizeof b,
                      "\"d\":%zu,\"rms_invariance_err\":%.3e,\"output_rel_err\":%.3e,\"exact\":%s",
                      d, std::fabs(rms_xt - rms_x) / rms_x, rel, (rel < 1e-12 ? "true" : "false"));
        run.event("validate_square_fold", b);
    }

    // (3) exact-rank truncation is lossless (discard exactly-zero singulars)
    {
        const size_t rr = 64, m2 = 512, n2 = 256;
        std::vector<double> A(m2 * rr), B(rr * n2);
        for (auto& v : A) v = gauss(rng);
        for (auto& v : B) v = gauss(rng);
        std::vector<double> W2(m2 * n2, 0.0);
        la::gemm(A.data(), B.data(), W2.data(), m2, rr, n2, 1.0, 0.0);
        std::vector<double> G2(n2 * n2, 0.0);
        la::parallel_for(n2, la::nthreads(), [&](size_t c) {
            for (size_t r3 = 0; r3 <= c; ++r3) {
                double s = 0;
                for (size_t i = 0; i < m2; ++i) s += W2[i * n2 + c] * W2[i * n2 + r3];
                G2[c * n2 + r3] = G2[r3 * n2 + c] = s;
            }
        });
        auto eig = la::topk_eigen(G2.data(), n2, rr, 100, 1e-13);
        // project: Wr = (W Vr) Vr^T ; eigenvectors stored n x k row-major
        std::vector<double> Vr(n2 * rr);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < rr; ++j) Vr[i * rr + j] = eig.vectors[i * rr + j];
        std::vector<double> WV(m2 * rr, 0.0);
        la::gemm(W2.data(), Vr.data(), WV.data(), m2, n2, rr, 1.0, 0.0);
        std::vector<double> VrT(rr * n2);
        for (size_t i = 0; i < n2; ++i)
            for (size_t j = 0; j < rr; ++j) VrT[j * n2 + i] = Vr[i * rr + j];
        std::vector<double> Wr(m2 * n2, 0.0);
        la::gemm(WV.data(), VrT.data(), Wr.data(), m2, rr, n2, 1.0, 0.0);
        double rel = std::sqrt(la::fro_sq(W2.data(), Wr.data(), m2 * n2)) / fro_norm(W2);
        double full = double(m2) * n2, fact = double(m2 + n2) * rr;
        char b[192];
        std::snprintf(b, sizeof b,
                      "\"rank\":%zu,\"recon_rel_err\":%.3e,\"params\":%.0f->%.0f (-%.1f%%)",
                      rr, rel, full, fact, 100.0 * (1.0 - fact / full));
        run.event("validate_exact_rank_lossless", b);
    }
    return 0;
}

// ---------------------------------------------------------------- --synth
/// \brief --synth mode: generates m=8192 activations of intrinsic rank 300 at d=2048
///        (plus 1e-3 noise), computes the fp64 Gram, and runs the top-k solver for
///        k=1843 (0.9*d). Logs the PCA energy captured at the {0.5, 0.75, 0.8125, 0.9}
///        slice fractions (\c synth_slice_energy) and the detected rank cliff — the
///        smallest r where lambda_{r+1} < 1e-8 * lambda_1 (\c synth_cliff).
/// \param run Event sink.
/// \return 0.
static int mode_synth(xlab::Run& run) {
    // activations of intrinsic rank 300 (+ small noise), d=2048, m=8192 samples
    const size_t d = 2048, m = 8192, rank = 300;
    std::mt19937_64 rng(5);
    std::normal_distribution<double> gauss(0, 1);
    std::vector<double> L(d * rank);
    for (auto& v : L) v = gauss(rng) * (1.0 / std::sqrt(double(rank)));
    std::vector<float> H(m * d);
    std::vector<double> u(rank);
    for (size_t i = 0; i < m; ++i) {
        for (auto& v : u) v = gauss(rng);
        for (size_t c = 0; c < d; ++c) {
            double s = 0;
            const double* lc = L.data() + c * rank;
            for (size_t t = 0; t < rank; ++t) s += lc[t] * u[t];
            H[i * d + c] = float(s + 1e-3 * gauss(rng));
        }
    }
    std::vector<double> G(d * d);
    la::gram_f32(H.data(), m, d, G.data());
    double trace = 0;
    for (size_t i = 0; i < d; ++i) trace += G[i * d + i];
    size_t k = 1843;  // 0.9 * 2048
    auto eig = la::topk_eigen(G.data(), d, k, 60, 1e-12);
    double acc = 0;
    std::map<size_t, double> energy_at;
    const double fracs[] = {0.5, 0.75, 0.8125, 0.9};
    for (size_t i = 0; i < k; ++i) {
        acc += eig.values[i];
        for (double f : fracs) {
            size_t r = size_t(f * d);
            if (i + 1 == r) energy_at[r] = acc / trace;
        }
    }
    // rank-cliff detection: smallest r where lambda_{r+1} drops below 1e-8 * lambda_1
    size_t cliff = 0;
    for (size_t i = 0; i + 1 < k; ++i)
        if (eig.values[i + 1] < 1e-8 * eig.values[0]) {
            cliff = i + 1;
            break;
        }
    for (auto& [r, e] : energy_at) {
        char b[128];
        std::snprintf(b, sizeof b, "\"d\":%zu,\"dprime\":%zu,\"energy\":%.9f", d, r, e);
        run.event("synth_slice_energy", b);
    }
    {
        char b[160];
        std::snprintf(b, sizeof b,
                      "\"true_rank\":%zu,\"detected_cliff\":%zu,\"iters\":%d,\"lambda1\":%.3e",
                      rank, cliff, eig.iters_used, eig.values[0]);
        run.event("synth_cliff", b);
    }
    return 0;
}

// ---------------------------------------------------------------- --st
/// \brief --st mode: per-tensor spectrum analysis over a real safetensors shard.
///
/// Selects the \p max_tensors largest 2-D BF16/F16/F32 tensors with at least
/// \p min_elems elements. Per tensor (Gram over the smaller dimension n, transposing
/// the access when needed): fp64 Gram + fp16 round-trip error energy in one pass,
/// top-k eigensolver (k = 0.8125n + 32), tail energy relative to the trace at
/// {0.5, 0.75, 0.8125} fractions, and the L2-lossless rank — the largest slice whose
/// tail energy still fits under the fp16 quantization error (i.e. low-rank storage
/// would be no worse than just storing fp16). Logs \c st_tensor_spectrum per tensor
/// and \c st_tensor_skipped_dtype / \c st_tensor_skipped_impl for exclusions.
/// \param run         Event sink.
/// \param path        Shard path.
/// \param max_tensors Cap on analyzed tensors (default 6).
/// \param min_elems   Minimum element count to consider (default 1e7).
/// \return 0 (throws on I/O or solver failure).
static int mode_st(xlab::Run& run, const std::string& path, size_t max_tensors,
                   uint64_t min_elems) {
    sal::st::SafeTensorsFile f{fs::path(path)};
    {
        char b[192];
        std::snprintf(b, sizeof b, "\"file\":\"%s\",\"bytes\":%llu,\"tensors\":%zu",
                      fs::path(path).filename().string().c_str(),
                      (unsigned long long)f.file_size(), f.tensors().size());
        run.event("st_file", b);
    }
    // rank candidates by size, filter to 2-D float tensors
    std::vector<std::pair<uint64_t, std::string>> cands;
    for (auto& [name, t] : f.tensors()) {
        if (t.shape.size() != 2) continue;
        if (t.dtype != "BF16" && t.dtype != "F16" && t.dtype != "F32") {
            if (t.numel() >= min_elems) {
                run.event("st_tensor_skipped_dtype",
                          "\"name\":\"" + name + "\",\"dtype\":\"" + t.dtype + "\"");
            }
            continue;
        }
        if (t.numel() >= min_elems) cands.emplace_back(t.numel(), name);
    }
    std::sort(cands.begin(), cands.end(),
              [](auto& a, auto& b) { return a.first > b.first; });
    if (cands.size() > max_tensors) cands.resize(max_tensors);

    for (auto& [numel, name] : cands) {
        const sal::st::TensorInfo& t = *f.find(name);
        uint64_t rows = t.shape[0], cols = t.shape[1];
        // Gram over the smaller dimension n; transpose access if needed
        size_t n = size_t(std::min(rows, cols)), m = size_t(std::max(rows, cols));
        bool transposed = (n != cols);  // if the smaller dim is the row dim
        double t0 = now_s();
        std::vector<double> G(n * n);
        // one pass: Gram in fp64 (blocked) + fp16 quantization error energy (parallel)
        double e16 = 0;
        if (t.dtype == "BF16") {
            const uint16_t* X = reinterpret_cast<const uint16_t*>(t.data);
            la::gram_bf16(X, m, n, G.data(), 0, transposed);
            std::vector<double> eparts(la::nthreads(), 0.0);
            la::parallel_for(la::nthreads(), la::nthreads(), [&](size_t th) {
                size_t nthr = la::nthreads();
                size_t total = size_t(m) * n;
                size_t b = th * total / nthr, e = (th + 1) * total / nthr;
                double s = 0;
                for (size_t i = b; i < e; ++i) {
                    float w = sal::bf16_to_f32(X[i]);
                    float q = sal::f16_to_f32(sal::f32_to_f16(w));
                    double d = double(w) - double(q);
                    s += d * d;
                }
                eparts[th] = s;
            });
            for (double s : eparts) e16 += s;
        } else {  // F16 / F32 -> not needed for this shard so far; skip with note
            run.event("st_tensor_skipped_impl",
                      "\"name\":\"" + name + "\",\"dtype\":\"" + t.dtype + "\"");
            continue;
        }
        double trace = 0;
        for (size_t i = 0; i < n; ++i) trace += G[i * n + i];
        size_t k = std::min(n, size_t(double(n) * 0.8125) + 32);
        auto eig = la::topk_eigen(G.data(), n, k, 25, 1e-11);
        double t1 = now_s();

        // energy at rank fractions + L2 rank (tail <= fp16 error energy)
        std::string fr;
        double cum = 0;
        size_t l2rank = 0;
        size_t idx = 0;
        for (double f : {0.5, 0.75, 0.8125}) {
            size_t r = size_t(f * n);
            while (idx < r && idx < k) {
                cum += eig.values[idx];
                ++idx;
            }
            double tail = trace - cum;
            char b[96];
            std::snprintf(b, sizeof b, "{\"r\":%zu,\"f\":%.4f,\"tail_rel\":%.3e}", r, f,
                          tail / trace);
            fr += (fr.empty() ? "" : ",") + std::string(b);
            if (tail <= e16) l2rank = std::max(l2rank, r);
        }
        double full = double(m) * n, factL2 = double(m + n) * l2rank;
        double l2_save = l2rank > 0 ? 100.0 * (1.0 - factL2 / full) : 0.0;
        char b[512];
        std::snprintf(b, sizeof b,
                      "\"name\":\"%s\",\"shape\":[%llu,%llu],\"dtype\":\"%s\","
                      "\"lambda1\":%.4e,\"lambda_k\":%.4e,\"trace\":%.4e,"
                      "\"fp16_err_energy\":%.4e,\"ranks\":%s,\"l2_rank\":%zu,"
                      "\"params\":%.0f,\"params_at_l2\":%.0f,\"l2_saving_pct\":%.2f,"
                      "\"l2_available\":%s,\"sec\":%.1f,\"iters\":%d",
                      name.c_str(), (unsigned long long)rows, (unsigned long long)cols,
                      t.dtype.c_str(), eig.values[0], eig.values[k - 1], trace, e16,
                      fr.c_str(), l2rank, full, factL2, l2_save,
                      l2rank > 0 ? "true" : "false", t1 - t0, eig.iters_used);
        run.event("st_tensor_spectrum", b);
    }
    return 0;
}

// ---------------------------------------------------------------- main
/// \brief Entry point: parses --validate | --synth | --st FILE plus
///        --max-tensors N / --min-elems N, opens the xlab run ("edecomp-<mode>"),
///        dispatches to the mode function. Exit code 2 on usage error, 1 on fatal.
/// \return Mode's return value, or 2 if no mode was given, or 1 on fatal error.
int main(int argc, char** argv) {
    std::string mode, st_path;
    size_t max_tensors = 6;
    uint64_t min_elems = 10'000'000;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--validate" || a == "--synth") mode = a.substr(2);
        else if (a == "--st") {
            mode = "st";
            st_path = argv[++i];
        } else if (a == "--max-tensors") max_tensors = size_t(std::stoull(argv[++i]));
        else if (a == "--min-elems") min_elems = std::stoull(argv[++i]);
    }
    if (mode.empty()) {
        std::fprintf(stderr, "usage: spike_decomp --validate | --synth | --st FILE "
                             "[--max-tensors N] [--min-elems N]\n");
        return 2;
    }
    xlab::Run run("edecomp-" + mode);
    try {
        if (mode == "validate") return mode_validate(run);
        if (mode == "synth") return mode_synth(run);
        if (mode == "st") return mode_st(run, st_path, max_tensors, min_elems);
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
    return 0;
}

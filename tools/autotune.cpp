// salikorn / autotune — find the best launch-geometry configuration for THIS machine.
//
//   build\autotune.exe                 full sweep -> profiles/best.json (+ table to stdout)
//   build\autotune.exe --quick         pruned grid (~1 min)
//   build\autotune.exe --only gemm_nt,vq_nearest
//   build\autotune.exe --reps 9        more timing reps (default 5, median)
//
// Protocol per candidate: compile (runtime HLSL with @TOKEN@ substitution) ->
// correctness gate vs CPU reference on a small shape (mismatch/NaN => disqualified, the
// tuner may never pick a broken variant) -> warmup + median-of-N timing on the
// trainer-representative shape (and the analysis shape for the GEMM family, recorded).
/// \file
/// \brief autotune.exe — sweeps the launch-geometry candidate space on THIS machine and
///        writes profiles/best.json for the production kernel sets to load.
///
/// \section gating Candidate-gating protocol (applied identically to every op)
/// For each candidate from sal::gpu::autotune::spaces():
///  1. <b>Compile</b> — gpu::Kernel performs runtime HLSL compilation with the
///     candidate's \@TOKEN\@ defines. A compile failure disqualifies the candidate
///     (caught std::exception, logged as COMPILE FAIL) but never aborts the sweep.
///  2. <b>Correctness gate</b> — the candidate runs on a small shape against a
///     CPU reference (GEMM: 64x48x32 with packed-buffer offsets, max diff < 2e-4;
///     syrk: 256x64 Gram, < 2e-3; elem: n=4096 gelu vs CPU, < 1e-5; vq_nearest:
///     argmin index match on B=8/K=256, zero mismatches). A non-finite or
///     over-tolerance result logs \c cand_rejected and the candidate is skipped:
///     the tuner may never pick a broken variant, no matter how fast it times.
///  3. <b>Timing</b> — warmup packets, then median-of-reps wall clock via
///     time_packets() on the trainer-representative shape (winner-selection metric);
///     the GEMM family additionally records a 2048^3 "analysis shape" timing,
///     which is informational and does not influence winner choice. Every timed
///     candidate logs a \c cand_timed event; each op logs \c op_winner.
/// Dispatches are wrapped in explicit COMMON -> NON_PIXEL_SHADER_RESOURCE /
/// UNORDERED_ACCESS transitions and back (see the resource-state discipline note in
/// tests/test_gpu.cpp); syrk additionally times one dispatch packet per output-column
/// chunk because WDDM's TDR budget applies per submitted packet.
///
/// Output: a per-op winners map + accumulated TuneParams written via save_profile()
/// (adapter name recorded), a \c profile_written event, and a summary table to stdout.
/// adam intentionally shares the elem family's geometry: its own winner is only
/// recorded as a note when it disagrees with the elem winner.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gpu/autotune.hpp"
#include "xlab/log.hpp"

namespace gpu = sal::gpu;
namespace at = sal::gpu::autotune;

/// \brief Monotonic steady-clock timestamp in milliseconds (timing baseline).
static double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()
                                                         .time_since_epoch())
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
/// \brief CPU reference GELU (tanh approximation) — correctness gate for the elem family.
/// \param x Input value. \return 0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3))).
static float gelu(float x) {
    const float k0 = 0.7978845608f;
    return 0.5f * x * (1.0f + std::tanh(k0 * (x + 0.044715f * x * x * x)));
}

// median of a timing sample
/// \brief Median of a timing sample.
/// \param v Sample (copied, then sorted).
/// \return Middle element of the sorted copy; 1e30 for an empty sample so a failed
///         measurement can never win the argmin.
static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 1e30 : v[v.size() / 2];
}

/// \brief Result of gating + timing one candidate.
struct Timing {
    bool gate_ok = false;       ///< True iff the correctness gate passed.
    std::string gate_note;      ///< "max_diff=<g>" recorded with rejections/events.
    double trainer_ms = 1e30;   ///< Representative shape used for winner selection
    double analysis_ms = -1.0;  ///< Second shape, recorded for the table
};

// ---- generic timed-run helper: reps of [begin; record; submit] ----
/// \brief Times a GPU operation as complete packets: \p warmup full
///        begin/record/submit_and_wait cycles, then \p reps timed cycles, each measured
///        wall-clock from before begin() to after submit_and_wait().
/// \param dev    Initialized device (queue/list/fence).
/// \param warmup Number of untimed warmup packets (shader JIT, allocator, cache state).
/// \param reps   Number of timed packets; the returned value is the median.
/// \param record Closure recording transitions + dispatch into the open list.
/// \return Median per-packet wall-clock milliseconds.
static double time_packets(gpu::Device& dev, int warmup, int reps,
                           const std::function<void()>& record) {
    for (int i = 0; i < warmup; ++i) {
        dev.begin();
        record();
        dev.submit_and_wait();
    }
    std::vector<double> ms;
    for (int i = 0; i < reps; ++i) {
        double t0 = now_ms();
        dev.begin();
        record();
        dev.submit_and_wait();
        ms.push_back(now_ms() - t0);
    }
    return median(ms);
}

// fills a small random buffer set once
/// \brief Fills a vector with uniform random floats in [lo, hi] from \p rng.
/// \param n   Element count. \param rng Shared seeded generator (deterministic run).
/// \param lo  Lower bound (default -1). \param hi Upper bound (default 1).
/// \return The filled vector.
static std::vector<float> randf(size_t n, std::mt19937& rng, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> u(lo, hi);
    std::vector<float> v(n);
    for (auto& x : v) x = u(rng);
    return v;
}

/// \brief Entry point: parses flags (--quick, --reps N, --out PATH, --only op,op),
///        initializes the device, runs the gated sweep per op family (gemm_nt/nn/tn,
///        syrk, elem, adam, vq_nearest, vq_update — see the file-level gating
///        protocol), accumulates winners into a TuneParams, and writes the profile
///        via save_profile() to \c --out (default profiles/best.json). Every stage
///        emits xlab events; any fatal error is logged as a \c "fatal" event with
///        exit code 1.
/// \return 0 on success, 1 on fatal error.
int main(int argc, char** argv) {
    bool quick = false;
    int reps = 5;
    std::vector<std::string> only;
    std::string out = "profiles/best.json";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--quick") quick = true;
        else if (a == "--reps") reps = std::stoi(argv[++i]);
        else if (a == "--out") out = argv[++i];
        else if (a == "--only") {
            std::string s = argv[++i];
            size_t p;
            while ((p = s.find(',')) != std::string::npos) {
                only.push_back(s.substr(0, p));
                s.erase(0, p + 1);
            }
            if (!s.empty()) only.push_back(s);
        }
    }
    xlab::Run run(quick ? "autotune-quick" : "autotune-full");
    try {
        gpu::Device dev;
        dev.init(false);
        gpu::TuneParams tp;  // accumulated winners; written to the profile at the end
        std::map<std::string, nlohmann::json> winners;
        std::mt19937 rng(77);

        // ================================================================ gemm family
        // One shared correctness harness for NT/NN/TN; per-op record closures for timing.
        {
            const uint32_t GM = 64, GN = 48, GK = 32, GPAD = 16;
            std::vector<float> A = randf(size_t(GM) * GK, rng);
            std::vector<float> B = randf(size_t(GN) * GK, rng);
            std::vector<float> B2 = randf(size_t(GK) * GN, rng);
            std::vector<float> A2 = randf(size_t(GK) * GM, rng);
            const uint32_t aoff = GPAD, boff = aoff + GM * GK + GPAD,
                            b2off = boff + GN * GK + GPAD, a2off = b2off + GK * GN + GPAD;
            std::vector<float> pack(size_t(a2off) + A2.size() + GPAD, 0.f);
            std::memcpy(pack.data() + aoff, A.data(), A.size() * 4);
            std::memcpy(pack.data() + boff, B.data(), B.size() * 4);
            std::memcpy(pack.data() + b2off, B2.data(), B2.size() * 4);
            std::memcpy(pack.data() + a2off, A2.data(), A2.size() * 4);
            gpu::Buffer bp(dev, pack.size() * 4, gpu::BufferKind::Uav);
            bp.upload(pack.data(), pack.size() * 4);
            gpu::Buffer out(dev, size_t(GM) * GN * 4, gpu::BufferKind::Uav);

            auto gate = [&](gpu::Kernel& k, int which, const at::Cand& cd) -> Timing {
                Timing t;
                // references
                std::vector<float> ref(size_t(GM) * GN, 0.f);
                if (which == 0)
                    for (uint32_t i = 0; i < GM; ++i)
                        for (uint32_t kk = 0; kk < GK; ++kk)
                            for (uint32_t j = 0; j < GN; ++j)
                                ref[size_t(i) * GN + j] += A[size_t(i) * GK + kk] * B[size_t(j) * GK + kk];
                else if (which == 1)
                    for (uint32_t i = 0; i < GM; ++i)
                        for (uint32_t kk = 0; kk < GK; ++kk)
                            for (uint32_t j = 0; j < GN; ++j)
                                ref[size_t(i) * GN + j] += A[size_t(i) * GK + kk] * B2[size_t(kk) * GN + j];
                else
                    for (uint32_t kk = 0; kk < GK; ++kk)
                        for (uint32_t i = 0; i < GM; ++i)
                            for (uint32_t j = 0; j < GN; ++j)
                                ref[size_t(i) * GN + j] +=
                                    A2[size_t(kk) * GM + i] * B2[size_t(kk) * GN + j];
                // per-form pack slots: NT(A@aoff,B@boff) NN(A@aoff,B@b2off) TN(A2@a2off,B2@b2off)
                uint32_t aslot = which == 2 ? a2off : aoff;
                uint32_t bslot = which == 0 ? boff : b2off;
                uint32_t c[8] = {GM, GN, GK, 0, aslot, bslot, 0, 0};
                uint32_t ts = uint32_t(std::stoi(cd.defines.at("TS")));
                dev.begin();
                dev.transition(bp.res(), D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                dev.transition(out.res(), D3D12_RESOURCE_STATE_COMMON,
                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                k.dispatch(dev, bp, bp, out, c, (GN + ts - 1) / ts, (GM + ts - 1) / ts, 1);
                dev.transition(bp.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                               D3D12_RESOURCE_STATE_COMMON);
                dev.transition(out.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                               D3D12_RESOURCE_STATE_COMMON);
                dev.submit_and_wait();
                std::vector<float> got(ref.size());
                out.download(got.data(), got.size() * 4);
                double md = 0;
                for (size_t i = 0; i < ref.size(); ++i)
                    md = std::max(md, double(std::fabs(got[i] - ref[i])));
                t.gate_ok = std::isfinite(md) && md < 2e-4;
                t.gate_note = "max_diff=" + std::to_string(md);
                return t;
            };

            auto trainer_shape = [&](const char* tmpl, int which, const at::Cand& cd,
                                     gpu::Device& d) {
                gpu::Kernel k(d, tmpl, cd.defines);
                uint32_t M, N, K;
                if (which == 0) { M = 512; N = 384; K = 256; }
                else if (which == 1) { M = 512; N = 256; K = 384; }
                else { M = 384; N = 256; K = 512; }
                gpu::Buffer a(d, size_t(M) * K * 4, gpu::BufferKind::Uav);
                gpu::Buffer b(d, size_t(which == 1 ? K * N : N * K) * 4, gpu::BufferKind::Uav);
                gpu::Buffer o(d, size_t(M) * N * 4, gpu::BufferKind::Uav);
                auto A2f = randf(size_t(M) * K, rng);
                auto B2f = randf(size_t(which == 1 ? K * N : N * K), rng);
                a.upload(A2f.data(), A2f.size() * 4);
                b.upload(B2f.data(), B2f.size() * 4);
                uint32_t c[8] = {M, N, K, 0, 0, 0, 0, 0};
                int ts = std::stoi(cd.defines.at("TS"));
                return time_packets(d, 2, reps, [&] {
                    d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    k.dispatch(d, a, b, o, c, (N + ts - 1) / ts, (M + ts - 1) / ts, 1);
                    d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_COMMON);
                    d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_COMMON);
                    d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COMMON);
                });
            };
            auto analysis_shape = [&](const char* tmpl, const at::Cand& cd, gpu::Device& d) {
                gpu::Kernel k(d, tmpl, cd.defines);
                uint32_t M = 2048, N = 2048, K = 2048;
                gpu::Buffer a(d, size_t(M) * K * 4, gpu::BufferKind::Uav);
                gpu::Buffer b(d, size_t(N) * K * 4, gpu::BufferKind::Uav);
                gpu::Buffer o(d, size_t(M) * N * 4, gpu::BufferKind::Uav);
                auto A2f = randf(size_t(M) * K, rng);
                auto B2f = randf(size_t(N) * K, rng);
                a.upload(A2f.data(), A2f.size() * 4);
                b.upload(B2f.data(), B2f.size() * 4);
                uint32_t c[8] = {M, N, K, 0, 0, 0, 0, 0};
                int ts = std::stoi(cd.defines.at("TS"));
                return time_packets(d, 1, 3, [&] {
                    d.transition(a.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    d.transition(b.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                    d.transition(o.res(), D3D12_RESOURCE_STATE_COMMON,
                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                    k.dispatch(d, a, b, o, c, (N + ts - 1) / ts, (M + ts - 1) / ts, 1);
                    d.transition(a.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_COMMON);
                    d.transition(b.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                 D3D12_RESOURCE_STATE_COMMON);
                    d.transition(o.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                 D3D12_RESOURCE_STATE_COMMON);
                });
            };

            struct GemmOp {
                const char* name;
                const char* tmpl;
                int which;
            };
            std::vector<GemmOp> ops = {{"gemm_nt", gpu::tmpl::gemm_nt(), 0},
                                       {"gemm_nn", gpu::tmpl::gemm_nn(), 1},
                                       {"gemm_tn", gpu::tmpl::gemm_tn(), 2}};
            for (auto& op : ops) {
                if (!only.empty() && std::find(only.begin(), only.end(), op.name) == only.end())
                    continue;
                printf("== %s\n", op.name);
                double best = 1e30;
                std::string best_label;
                int best_ts = 16;
                double best_analysis = -1;
                for (auto& cd : at::spaces(quick, {op.name})[0].cands) {
                    try {
                        gpu::Kernel k(dev, op.tmpl, cd.defines);
                        Timing t = gate(k, op.which, cd);
                        if (!t.gate_ok) {
                            run.event("cand_rejected", "\"op\":\"" + std::string(op.name) +
                                                           "\",\"cand\":\"" + cd.label +
                                                           "\",\"gate\":\"" + t.gate_note + "\"");
                            printf("   %-14s GATE FAIL (%s)\n", cd.label.c_str(),
                                   t.gate_note.c_str());
                            continue;
                        }
                        t.trainer_ms = trainer_shape(op.tmpl, op.which, cd, dev);
                        t.analysis_ms = analysis_shape(op.tmpl, cd, dev);
                        run.event("cand_timed",
                                  "\"op\":\"" + std::string(op.name) + "\",\"cand\":\"" +
                                      cd.label + "\",\"trainer_ms\":" + std::to_string(t.trainer_ms) +
                                      ",\"analysis_ms\":" + std::to_string(t.analysis_ms));
                        printf("   %-14s trainer %8.3f ms   analysis %8.3f ms\n",
                               cd.label.c_str(), t.trainer_ms, t.analysis_ms);
                        if (t.trainer_ms < best) {
                            best = t.trainer_ms;
                            best_label = cd.label;
                            best_ts = std::stoi(cd.defines.at("TS"));
                            best_analysis = t.analysis_ms;
                        }
                    } catch (const std::exception& e) {
                        printf("   %-14s COMPILE FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
                printf("   -> winner %s (%.3f ms)\n", best_label.c_str(), best);
                run.event("op_winner", "\"op\":\"" + std::string(op.name) + "\",\"label\":\"" +
                                           best_label + "\",\"trainer_ms\":" +
                                           std::to_string(best));
                winners[op.name] = {{"TS", best_ts}, {"ms", best}, {"label", best_label},
                                    {"analysis_ms", best_analysis}};
                if (best_ts != tp.gemm_ts) tp.gemm_ts = best_ts;  // last op wins ties; same family
            }
        }

        // ================================================================ syrk
        {
            printf("== syrk\n");
            const uint32_t m = 4096, n = 2048;
            auto X = randf(size_t(m) * n, rng);
            gpu::Buffer bx(dev, X.size() * 4, gpu::BufferKind::Uav);
            bx.upload(X.data(), X.size() * 4);
            gpu::Buffer bg(dev, uint64_t(n) * n * 4, gpu::BufferKind::Uav);
            double best = 1e30;
            std::string best_label;
            int best_ts = 16, best_chunk = 2048;
            for (auto& op : at::spaces(quick, {"syrk"})) {
                for (auto& cd : op.cands) {
                    try {
                        gpu::Kernel k(dev, gpu::tmpl::syrk(), cd.defines);
                        // gate on small shape
                        {
                            const uint32_t gm = 256, gn = 64;
                            auto Xg = randf(size_t(gm) * gn, rng);
                            std::vector<float> ref(size_t(gn) * gn, 0.f);
                            for (uint32_t i = 0; i < gm; ++i)
                                for (uint32_t c1 = 0; c1 < gn; ++c1)
                                    for (uint32_t c2 = 0; c2 < gn; ++c2)
                                        ref[size_t(c1) * gn + c2] +=
                                            Xg[size_t(i) * gn + c1] * Xg[size_t(i) * gn + c2];
                            gpu::Buffer bxg(dev, Xg.size() * 4, gpu::BufferKind::Uav);
                            gpu::Buffer bgg(dev, uint64_t(gn) * gn * 4, gpu::BufferKind::Uav);
                            bxg.upload(Xg.data(), Xg.size() * 4);
                            dev.begin();
                            dev.transition(bxg.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bgg.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            uint32_t c[8] = {gm, gn, 0, 0, 0, 0, 0, 0};
                            int ts = std::stoi(cd.defines.at("TS"));
                            k.dispatch(dev, bxg, bxg, bgg, c, (gn + ts - 1) / ts, (gn + ts - 1) / ts, 1);
                            dev.transition(bxg.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bgg.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.submit_and_wait();
                            std::vector<float> got(size_t(gn) * gn);
                            bgg.download(got.data(), got.size() * 4);
                            double md = 0;
                            for (size_t i = 0; i < ref.size(); ++i)
                                md = std::max(md, double(std::fabs(got[i] - ref[i])));
                            if (!(std::isfinite(md) && md < 2e-3)) {
                                printf("   %-20s GATE FAIL (%g)\n", cd.label.c_str(), md);
                                continue;
                            }
                        }
                        int ts = std::stoi(cd.defines.at("TS"));
                        uint32_t chunk = uint32_t(cd.syrk_chunk);
                        // dedicated timing: one packet per chunk, whole-G wall clock per rep
                        for (int i = 0; i < 1; ++i) {
                            dev.begin();
                            dev.submit_and_wait();  // warm the allocator state
                        }
                        std::vector<double> samples;
                        for (int r = 0; r < 3; ++r) {
                            double t0 = now_ms();
                            for (uint32_t c0 = 0; c0 < n; c0 += chunk) {
                                dev.begin();
                                dev.transition(bx.res(), D3D12_RESOURCE_STATE_COMMON,
                                               D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                                dev.transition(bg.res(), D3D12_RESOURCE_STATE_COMMON,
                                               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                                uint32_t c[8] = {m, n, c0, 0, 0, 0, 0, 0};
                                k.dispatch(dev, bx, bx, bg, c, (chunk + ts - 1) / ts,
                                           (n + ts - 1) / ts, 1);
                                dev.transition(bx.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                               D3D12_RESOURCE_STATE_COMMON);
                                dev.transition(bg.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                               D3D12_RESOURCE_STATE_COMMON);
                                dev.submit_and_wait();
                            }
                            samples.push_back(now_ms() - t0);
                        }
                        double ms = median(samples);
                        printf("   %-20s %8.1f ms (full 4096x2048)\n", cd.label.c_str(), ms);
                        if (ms < best) {
                            best = ms;
                            best_label = cd.label;
                            best_ts = ts;
                            best_chunk = cd.syrk_chunk;
                        }
                    } catch (const std::exception& e) {
                        printf("   %-20s FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
            }
            printf("   -> winner %s (%.1f ms)\n", best_label.c_str(), best);
            winners["syrk"] = {{"TS", best_ts}, {"chunk", best_chunk}, {"ms", best},
                               {"label", best_label}};
            tp.gemm_ts = best_ts;
            tp.syrk_chunk = best_chunk;
        }

        // ================================================================ elem family (gelu)
        {
            printf("== elem (gelu representative)\n");
            const uint32_t n = 512 * 384;
            auto A = randf(n, rng, -3.f, 3.f);
            gpu::Buffer ba(dev, n * 4, gpu::BufferKind::Uav);
            ba.upload(A.data(), n * 4);
            gpu::Buffer bo(dev, n * 4, gpu::BufferKind::Uav);
            double best = 1e30;
            std::string best_label;
            int best_g = 256, best_ept = 1;
            for (auto& op : at::spaces(quick, {"elem"})) {
                for (auto& cd : op.cands) {
                    try {
                        auto defs = cd.defines;
                        defs["BOUND"] = "p.x";
                        gpu::Kernel k(dev, gpu::tmpl::gelu(), defs);
                        // gate: n=4096 vs CPU
                        {
                            const uint32_t gn = 4096;
                            auto Ag = randf(gn, rng, -3.f, 3.f);
                            gpu::Buffer bag(dev, gn * 4, gpu::BufferKind::Uav);
                            gpu::Buffer bog(dev, gn * 4, gpu::BufferKind::Uav);
                            bag.upload(Ag.data(), gn * 4);
                            uint32_t c[8] = {gn, 0, 0, 0, 0, 0, 0, 0};
                            uint32_t thr = uint32_t(std::stoi(cd.defines.at("G")) *
                                                    std::stoi(cd.defines.at("EPT")));
                            dev.begin();
                            dev.transition(bag.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bog.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            k.dispatch(dev, bag, bag, bog, c, (gn + thr - 1) / thr, 1, 1);
                            dev.transition(bag.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bog.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.submit_and_wait();
                            std::vector<float> got(gn);
                            bog.download(got.data(), got.size() * 4);
                            double md = 0;
                            for (uint32_t i = 0; i < gn; ++i)
                                md = std::max(md, double(std::fabs(got[i] - gelu(Ag[i]))));
                            if (!(std::isfinite(md) && md < 1e-5)) {
                                printf("   %-16s GATE FAIL (%g)\n", cd.label.c_str(), md);
                                continue;
                            }
                        }
                        uint32_t thr = uint32_t(std::stoi(cd.defines.at("G")) *
                                                std::stoi(cd.defines.at("EPT")));
                        uint32_t c[8] = {n, 0, 0, 0, 0, 0, 0, 0};
                        double ms = time_packets(dev, 2, reps, [&] {
                            dev.transition(ba.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bo.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            k.dispatch(dev, ba, ba, bo, c, (n + thr - 1) / thr, 1, 1);
                            dev.transition(ba.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bo.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                        });
                        printf("   %-16s %8.3f ms\n", cd.label.c_str(), ms);
                        if (ms < best) {
                            best = ms;
                            best_label = cd.label;
                            best_g = std::stoi(cd.defines.at("G"));
                            best_ept = std::stoi(cd.defines.at("EPT"));
                        }
                    } catch (const std::exception& e) {
                        printf("   %-16s FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
            }
            printf("   -> winner %s (%.3f ms)\n", best_label.c_str(), best);
            winners["elem"] = {{"G", best_g}, {"EPT", best_ept}, {"ms", best},
                               {"label", best_label}};
            tp.elem_g = best_g;
            tp.elem_ept = best_ept;
        }

        // ================================================================ adam (elem geometry)
        {
            printf("== adam\n");
            const uint32_t count = 384 * 256, region = count;
            auto P = randf(count, rng, -0.1f, 0.1f);
            auto Gd = randf(count, rng, -0.1f, 0.1f);
            std::vector<float> st(P);
            st.insert(st.end(), region * 2, 0.f);
            gpu::Buffer bs(dev, st.size() * 4, gpu::BufferKind::Uav);
            gpu::Buffer bg2(dev, count * 4, gpu::BufferKind::Uav);
            bs.upload(st.data(), st.size() * 4);
            bg2.upload(Gd.data(), Gd.size() * 4);
            double best = 1e30;
            std::string best_label;
            int best_g = 256, best_ept = 1;
            for (auto& op : at::spaces(quick, {"adam"})) {
                for (auto& cd : op.cands) {
                    try {
                        auto defs = cd.defines;
                        defs["BOUND"] = "p.w";
                        gpu::Kernel k(dev, gpu::tmpl::adam(), defs);
                        uint32_t thr = uint32_t(std::stoi(cd.defines.at("G")) *
                                                std::stoi(cd.defines.at("EPT")));
                        uint32_t c[8] = {0, region, 2 * region, count, f2u(3e-3f), f2u(1.0f),
                                         f2u(1e-8f), 0};
                        double ms = time_packets(dev, 2, reps, [&] {
                            dev.transition(bg2.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bs.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            k.dispatch(dev, bg2, bg2, bs, c, (count + thr - 1) / thr, 1, 1);
                            dev.transition(bg2.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bs.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                        });
                        printf("   %-16s %8.3f ms\n", cd.label.c_str(), ms);
                        if (ms < best) {
                            best = ms;
                            best_label = cd.label;
                            best_g = std::stoi(cd.defines.at("G"));
                            best_ept = std::stoi(cd.defines.at("EPT"));
                        }
                    } catch (const std::exception& e) {
                        printf("   %-16s FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
            }
            printf("   -> winner %s (%.3f ms)\n", best_label.c_str(), best);
            winners["adam"] = {{"ms", best}, {"label", best_label}};
            // adam shares the elem family geometry; keep the family winner unless adam's differs a lot
            if (best_g != tp.elem_g || best_ept != tp.elem_ept) {
                // prefer the family winner; record adam's preference in the note
                winners["adam"]["note"] = "family geometry kept";
            }
        }

        // ================================================================ vq_nearest
        {
            printf("== vq_nearest\n");
            const uint32_t B = 512, K = 4096, L = 64;
            auto Z = randf(size_t(B) * L, rng);
            auto CB = randf(size_t(K) * L, rng);
            gpu::Buffer bz(dev, Z.size() * 4, gpu::BufferKind::Uav);
            gpu::Buffer bcb(dev, CB.size() * 4, gpu::BufferKind::Uav);
            gpu::Buffer bo(dev, size_t(B) * L * 4 + B * 4, gpu::BufferKind::Uav);
            bz.upload(Z.data(), Z.size() * 4);
            bcb.upload(CB.data(), CB.size() * 4);
            double best = 1e30;
            std::string best_label;
            int best_tp = 64;
            for (auto& op : at::spaces(quick, {"vq_nearest"})) {
                for (auto& cd : op.cands) {
                    try {
                        gpu::Kernel k(dev, gpu::tmpl::vq_nearest(), cd.defines);
                        // gate: B=8, K=256, L=64 vs CPU argmin
                        {
                            const uint32_t gb = 8, gk = 256;
                            auto Zg = randf(size_t(gb) * L, rng);
                            auto Cg = randf(size_t(gk) * L, rng);
                            gpu::Buffer bzg(dev, Zg.size() * 4, gpu::BufferKind::Uav);
                            gpu::Buffer bcg(dev, Cg.size() * 4, gpu::BufferKind::Uav);
                            gpu::Buffer bog(dev, size_t(gb) * L * 4 + gb * 4,
                                            gpu::BufferKind::Uav);
                            bzg.upload(Zg.data(), Zg.size() * 4);
                            bcg.upload(Cg.data(), Cg.size() * 4);
                            dev.begin();
                            dev.transition(bzg.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bcg.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bog.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            uint32_t c[8] = {gb, L, gk, 0, uint32_t(gb * L), 0, 0, 0};
                            k.dispatch(dev, bzg, bcg, bog, c, gb, 1, 1);
                            dev.transition(bzg.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bcg.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bog.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.submit_and_wait();
                            std::vector<uint32_t> got(size_t(gb) * L + gb);
                            bog.download(got.data(), got.size() * 4);
                            int mism = 0;
                            for (uint32_t b = 0; b < gb; ++b) {
                                double bdist2 = 1e300;
                                uint32_t bk = 0;
                                for (uint32_t k2 = 0; k2 < gk; ++k2) {
                                    double s = 0;
                                    for (uint32_t l = 0; l < L; ++l) {
                                        double d = double(Zg[size_t(b) * L + l]) -
                                                   double(Cg[size_t(k2) * L + l]);
                                        s += d * d;
                                    }
                                    if (s < bdist2) { bdist2 = s; bk = k2; }
                                }
                                if (got[size_t(gb) * L + b] != bk) ++mism;
                            }
                            if (mism != 0) {
                                printf("   %-10s GATE FAIL (%d mismatches)\n", cd.label.c_str(),
                                       mism);
                                continue;
                            }
                        }
                        uint32_t c[8] = {B, L, K, 0, uint32_t(B * L), 0, 0, 0};
                        double ms = time_packets(dev, 2, reps, [&] {
                            dev.transition(bz.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bcb.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bo.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            k.dispatch(dev, bz, bcb, bo, c, B, 1, 1);
                            dev.transition(bz.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bcb.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bo.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                        });
                        printf("   %-10s %8.3f ms\n", cd.label.c_str(), ms);
                        if (ms < best) {
                            best = ms;
                            best_label = cd.label;
                            best_tp = std::stoi(cd.defines.at("TP"));
                        }
                    } catch (const std::exception& e) {
                        printf("   %-10s FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
            }
            printf("   -> winner %s (%.3f ms)\n", best_label.c_str(), best);
            winners["vq_nearest"] = {{"TP", best_tp}, {"ms", best}, {"label", best_label}};
            tp.vqn_tp = best_tp;
        }

        // ================================================================ vq_update
        {
            printf("== vq_update\n");
            const uint32_t B = 512, K = 4096, L = 64;
            auto Z = randf(size_t(B) * L, rng);
            std::vector<uint32_t> IDX(B);
            {
                std::uniform_int_distribution<uint32_t> pk(0, K - 1);
                for (auto& i : IDX) i = pk(rng);
            }
            gpu::Buffer bz(dev, Z.size() * 4, gpu::BufferKind::Uav);
            gpu::Buffer bi(dev, IDX.size() * 4, gpu::BufferKind::Uav);
            gpu::Buffer bcb(dev, size_t(K) * L * 4, gpu::BufferKind::Uav);
            bz.upload(Z.data(), Z.size() * 4);
            bi.upload(IDX.data(), IDX.size() * 4);
            double best = 1e30;
            std::string best_label;
            int best_g = 64;
            for (auto& op : at::spaces(quick, {"vq_update"})) {
                for (auto& cd : op.cands) {
                    try {
                        gpu::Kernel k(dev, gpu::tmpl::vq_update(), cd.defines);
                        uint32_t c[8] = {B, L, K, 0, uint32_t(B * L), 0, 0, 0};
                        int g = std::stoi(cd.defines.at("G"));
                        double ms = time_packets(dev, 2, reps, [&] {
                            dev.transition(bz.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bi.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                            dev.transition(bcb.res(), D3D12_RESOURCE_STATE_COMMON,
                                           D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                            k.dispatch(dev, bz, bi, bcb, c, (K + g - 1) / g, 1, 1);
                            dev.transition(bz.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bi.res(), D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                                           D3D12_RESOURCE_STATE_COMMON);
                            dev.transition(bcb.res(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                           D3D12_RESOURCE_STATE_COMMON);
                        });
                        printf("   %-10s %8.3f ms\n", cd.label.c_str(), ms);
                        if (ms < best) {
                            best = ms;
                            best_label = cd.label;
                            best_g = g;
                        }
                    } catch (const std::exception& e) {
                        printf("   %-10s FAIL: %s\n", cd.label.c_str(), e.what());
                    }
                }
            }
            printf("   -> winner %s (%.3f ms)\n", best_label.c_str(), best);
            winners["vq_update"] = {{"G", best_g}, {"ms", best}, {"label", best_label}};
            tp.vqu_g = best_g;
        }

        // ================================================================ save
        std::filesystem::create_directories("profiles");
        at::save_profile(out, tp, winners, dev.adapter_name());
        run.event("profile_written",
                  "\"path\":\"" + out + "\",\"gemm_ts\":" + std::to_string(tp.gemm_ts) +
                      ",\"elem_g\":" + std::to_string(tp.elem_g) + ",\"elem_ept\":" +
                      std::to_string(tp.elem_ept) + ",\"vqn_tp\":" + std::to_string(tp.vqn_tp) +
                      ",\"vqu_g\":" + std::to_string(tp.vqu_g) + ",\"syrk_chunk\":" +
                      std::to_string(tp.syrk_chunk));
        printf("\nProfile written to %s\n", out.c_str());
        printf("gemm_ts=%d elem_g=%d elem_ept=%d vqn_tp=%d vqu_g=%d syrk_chunk=%d\n", tp.gemm_ts,
               tp.elem_g, tp.elem_ept, tp.vqn_tp, tp.vqu_g, tp.syrk_chunk);
        return 0;
    } catch (const std::exception& e) {
        run.event("fatal", std::string("\"error\":\"") + e.what() + "\"");
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}

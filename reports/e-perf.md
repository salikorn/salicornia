# e-perf — trainer ms/step optimization ladder + test suite (2026-10-25)

Runs: `runs/*test-*/` (suite), `runs/*vqvae-gpu-run*/` (bench/gate). All parity gates re-run
green after every rung: AE 4.35e-7, VQ 5.76e-7 Frobenius vs CPU fp64; q_max_abs_diff 0.0,
idx_mismatch 0/512.

## Test suite (new; `scripts\test.bat` → ctest)
- `src/xlab/test.hpp`: self-registering tests, CHECK/REQUIRE/CHECK_CLOSE/SKIP_IF, JSONL to runs/.
- unit (4 exes): fp16/bf16/fp8 conversions (exhaustive round-trips; **caught a real bug — fp8
  NaN encoded as Inf**), safetensors (synthetic fixture + real shard-62 header), linalg
  (spectrum recovery, square-Q exact fold, exact-rank lossless, orthonormalize), codecs
  (parser, int8 RTN, bpw math).
- gpu (1 exe, SKIP without adapter): GEMM NT/NN/TN incl. packed offsets, elem kernels, packed
  Adam, one-step AE trainer parity, gather kernel (tiny + 40 MB production pattern + gather→GEMM
  continuation). **Found two real bugs**: fp8 NaN-as-Inf (unit tier) and a segfault-ordering
  issue in its own harness (fixed).

## Optimization ladder (iGPU, batch 512, L=64, median of 200 steps)

| Config | baseline | after T4a+b | after T4d (tiled VQ) | target |
|---|---|---|---|---|
| ae_L64 | 8.6 ms | 8.8 ms | 9.0 ms | ≤3 ms |
| vae_L64_b1e-4 | 9.6 ms | 9.9 ms | 9.8 ms | — |
| vqvae_K256_L64 | 17.7 ms | 17.8 ms | **13.7 ms (−23%)** | ≤8 ms |
| vqvae_K4096_L64 | 114.8 ms | 115.8 ms | **51.3 ms (−55%)** | ≤15 ms |

- **T4a+T4b (single-packet step, zero per-step allocations, dataset-resident gather)**: landed
  structurally — the step is now ONE submitted packet; indices/eps staged through persistent
  upload buffers; the 40 MB training set lives on the GPU. Net ms/step ≈ unchanged: the batch
  path was never the bottleneck (the step is GPU-kernel-bound). The per-step `CreateCommittedResource`
  churn is gone, which matters for stability and for later CPU-side pipelining.
- **T4d (tiled vq_nearest)**: one 64-thread group per patch, threads stride K codes in
  parallel, groupshared argmin reduction. K4096: 114.8 → 51.3 ms; parity exact (idx_mismatch 0).
- **T4e (merged Adam)**: skipped after measurement — the 4-5 tiny Adam dispatches cost <0.3 ms
  total; not worth the packed-offset-table complexity now.
- **T4c (fast-mode recording, SRV/UAV persistent + UAV barriers)**: deferred to the autotuner
  round, where barrier cost per kernel gets measured per-shape anyway.

## Debugging saga (recorded because it changed three beliefs)
1. The dataset-gather path "crashed the driver" for six iterations of bisecting (isolated
   kernels passed; trainer died; debug layer said TDR). Root cause was **not** the driver:
   `std::vector<uint32_t> chosen{size_t(batch)}` is brace-init — a ONE-element vector holding
   512 — so the index upload sent 511 words of stack garbage as row indices → wild GPU reads →
   genuine device hang. Same bug class bit twice more (borrowed-buffer double-delete between
   configs; a staging upload inside an open command list → allocator-reset E_FAIL). **Lesson
   recorded for the engine: brace-init on size_t is banned in this codebase (`assign(n, v)`
   convention), and Buffer::upload must assert no open list.**
2. Splitting the batch into its own submit packet (the "safe" version) cost +1 ms/step — two
   extra submits per step are measurable; the single-packet form is the keeper.
3. AE is now GPU-kernel-bound at ~9 ms for ~14 GEMM dispatches; the remaining AE headroom lives
   in T4c (barrier overhead) and kernel fusion, not in the batch path.

## Remaining work (next round, with the autotuner)
- T4c fast-mode recording; K×L-tiled vq_nearest (51 → ~15 ms plausible); autotuner sweep over
  tile/group/chunk geometry per shape class; profile loading in KernelSet; perf smoke test
  under the `perf` label (non-blocking).

## Autotuner round (T5) — tools/autotune.exe

A standalone program anyone can run to find the best launch-geometry configuration for their
machine: `build\autotune.exe [--quick] [--only op,...] [--reps N] [--out profiles/best.json]`.

- **Mechanism**: all kernels are runtime-compiled HLSL templates with @TOKEN@ geometry (gemm/syrk
  tile TS, elementwise group G + elements-per-thread EPT + per-kernel loop BOUND, vq_nearest
  threads-per-patch TP, vq_update group, syrk chunk). The tuner compiles each candidate, gates it
  against a CPU reference on a small shape (mismatch/NaN ⇒ disqualified — it can never pick a
  broken variant), then times warmup + median-of-N on the trainer-representative shape (analysis
  shape recorded for GEMMs). Winners land in profiles/best.json; production code loads it at
  startup (`TuneParams` sanity-checked; defaults when absent/invalid).
- **Sweep space**: 3 GEMM forms × TS{8,16,32} · syrk TS×chunk · elem G{64,128,256}×EPT{1,2,4} ·
  vq_nearest TP{32,64,128} · vq_update G — 27 compile-gate-time cycles per full run, minutes.
- **Result on this iGPU** (full sweep): gemm TS=32 (TN/NN) / 16 (NT), elem G=64, vqn TP=32,
  vqu G=128, syrk chunk 2048. **Adopted profile re-gated: parity exact (AE 4.35e-7, VQ 5.76e-7,
  idx_mismatch 0); bench unchanged within noise (K4096 51.4 vs 51.3 ms; K256 14.1 vs 13.7;
  AE 8.87 vs 8.8).** The hand-tuned defaults were already at the machine's optimum — the
  deliverable is the portable tool + the empirical confirmation, with per-machine wins expected
  elsewhere (the whole point of shipping the tuner).
- Tuner harness bugs found by its own gates during bring-up (all fixed): TN gate read the wrong
  packed slots; gemm gate used a hardcoded tile grid; syrk timing nested allocator resets; elem
  candidates missing the BOUND token.
- Kernel-side refactor shipped with it: templates tokenized (`sal::gpu::tmpl`), `TuneParams`
  threaded through LinAlgKernels/TrainKernels, elem family redesigned to single-pass contiguous
  EPT (a strided-loop form double-covered elements — caught by the adam unit test before any
  production use; grid contract is now `elem_grid(count)` everywhere).

Test suite: 6/6 gpu + 5/5 unit (new test_autotune: enumeration coverage, substitution strictness,
profile round-trip + garbage rejection, geometry helpers).
*(report appended above — autotuner round complete)*

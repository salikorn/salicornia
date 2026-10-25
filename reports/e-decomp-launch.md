# E-DECOMP + S8 — linear-algebraic decomposition experiments & raw D3D12 compute (2026-10-25)

Runs: `runs/20261025-*edecomp*/`, `runs/20261025-*s8-d3d12*/`. Kit: `src/compress/linalg.hpp`
(Eigen 3.4.0 vendored — per user direction to use BLAS-class libraries; SIMD + OpenMP),
`src/st/safetensors.hpp` (mmap reader), `tools/spikes/spike_decomp.cpp`, `tools/spikes/spike_d3d12.cpp`.

## Solver validation (all green before any real-weight claim)

| Test | Result |
|---|---|
| Full symmetric eigensolver (Eigen) on known spectrum (216 decaying + 40 exact-zero singulars) | max rel err **5e-14**; zero tail at 1.5e-16 |
| Top-k subspace iteration (Eigen-powered) vs ground truth | dominant eigenpairs accurate; worst-of-top-k rel err 1.3e-2 on the slowest-decay synthetic (documented limitation; energy metrics use dominant values) |
| **Square-Q exact-fold test** (RMSNorm block, random orthogonal Q) | RMS invariance 1.2e-16, output rel err **1.5e-15** → the SliceGPT folding math is exact in code |
| Exact-rank truncation (rank-64 synthetic, zero tail) | reconstruction 9.5e-16, **−62.5% params, bit-lossless** |
| Synthetic rank-300 activations (d=2048) | energy captured 0.9999997 at d′=1024 (cliff recovered); matches hand-rolled kernels digit-for-digit |

## Real weights — shard 62 (`model-00062-of-00062.safetensors`, 351 tensors, mmap+bf16 verified)

Six largest 2-D bf16 tensors (~200M params total): vision-merger projections and a real
language-layer (L9) attention projection pair.

| Tensor | Shape | tail@50% | tail@75% | tail@81.25% (3328) |
|---|---|---|---|---|
| visual.merger.down_proj | 4096×10240 | 18.9% | 5.6% | 3.6% |
| visual.merger.gate_proj | 10240×4096 | 19.5% | 5.8% | 3.7% |
| visual.merger.up_proj | 10240×4096 | 19.8% | 5.8% | 3.7% |
| L9 self_attn.q_proj | 8192×4096 | 12.6% | 3.4% | **2.1%** |
| L9 self_attn.v_proj | 8192×4096 | 13.8% | 3.6% | **2.3%** |
| visual.merger.proj | 4096×4096 | 8.4% | 0.93% | **0.38%** |

### Verdicts (what sticks, what doesn't)
1. **L1 exact / L2 storage-precision lossless rank reduction is NOT available on trained
   matrices here**: no exact null spaces (λ at rank 3360 ≈ 0.6–0.9 vs λ₁ ≈ 13–100), and bf16
   weight values are exactly representable in fp16 (quantization energy ~1e-12), so any truncation
   is strictly information-losing. "Lossless" applies only to the orthogonal-basis/rotation
   machinery itself (proven exact above) and to exactly-low-rank structures, which these tensors
   are not.
2. **L3 bounded-energy is the real lever and it is moderate**: at the plan's 81.25% width target,
   discarded energy is 0.4–3.7% per tensor (language attention ≈ 2.2%, vision merger ≈ 3.7%).
   This is consistent with SliceGPT-class results and feeds D1/D2 gates: slicing remains viable,
   with the router ridge re-fit + recovery distillation absorbing the tails.
3. Language-layer attention projections decay noticeably faster than vision projections —
   per-layer-type rank/energy budgets (not flat budgets) are justified → supports the E9a
   graph-conditioned allocation hypothesis with concrete evidence.
4. Runtime: ~255 s per 4096-dim tensor (Eigen/OpenMP on 12 threads) — fast enough for Tier-B
   sweeps; full Pass-1 statistics will parallelize the same way.

## S8 — raw D3D12 compute shaders on the iGPU (user direction)

- Hand-written tiled fp32 GEMM (16×16 tiles, groupshared, SM5.1 via runtime `D3DCompile`,
  root-descriptor binding) — **works on the Intel UHD where DirectML cannot**:
  correctness max_abs_diff 1.3e-5 / mean 1.1e-6 at 256³; **84 GFLOPS @512³, 68 GFLOPS @1024³,
  62 GFLOPS @1536³** with a naive un-tuned kernel (3–4× the WARP-DML path; float4 loads, larger
  tiles, and fp16 packing still untapped).
- 2048³ hit the 2 s TDR (single 3.7 s dispatch) — confirmed by 1536³ completing cleanly; the
  engine must chunk dispatches under ~2 s (noted as a backend constraint).
- **Decision consequence (D0 input)**: the engine gains a third backend — D3D12-CS kernels —
  which is also the only path that can host fused dequant+GEMM / fused-attention kernels later.
  DML (WARP recipe) stays the correctness cross-check.

## Budgets measured in passing (S4 partial)
Shard 62 (1.26 GB) + metadata downloaded in <60 s ⇒ ≥ 200 Mbps ⇒ a full 328 GB pass ≈ 35–40 min;
both passes ≈ 75–80 min. Formal disk-stream bench still pending (S4 remainder).

## Next
D0 consolidation → M2 plumbing (FP8 dequant, storage, tokenizer, registry) → M3 CPU layers.
L2 criterion refinement: compare against int8 quantization error (the actual deployment codec),
not fp16, when the codec is chosen at D2.

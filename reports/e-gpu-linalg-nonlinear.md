# GPU analysis linear algebra + non-linear decomposition experiments (2026-10-25, round 2)

Round 3 additions (VQ-VAE/VAE on GPU): see last section. Runs: `runs/20261025-*gpu-linalg*/`,
`runs/*nonlinear-codec*/`, `runs/*vqvae-gpu-*/`. New code: `src/gpu/cs.hpp` (reusable D3D12
compute backend), `src/gpu/kernels.hpp` (GEMM NT/NN/TN + syrk + elementwise/training kernels),
`tools/spikes/spike_gpu_linalg.cpp`, `tools/spikes/spike_nonlinear.cpp`, `tools/spikes/spike_vqvae.cpp`.

## 1. GPU analysis linear algebra — APPROVED with precision gates (user direction, "make sure it is OK")

Pipeline: fp32 Gram (X^T X) via hand-written syrk compute shader on the iGPU + hybrid subspace
eigensolver (GPU does G·V multiplies, CPU does Rayleigh + Cholesky QR), vs CPU fp64 (Eigen) reference
on the real L9 q_proj tensor [8192×4096].

| Metric (gate) | GPU fp32 | CPU fp64 | Verdict |
|---|---|---|---|
| trace rel err (<1e-5) | 6.3e-07 | — | PASS |
| λ₁ rel err (<1e-4) | 1.0e-06 | — | PASS |
| tail energy @ r∈{512,1024,1536,2048} (<1e-4 of trace) | matches to 4–5 significant digits | — | PASS |

- syrk kernel self-test: max_abs_diff 4.6e-4 on fp32 accumulation over m=1024 — expected fp32 level.
- **Verdict: fp32 GPU analysis math is approved for spectrum/energy screening.** Final gate numbers
  (the ≤10% ppl etc. decisions) still cross-checked on CPU fp64, which stays cheap at Tier-B scale.
- Performance, honestly: GPU Gram = **3.9 s vs ~20 s CPU (5×)**. Hybrid eigensolver end-to-end is
  only ~1.1× (84 s vs 92 s at k=2048) because CPU Cholesky orthonormalization (59 s) now dominates —
  next optimization: GPU-side QR or blocked Cholesky. Logged as the known bottleneck.
- TDR learnings (before the timeout was extended): WDDM applies the 2 s limit to whole submitted
  packets, so multi-dispatch submits also trip it; chunked per-submit dispatches were used.
  **TdrDelay now = 10 s, TdrDdiDelay = 15 s (registry, user-approved UAC; effective after reboot)** —
  after reboot, dispatch chunking can be relaxed (single 3–8 s dispatches OK; keep < 10 s).

## 2. Non-linear decomposition bake-off — COMPLETE (real 16×16 patches of L9 q_proj)

| Codec | bpw | rel MSE (weight-space, held-out patches) |
|---|---|---|
| fp16 | 16 | 0 (reference) |
| int8 RTN per-tensor | 8 | **4.3e-4** |
| naive patch k-means VQ (k=16/64/256) | 4.0 / 6.0 / 8.0 | ~0.98 — **catastrophic** |
| global SVD (measured curve) | 0.75 / 1.5 / 3.0 / 6.0 | 0.765 / 0.655 / 0.511 / 0.326 |
| linear AE (L=32/64/96, fp16 latents) | 2.1 / 4.1 / 6.1 | 0.875 / 0.749 / 0.624 |
| MLP (GELU) AE, same latents | 2.1 / 4.1 / 6.1 | 0.875 / 0.750 / 0.626 |

Verdicts:
1. **Non-linearity does not pay on real weights here**: MLP-AE ≈ linear-AE at every bitrate
   (marginally worse; latent-fp16 quantization is free either way). Combined with the NWC-etal
   literature (learned decoders concede <3.5 bits to trellis/VQ), the neural-decoder track stays
   gated/low-priority — E6 now has local evidence.
2. **Naive whole-patch k-means VQ fails in 256-d** (curse of dimensionality; centroids ≈ mean).
   The VQ track must use sub-vector (product) quantization on 8–16-dim groups + GPTQ-style error
   feedback — i.e., the VPTQ-family design already in the plan. Whole-vector k-means: dead.
3. **Patch-local codecs trail global SVD badly at equal bits** (0.749 @ 4.1bpw vs 0.511 @ 3bpw):
   global linear structure dominates on this tensor. Expert-level codecs must be global/tensor-aware.
4. **int8 RTN at 4.3e-4 rel MSE is the practical bar** — sub-8bpw codecs must beat the SVD curve
   to justify themselves. Caveat recorded: this bake-off measures raw weight-space MSE; the D2
   decision gate will re-rank finalists by activation-weighted distortion (‖WX−ŴX‖²), which is
   what GPTVQ/VPTQ-class methods actually optimize.

## Status
- `src/gpu/cs.hpp` (Device/Buffer/Kernel) is the seed of the M4 D3D12-CS backend.
- All events in the run registry; decision-gate inputs recorded for D0/D1.
- TDR: extended via `scripts/set_tdr.ps1` (TdrDelay=10, TdrDdiDelay=15; effective after reboot).

## 3. VQ-VAE / VAE trained ON GPU (round 3, user direction)

Built a mini training framework on the D3D12-CS backend (`src/gpu/kernels.hpp`): GEMM NT/NN/TN,
GELU fwd/bwd, MSE, sum-of-squares, packed-state Adam, VQ nearest/update, VAE sampling/grads —
one submitted packet per training step.

**Trainer gate (CPU fp64 reference, one full step): PASSED** — every forward layer, every
gradient (max abs diff ~1e-11 on scales ~1e-5), and post-Adam params match to Frobenius-relative
4.3e-7. The iGPU is now a *verified* training device (also the seed of the M9 training stack).
Bugs found and fixed on the way (all recorded for the engine): Eigen column-major vs row-major
packing at every buffer boundary; root constants must be a single 32-byte cbuffer (a second
`register(b1)` reads nothing); Adam needs per-tensor GRAD offsets when gradients are packed;
scratch buffers must match the shape they replace (an under-sized [B,256] scratch for [B,384]
caused OOB reads the driver rejected); duplicate resource transitions within one list are fatal.

Results (real L9 q_proj patches, 2000 steps, batch 512, ~9-25 ms/step on the iGPU):

| Codec | bpw | rel MSE | note |
|---|---|---|---|
| AE L64 (GPU) | 4.117 | 0.7484 | **matches CPU-trained AE (0.7493) — trainer cross-validated** |
| VAE L64 β=1e-4 | 8.117 | 0.9957 | poor; μ+λ storage doubles bpw |
| VAE L64 β=1e-2 | 8.117 | 0.99999 | posterior collapse |
| VQ-VAE K=256 | 0.156 | NaN | diverged (see below) |
| VQ-VAE K=4096 | 0.289 | NaN | diverged; also 123 ms/step (naive O(B·K·L) nearest) |

Verdicts:
- **GPU training works and is verified**; for this tiny model it is ~1.2× CPU (overhead-bound);
  larger models amortize the per-step submit cost.
- **Continuous VAEs are poor weight codecs** (KL vs near-zero reconstruction σ; also 2× latent
  storage) — deprioritized vs plain AE, consistent with the round-2 non-linear verdict.
- **VQ-VAE diverged to NaN on first contact** — expected first-run failure modes now queued:
  commitment loss (β≈0.25), codebook EMA updates, gradient clipping, lower codebook-path lr,
  distance-overflow guards. The architecture stays interesting because its storage is tiny
  (0.16-0.29 bpw payloads); stabilization is the next experiment, not a redesign.

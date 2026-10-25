# VQ-VAE stabilization + oracle/zlib spikes (2026-10-25, round 4)

Runs: `runs/20261025-043317-vqvae-gpu-run/` (pre-fix, NaN), `runs/20261025-054012..054221-vqvae-gpu-gate/`
(9 gate runs on the final binary, built 05:40:04), `runs/20261025-054222-vqvae-gpu-run/` (final bake-off).
Changed code: `src/gpu/kernels.hpp` (vq_commit kernel, EMA vq_update, guarded Adam + vq_nearest),
`tools/spikes/spike_vqvae.cpp` (direct VQ dispatch, mode-2 gate section, data-aligned codebook init, warmup).
Oracle/zlib numbers come from this round's spike sessions (commands quoted inline); the reporter did not
re-run those two spikes — attribution is stated per number below.

## 1. VQ-VAE stabilization — STABLE (NaN root-caused and fixed); codec result is a NEGATIVE finding

**Stable: true** — the round-3 NaN is gone: `vqvae_K256_L64` and `vqvae_K4096_L64` rel_mse are
`-nan(ind)` in `runs/20261025-043317-vqvae-gpu-run/events.jsonl` (pre-fix) and finite
(9.752087e-01 / 9.665478e-01) in `runs/20261025-054222-vqvae-gpu-run/events.jsonl` (final binary).

### 1a. Root cause

`vq_nearest` is `[numthreads(64,1,1)]` (`src/gpu/kernels.hpp:345`, comment at :343) but was dispatched
via `elem()`, whose `(count+255)/256` grid assumes 256-thread groups (`src/gpu/kernels.hpp:487`). With
batch B=512 only 128 of 512 rows were quantized, so `vqout` rows 128..511 — Q floats AND indices — were
uninitialized GPU-heap memory feeding the decoder forward and `vq_update` every step. Fix: direct
dispatch with the matching `(B+63)/64` grid and explicit transitions (`tools/spikes/spike_vqvae.cpp:197`).

### 1b. Remedies applied

1. **Dispatch fix** (above) — the actual NaN source.
2. **Commitment loss** (β=0.25, encoder-side only): new `vq_commit` kernel (`src/gpu/kernels.hpp:417-427`)
   adds `beta_eff*(z − sg(Q))` to the straight-through gradient in gz, with
   `beta_eff = 0.25·2/(B·L)` matching the `2/(B·IO)` mean-MSE recon-gradient scale
   (`tools/spikes/spike_vqvae.cpp:230-234`); wired into record_backward for mode 2 only — decoder and
   codebook see no commitment gradient.
3. **Codebook EMA** instead of hard batch-mean replacement: `vq_update` does `CB += gamma·(batch_mean − CB)`
   for assigned codes, gamma=0.25 (`src/gpu/kernels.hpp:376-403`, gamma passed at
   `tools/spikes/spike_vqvae.cpp:269`); empty codes and non-finite means left unchanged. (This is the
   "EMA instead of lower codebook-path lr" option.)
4. **Update-norm clipping + non-finite gradient guard in Adam** (`src/gpu/kernels.hpp:303,315`): normalized
   update `u = mhat/(sqrt(vhat)+eps)` clamped to [−1,1] (one step moves |dP| ≤ lr per element); non-finite
   gradients dropped so a transient Inf/NaN grad cannot poison the moments. AE gate parity unaffected
   (clamp never binds at step 1; AE max_param_frob_rel unchanged at 4.347e-07, run `…-054221-…-gate`).
5. **Distance-overflow guards in `vq_nearest`** (`src/gpu/kernels.hpp:361`): a non-finite summed distance
   disqualifies the code instead of poisoning the argmin; if all codes are disqualified, bk=0 keeps Q finite
   (a codebook row). Additional isfinite guards in `vq_commit` and `vq_update`.
6. **VQ-only warmup LR**: 100-step linear ramp multiplied onto the cosine schedule, mode 2 only
   (`tools/spikes/spike_vqvae.cpp:846-847`); AE/VAE dynamics deliberately identical to the round-3 reference.
7. **Data-aligned codebook init** (`tools/spikes/spike_vqvae.cpp:800-830`): one forward pass on the first
   training rows; each of the K codes seeded from a batch z (coprime-stride sampling) + 0.1·std jitter,
   replacing N(0,0.3) random init that starts far from the z-cloud (dead codes + violent first-contact EMA).
8. **Gate extended with a mode-2 single-step CPU parity check** (`tools/spikes/spike_vqvae.cpp:556-698`):
   verifies `vq_nearest` (Q values AND indices over all 512 rows), STE+commitment backward, clamped Adam,
   and the EMA codebook update against an Eigen CPU reference; fails above frob-rel 1e-3 (looser than the
   AE 1e-4, to tolerate a possible fp32 nearest-neighbor near-tie flip).
9. **Two bugs introduced this round, caught by the new gate and fixed**: (a) `std::vector<uint32_t>
   qIdx{B}` brace-initializes a ONE-element vector holding the value 512 — the CPU argmin loop then wrote
   ~2 KB past it, causing the wandering STATUS_HEAP_CORRUPTION (0xC0000374) crashes during verification;
   fixed with the count constructor (`tools/spikes/spike_vqvae.cpp:573`); (b) the codebook comparison
   indexed Eigen column-major `CBm.data()[i]` against row-major GPU data (comparing a matrix to its
   transpose); fixed to indexed (k,l) access (`tools/spikes/spike_vqvae.cpp:664-671`).

### 1c. Checks (run by the stabilization session on the final binary, built 05:40:04; artifacts on disk —
the reporter verified the artifacts by grep this session but did not re-build or re-run)

- `cmd /c scripts\build.bat` — clean build.
- `build\spike_vqvae.exe --gate` — **9/9 consecutive passes** on the final binary (runs
  `20261025-054012/13/14/15`, `-054120/21/22/23`, `-054221`; reporter's grep: every run reports
  `vq_max_param_frob_rel 5.759e-07`, zero fatals). The stabilization session claimed 10/10; the reviewer
  counted 9 gate runs after the 05:40:04 build and the reporter's own `ls runs/` confirms 9 — the count
  was off by one, the quoted numbers are genuine: from `runs/20261025-054221-vqvae-gpu-gate/events.jsonl` —
  AE trainer gate max_param_frob_rel **4.347e-07** (threshold 1e-4, unchanged vs baseline);
  `gate_vq_forward` **q_max_abs_diff 0.0, idx_mismatch 0** (B=512); `vq_codebook` EMA frob **1.158e-08**;
  VQ trainer gate **vq_max_param_frob_rel 5.759e-07** (threshold 1e-3); `vq_W1..W4` frob 3.7–5.8e-07
  (per the session's summary; W-rows not re-grepped by the reporter).
- `--run` — the stabilization session stated it was NOT executed "per instructions"; **the reviewer found
  and the reporter verified a completed bake-off from the final binary on disk**:
  `runs/20261025-054222-vqvae-gpu-run/events.jsonl` (started 05:42:22, ~2.5 min after the build, 1 s after
  the final gate run 054221; all five codec points emitted, `"done","status":"ok"`). Whoever executed it,
  the artifact exists and the numbers below are its outcome.

### 1d. Final codec points (all from `runs/20261025-054222-vqvae-gpu-run/events.jsonl`, 2000 steps)

| Codec | bpw | rel MSE | ms/step (same run) |
|---|---|---|---|
| AE L64 | 4.117 | **7.483640e-01** | 9.00 |
| VAE L64 β=1e-4 | 8.117 | 9.957348e-01 | 10.00 |
| VAE L64 β=1e-2 | 8.117 | 9.999969e-01 | 10.01 |
| VQ-VAE K=256 | 0.156 | 9.752087e-01 (was `-nan(ind)` in `…-043317-…`) | 18.29 |
| VQ-VAE K=4096 | 0.289 | 9.665478e-01 (was `-nan(ind)` in `…-043317-…`) | 118.08 |

### 1e. Residual risk, stated honestly

Single-step CPU parity proves the VQ wiring/mechanics and that the NaN source is removed. But
"rel_mse meaningfully below 1.0" for K=256/K=4096 over 2000 steps is **not** achieved by the on-disk
bake-off: 0.966–0.975 rel MSE explains only ~3% of variance — finite, but far from useful, and far behind
AE at 0.748 (4.117 bpw). A codec decision gate should read these rows as **VQ-VAE at one index per
256-weight patch does not work at these budgets/steps** — a negative result, not a missing measurement.
If it is iterated anyway (not NaN, just weak), the next knobs are beta (0.25), vq_gamma (0.25), and warmup
length (100), all isolated as named constants/members (`tools/spikes/spike_vqvae.cpp:98-99,847`).

### 1f. Reviewer problems (all against the final code + run artifacts; reporter re-verified the marked ones)

1. **WORST — "run not executed" contradicted by the artifact**: the summary's claim was contradicted by
   `runs/20261025-054222-vqvae-gpu-run/events.jsonl` (timing chain verified: build 05:40:04 → gate 054221
   → run 054222). *(Reporter verified: file exists, 5 codec points, done ok.)*
2. **DECISION-RELEVANT — the bake-off answers the residual-risk question negatively** (numbers in §1d/§1e).
3. Gate count 10/10 → actually **9/9**; quoted gate numbers match the 054221 artifact exactly. *(Reporter
   verified: 9 runs, all 5.759e-07, zero fatals.)*
4. **Latent D3D12 hazard in `elem()`** (`src/gpu/kernels.hpp:470-484`): if a kernel is ever called with
   `o == b`, `o_unique` is false and the `o != a` else-branch never fires, so the output is transitioned to
   SRV but never to UAV before dispatch. No current call site hits this (all `elem()` users at
   `tools/spikes/spike_vqvae.cpp:174,182,210,218,224,234,240,253,259` have o distinct from a and b) — an
   unguarded trap for the next caller. *(Reporter verified the code shape at :470-484.)*
5. **Mode-1 state mismatch**: in-place `elem_add` at `tools/spikes/spike_vqvae.cpp:249` (o == a == gh1)
   transitions gh1 COMMON→SRV→UAV (`kernels.hpp:482-484`), leaving the buffer in UAV while still bound as
   t0 SRV — formally invalid D3D12 (debug layer would flag); same-index read-then-write so likely correct
   on real hardware. Affects only the VAE rows it serves — which sit at rel_mse 9.957e-01/9.9999e-01
   (run `…-054222-…`), i.e. essentially no learning, not diagnosed beyond this.
6. **Latent capacity cap**: `vq_update` accumulates into a fixed `float acc[64]`
   (`src/gpu/kernels.hpp:386-392`); any mode-2 run with L > 64 writes past it in-shader, no assert/guard.
   Safe today only because every current config uses L=64 (`tools/spikes/spike_vqvae.cpp:794-795`; gate
   Lq=64 at :558).
7. **Gate parity asymmetry**: GPU `vq_nearest` disqualifies non-finite distances (`kernels.hpp:361`) but
   the CPU gate argmin uses a plain `s < best` (`spike_vqvae.cpp:583`) — the guard is untested against the
   case it exists for (gate inputs are finite by construction).
8. **Pre-existing accounting optimism** (not introduced this round): bpw budgets z, codebook, and shared
   weights at 16 bits (`spike_vqvae.cpp:856-865`) while `eval_rel_mse` reconstructs from full fp32 state
   (`spike_vqvae.cpp:703-728`), so absolute rel_mse is slightly better than a deployed fp16 codec would
   achieve; consistent across codecs, so rankings are unaffected.
9. **Within-round confound** (declared in the session summary): the 100-step LR warmup
   (`spike_vqvae.cpp:846-847`) is mode-2 only, so the bake-off compares "VQ-VAE with warmup" vs
   "AE/VAE without"; at VQ rel_mse ~0.97 the confound did not change the outcome, but architecture-vs-
   schedule attribution from these rows alone is not clean.

### 1g. Reviewer assessment

The VQ-VAE mechanics can be trusted; the codec cannot. The reviewer verified by reading both files
end-to-end: dispatch fix real and matched (numthreads(64) at `kernels.hpp:345`, direct `(B+63)/64` grid at
`spike_vqvae.cpp:197`; `vq_update` likewise at `kernels.hpp:380`/`spike_vqvae.cpp:271`; every kernel still
routed through `elem()` is a 256-thread kernel); STE and commitment wiring mathematically right (gz =
dL/dQ flows to the encoder; commitment added at `spike_vqvae.cpp:234` only after gW4/gW3 are recorded at
:221/:228, so decoder and EMA codebook see no commitment gradient; `beta_eff = 0.25·2/(B·L)` at :232
matches the CPU gate at :628); the EMA update (`kernels.hpp:394-399`) matches its CPU reference
(`spike_vqvae.cpp:639-647`); the COMMON-hub transition pattern is consistent with buffers created in
COMMON (`cs.hpp:131`) on every traced path; both gate-found bug fixes are in place (:573, :676-681). Run
artifacts corroborate: the 04:33 pre-fix run shows the round-3 NaN, the 05:42 final run shows it gone, and
AE/VAE rows reproduce to seven digits across both runs. Conclusion: trust the gate numbers (genuine,
reproducible artifacts), trust that the NaN root cause is fixed, and trust the bake-off to reproduce — but
the numbers themselves say the VQ-VAE configuration being ranked is not yet a working codec; consume it as
a negative result, do not await a further run. The VAE rows (~0.996–1.0) have learned nothing across both
rounds. (Reviewer performed no build/run; reporter likewise — verification here was read + grep only.)

## 2. Oracle availability spike — best: **torch-cpu** (zero setup on this machine)

Run by the oracle spike session on this machine (Python 3.14.3 at `C:\Python314\python.exe`, pip 25.3);
no file in the working dir modified; not re-run by the reporter — numbers are the spike session's, quoted
with their commands.

1. **torch CPU wheel for Python 3.14: YES.** The named `python -m pip download torch --no-deps --dry-run
   --index-url https://download.pytorch.org/whl/cpu` failed (pip 25.3's `download` has no `--dry-run`);
   the named alternative `python -m pip index versions torch --index-url …/whl/cpu` returned
   `torch (2.14.1+cpu)` with 2.14.1/2.14.0/2.13.0/2.12.1/2.12.0/2.11.0/2.10.0/2.9.1/2.9.0 available,
   `INSTALLED: 2.11.0`. Stronger: torch is already installed and verified — `python -B -c "import torch,
   numpy; …"` printed `torch 2.11.0+cpu`, `numpy 2.4.6`, `matmul ok: torch.Size([4, 4])` (a real 4×4
   matmul on this CPU). Nothing installed; `-B` avoided bytecode caches.
2. **numpy for Python 3.14: YES.** `python -m pip index versions numpy` → `numpy (2.5.4)` available,
   `INSTALLED: 2.4.6`; same import check confirmed numpy 2.4.6 imports and runs under 3.14.3.
3. **llama.cpp prebuilt Windows binaries + glm5-next: BOTH YES, with a caveat.**
   `curl -sI https://github.com/ggml-org/llama.cpp/releases` → HTTP/1.1 200 OK. Latest tagged release
   **v0.7.0** (published 2026-10-15) explicitly ships GLM5-Next support (release body: "MTP for
   GLM5-Next", "GLM5-Next: MTP speculative decoding with an optimized sparse attention path [#29928]",
   "remove the gather path of the glm5-next sparse attention (#30042)") but attaches **no binaries**
   (single asset `nightly-tag.txt`, 7 bytes). Prebuilt Windows binaries live on the b-numbered releases:
   **b11869** (published 2026-10-24) has 36 assets including `llama-b11869-bin-win-cpu-x64.zip`,
   `-win-cuda-12.4-x64.zip`, `-win-cuda-13.4-x64/arm64.zip`, `-win-vulkan-x64/arm64.zip`,
   `-win-sycl-x64.zip`, `-win-rocm-10.0-x64.zip`, `-win-openvino-2026.4.1-x64.zip`, `-win-cpu-arm64.zip`,
   `-win-opencl-adreno-arm64.zip`. Caveat: b11869's own body does not mention glm5-next (nightly bodies
   are bare); "prebuilts include glm5-next" rests on b11869 postdating v0.7.0 by 9 days and building from
   master — an inference, not verified by running a glm5-next model.

**Verdict: torch-cpu** — cp314 wheels offered AND 2.11.0+cpu already installed and verified doing matmul
on this exact interpreter; verification can start with zero setup. numpy 2.4.6 likewise ready as second
oracle. llama.cpp viable but heavier: would additionally require downloading a glm5-next GGUF model
(not attempted).

## 3. zlib ratio spike — the "~1.0 for trained weights" expectation is FALSIFIED on this shard

Run by the zlib spike session: two `python - <<EOF` heredoc runs (Python 3.14.3, stdlib only:
json/struct/mmap/zlib) against `models/GLM-5.3-Flash/model-00062-of-00062.safetensors` (verified
1,261,584,968 bytes; 8-byte LE header length 39,488, data at offset 39,496; 351 tensors, **all BF16**
— sum of tensor bytes 1,261,545,472 = file size minus data start, no hidden regions; no F8_E4M3 exists
in this shard, so per the ask's "if present" clause all samples are BF16). Only 5 tensors are ≥64 MiB, so
exactly 5 clean 64-MiB samples exist. Each sample: mmap slice at data_start+offset, `zlib.compress(raw,
6)`, `zlib.decompress(packed)==raw` verified True on every sample. **Not re-run by the reporter.**

| Tensor (BF16) | raw bytes | packed bytes | ratio |
|---|---|---|---|
| `model.visual.merger.up_proj.weight` [10240,4096] | 67,108,864 | 52,423,367 | **0.781** |
| `model.visual.merger.gate_proj.weight` [10240,4096] | 67,108,864 | 52,483,294 | 0.782 |
| `model.visual.merger.down_proj.weight` [4096,10240] | 67,108,864 | 52,473,694 | 0.782 |
| `…layers.9.self_attn.v_proj.weight` [8192,4096] (whole tensor) | 67,108,864 | 38,455,169 | **0.573** |
| `…layers.9.self_attn.q_proj.weight` [8192,4096] (whole tensor) | 67,108,864 | 38,334,745 | 0.571 |

(All rows from the corrected second heredoc run; ratios 0.571–0.782.)

- **Byte-plane diagnostics** (same run): BF16 high bytes (sign/exponent/msb) take only 28–33 distinct
  values, ~2.7 bits/byte entropy (weights concentrated in a narrow magnitude range); merger low mantissa
  bytes near-random (~7.97 bits/byte) while layer-9 q/v_proj low bytes are heavily skewed (~4.9 bits/byte;
  top values 160/32/16 each ~3.4%) — consistent with mantissas effectively coarser than BF16
  (quantization upcast), which is why attention projections compress to 0.57.
- **Storage-layer implication**: do not assume weight bytes are incompressible, but don't credit blind
  deflate as the win either — the compressibility is dtype-structure (clustered exponent bytes,
  low-entropy mantissas), so a dtype-aware layer (byte-plane split + entropy coding, or explicit
  quantization) should beat plain zlib; ratios must be re-measured per model/shard, not assumed. CPU cost
  of level-6 deflate measured at ~2–4 s per 64 MiB here — weigh against I/O saved.
- **Caveats**: single shard (62/62, vision-merger- and tail-layer-heavy); the first exploratory run
  overread a 33,554,432-byte tensor (`model.visual.merger.proj.weight`) into its neighbor under the fixed
  64-MiB slice and that row was discarded — all reported rows come from the corrected run, which asserts
  each slice stays inside the tensor's data_offsets.

## Status

- VQ-VAE NaN root cause fixed and gate-verified; VQ mechanics (dispatch, STE+commitment, clamped Adam,
  EMA codebook) proven against CPU parity — but the codec rows are a negative result at these
  budgets/steps (§1e); consume as data, not as a pending measurement.
- VAE rows unchanged and still learning nothing (both rounds) — deprioritization from round 3 stands.
- Carried hazards for the engine: `elem()` o==b transition trap (§1f.4), in-place elem_add UAV/SRV
  mismatch (§1f.5), `acc[64]` capacity cap in vq_update (§1f.6), untested non-finite branch in the gate's
  CPU argmin (§1f.7), 16-bit bpw accounting vs fp32 eval (§1f.8).
- Oracles ready: torch-cpu 2.11.0+cpu and numpy 2.4.6 already installed and verified on Python 3.14.3;
  llama.cpp prebuilts exist (b11869) if a glm5-next GGUF oracle is ever needed.
- Weight bytes are NOT incompressible: zlib-6 ratios 0.571–0.782 on 64-MiB BF16 samples of shard 62 —
  entropy-code after dtype-aware transforms, and re-measure per shard.

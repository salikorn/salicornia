# D0 input — S1/S2 DirectML spike results (2026-10-25)

Run logs: `runs/20261025-020804-s1s2-dml-warp/` (final green run), earlier runs in `runs/`.

## Verdict

**DirectML is functional on this machine via WARP (software D3D12) with a specific recipe; the
Intel UHD driver (31.0.101.3358, 2022) cannot execute ANY DirectML dispatch (device hang) in every
tested configuration.** Development proceeds on WARP; the hardware path needs an Intel driver
update (user decision, pending).

## The recipe that works (mandatory for src/dml)

1. **COMPUTE queue + COMPUTE command lists** (DIRECT also works on WARP, but COMPUTE matches the
   samples and is what we tested green end-to-end).
2. **Always compile via `IDMLDevice1::CompileGraph`** — even single operators. Bare
   `CompileOperator` dispatches are broken on this system (fp16/fp32 GEMM writes zeros and
   eventually removes the device; happens on WARP + Intel alike).
3. **`DML_EXECUTION_FLAG_DISABLE_META_COMMANDS`** on every compile. DML's metacommand negotiation
   with this OS/driver generation is broken (`CreateMetaCommand: MetaCommand parameters are not
   supported by the current system configuration`).
4. **Initializer outputs must be BUFFER_ARRAY bindings** (`DML_BUFFER_ARRAY_BINDING{1, &binding}`),
   never a plain buffer binding — otherwise the persistent resource is never initialized and exec
   dispatches read garbage.
5. **Fully explicit resource-state transitions** (COMMON→UAV around dispatches, COMMON→COPY_DEST /
   COPY_SOURCE around copies); do not rely on implicit promotion/decay.
6. Inputs must live in DEFAULT-heap buffers with ALLOW_UNORDERED_ACCESS; stage through UPLOAD
   heaps; read back through READBACK heaps (`Map` read-range must be the exact byte range).

## Measured results (WARP, i5-12450H)

| Test | Result |
|---|---|
| Identity exec | exact `[0,1,2,3]` ✓ |
| Softmax1 exec | matches reference to 1e-6 ✓ |
| FP16 GEMM 8³ (tiny) | bit-exact vs CPU ref ✓ |
| FP16 GEMM 256³ | max_abs_diff 0.0155, mean 0.0015 (values ~±20) ✓ |
| FP16 GEMM 1024³ throughput | 92.2 ms/op ≈ **23.3 GFLOPS** (WARP software) |
| FP32 GEMM 256³ throughput | 1.90 ms/op ≈ 17.6 GFLOPS |

## Probe results (CreateOperator/CompileGraph compile-level)

- Feature level: **6_3** supported by DML device (on WARP and Intel).
- dtypes: float16, float32, int8, uint32, int32, int64 supported.
- Ops compile OK: gemm(f16/f32), softmax1, log_softmax1, top_k1, cumulative_summation, swish,
  matrix_multiply_integer.
- Ops E_INVALIDARG at creation (need desc variants — open, non-blocking): gather_elements,
  multihead_attention1, adam_optimizer. Attention will be composed manually anyway (MLA/KDA/DSA
  don't map to MHA), and AdamW can be hand-composed from elementwise ops if the op stays broken.

## Configurations tested (all deterministic, reproducible)

- DirectML redist 1.15.4, 1.10.1, 1.9.1 × {WARP, Intel} × {DIRECT, COMPUTE queue} ×
  {metacommands on, off} × {CompileOperator, CompileGraph} × {implicit, explicit transitions}.
- Intel outcome identical in all: `device removed 0x887A0006` (DXGI_ERROR_DEVICE_HUNG) at the
  first operator dispatch.

## Consequences for the plan (D0, partial)

- Engine DML layer: graphs-only, no metacommands, compute queue, explicit transitions — the
  spike's `Env` + `compile_single_op_graph` + `run_gemm` are the seeds of `src/dml`.
- Runtime path until driver update: **WARP for correctness bring-up**; expected LLM-relevant
  throughput on WARP is poor (~23 GFLOPS) — the CPU fp32 reference path (M3) may outperform WARP
  and will be benchmarked against it; final performance rests on the Intel driver update.
- Ask user: update Intel graphics driver (2022 → current). If updated, re-run
  `build/spike_dml.exe --compute --check 256 --m 1024 --iters 10` to re-evaluate.

## Remaining M1 spikes

S3 real-shard safetensors/FP8 · S4 bandwidth · S5 oracle availability · S6 codec micro-benches ·
S7 HF write round-trip.

# salikorn

Making **GLM-5.3-Flash** (zai-org, 320B total / 18B active params, MIT) runnable **fully locally**
on a modest laptop (i5-12450H, 8 GB RAM, Intel UHD iGPU, ~87 GB free disk) with a custom
**C++23 + DirectML** inference & training stack — no ONNX Runtime, no PyTorch at runtime.

The 328 GB FP8 checkpoint never fits on disk: it is consumed **one safetensors shard at a time**
(two streaming passes total) and transformed into a compressed local "student" via:

- **Structural surgery** (parameter count): REAP expert pruning, SliceGPT-style orthogonal
  width slicing with exact weight folding (the "embedding VAE" as a trainable orthogonal map),
  router ridge re-fits, expert slimming.
- **Weights-as-data codecs** (bits/param): VQ codebooks (VPTQ-style), cross-expert shared
  codebooks, entropy coding (rANS/Huffman), binary/ternary tracks — decided by bake-off, not by fiat.
- **Recovery training**: hand-written backward passes on DirectML with ZeRO-1/2/3-style
  disk offload, LoRA + codebook fine-tuning, logit-KL self-distillation from cached teacher outputs.

Everything is **experiment-gated** (spikes first, decision gates D0–D4), logged in a queryable
run registry, and published openly (student + caches + full experiment history to Hugging Face).

See [PLAN.md](PLAN.md) for the full approved plan and `reports/` for generated results.

## Status

- [x] M0 toolchain skeleton
- [ ] M1 spikes S1–S7 → decision gate D0
- [ ] M2 core plumbing (safetensors, FP8 dequant, storage, tokenizer)
- [ ] M3 CPU reference layers (KDA / DSA / MLA / MoE / mHC)
- [ ] M4 DirectML engine + tiny-model generation
- [ ] M5–M7 calibration passes, bake-off, compressed `.slm` student
- [ ] M8 local chat runtime
- [ ] M9 training stack + recovery
- [ ] M11 open release

## Build (Windows, MSVC)

```bat
scripts\build.bat
scripts	est.bat             # unit tests (ctest; scripts	est.bat gpu for the GPU suite)
buildutotune.exe            # sweep launch-geometry params on THIS machine (minutes)
                              # -> profilesest.json, loaded automatically at startup
build\spike_vqvae.exe --gate  # GPU trainer parity vs CPU fp64
build\spike_vqvae.exe --run   # codec bake-off on real weights
build\spike_dml.exe          # S1/S2 DirectML spike (add --warp to force CPU adapter)
```

Vendored deps under `extern/`: DirectML NuGet 1.15.4 (headers + x64 dll/lib), DirectMLX.h,
nlohmann/json, miniz. License: MIT.

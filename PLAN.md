# salikorn — GLM-5.3-Flash, locally: C++23 + DirectML compress-train-infer stack (open release, experiment-gated)

Approved plan. Governing principle: **experiment before committing** — nothing expensive or hard
to reverse happens before its enabling experiments pass; every path choice is made at a named
decision gate backed by run-registry evidence.

## Decision gates
- **D0** after spikes S1–S7 (runtime path DML-primary vs hybrid, oracle choice, pass strategy)
- **D1** after Tier-A matrix (structural cut shortlist + codec shortlist)
- **D2** after Tier-B bake-off on real-model slices (final codec + cut point)
- **D3** after recovery sweeps (recovery regime)
- **D4** final end-to-end re-gate before open release

## Spikes (M1, before build-out)
S1 DML fp16 GEMM throughput + numerics on this iGPU · S2 driver feature-level + operator probes ·
S3 real-file spike (config/tokenizer + smallest shard tail; safetensors mmap + FP8 dequant) ·
S4 measured HF download + disk-stream bandwidth · S5 oracle availability (torch on py3.14? numpy?
llama.cpp glm5-next as semantic oracle) · S6 codec micro-benches (CPU VQ encode speed, zstd ratios,
fp64 symmetric eigensolver for 4096² Gramians, REAP scoring on synthetic routing) ·
S7 HF write path round-trip (chunked upload/download/evict).

## Verified facts
- **Model** `zai-org/GLM-5.3-Flash` (MIT, ungated): 320B/18B active; 62 FP8-e4m3 shards, 328.3 GB.
  Arch `glm5_next`: 45 layers = 34× KDA gated-delta-net linear attention + 11× DeepSeek sparse
  attention (MLA latent, NoPE; q_lora 1536, kv_lora 512; indexer top-2048); mHC hyper-connections (×4);
  MoE 288 routed + 1 shared, top-8 sigmoid noaux-tc (scale 2.5), first 3 layers dense FFN 12288;
  clipped SwiGLU (10); RMSNorm 1e-5; vocab 154,880 byte-level BPE; untied embeddings; MTP ignored;
  ViT tower = phase 2. Embeddings/heads/routers/norms/attention bf16; experts FP8 [128,128]-block.
- **Hardware**: Win10 21H2, i5-12450H, 8 GB RAM + 16 GB pagefile, Intel UHD iGPU, 86.6 GB free disk;
  VS 2022 Community (MSVC 14.44) + SDK 26100; CMake/Ninja/Py3.14/git/curl. DirectML NuGet 1.15.4 vendored.
- **DirectML**: GEMM fp16/fp32 only (no BF16/FP8), GATHER, SOFTMAX1, TOP_K1, CUMULATIVE_SUMMATION,
  SWISH, ADAM_OPTIMIZER, INT8/INT4 paths; DirectMLX + CompileGraph; UAV barriers/fences ours;
  no autograd → hand-written backward; no fused dequant → decode at prefetch.
  NOTE (verified against the 1.15.4 header): the only feature queries are FEATURE_LEVELS and
  TENSOR_DATA_TYPE_SUPPORT — operator availability must be probed via CreateOperator/CompileOperator.

## Compression — two axes
**Axis 1 structure** (experts = 95% of params): REAP gate-aware expert pruning (near-lossless @25%,
published @50%); SliceGPT-style orthogonal width slicing d→d′ with exact folding (square-Q bit-exact
self-test first); router-row ridge re-fit + SparseGPT-style closed-form reconstruction; d_ffn
slimming (SlimMoE). Merging avoided. The "embedding VAE" = trainable orthogonal P (Cayley-param
linear VAE on activations, PCA-init) — folds exactly; orthogonality ⇒ lossless-by-construction.
**Axis 2 weights-as-data**: VQ learned codebooks (VPTQ-style primary; QTIP trellis, SAGE spherical
VQ stretch); cross-expert shared codebooks (unpublished gap); entropy layer (Huffman/SIMD rANS,
+5–20%); ≥8-bit ≈ Shannon capacity → INT8/fp16 per-channel; neural-decoder codecs gated;
binary/ternary (1.0–1.6 bpw) — sign-domain fp16 on DML. QMoE anchors sub-1-bit feasibility.

## Storage
Local: zstd (streaming + sidecar index) + miniz (zip/zip64) for intermediates (caches 20–40%,
checkpoints 30–50%, corpora 60–75%; weight payloads only ~5–15%). Remote: Hugging Face Hub as
evictable object store — content-addressed manifests, ≤4 GiB zstd chunks, resumable, public repos
for student/caches/runs; LRU eviction with re-pull. Raw shards permanently available upstream.

## Experimentation & logging
`src/xlab/` JSONL events from every binary; SQLite run registry + decision log; auto league
tables/plots. Matrix E1–E10 in tiers A (tiny model, minutes) / B (2–3 real shards, bake-off
arena) / C (full pipeline, finalists): slice ratios, REAP keep-counts, slimming, codec bake-off
(INT4 / VQ / spherical / trellis / binary / ternary), cross-expert codebooks, neural codec,
recovery regimes, runtime tuning, GNN tracks (routing-graph GNN, weights-graph priors),
graph↔language cross-attention prosthesis (control arm: equal compute pure KD; kill criteria).

## Working point (final knobs chosen by gates)
Aim ≤35 GB student, ≤10% held-out ppl delta: ~288→96 experts, slice 4096→3328, d_ffn 1280,
experts ~2.5–3.0 bpw; shared expert + attention fp16/int8; router fp32. Est. 0.5–3 tok/s.
Profiles: quality / balanced / maxfit.

## Pipeline
Pass 1 calibration (layer-streamed, shards deleted after use): fp64 Gramians, REAP scores, routing
logs, teacher top-8 logits + block hiddens, per-expert-tile stats. Offline: PCA→P (+Cayley),
keep-sets, slimming, code design, bit allocation. Pass 2 transform (shard-by-shard): fold, prune,
ridge, encode, entropy-code → `.slm` → local + HF. 2 full download passes ≈ 656 GB total.

## Training / recovery — ZeRO-1/2/3 on iGPU+CPU+NVMe
Hand-written backward for all block types (gradient-checked vs central differences); ZeRO-1
optimizer states in disk arenas; ZeRO-2 transient per-layer grads; ZeRO-3 layer-gather→compute→
release with double-buffered prefetch; activation rematerialization. Trainable: LoRA + routers +
norms + hc scales + codebooks/binary scales (+ E10 adapters). Losses: logit-KL vs cached teacher,
block-hidden MSE (Pᵀ-projected), CE. fp16 + dynamic loss scaling + fp32 masters. Multi-day runs
with heartbeats + resume.

## Milestones
M0 toolchain → M1 spikes→D0 → M2 plumbing → M3 CPU reference → M4 DML engine → M5 Tier-A→D1 →
M6 Pass-1 → M7 Tier-B→D2 + Pass-2 → M8 runtime → M9 training→D3 → M10 GNN/cross-attn (non-blocking) →
M11 D4 + open release → M12 vision tower (phase 2).

## Risks (mitigations)
DML-on-this-iGPU unknown until S1 (hybrid CPU fallback first-class); bandwidth 656 GB (resumable,
ModelScope fallback, Tier-B slices); MLA/KDA slicing unpublished (square-Q gate, cliff curves);
REAP >50% untested (staged gates); VQ/binary encode on CPU (pooled codebooks, subsampled init);
binarization quality (explicit experiment); speculative tracks (controls + kill criteria);
HF limits (chunking, retries); no fused dequant (prefetch+LRU); fp16 numerics (fp32 router/norms,
loss scaling); 8 GB RAM (mmap, ≤2 GB working set).

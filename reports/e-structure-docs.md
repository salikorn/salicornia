# Round E — Structure (header split), Doxygen docs, D0 disk bench

Date: 2026-10-25 · Repo: `salikorn` @ `decc972` (initial commit, created this round)

This report consolidates the round's results from four work streams:
header separation, documentation, Git setup, and the D0 disk benchmark.

---

## 1. Header separation (header-only → split .hpp/.cpp/.inl)

### Files split

| Split | Destination | What moved |
|---|---|---|
| `src/gpu/kernels.hpp` | `src/gpu/kernels.inl` | The 17 `sal::gpu::tmpl` shader-template functions, moved verbatim. `TuneParams` + `LinAlgKernels`/`TrainKernels` stay in the `.hpp`. The `.inl` is included from **inside** `kernels.hpp` right after `TuneParams`, **not** at the very bottom, because the kernel-set structs' inline constructors call `tmpl::...` — a bottom include would not compile. |
| `src/gpu/cs.hpp` | `src/gpu/cs.cpp` | **Not** a `.inl`: per the ask's "prefer .cpp when no template is involved" — `cs.hpp` contains no templates, so `check`, `substitute_tokens` and all `Device`/`Buffer`/`Kernel` bodies became out-of-line definitions in `cs.cpp`. The `.hpp` keeps declarations, members, and the class/enum shapes byte-identical. |
| `src/compress/linalg.hpp` | `src/compress/linalg.cpp` | All non-template free functions moved: `nthreads`, `gemm`, `load_matrix_bf16`, `gram_bf16`, `gram_f32`, `orthonormalize`, `topk_eigen`, `jacobi_eigen`, `random_orthogonal`, `fro_sq`. `parallel_for` stayed in the `.hpp` because it is a template and is instantiated directly by `spike_decomp.cpp:79/169/298` and `spike_gpu_linalg.cpp:100`. The Eigen include moved to the `.cpp`; the `.hpp` forward-declares `Eigen::Matrix` exactly as Eigen's own `ForwardDeclarations.h` does, and `load_matrix_bf16`'s declared return type is spelled `Eigen::Matrix<double,-1,-1,1,-1,-1>` (`Dynamic=-1`, `RowMajor=1` are fixed public Eigen constants) with a comment; the `.cpp` keeps the original longhand spelling — same type. `xlab/fp16.hpp` include moved to the `.cpp` (only the `.cpp` uses `bf16_to_f32`). |
| `src/st/safetensors.hpp` | `src/st/safetensors.cpp` | `SafeTensorsFile` ctor/dtor — the Windows API + JSON parsing — moved verbatim. `<nlohmann/json.hpp>` moved to the `.cpp`; `<Windows.h>` stays in the `.hpp` because of the private `HANDLE map_` member. `TensorInfo::numel` stays inline in the `.hpp`. |
| `src/gpu/autotune.hpp` | `src/gpu/autotune.cpp` | `save_profile` + `load_profile` moved verbatim; `grids`, `spaces()`, `apply_dispatch` stay inline. **Deviation, deliberate:** `<nlohmann/json.hpp>` stays in the `.hpp` because `save_profile`'s public signature exposes `nlohmann::json` and that alias cannot be forward-declared robustly. `<fstream>`/`<filesystem>` moved to the `.cpp`. |
| `src/compress/codec_points.hpp` | `src/compress/codec_points.cpp` | `parse_codec_points` moved verbatim; declaration stays. |

### Kept as-is (deliberately)

- `src/xlab/test.hpp` — self-registering macro test framework; header-only by nature.
- `src/xlab/fp16.hpp` — tiny pure inline converters; moving them to a `.cpp` would
  insert a function call into hot conversion loops. **Deliberately left inline**, not
  merely left alone.
- `src/xlab/log.hpp` — tiny logger, header-only by nature.

### Build / test verification (run this round)

All under vcvars64 (the Ninja/MSVC build needs the vcvars environment):

1. `cmake -S . -B build` + `cmake --build build --clean-first` → exit 0, all targets
   compiled + linked including all six spikes and `autotune` (spikes were built, not
   executed — running them was not requested).
2. `ctest -L unit --output-on-failure` → **5/5 passed** (`test_fp16`,
   `test_safetensors`, `test_codecs`, `test_linalg`, `test_autotune`).
3. `ctest -L gpu --output-on-failure` → **1/1 passed** (`test_gpu`).

The 394 build warnings are the pre-existing `/W4` set (Eigen C4996/C5054 deprecations,
`spike_decomp` C4456 shadowing, `cs.cpp` C4701 on the verbatim `Buffer`-ctor switch
locals); no new warnings from the split code beyond what the same bodies already
produced when header-inline.

### CMake wiring

- New static library `sal_core` (`src/gpu/cs.cpp`, `src/gpu/autotune.cpp`,
  `src/compress/linalg.cpp`, `src/compress/codec_points.cpp`,
  `src/st/safetensors.cpp`) with `PRIVATE` include dir `src/extern/extern-eigen` and
  MSVC `/openmp` (linalg.cpp's Eigen products were previously inlined into `/openmp`
  consumers; `/openmp` keeps Eigen's OpenMP path active for identical behavior).
- `sal_core` is linked into `spike_decomp`, `spike_gpu_linalg`, `spike_nonlinear`,
  `spike_vqvae`, `test_linalg`, `test_gpu`, `autotune`, and the `sal_test` function
  (hence `test_fp16`/`test_safetensors`/`test_codecs`) plus `test_autotune`; static
  link symbol resolution pulls only needed objects.
- `test_autotune` additionally links `d3d12`/`dxgi`: `tests/test_autotune.cpp:44-52`
  calls `sal::gpu::substitute_tokens` and `sal::gpu::tmpl::gemm_nt()`, which drags in
  `cs.cpp.obj` with its D3D12 imports (the test never touches the GPU at runtime).
- **No consumer `.cpp` `#include` changes were required** — every split header still
  supplies its own declaration-time dependencies.

### Project conventions held

- No vector brace-init with `size_t` introduced.
- `Buffer::upload` still calls `dev_->begin()` (list reset) before any copy — bodies
  moved byte-for-byte.

---

## 2. Documentation (Doxygen)

**17 files documented** across three groups:

- GPU core: `src/gpu/cs.hpp`, `src/gpu/cs.cpp`, `src/gpu/kernels.hpp`,
  `src/gpu/kernels.inl`, `src/gpu/autotune.hpp`, `src/gpu/autotune.cpp`
- Compress / storage: `src/compress/linalg.hpp`, `src/compress/linalg.cpp`,
  `src/compress/codec_points.hpp`, `src/compress/codec_points.cpp`,
  `src/st/safetensors.hpp`, `src/st/safetensors.cpp`
- Spikes / tests / tools: `src/xlab/log.hpp`, `src/xlab/fp16.hpp`,
  `src/xlab/test.hpp`, `tests/test_autotune.cpp`, `tests/test_codecs.cpp`,
  `tests/test_fp16.cpp`, `tests/test_gpu.cpp`, `tests/test_linalg.cpp`,
  `tests/test_safetensors.cpp`, `tools/autotune.cpp`, `tools/bench_disk.py`,
  `tools/spikes/spike_d3d12.cpp`, `tools/spikes/spike_decomp.cpp`,
  `tools/spikes/spike_dml.cpp`, `tools/spikes/spike_gpu_linalg.cpp`,
  `tools/spikes/spike_nonlinear.cpp`, `tools/spikes/spike_vqvae.cpp`

**Infrastructure:** `Doxyfile` (repo root, `PROJECT_NAME=salikorn`,
`EXTRACT_ALL=YES`, `RECURSIVE=YES`, `INPUT=src,tools,tests`,
`OUTPUT_DIRECTORY=build/docs`, `EXCLUDE_PATTERNS` for `extern/` and `build/`,
`GENERATE_HTML=YES`, `GENERATE_LATEX=NO`, `QUIET=YES`, `WARN_AS_ERROR=NO`, warnings
log to `build/docs/doxygen_warnings.log`) and `scripts/gen_docs.bat` (runs doxygen
from the repo root and reports the warning count) are in place. Doxygen 1.18.0 was
installed via winget (`DimitriVanHeesch.Doxygen` — the lowercase-id form from the
original ask failed to match); a clean run produced 247 HTML files, 0 warnings after
fixing the 7 obsolete-config-tag warnings. `scripts/gen_docs.bat` smoke-tested clean.
Caveat carried over from an earlier pass: doxygen was reported available by the infra
agent but not spawnable from that process (ENOENT / not on PATH) — rerun
`scripts\gen_docs.bat` after a PATH refresh if needed.

**Notable documentation findings:**

- No `src/gpu/kernels.cpp` exists — kernels are header-only (`kernels.hpp` includes
  `kernels.inl` shader templates); the ask's "kernels.hpp/.inl" pair was documented
  as-is, nothing was missing.
- Doxygen comments that mention shader tokens use the `&#64;` HTML entity (renders
  as `@`) because Doxygen 1.18.0 parses a literal `@WORD` in comments as an unknown
  command (`&commat;` is unsupported in this build). Verified via
  `build/docs/doxygen_warnings.log`: "Found unknown command" warnings appeared, then
  disappeared after the `&#64;` fix.
- cpp-side `\copydoc` comments were recursive under this Doxyfile
  (`EXTRACT_ALL=YES` merges declaration + definition, so the copydoc resolved to
  itself; 82 warnings). Fixed by keeping full docs in the headers and removing the
  cpp `\copydoc` lines. Four `\param nt` entries did not match the unnamed
  `unsigned /*nt*/` parameters and were folded into prose. Final run:
  `doxygen.exe Doxyfile` exit 0, 0 warnings mentioning the documented files.
- Source inconsistency noted and documented faithfully: the syrk shader's in-code
  comment says "chunk over output rows" while `TuneParams::syrk_chunk` says
  "output-column chunk" — both are the same axis (rows of G = columns of X); the
  `kernels.inl` syrk doc states the equivalence instead of picking one.
- Interpretation notes documented as such: `random_orthogonal()` returns true
  unconditionally; `topk_eigen()` uses a hardcoded RNG seed 20261025 for
  reproducibility and ignores its `nt` parameter; `orthonormalize()` writes V only on
  a successful LLT (the jitter retry operates on the scratch Gram matrix C).
- In `spike_vqvae.cpp` the trainer gate tolerances (AE 1e-4 / VQ 1e-3
  Frobenius-relative) are taken from the `run_gate` code and its inline comment (a
  single fp32 near-tie can flip one of 512 nearest-neighbor indices) — documented
  what the code does, not a verified gate run.

**Deep-dive topics covered (where the code actually lives):**

- D3D12 resource-state discipline (COMMON-at-rest convention, per-op transitions,
  aliasing guard in `TrainKernels::elem`) documented on `Device`, `Kernel::dispatch`,
  the kernel sets, `spike_d3d12.cpp`, `spike_dml.cpp`, `spike_gpu_linalg.cpp`
  (TDR per-packet chunking), `spike_vqvae.cpp`, `test_gpu.cpp`, and
  `tools/autotune.cpp`.
- Tuner candidate-gating: `spaces()` producer grids incl. quick/full pruning
  (`autotune.hpp`); `load_profile`'s parse-into-temporary + hard-range
  commit-or-reject consumer gate (`autotune.hpp`, ranges `4<=gemm_ts<=32` due to the
  1024-thread group limit); the compile → correctness-gate → median-timing protocol
  in `tools/autotune.cpp` (file-level `\section`) and `test_autotune.cpp`.
- ElemGrid contract (`ceil(count/(G*EPT))` + per-thread contiguous EPT run under a
  BOUND guard = exact single-pass coverage, injective (thread,e)→element map)
  documented on `TuneParams::elem_grid` and the `SAL_ELEM_PRELUDE` comment in
  `kernels.inl` (`kernels.hpp:34-37` / `kernels.inl:131-137`), `test_autotune.cpp`,
  `test_gpu.cpp`, and `spike_vqvae.cpp` (including why `vq_nearest` is dispatched
  directly with grid `B` instead of via `elem()`).

**Verification of comment-only edits:**

- Per-file `git diff -U0` with a script stripping `//` comments and whitespace,
  confirming every removed line's code content reappears among added lines:
  `COMMENTS-ONLY: OK` for all 16 C++ files.
- `python -m py_compile tools/bench_disk.py`: passed. (`bench_disk.py` carried
  pre-existing uncommitted modifications from a prior pass, including the
  direct-I/O ctypes block; only this round's edits were comment additions.)
- Rebuild of `test_fp16`/`test_codecs`/`test_autotune`/`test_safetensors`/
  `test_linalg` via `cmake --build` (MSVC 14.44 / Ninja through vcvars64): no
  errors, only pre-existing C4127 warnings from vendored Eigen headers.
- All five CPU-only test binaries run from the repo root, exit 0: fp16 6 passed,
  codecs 3 passed, autotune 4 passed, safetensors 1 passed + 1 SKIP (the
  real-shard test skips by design because
  `models/GLM-5.3-Flash/model-00062-of-00062.safetensors` is absent), linalg 4 passed.
- **Not run:** GPU-dependent suites (`test_gpu`, `spike_*`) were not executed this
  pass — they require the iGPU adapter and were only compile-verified.
- Build-environment finding: building from a bare Git Bash shell fails with C1083
  (MSVC `INCLUDE` unset), and VS "18" preview's vcvars64 fails with
  `STL1001: Unexpected compiler version, expected MSVC Compiler 19.50` — builds
  succeed using **VS 2022's** `vcvars64.bat`. Environment issue, not a code issue.

---

## 3. Git setup

- `git init` done (the directory was not a repo). Identity resolves via global
  config (`user.name='Salikorn Labs'`,
  `user.email='338634960+salikorn@users.noreply.github.com'`), so per the "only if
  unset" rule **no local identity was set**.
- `.gitignore` reviewed (`:1-24`): `build/`, `runs/`, `cache/`, `shards/`,
  `models/`, `profiles/`, `reports/generated/` and most `extern/directml` binaries
  were already covered; extended with `extern/dml-variants/*.zip` and `*.dll`
  (~258 MB of fetched DirectML binaries covered by the existing "vendored binaries
  not committed" rule) and, this closing pass, `__pycache__/` (bytecode cache from
  `tools/bench_disk.py`).
- Initial commit `decc972730ac5e67e84a58936cdf0edbc375a66e` created with message
  `salikorn: spikes, GPU trainer, codec experiments, test suite, autotuner`;
  verified no file >2 MB and no zips/DLLs staged (353 extern files = Eigen/miniz/
  nlohmann/DirectML headers + licenses only).
- `Doxyfile` and `scripts/gen_docs.bat` were left untracked initially per
  instruction; they are committed by the closing commit below.

---

## 4. D0 disk benchmark

- Report: `reports/d0-disk-bench.md` (raw log: `reports/d0-disk-bench-run.log`)
- Sequential read: **2580.2 MB/s**
- Random read: **163.8 MB/s**
- Pass estimate: **2.12 minutes**

---

## 5. Closing commit

This report, the split sources, documentation comments, `Doxyfile`,
`scripts/gen_docs.bat`, and the D0 bench report are committed with message
`structure: split headers, doxygen docs, D0 disk bench` (see command log below for
the executed check).

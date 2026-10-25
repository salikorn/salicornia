# D0 spike S4: disk behavior benchmark (formal)

**Date:** 2026-10-25
**Target:** `models/GLM-5.3-Flash/model-00062-of-00062.safetensors` — 1,261,584,968 bytes (1.262 GB decimal), the actual 62nd shard of the GLM-5.3-Flash model.
**Script:** `tools/bench_disk.py` (stdlib only, read-only on the shard). Raw run log: `reports/d0-disk-bench-run.log`.
**Machine:** Windows 10.0.19044 x64, 12 logical CPUs, Python 3.14.3, **8.3 GB RAM** (3.1 GB available at benchmark time, 63% load — queried via `GlobalMemoryStatusEx`).

## Method

The benchmark ran three modes in a single invocation (`python tools/bench_disk.py`):

1. **Sequential large-read, buffered** (`bench_seq`): full passes over the whole 1.262 GB file with Python buffered `f.read()` at 8 MB, 32 MB, and 64 MB chunk sizes. The *first* two passes (8 MB, 64 MB) are labeled **cold-ish** (single pass, cache state unknown at start); the following three passes are labeled **warm** (file freshly read milliseconds earlier).
2. **4 KB random-read, buffered** (`bench_random`): 20,000 randomly ordered, 4 KB-aligned `seek()+read()` calls over a 1 GB window at offset 0–1e9 of the shard (the shard-at-a-time streaming pattern at 4 KB granularity), 200-read warmup excluded from timing, fixed seed 1234.
3. **Direct I/O (uncached) check** (`bench_seq_direct`, `bench_random_direct`): the same reads issued via Win32 `CreateFileW` with **`FILE_FLAG_NO_BUFFERING`** (ctypes, sector-aligned 4 KB offsets/sizes, page-aligned buffer from `VirtualAlloc`), which bypasses the Windows page cache entirely and gives a **true cold** device number. This mode exists because of a limitation found during the run, below.

**Known measurement limitation (important):** the shard is 1.26 GB and the machine has 8.3 GB RAM, so the file fits comfortably in the page cache. The buffered "cold-ish" passes showed **no difference from warm passes** (within noise) — i.e., the buffered numbers measure memory-copy bandwidth, not disk, whenever the cache is warm. Only the direct-I/O mode gives an honest uncached result, and it is the number used for the cold pass estimate. A single pass per configuration (0.4–0.6 s each) was timed; run-to-run noise across three invocations of the script was visible at roughly ±10% on the buffered numbers.

## Numbers

### (a) Sequential large-read throughput (whole 1.262 GB file)

| Pass | Chunk | Time | Throughput |
|---|---|---|---|
| cold-ish (buffered, pass 1) | 8 MB | 0.532 s | **2,370.6 MB/s** (2,260.8 MiB/s) |
| cold-ish (buffered, pass 1) | 64 MB | 0.494 s | **2,553.5 MB/s** (2,435.2 MiB/s) |
| warm (buffered, pass 3+) | 8 MB | 0.489 s | **2,580.2 MB/s** (2,460.7 MiB/s) |
| warm (buffered, pass 3+) | 32 MB | 0.516 s | **2,445.5 MB/s** (2,332.3 MiB/s) |
| warm (buffered, pass 3+) | 64 MB | 0.532 s | **2,372.5 MB/s** (2,262.6 MiB/s) |
| **true cold (direct I/O, no page cache)** | 8 MB | 0.428 s | **2,948.9 MB/s** (2,812.3 MiB/s) |

Interpretation: warm and cold-ish buffered numbers are statistically indistinguishable (2.37–2.58 GB/s band) — the cache-contamination limitation above applies. The direct-I/O pass is the honest cold number and is actually *faster* than the buffered path (~2.95 GB/s), which is consistent with an NVMe-class drive plus the buffered path's extra kernel copy. **Planning envelope: ~2.4–2.9 GB/s sequential regardless of cache state.**

### (b) 4 KB random-read throughput (1 GB window of the shard)

| Mode | Reads | Time | Throughput | IOPS |
|---|---|---|---|---|
| Buffered (app-level, Python `seek+read`) | 20,000 × 4 KB | 0.500 s | **163.8 MB/s** | **39,996 IOPS** |
| Direct I/O (true cold, uncached) | 5,000 × 4 KB | 0.561 s | 36.5 MB/s | 8,905 IOPS |

The buffered figure is what the application would see with normal file I/O (~25 µs per read, dominated by Python/OS call overhead on a cached file). The direct figure exposes raw per-I/O device latency when the cache cannot absorb it (~112 µs per 4 KB read). Both are **~15–80× slower than sequential** — the pattern penalty is the finding, not the absolute number.

### Full 328 GB model pass at measured sequential rates

328 GB taken as decimal (328 × 10⁹ bytes), matching PLAN.md's "≈ 656 GB total" for two passes:

| Rate used | MB/s | 328 GB pass |
|---|---|---|
| **Warm** (buffered 8 MB) | 2,580.2 | **2.12 min** |
| **Warm** (buffered 32/64 MB, range) | 2,372–2,446 | 2.24–2.31 min |
| **Cold** (buffered single pass — cache-contaminated, upper bound on cold time) | 2,371–2,553 | 2.14–2.31 min |
| **True cold** (direct I/O, uncached — honest cold figure) | 2,948.9 | **1.85 min** |

**Headline: one full 328 GB pass costs ≈ 2 minutes of disk time — ~2.1 min warm, ~1.9–2.3 min cold** (the direct-I/O cold number is faster than buffered, so the honest cold estimate sits in the same ~2-minute band; the exact cold duration cannot be more tightly bounded from a 1.26 GB test file against an 8 GB cache).

## D0 implication for the two-pass compression plan

1. **Disk is not the bottleneck for the pass strategy.** At ~2.4–2.9 GB/s sequential, the entire 328 GB Pass-1 (and Pass-2) stream costs ~2 minutes of I/O; the planned "2 full download passes ≈ 656 GB" (PLAN.md:71) is ~4 minutes of pure disk time. All other per-shard work (fp64 Gramians, REAP scoring, entropy coding — PLAN.md:68–71) is orders of magnitude more expensive. D0 should choose the pass strategy on CPU/memory/oracle grounds, not disk.
2. **Streaming layout is mandatory.** Sequential reads deliver ~2.4–2.9 GB/s; 4 KB random access delivers ~164 MB/s buffered (and 8.9k IOPS uncached device latency). Anything that faults tensors in random order rather than streaming shards front-to-back forfeits 15×+ of read throughput. This validates the shard-at-a-time, "shards deleted after use" discipline already in the plan (PLAN.md:68).
3. **The page cache cannot be relied on for cold-pass planning on this machine.** With 8.3 GB RAM, warm numbers are memory bandwidth, and only ~3 GB was available — a 328 GB working set always streams from device. The honest planning rate is the direct-I/O ~2.9 GB/s (with buffered real-world overhead, budget ~2.1–2.4 min per pass, i.e. round to 3 min with margin).
4. **Buffered I/O overhead is real but small.** The buffered path runs ~15% slower than direct I/O at 8 MB chunks (2.58 vs 2.95 GB/s). For M3 CPU-reference work, plain Python `read()` is fine; if Pass-1 ever becomes I/O-bound, switching the streaming loops to unbuffered/direct reads recovers the difference.

**Repro:** `python tools/bench_disk.py` from the repo root (three runs performed this session; all numbers above from the final run, logged verbatim in `reports/d0-disk-bench-run.log`).

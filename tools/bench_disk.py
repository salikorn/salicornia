#!/usr/bin/env python3
"""salikorn spike S4: disk behavior benchmark against a real model shard.

Target: models/GLM-5.3-Flash/model-00062-of-00062.safetensors (1,261,584,968 bytes).

(a) Sequential large-read throughput (8/32/64 MB chunks, whole file).
    Page-cache aware: pass 1 is a cold-ish estimate (single pass, whatever the
    cache state was before), pass 2+ are warm (file freshly read in pass 1).
(b) 4KB random-read throughput over a 1 GB window of the file
    (shard-at-a-time streaming pattern).

Pass estimate: minutes for one full 328 GB model pass at the measured
sequential rate (warm and cold), assuming 328 GB = 328e9 bytes decimal.

Stdlib only. Read-only on the shard.
"""
import os
import random
import sys
import time

TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "GLM-5.3-Flash", "model-00062-of-00062.safetensors",
)
GB_DECIMAL = 1e9
MB_DECIMAL = 1e6
MODEL_TOTAL_BYTES = 328 * GB_DECIMAL  # 328 GB, decimal, as stated in the plan

MIB = 1024 * 1024


def fmt_mibps(nbytes, dt):
    return nbytes / dt / MB_DECIMAL


def seq_pass(fh, size, chunk_bytes, label):
    """One sequential read pass over the whole file. Returns (seconds, bytes)."""
    fh.seek(0)
    total = 0
    t0 = time.perf_counter()
    while True:
        buf = fh.read(chunk_bytes)
        if not buf:
            break
        total += len(buf)
    dt = time.perf_counter() - t0
    mbs = fmt_mibps(total, dt)
    print(f"  seq {label:>9}: {total:>12,} bytes in {dt:8.3f}s -> {mbs:8.1f} MB/s "
          f"({mbs / 1.048576:8.1f} MiB/s)")
    return dt, total


def bench_seq():
    print(f"[a] sequential large-read throughput on {TARGET}")
    size = os.path.getsize(TARGET)
    print(f"    file size: {size:,} bytes ({size / GB_DECIMAL:.3f} GB decimal)")
    chunk_specs = [
        ("cold-ish 8MB", 8 * MIB),
        ("cold-ish 64MB", 64 * MIB),
        ("warm 8MB", 8 * MIB),
        ("warm 32MB", 32 * MIB),
        ("warm 64MB", 64 * MIB),
    ]
    results = {}
    fh = open(TARGET, "rb")
    try:
        for label, chunk in chunk_specs:
            dt, total = seq_pass(fh, size, chunk, label)
            results[label] = {"dt": dt, "bytes": total,
                              "MBps": fmt_mibps(total, dt)}
    finally:
        fh.close()
    return results


def bench_random(window_bytes=1 * GB_DECIMAL, n_reads=20000, seed=1234):
    print(f"[b] 4KB random-read throughput over a {window_bytes / GB_DECIMAL:.0f} GB window")
    size = os.path.getsize(TARGET)
    assert size >= window_bytes, f"file too small for a {window_bytes}-byte window"
    rng = random.Random(seed)
    max_off = int(window_bytes) - 4096
    offsets = [rng.randrange(0, max_off, 4096) for _ in range(n_reads)]
    fh = open(TARGET, "rb")
    total = 0
    try:
        # small warmup so timer excludes interpreter/first-touch effects
        for off in offsets[:200]:
            fh.seek(off)
            fh.read(4096)
        fh.seek(0)
        t0 = time.perf_counter()
        for off in offsets:
            fh.seek(off)
            buf = fh.read(4096)
            total += len(buf)
        dt = time.perf_counter() - t0
    finally:
        fh.close()
    mbs = fmt_mibps(total, dt)
    iops = n_reads / dt
    print(f"    {n_reads:,} x 4KB reads, {total:,} bytes in {dt:.3f}s -> "
          f"{mbs:.2f} MB/s ({iops:,.0f} IOPS)")
    return {"MBps": mbs, "iops": iops, "dt": dt, "bytes": total, "n_reads": n_reads}


def main():
    print(f"python {sys.version.split()[0]}, {os.cpu_count()} logical CPUs")
    print()
    seq = bench_seq()
    print()
    rnd = bench_random()
    print()
    print("[pass estimate] one full 328 GB model pass at measured seq rates")
    for label in ("cold-ish 8MB", "cold-ish 64MB", "warm 8MB", "warm 32MB", "warm 64MB"):
        rate = seq[label]["MBps"]  # decimal MB/s
        minutes = MODEL_TOTAL_BYTES / (rate * MB_DECIMAL) / 60
        print(f"  {label:>13}: {rate:8.1f} MB/s -> {minutes:7.1f} min")
    print()
    print("JSON:")
    import json
    print(json.dumps({"seq": seq, "random4k": rnd,
                      "pass328GB_min": {k: MODEL_TOTAL_BYTES / (v["MBps"] * MB_DECIMAL) / 60
                                        for k, v in seq.items()}}, indent=2))


if __name__ == "__main__":
    main()

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

## @file bench_disk.py
#  @brief Disk-behavior benchmark spike (S4) against a real model shard.
#
#  Measures (a) sequential large-read throughput over the whole shard at 8/32/64 MB
#  chunk sizes — page-cache aware, with a separate FILE_FLAG_NO_BUFFERING direct-I/O
#  pass for a true cold number — and (b) 4KB random-read throughput over a 1 GB
#  window (the shard-at-a-time streaming pattern the trainer will use). Ends with a
#  full-model pass-time estimate at each measured rate and a JSON dump of all results.
#
#  Stdlib only (ctypes for the direct-I/O Win32 calls). Read-only on the shard.

TARGET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "GLM-5.3-Flash", "model-00062-of-00062.safetensors",
)
GB_DECIMAL = 1e9
MB_DECIMAL = 1e6
MODEL_TOTAL_BYTES = 328 * GB_DECIMAL  # 328 GB, decimal, as stated in the plan

MIB = 1024 * 1024


## @brief Formats a throughput as decimal MB/s (1 MB = 1e6 bytes, matching the
#         pass-time estimate convention).
#  @param nbytes Bytes transferred.
#  @param dt Elapsed seconds.
#  @return Throughput in decimal MB/s.
def fmt_mibps(nbytes, dt):
    return nbytes / dt / MB_DECIMAL


## @brief One sequential read pass over the whole file, timed and reported.
#  @param fh Open binary file handle (position is reset to 0 first).
#  @param size File size in bytes (used only for reporting bookkeeping).
#  @param chunk_bytes Read size per fh.read() call.
#  @param label Row label printed with the result (e.g. "warm 8MB").
#  @return Tuple (elapsed seconds, total bytes read).
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


## @brief Runs the sequential buffered benchmark: cold-ish 8MB/64MB passes followed
#         by warm 8MB/32MB/64MB passes (later passes are warm because pass 1 just
#         pulled the file into the page cache).
#  @return Dict label -> {"dt", "bytes", "MBps"} per pass.
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


## @brief 4KB random-read benchmark over a leading window of the file (buffered I/O,
#         page-cache warm after the 200-read warmup). Offsets are 4KB-aligned, seeded.
#  @param window_bytes Window size (default 1 GB decimal).
#  @param n_reads Number of 4KB reads (default 20000).
#  @param seed RNG seed for the offset sequence (deterministic re-runs).
#  @return Dict with MBps, iops, dt, bytes, n_reads.
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


# ---------------------------------------------------------------- direct I/O
# FILE_FLAG_NO_BUFFERING opens bypass the Windows page cache, giving a TRUE
# cold-read number. Requires sector-aligned offsets/sizes and a page-aligned
# buffer. Used to validate that the buffered "cold-ish" pass is honest.
import ctypes

GENERIC_READ = 0x80000000  # noqa: E305 (constants block)
FILE_SHARE_READ = 1
FILE_SHARE_WRITE = 2
OPEN_EXISTING = 3
FILE_FLAG_NO_BUFFERING = 0x20000000
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
FILE_FLAG_RANDOM_ACCESS = 0x10000000
FILE_BEGIN = 0
PAGE_READWRITE = 0x04
MEM_COMMIT = 0x1000
MEM_RELEASE = 0x8000

class _LARGE_INT(ctypes.Union):
    """LARGE_INTEGER as 64-bit; passed by value as c_longlong, by ref as pointer."""
    _fields_ = [("as_ul", ctypes.c_ulonglong), ("as_l", ctypes.c_longlong)]


_k32 = ctypes.windll.kernel32
# Explicit prototypes: without these ctypes truncates returned 64-bit
# pointers/handles to c_int, which caused ERROR_NOACCESS (998).
_k32.CreateFileW.restype = ctypes.c_void_p
_k32.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_ulong, ctypes.c_ulong,
                             ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                             ctypes.c_void_p]
_k32.ReadFile.restype = ctypes.c_int
_k32.ReadFile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong,
                          ctypes.POINTER(ctypes.c_ulong), ctypes.c_void_p]
_k32.SetFilePointerEx.restype = ctypes.c_int
_k32.SetFilePointerEx.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                  ctypes.POINTER(_LARGE_INT), ctypes.c_ulong]
# (LARGE_INTEGER passed by value is declared as c_longlong)
_k32.VirtualAlloc.restype = ctypes.c_void_p
_k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong, ctypes.c_ulong]
_k32.VirtualFree.restype = ctypes.c_int
_k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
_k32.CloseHandle.restype = ctypes.c_int
_k32.CloseHandle.argtypes = [ctypes.c_void_p]


## @brief Opens TARGET with FILE_FLAG_NO_BUFFERING plus the caller's extra hint flags.
#  @param flags Additional FILE_FLAG_* bits (SEQUENTIAL_SCAN or RANDOM_ACCESS).
#  @return Raw Win32 HANDLE (c_void_p) for the other _direct_* helpers.
#  @exception OSError With the Win32 last error if CreateFileW fails.
def _direct_open(flags):
    h = _k32.CreateFileW(TARGET, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
                         None, OPEN_EXISTING, FILE_FLAG_NO_BUFFERING | flags, None)
    if not h or h == ctypes.c_void_p(-1).value:
        raise OSError(f"CreateFileW failed: {ctypes.GetLastError()}")
    return h


## @brief One direct-I/O ReadFile into a page-aligned buffer.
#  @param h Handle from _direct_open.
#  @param buf Aligned buffer from _aligned_buf.
#  @param nbytes Requested byte count (must respect NO_BUFFERING sector alignment).
#  @return Bytes actually read (0 at EOF).
#  @exception OSError With the Win32 last error if ReadFile fails.
def _direct_read(h, buf, nbytes):
    got = ctypes.c_ulong(0)
    ok = _k32.ReadFile(h, buf, nbytes, ctypes.byref(got), None)
    if not ok:
        raise OSError(f"ReadFile failed: {ctypes.GetLastError()}")
    return got.value


## @brief Seeks a direct-I/O handle to an absolute byte offset.
#  @param h Handle from _direct_open.
#  @param off Absolute offset (FILE_BEGIN).
#  @return New position as reported by SetFilePointerEx.
#  @exception OSError With the Win32 last error if the seek fails.
def _direct_seek(h, off):
    new = _LARGE_INT()
    if not _k32.SetFilePointerEx(h, off, ctypes.byref(new), FILE_BEGIN):
        raise OSError(f"SetFilePointerEx failed: {ctypes.GetLastError()}")
    return new.as_ul


## @brief Allocates a page-aligned buffer via VirtualAlloc — required for
#         FILE_FLAG_NO_BUFFERING reads, which ctypes/BytesIO buffers cannot satisfy.
#  @param size Buffer size in bytes.
#  @return c_void_p holding the aligned allocation.
#  @exception OSError If VirtualAlloc returns NULL.
def _aligned_buf(size):
    p = _k32.VirtualAlloc(None, size, MEM_COMMIT, PAGE_READWRITE)
    if not p:
        raise OSError("VirtualAlloc failed")
    return ctypes.c_void_p(p)


## @brief Sequential TRUE-cold pass: direct I/O (no page cache) in page-aligned
#         chunks over the 4KB-aligned file span, plus a buffered read of the tiny
#         unaligned tail so full-file coverage is preserved in the timing.
#  @param chunk_bytes Chunk size (default 8 MiB).
#  @return Dict with MBps (decimal), dt (seconds), bytes.
def bench_seq_direct(chunk_bytes=8 * MIB):
    """Sequential direct-I/O pass (uncached), over the 4KB-aligned file span."""
    print(f"[a-direct] TRUE cold: sequential direct I/O (FILE_FLAG_NO_BUFFERING), {chunk_bytes // MIB} MB chunks")
    size = os.path.getsize(TARGET)
    align = 4096
    span = (size // align) * align  # aligned span; tiny unaligned tail via buffered read
    h = _direct_open(FILE_FLAG_SEQUENTIAL_SCAN)
    buf = _aligned_buf(chunk_bytes)
    total = 0
    t0 = time.perf_counter()
    try:
        off = 0
        while off < span:
            want = min(chunk_bytes, span - off)
            _direct_seek(h, off)
            got = _direct_read(h, buf, want)
            if got != want:
                break
            total += got
            off += got
    finally:
        _k32.CloseHandle(h)
        _k32.VirtualFree(buf, 0, MEM_RELEASE)
    # unaligned tail via buffered read (negligible, keeps full-file coverage)
    fh = open(TARGET, "rb")
    fh.seek(span)
    tail = fh.read()
    fh.close()
    t_total = time.perf_counter() - t0
    total += len(tail)
    mbs = fmt_mibps(total, t_total)
    print(f"  direct cold 8MB: {total:,} bytes in {t_total:.3f}s -> {mbs:8.1f} MB/s ({mbs / 1.048576:8.1f} MiB/s)")
    return {"MBps": mbs, "dt": t_total, "bytes": total}


## @brief 4KB random TRUE-cold reads: direct I/O over a 1 GB window, giving the
#         uncached device latency/throughput number the buffered bench_random
#         cannot show.
#  @param window_bytes Window size (default 1 GB decimal).
#  @param n_reads Number of 4KB reads (default 5000).
#  @param seed RNG seed for the 4KB-aligned offset sequence.
#  @return Dict with MBps (decimal), iops, dt, bytes.
def bench_random_direct(window_bytes=GB_DECIMAL, n_reads=5000, seed=1234):
    """4KB random direct-I/O reads over a 1 GB window (uncached device latency)."""
    print(f"[b-direct] TRUE cold: 4KB random direct I/O over {window_bytes / GB_DECIMAL:.0f} GB window")
    rng = random.Random(seed)
    max_off = int(window_bytes) - 4096
    offsets = [rng.randrange(0, max_off, 4096) for _ in range(n_reads)]
    h = _direct_open(FILE_FLAG_RANDOM_ACCESS)
    buf = _aligned_buf(4096)
    total = 0
    t0 = time.perf_counter()
    try:
        for off in offsets:
            _direct_seek(h, off)
            got = _direct_read(h, buf, 4096)
            total += got
    finally:
        _k32.CloseHandle(h)
        _k32.VirtualFree(buf, 0, MEM_RELEASE)
    dt = time.perf_counter() - t0
    mbs = fmt_mibps(total, dt)
    print(f"    {n_reads:,} x 4KB direct reads, {total:,} bytes in {dt:.3f}s -> "
          f"{mbs:.2f} MB/s ({n_reads / dt:,.0f} IOPS)")
    return {"MBps": mbs, "iops": n_reads / dt, "dt": dt, "bytes": total}


## @brief Runs all benchmark phases and prints the summary: buffered sequential +
##        random passes, direct-I/O passes (skipped with a note if unsupported),
##        a full-model (328 GB) pass-time estimate per measured sequential rate,
##        and a final JSON dump of every result.
#  @return None (all output goes to stdout).
def main():
    print(f"python {sys.version.split()[0]}, {os.cpu_count()} logical CPUs")
    print()
    seq = bench_seq()
    print()
    rnd = bench_random()
    print()
    try:
        seqd = bench_seq_direct()
        rndd = bench_random_direct()
    except OSError as e:
        print(f"  (direct I/O unavailable: {e})")
        seqd, rndd = None, None
    print()
    print("[pass estimate] one full 328 GB model pass at measured seq rates")
    est = {}
    for label in ("cold-ish 8MB", "cold-ish 64MB", "warm 8MB", "warm 32MB", "warm 64MB"):
        rate = seq[label]["MBps"]  # decimal MB/s
        minutes = MODEL_TOTAL_BYTES / (rate * MB_DECIMAL) / 60
        est[label] = minutes
        print(f"  {label:>13}: {rate:8.1f} MB/s -> {minutes:7.1f} min")
    if seqd:
        minutes = MODEL_TOTAL_BYTES / (seqd["MBps"] * MB_DECIMAL) / 60
        est["true-cold direct 8MB"] = minutes
        print(f"  {'true-cold direct':>13}: {seqd['MBps']:8.1f} MB/s -> {minutes:7.1f} min")
    print()
    import json
    print("JSON:")
    print(json.dumps({"seq": seq, "random4k": rnd,
                      "seq_direct": seqd, "random4k_direct": rndd,
                      "pass328GB_min": est}, indent=2))


if __name__ == "__main__":
    main()

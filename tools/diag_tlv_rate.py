"""Standalone TLV-rate diagnostic.

Opens COM10 (the radar data port), reads raw bytes for N seconds, and
reports EXACTLY what's happening on the wire — chunk sizes, read
intervals, TLV packets per chunk, per-packet sizes. Then computes:

  - Actual chip TLV emission rate (packets/sec)
  - Implied "latest-pkt-wins" publish rate (= chunks/sec)
  - WHETHER the hypothesis "publish rate = chunks/sec because of
    latest-wins" is CONFIRMED or REFUTED.

This is a passive observer. It does NOT send anything to the chip. It
just opens COM10, reads bytes for the duration, and prints stats.

PRE-REQ: stop the seeker app first (Ctrl+C) so COM10 is free. The chip
will keep emitting because we no longer send sensorStop on shutdown.

Usage:
    python tools/diag_tlv_rate.py
    python tools/diag_tlv_rate.py --duration 10 --bufsize 4096
    python tools/diag_tlv_rate.py --bufsize 1   # one byte at a time — bypasses batching

The --bufsize 1 mode is the smoking-gun test: if the same chip emits
~20 packets/sec when we read 1 byte at a time, the host buffer was the
bottleneck. If it still gives 4.9 packets/sec, the chip itself is slow.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import serial

# Make repo importable
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from radar.tlv_parser import TLVStream  # noqa: E402


def _hist(values, bins):
    """Return [(lo, hi, count), ...] given sorted bins boundaries."""
    out = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n = sum(1 for v in values if lo <= v < hi)
        out.append((lo, hi, n))
    # last open bin
    out.append((bins[-1], None, sum(1 for v in values if v >= bins[-1])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM10")
    ap.add_argument("--baud", type=int, default=3_125_000)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--bufsize", type=int, default=4096,
                    help="Bytes to request per read() call. Match the live "
                         "code (4096) to reproduce its behaviour, or use 1 "
                         "to bypass batching.")
    ap.add_argument("--timeout", type=float, default=0.5)
    args = ap.parse_args()

    print(f"=== TLV rate diagnostic ===")
    print(f"  port:     {args.port} @ {args.baud}")
    print(f"  duration: {args.duration} s")
    print(f"  bufsize:  {args.bufsize} bytes per read()")
    print(f"  timeout:  {args.timeout} s")
    print()

    chunk_sizes = []
    read_intervals_ms = []
    packets_per_chunk = []
    packet_sizes = []
    total_bytes = 0
    n_reads = 0
    n_packets = 0

    try:
        ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
    except serial.SerialException as e:
        print(f"FATAL: cannot open {args.port}: {e}", file=sys.stderr)
        print(f"\n   → Is the seeker app still running? Stop it first (Ctrl+C).", file=sys.stderr)
        return 1

    stream = TLVStream()
    t_start = time.monotonic()
    t_deadline = t_start + args.duration
    last_read_t = t_start

    print(f"Capturing for {args.duration:.1f}s...", flush=True)
    try:
        while time.monotonic() < t_deadline:
            chunk = ser.read(args.bufsize)
            now = time.monotonic()
            if chunk:
                interval_ms = (now - last_read_t) * 1000.0
                read_intervals_ms.append(interval_ms)
                chunk_sizes.append(len(chunk))
                total_bytes += len(chunk)
                n_reads += 1

                # Try to parse TLV packets out of this chunk
                pkts = stream.feed(chunk)
                packets_per_chunk.append(len(pkts))
                for p in pkts:
                    n_packets += 1
                    # Crude per-packet size estimate: total payload of TLVs.
                    # We don't track exact wire bytes per packet, so estimate
                    # by even-distribution: total_bytes / n_packets at end.
                last_read_t = now
            else:
                # timeout fired — record empty interval but no chunk size
                pass
    finally:
        try:
            ser.close()
        except Exception:
            pass

    elapsed = time.monotonic() - t_start

    if not chunk_sizes:
        print(f"\nFAIL: no bytes received in {elapsed:.1f}s.")
        print(f"  → Chip is not streaming. Is the cfg pushed? Did sensorStart fire?")
        return 2

    bps = total_bytes / elapsed
    bytes_per_packet_avg = total_bytes / n_packets if n_packets else 0
    chip_emit_hz = n_packets / elapsed
    chunks_per_sec = n_reads / elapsed

    print(f"\n--- RAW WIRE STATS ---")
    print(f"  Elapsed:                    {elapsed:.2f} s")
    print(f"  Total bytes received:       {total_bytes:,}  ({bps/1024:.1f} KB/s)")
    print(f"  read() calls returned data: {n_reads}  ({chunks_per_sec:.1f}/s)")
    print(f"  TLV packets parsed:         {n_packets}  ({chip_emit_hz:.1f}/s)  ← chip emit rate")
    print(f"  Avg bytes per packet:       {bytes_per_packet_avg:.0f}")

    print(f"\n--- CHUNK SIZE (bytes per read() return) ---")
    cs = sorted(chunk_sizes)
    print(f"  min={cs[0]}   median={statistics.median(cs):.0f}   "
          f"mean={statistics.mean(cs):.0f}   max={cs[-1]}")
    print(f"  Histogram:")
    for lo, hi, n in _hist(chunk_sizes, [0, 256, 1024, 2048, 3072, 4096]):
        bar = "█" * min(40, n)
        hi_str = f"-{hi}" if hi is not None else "+"
        print(f"    {lo:>5}{hi_str:>6} bytes : {n:>4}  {bar}")

    print(f"\n--- TIME BETWEEN read() RETURNS (ms) ---")
    ri = sorted(read_intervals_ms)
    print(f"  min={ri[0]:.1f}   median={statistics.median(ri):.1f}   "
          f"mean={statistics.mean(ri):.1f}   max={ri[-1]:.1f}")
    print(f"  Histogram:")
    for lo, hi, n in _hist(read_intervals_ms, [0, 10, 50, 100, 200, 500]):
        bar = "█" * min(40, n)
        hi_str = f"-{hi}" if hi is not None else "+"
        print(f"    {lo:>5}{hi_str:>6} ms    : {n:>4}  {bar}")

    print(f"\n--- TLV PACKETS PER CHUNK ---")
    ppc_counter = Counter(packets_per_chunk)
    for k in sorted(ppc_counter.keys()):
        n = ppc_counter[k]
        bar = "█" * min(40, n)
        print(f"    {k} packet(s) : {n:>4} chunks  {bar}")

    # The crucial calculation
    chunks_with_multi = sum(1 for n in packets_per_chunk if n >= 1)  # all chunks with at least 1 packet
    multi_packet_chunks = sum(1 for n in packets_per_chunk if n > 1)
    avg_packets_per_chunk = n_packets / max(chunks_with_multi, 1)

    print(f"\n--- HYPOTHESIS CHECK: 'latest-pkt-wins' publish rate ---")
    print(f"  Chip TLV emit rate:                    {chip_emit_hz:.1f} packets/sec")
    print(f"  Chunks per sec (= live publish rate):  {chunks_per_sec:.1f} chunks/sec")
    print(f"  Avg packets per chunk:                 {avg_packets_per_chunk:.2f}")
    print(f"  Chunks containing >1 packet:           {multi_packet_chunks}/{chunks_with_multi} "
          f"({100*multi_packet_chunks/max(chunks_with_multi,1):.0f}%)")
    print()
    print(f"  Live app's RadarManager publishes 1 frame per chunk (latest pkt only).")
    print(f"  → Predicted publish rate from this run = {chunks_per_sec:.1f} Hz")
    print()

    if abs(chunks_per_sec - 4.9) < 1.5 and chip_emit_hz > 15:
        print(f"  *** HYPOTHESIS CONFIRMED ***")
        print(f"     Chip emits at {chip_emit_hz:.1f} Hz but read(4096) batches ~"
              f"{avg_packets_per_chunk:.1f} packets per chunk.")
        print(f"     Latest-wins drops {avg_packets_per_chunk-1:.1f} of every {avg_packets_per_chunk:.1f} packets.")
        print(f"     Fix: replace single-slot _latest_pkt with a queue.")
    elif chip_emit_hz < 10:
        print(f"  *** HYPOTHESIS REFUTED ***")
        print(f"     Chip itself is emitting at only {chip_emit_hz:.1f} Hz. The host buffer")
        print(f"     is NOT the bottleneck. Investigate chip-side / cfg.")
    else:
        print(f"  *** INCONCLUSIVE ***")
        print(f"     Chip emit rate {chip_emit_hz:.1f} Hz, chunk rate {chunks_per_sec:.1f} Hz.")
        print(f"     Doesn't cleanly match either hypothesis.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

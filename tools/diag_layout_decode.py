"""Empirical layout test for the 4-RX recording (channelCfg 15 15).

The cfg asks the chip for 4 RX. The radar_dca host parsers disagree on
byte layout: bin_parser.py assumes complex-int16 RX-interleaved, while
validate_two_recordings.py assumes real-int16 non-interleaved. We need
to know which one actually maps the wire bytes onto 4 distinct RX
channels.

Strategy: pick one frame from the 4-RX recording, try every plausible
reshape (real vs complex; (chirps,rx,samples), (chirps,samples,rx),
(samples,rx,2)), and report mean |value| per "RX" channel under each.
The correct layout puts roughly equal energy on all 4 RX. A wrong
layout produces lopsided energy or some zero channels.

Recording: recordings/seeker_2026-05-08_11-03-48_radar.bin (channelCfg 15 15).
"""
from __future__ import annotations
import numpy as np
import sys

REC = r"C:/Users/asaf.ruf.BLUERIVERTECH/Desktop/Seeker01/seeker_bench/recordings/seeker_2026-05-08_11-03-48_radar.bin"

N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192

# Two BPF candidates
BPF_REAL    = N_CHIRPS * N_RX * N_SAMPLES * 2  # 1,179,648  (int16 real)
BPF_COMPLEX = N_CHIRPS * N_RX * N_SAMPLES * 4  # 2,359,296  (complex int16, I+Q)

import os
size = os.path.getsize(REC)
print(f"file size      : {size:,} bytes")
print(f"BPF real       : {BPF_REAL:,}  -> {size/BPF_REAL:.3f} frames")
print(f"BPF complex    : {BPF_COMPLEX:,}  -> {size/BPF_COMPLEX:.3f} frames")
print()

# Pick frame 5 to skip any startup transient
FRAME_IDX = 5

def report(name, per_rx_abs):
    """per_rx_abs is a 1D array of length 4 — mean |value| per RX."""
    s = "  ".join(f"RX{i}={v:9.1f}" for i, v in enumerate(per_rx_abs))
    flag = "ALL FOUR ACTIVE" if all(v > per_rx_abs.max() * 0.1 for v in per_rx_abs) else \
           ("ONLY 2 ACTIVE" if sum(v > per_rx_abs.max() * 0.1 for v in per_rx_abs) == 2 else "MIXED")
    print(f"  [{flag}]  {name}")
    print(f"       {s}")

# Read the appropriate BPF chunk
print(f"--- Decoding frame {FRAME_IDX} of 4-RX recording ---\n")

# REAL interpretation
if size >= (FRAME_IDX + 1) * BPF_REAL:
    with open(REC, "rb") as f:
        f.seek(FRAME_IDX * BPF_REAL)
        buf = f.read(BPF_REAL)
    raw = np.frombuffer(buf, dtype=np.int16)
    print("=== REAL int16 interpretation ===")

    # Layout A: (chirps, rx, samples) — non-interleaved per-chirp, all RX0 first then RX1...
    a = raw.reshape(N_CHIRPS, N_RX, N_SAMPLES).astype(np.float32)
    per_rx = np.abs(a).mean(axis=(0, 2))
    report("(chirps, rx, samples)  validate_two_recordings.py layout", per_rx)

    # Layout B: (chirps, samples, rx) — RX-interleaved, sample-major
    b = raw.reshape(N_CHIRPS, N_SAMPLES, N_RX).astype(np.float32)
    per_rx = np.abs(b).mean(axis=(0, 1))
    report("(chirps, samples, rx)  RX-interleaved sample-major", per_rx)

    # Layout C: (chirps, rx*samples) viewed as (rx, chirps*samples) — full RX-major across frame
    c = raw.reshape(N_RX, N_CHIRPS * N_SAMPLES).astype(np.float32)
    per_rx = np.abs(c).mean(axis=1)
    report("(rx, chirps*samples)   full RX-major across frame", per_rx)

    # Layout D: (chirps*samples, rx) — full RX-interleaved across frame
    d = raw.reshape(N_CHIRPS * N_SAMPLES, N_RX).astype(np.float32)
    per_rx = np.abs(d).mean(axis=0)
    report("(chirps*samples, rx)   full RX-interleaved across frame", per_rx)

    # Layout E: 2-LVDS-lane time-mux: lane0 = [RX0, RX2 interleaved] across samples,
    #           lane1 = [RX1, RX3 interleaved]; DCA packs as alternating 16-bit words.
    # Wire stream order: lane0_word0, lane1_word0, lane0_word1, lane1_word1, ...
    # If lane0 carries RX0 then RX2 alternating per sample tick, and similarly lane1
    # carries RX1 then RX3, then the byte stream is:
    #   RX0_s0, RX1_s0, RX2_s0, RX3_s0, RX0_s1, RX1_s1, RX2_s1, RX3_s1, ...
    # which is identical to layout B above. So no separate test needed.

print()

# COMPLEX interpretation
if size >= (FRAME_IDX + 1) * BPF_COMPLEX:
    with open(REC, "rb") as f:
        f.seek(FRAME_IDX * BPF_COMPLEX)
        buf = f.read(BPF_COMPLEX)
    raw = np.frombuffer(buf, dtype=np.int16)
    print("=== COMPLEX int16 (I,Q pair) interpretation ===")

    # Layout A: (chirps, samples, rx, 2) — bin_parser.py layout
    a = raw.reshape(N_CHIRPS, N_SAMPLES, N_RX, 2).astype(np.float32)
    per_rx = np.abs(a[..., 0] + 1j * a[..., 1]).mean(axis=(0, 1))
    report("(chirps, samples, rx, IQ)  bin_parser.py layout", per_rx)

    # Layout B: (chirps, rx, samples, 2)
    b = raw.reshape(N_CHIRPS, N_RX, N_SAMPLES, 2).astype(np.float32)
    per_rx = np.abs(b[..., 0] + 1j * b[..., 1]).mean(axis=(0, 2))
    report("(chirps, rx, samples, IQ)  RX-major", per_rx)
else:
    print("=== COMPLEX layout: file too small for one complex frame; skipping ===")

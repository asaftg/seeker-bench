"""Verify the bin_parser.py fix on the airborne1 recording.

Loads frame 5 via the live bin_parser and checks per-RX mean |value|.
Before the fix this would have shown RX1 = RX3 = 0; after the fix all
four RX should have comparable energy.
"""
from __future__ import annotations
import os
import sys
import numpy as np

# Ensure we import the repo's bin_parser, not anything else.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from radar_dca.bin_parser import parse_bin_streaming, CaptureDims  # noqa: E402

# Reconstruct CaptureDims by hand from the cfg (no .mmwave.json for these
# Seeker recordings — we use the .cfg directly, but the dims are constant).
N_CHIRPS = 768
N_RX = 4
N_TX = 4
N_SAMPLES = 192
PRF_HZ = 30_478.51264858275
RANGE_RES_M = 2.638
FRAME_PERIOD_S = 50e-3

dims = CaptureDims(
    n_rx=N_RX,
    n_tx=N_TX,
    n_samples=N_SAMPLES,
    n_chirps_per_frame=N_CHIRPS,
    chirp_period_s=1.0 / PRF_HZ,
    range_resolution_m=RANGE_RES_M,
    framePeriodicity_s=FRAME_PERIOD_S,
)

print(f"bytes_per_chirp = {dims.bytes_per_chirp}")
print(f"bytes_per_frame = {dims.bytes_per_frame}")

# 1. New 4-RX recording (today)
REC_NEW = r"C:/Users/asaf.ruf.BLUERIVERTECH/Desktop/Seeker01/seeker_bench/recordings/seeker_2026-05-08_11-03-48_radar.bin"
# 2. The infamous airborne1 recording
REC_AIRBORNE = r"C:/Users/asaf.ruf.BLUERIVERTECH/Desktop/Seeker01/seeker_bench/recordings/seeker_2026-05-06_12-58-59_radar.bin"

for label, path in (("4-RX recording (today)", REC_NEW),
                    ("airborne1 (the 24-hour heartbreak)", REC_AIRBORNE)):
    if not os.path.exists(path):
        print(f"\n[{label}] not found at {path}")
        continue
    size = os.path.getsize(path)
    nf = size // dims.bytes_per_frame
    print(f"\n[{label}] {size:,} bytes  /  {dims.bytes_per_frame:,} bpf  =  {nf} frames")

    # Pull frame 5 via the streaming API (which uses the patched parser).
    target = 5 if nf > 5 else 0
    cube = None
    for idx, c in parse_bin_streaming(path, dims):
        if idx == target:
            cube = c
            break
    if cube is None:
        print(f"  could not read frame {target}")
        continue
    # cube shape: (n_chirps, n_samples, n_rx) complex64
    print(f"  frame {target} cube shape = {cube.shape}, dtype = {cube.dtype}")
    per_rx = np.abs(cube).mean(axis=(0, 1))
    s = "   ".join(f"RX{i}={v:9.1f}" for i, v in enumerate(per_rx))
    print(f"  per-RX mean |value|: {s}")
    if all(v > per_rx.max() * 0.1 for v in per_rx):
        print("  [PASS] all four RX channels carry signal")
    else:
        zeros = [i for i, v in enumerate(per_rx) if v < per_rx.max() * 0.1]
        print(f"  [FAIL] RX channels {zeros} are dead — bin_parser still wrong")

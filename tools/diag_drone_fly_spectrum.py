"""Dump the slot-0 slow-time spectrum at the known drone bins of drone_fly,
so we can see what peaks actually exist before redesigning the detector.

Per herm_detector_v2 docstring, peaks should appear at 1374 Hz and 1166 Hz
at range bins 1-2 of drone_fly (5 m hover). If those peaks are present
with high SNR, the data has the signal and we just need the right
detector rule.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
from scipy import fft as scipy_fft

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from tools.diagnose_pmm._common_fixed import (
    iter_real_frames, resolve_recording, stage1_range_fft,
)

rec = resolve_recording("drone_fly")
print(f"recording: {rec.bin_path.name}")
print(f"per-VA PRF: {rec.dims.per_va_prf_hz:.1f} Hz")

n_frames_to_avg = 50
fft_n = 1024
n_va_chirps = rec.dims.n_chirps // 6
win = np.hanning(n_va_chirps).astype(np.float32)

freqs_full = np.fft.fftfreq(fft_n, d=1.0 / rec.dims.per_va_prf_hz)
pos_mask = freqs_full > 0
freqs = freqs_full[pos_mask]

# Average magnitude spectrum across n frames per range bin
sum_spec = np.zeros((fft_n, rec.dims.n_range_bins), dtype=np.float64)
n = 0
for idx, cube in iter_real_frames(rec.bin_path, rec.dims, max_frames=n_frames_to_avg):
    rfft = stage1_range_fft(cube)
    slot0 = rfft[0::6, :, :]
    slot0 = slot0.sum(axis=-1)
    slot0_w = slot0 * win[:, None]
    spec = np.abs(scipy_fft.fft(slot0_w, n=fft_n, axis=0))
    sum_spec += spec
    n += 1

avg_spec = sum_spec / n
avg_spec_pos = avg_spec[pos_mask, :]
avg_spec_db = 20 * np.log10(avg_spec_pos + 1e-12)

# For each of bins 0..6, find local maxima (peaks separated by ≥30 Hz).
print(f"\nAveraged over {n} frames. Top WELL-SEPARATED peaks per range bin")
print(f"(masking the 2400-2540 Hz Nyquist-spur zone):\n")

def find_peaks(spec_db: np.ndarray, freqs: np.ndarray, min_sep_hz: float = 30.0,
               n_top: int = 6) -> list[tuple[float, float]]:
    """Greedy peak picker: pick global max, mask ±min_sep_hz around it,
    pick next, repeat n_top times."""
    s = spec_db.copy()
    out = []
    for _ in range(n_top):
        i = int(np.argmax(s))
        if not np.isfinite(s[i]):
            break
        out.append((float(freqs[i]), float(s[i])))
        # Mask ±min_sep_hz around picked peak
        mask = np.abs(freqs - freqs[i]) < min_sep_hz
        s[mask] = -np.inf
    return out

for bin_idx in [0, 1, 2, 3, 4, 5, 6, 24, 50, 96]:
    rng_m = bin_idx * rec.dims.range_resolution_m
    spec_db = avg_spec_db[:, bin_idx].copy()
    # Kill DC + below 100 Hz + Nyquist spur zone (2400-2540 Hz).
    spec_db[freqs < 100.0] = -np.inf
    spec_db[(freqs >= 2400.0)] = -np.inf
    noise_floor = float(np.median(spec_db[(freqs >= 200) & (freqs <= 2300)
                                          & np.isfinite(spec_db)]))
    peaks = find_peaks(spec_db, freqs, min_sep_hz=40.0, n_top=6)
    print(f"  bin {bin_idx:3d} ({rng_m:6.2f} m)  noise_floor={noise_floor:.1f} dB")
    for f, d in peaks:
        snr = d - noise_floor
        print(f"     peak: {f:7.1f} Hz   {d:7.2f} dB  (SNR {snr:+5.1f} dB)")
    print()

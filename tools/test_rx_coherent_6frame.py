"""6-frame integration WITHOUT DDMA unfold.

Per Wave 1B forensics: DDMA-unfold-and-average loses ~3 dB vs straight
RX-coherent-sum at full PRF. The PMM detector was using DDMA + per-VA
averaging — that's wrong for *detection* (it's right only for AoA).

This test:
  - Stage 1 range FFT (existing, unchanged)
  - MTI (mean subtract) per (range, RX) — existing, unchanged
  - NO notch (Wave 1B confirmed it kills 86% of range bins)
  - NO DDMA unfold for detection
  - Coherent RX sum: chirp_x_range = range_cube.sum(axis=2) -> (768, 97)
  - Full 768-chirp slow-time at full PRF (Nyquist +-15239 Hz)
  - 6-frame integration: sum |X|^2 across 6 frames per (range, freq) bin
  - Look for drone signature at chip-CFAR-known range bins

DJI FPV expected blade-pass at hover ~800 Hz (3 blades x ~16k RPM motor).
With full PRF Nyquist 15 kHz, harmonics at 800, 1600, 2400, 3200, 4000 Hz
all comfortably in band (3-5 harmonics fit, vs only 2 with DDMA per-VA path).

Run:
    py -3.11 tools/test_rx_coherent_6frame.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_FFT_SLOW = 4096  # zero-pad of 768 -> finer resolution
BIN_HZ = PRF_HZ / N_FFT_SLOW  # 7.44 Hz/bin
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

AIRBORNE_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)
BG_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-54-23_radar.bin"
)


def load_frame(path, idx):
    with open(path, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube):
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def per_frame_pwr_rxsum(path, frame_idx):
    """Returns power spectrum (n_fft//2, n_range) — RX-coherent sum,
    full 768-chirp slow-time at full PRF, no DDMA, no notch.
    """
    cube = load_frame(path, frame_idx)
    rc = stage1(cube)                       # (768, 97, 4) complex64
    rc -= rc.mean(axis=0, keepdims=True)    # MTI per (range, rx)
    # NO notch - critical
    rxsum = rc.sum(axis=2)                  # (768, 97) coherent RX combine
    # Slow-time FFT per range bin with Hann window
    win = np.hanning(N_CHIRPS).astype(np.float32)
    spec = np.fft.fft(rxsum * win[:, None], n=N_FFT_SLOW, axis=0)
    pwr = (spec.real ** 2 + spec.imag ** 2)  # (N_FFT_SLOW, n_range)
    return pwr[: N_FFT_SLOW // 2, :].astype(np.float64)


def integrate_n(path, start, n):
    acc = None
    for i in range(n):
        try:
            spec = per_frame_pwr_rxsum(path, start + i)
        except EOFError:
            break
        if acc is None:
            acc = spec.copy()
        else:
            acc += spec
    return acc


def find_drone_peaks(spec_int, label, target_range_bins=None):
    """Look for in-band peaks at target_range_bins specifically (where
    chip CFAR found drone), and at chip-artifact bins for comparison.
    """
    n_freq, n_range = spec_int.shape
    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)

    # Mask out DC + low-freq leakage
    keep = np.ones(n_freq, dtype=bool)
    keep[:8] = False

    print(f"\n=== {label} ===")
    print(f"shape: {spec_int.shape}, bin_hz: {BIN_HZ:.2f}")

    # Drone-candidate bins (chip CFAR found drone here): 1-15 (2.6-39.6m)
    DRONE_BINS = list(range(1, 16))
    # Chip-artifact bins (from previous analysis): 24, 48, 72
    ARTIFACT_BINS = [24, 48, 72]

    rows = []
    for rb in DRONE_BINS + ARTIFACT_BINS:
        if rb >= n_range: continue
        spec = spec_int[:, rb].copy()
        spec[~keep] = 0
        # Find top peak in 50-3000 Hz band
        band = spec[bin_50:bin_3000]
        if band.max() <= 0:
            continue
        floor = float(np.median(spec[keep]))
        peak_idx = int(np.argmax(band))
        peak_freq = (bin_50 + peak_idx) * BIN_HZ
        peak_db = 10.0 * np.log10(band[peak_idx] / max(floor, 1e-30))
        # Comb: count harmonics of peak that also exceed floor + 3 dB
        thr = floor * 2.0
        harmonics_strong = 0
        peaks_at_harmonics = []
        for h in range(1, 8):
            target_bin = int(round(h * peak_freq / BIN_HZ))
            if target_bin >= n_freq: break
            lo = max(target_bin - 3, 0)
            hi = min(target_bin + 4, n_freq)
            h_pwr = spec[lo:hi].max()
            if h_pwr > thr:
                harmonics_strong += 1
                h_db = 10.0 * np.log10(h_pwr / max(floor, 1e-30))
                peaks_at_harmonics.append(f"h{h}@{h_db:.0f}dB")
        kind = "DRONE-cand" if rb in DRONE_BINS else "ARTIFACT"
        rows.append((rb, kind, floor, peak_db, peak_freq, harmonics_strong, peaks_at_harmonics))

    # Print sorted by descending peak_db
    rows.sort(key=lambda x: -x[3])
    print(f"{'rb':>3} {'kind':<11} {'range_m':>7} {'floor':>10} {'peak_dB':>7} "
          f"{'peak_Hz':>8} {'#harm':>5} harmonics")
    for rb, kind, floor, peak_db, peak_f, n_h, h_str in rows[:15]:
        print(f"{rb:>3} {kind:<11} {rb*RANGE_RES_M:>7.1f} {floor:>10.2e} {peak_db:>7.1f} "
              f"{peak_f:>8.0f} {n_h:>5} {' '.join(h_str)}")


def main():
    print(f"PRF (full) = {PRF_HZ:.0f} Hz, Nyquist = {PRF_HZ/2:.0f} Hz")
    print(f"FFT length = {N_FFT_SLOW}, bin_hz = {BIN_HZ:.2f}")
    print(f"This test: NO DDMA, NO notch, RX-coherent sum, full PRF")
    print()

    # Hover window: t=33-100s = frames 463..1387 at 13.86 fps
    print("=" * 90)
    print("AIRBORNE1 HOVER WINDOW (drone hovering per visual ground truth)")
    print("=" * 90)
    HOVER_STARTS = [470, 600, 750, 900, 1050, 1200, 1350]
    for start in HOVER_STARTS:
        spec_int = integrate_n(AIRBORNE_BIN, start, 6)
        if spec_int is None: continue
        t_rel = start / 13.86
        find_drone_peaks(spec_int, f"Airborne hover frames {start}-{start+5} (t~{t_rel:.1f}s)")

    print()
    print("=" * 90)
    print("AIRBORNE1 PRE-DRONE (drone not in scene)")
    print("=" * 90)
    for start in [50, 150, 250]:
        spec_int = integrate_n(AIRBORNE_BIN, start, 6)
        if spec_int is None: continue
        find_drone_peaks(spec_int, f"Pre-drone frames {start}-{start+5}")

    print()
    print("=" * 90)
    print("BACKGROUND (no drone)")
    print("=" * 90)
    for start in [50, 200, 400, 600]:
        spec_int = integrate_n(BG_BIN, start, 6)
        if spec_int is None: continue
        find_drone_peaks(spec_int, f"Background frames {start}-{start+5}")


if __name__ == "__main__":
    sys.exit(main())

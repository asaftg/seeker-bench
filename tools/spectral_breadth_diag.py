"""SPECTRAL BREADTH per range bin — true drone-specific discriminator.

Insight from user:
  - Drone propeller blade tips at 50+ kHz Doppler fold many times into our
    Nyquist, creating COHERENT energy across MANY freq bins simultaneously.
  - Chip artifacts are NARROWBAND tones (single freq peak).
  - Leaves/wind/operator are NARROWBAND too (single low-freq peak).
  - A drone is the ONLY source that produces broadband coherence at ONE range.

Method per (range_bin):
  1. Compute MIMO coherence spread per (range, freq) cell (already done).
  2. Count how many freq bins at this range have spread > threshold.
  3. drone bin: BREADTH > 10 (broadband)
  4. chip artifact bin: BREADTH = 1-3 (narrowband)
  5. clean bin: BREADTH = 0

Then for frame-by-frame study:
  - Run on multiple specific frames in 27-40s window (drone fly-away)
  - Run on hover window (33-100s)
  - Run on the OLDER 'drone fly' recording (yesterday)
  - Compare drone-range breadth vs chip-artifact-range breadth

Run:
    py -3.11 tools/spectral_breadth_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft
import json
import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from radar_dca.ddma import ddma_unfold

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
EFF_PRF = PRF_HZ / N_TX
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT
RANGE_RES_M = 2.638
N_VA = 16
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)


def load_frame(path, idx):
    with open(path, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME: raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube):
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out


def per_frame_va_spec(path, idx):
    cube = load_frame(path, idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)
    virtual = ddma_unfold(rc)  # (n_per_tx, n_range, n_rx, n_tx)
    n_per_tx, n_range, _, _ = virtual.shape
    slow = virtual.reshape(n_per_tx, n_range, N_VA)
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
    return spec[: N_FFT // 2, :, :]  # (512, 97, 16) complex


def integrate_covariance(path, start, n_frames, range_bins, freq_bins):
    """Sum R = X X^H per (range, freq) across n_frames."""
    n_rb = len(range_bins)
    n_fb = len(freq_bins)
    R = np.zeros((n_rb, n_fb, N_VA, N_VA), dtype=np.complex64)
    for i in range(n_frames):
        try: spec = per_frame_va_spec(path, start + i)
        except EOFError: break
        for ir, rb in enumerate(range_bins):
            for ifq, fb in enumerate(freq_bins):
                v = spec[fb, rb, :]
                R[ir, ifq] += np.outer(v, v.conj())
    return R


def coherence_spread(R):
    n_rb, n_fb = R.shape[:2]
    spread = np.zeros((n_rb, n_fb))
    for ir in range(n_rb):
        for ifq in range(n_fb):
            try:
                eigs = np.linalg.eigvalsh(R[ir, ifq])
                lam_max = float(eigs[-1])
                rest_mean = float(eigs[:-1].mean())
                if rest_mean > 0:
                    spread[ir, ifq] = lam_max / rest_mean
                else:
                    spread[ir, ifq] = 1.0
            except np.linalg.LinAlgError:
                spread[ir, ifq] = 1.0
    return spread


def analyze(path, start_frame, n_frames, label):
    """For each range bin, count #freq bins with high coherence (BREADTH).
    Compare drone-candidate range bins (rb=15-25) vs chip-artifact range
    bins (rb=24, 36, 48, 60, 72) vs clutter (other)."""
    range_bins = list(range(4, 80))   # 10-211m
    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)
    freq_bins = list(range(bin_50, bin_3000))   # 50-3000 Hz

    R = integrate_covariance(path, start_frame, n_frames, range_bins, freq_bins)
    spread = coherence_spread(R)
    spread_db = 10.0 * np.log10(np.maximum(spread, 1e-12))

    # For each range bin: count how many freq bins exceed N dB spread
    # ALSO show max spread per range bin
    BREADTH_THR_DB = 8.0  # cell coherence above 8 dB counts toward "broadband"
    breadth = (spread_db > BREADTH_THR_DB).sum(axis=1)  # (n_rb,)
    max_spread = spread_db.max(axis=1)
    max_freq_idx = spread_db.argmax(axis=1)
    max_freq_hz = (np.array(freq_bins)[max_freq_idx]) * BIN_HZ

    print(f"\n=== {label} (frames {start_frame}-{start_frame+n_frames-1}) ===")
    print(f"breadth threshold = {BREADTH_THR_DB} dB; n_freq_bins scanned = {len(freq_bins)}")
    print(f"{'rb':>3} {'range_m':>7} {'breadth':>7} {'max_dB':>7} {'max_Hz':>7}  bar")
    # Sort by breadth descending and show top 25
    order = np.argsort(-breadth)
    for i in order[:25]:
        rb = range_bins[i]
        bar = "#" * min(60, int(breadth[i]))
        print(f"{rb:>3} {rb*RANGE_RES_M:>7.1f} {breadth[i]:>7} {max_spread[i]:>7.1f} "
              f"{max_freq_hz[i]:>7.0f}  {bar}")


def main():
    AIRBORNE = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
                    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin")
    BG = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
              r"\recordings\seeker_2026-05-06_12-54-23_radar.bin")
    DRONE_FLY_BIN = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
                          r"\recordings\seeker_2026-05-05_21-14-35_radar.bin")

    # User-suggested windows
    print("##" * 50)
    print("AIRBORNE1 RECORDING")
    print("##" * 50)
    # t=27-40s drone flying away (close range to 40m)
    # 27s @ 13.86 fps = frame ~374
    # 40s @ 13.86 fps = frame ~554
    analyze(AIRBORNE, 380, 6, "AIRBORNE1 t=27.4s (drone flying away, close range)")
    analyze(AIRBORNE, 460, 6, "AIRBORNE1 t=33.2s (transitioning to hover)")
    analyze(AIRBORNE, 600, 6, "AIRBORNE1 t=43.3s (HOVER)")
    analyze(AIRBORNE, 800, 6, "AIRBORNE1 t=57.7s (HOVER)")
    analyze(AIRBORNE, 1000, 6, "AIRBORNE1 t=72.1s (HOVER)")
    analyze(AIRBORNE, 1200, 6, "AIRBORNE1 t=86.6s (HOVER)")

    print("\n")
    print("##" * 50)
    print("BACKGROUND RECORDING (no drone)")
    print("##" * 50)
    analyze(BG, 100, 6, "BG t=7s")
    analyze(BG, 300, 6, "BG t=21s")
    analyze(BG, 500, 6, "BG t=36s")

    print("\n")
    print("##" * 50)
    print("OLD 'DRONE FLY' RECORDING (yesterday's flight)")
    print("##" * 50)
    if DRONE_FLY_BIN.exists():
        n_drone_frames = DRONE_FLY_BIN.stat().st_size // BYTES_PER_FRAME
        print(f"  drone_fly bin has {n_drone_frames} frames")
        # Sample a few
        for f in [50, 200, 350]:
            if f < n_drone_frames - 6:
                analyze(DRONE_FLY_BIN, f, 6, f"DRONE_FLY frame {f}")
    else:
        print(f"  drone_fly bin not found at {DRONE_FLY_BIN}")


if __name__ == "__main__":
    sys.exit(main())

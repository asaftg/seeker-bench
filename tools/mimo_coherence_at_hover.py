"""MIMO coherence test at the rb=22-26 hover range bins.

Concept: a real source produces a phase-coherent return across all 16
virtual antennas (4 RX x 4 TX after DDMA unfold). Background noise is
incoherent across VAs. Eigenvalue spread of the 16x16 covariance matrix
discriminates: real source -> dominant eigenvalue; noise -> uniform.

Per (range, freq) bin:
  R = sum over chirps of X_va_vector @ X_va_vector^H  (16x16 Hermitian)
  spread = lambda_max / mean(other 15 eigenvalues)
  large spread -> coherent source (drone, operator, vehicle)
  spread ~ 1 -> noise

Test: integrate covariance across N=24 frames, compute spread per
(range, freq), look for cells where hover spread > pre-drone spread.

Run:
    py -3.11 tools/mimo_coherence_at_hover.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

from radar_dca.ddma import ddma_unfold

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
N_VA = N_RX * N_TX  # 16
EFF_PRF = PRF_HZ / N_TX  # 7619.5
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT  # 7.44 Hz
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

AIRBORNE_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)


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


def per_frame_va_spec(path, frame_idx):
    """Returns spec_va of shape (n_freq=N_FFT/2, n_range, 16) complex."""
    cube = load_frame(path, frame_idx)
    windowed = cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    rfft_out -= rfft_out.mean(axis=0, keepdims=True)  # MTI
    virtual = ddma_unfold(rfft_out)  # (n_per_tx, n_range, 4, 4)
    n_per_tx, n_range, _, _ = virtual.shape
    slow = virtual.reshape(n_per_tx, n_range, N_VA)  # (192, 97, 16)
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
    return spec[: N_FFT // 2, :, :]  # (512, 97, 16) complex


def integrate_covariance(path, start, n, range_bins, freq_bins):
    """Sum covariance R = X X^H across N frames per (range, freq).
    Returns R[range, freq, 16, 16] complex.
    """
    n_rb = len(range_bins)
    n_fb = len(freq_bins)
    R = np.zeros((n_rb, n_fb, N_VA, N_VA), dtype=np.complex64)
    for i in range(n):
        try: spec = per_frame_va_spec(path, start + i)
        except EOFError: break
        for ir, rb in enumerate(range_bins):
            for ifq, fb in enumerate(freq_bins):
                v = spec[fb, rb, :]  # (16,) complex
                R[ir, ifq] += np.outer(v, v.conj())
    return R


def coherence_score(R):
    """Return lambda_max / mean(other 15 eigenvalues) for each (range, freq)."""
    n_rb, n_fb = R.shape[:2]
    score = np.zeros((n_rb, n_fb))
    f0_dom = np.zeros((n_rb, n_fb))
    for ir in range(n_rb):
        for ifq in range(n_fb):
            try:
                eigs = np.linalg.eigvalsh(R[ir, ifq])
                # sorted ascending; lambda_max at end
                lam_max = float(eigs[-1])
                lam_rest_mean = float(eigs[:-1].mean())
                if lam_rest_mean > 0:
                    score[ir, ifq] = lam_max / lam_rest_mean
                else:
                    score[ir, ifq] = 1.0
            except np.linalg.LinAlgError:
                score[ir, ifq] = 1.0
    return score


def main():
    print(f"PRF (full)={PRF_HZ:.0f}, eff PRF per VA={EFF_PRF:.0f}")
    print(f"Per-VA Nyquist = {EFF_PRF/2:.0f} Hz, bin_hz = {BIN_HZ:.2f}")
    print(f"FFT={N_FFT}, n_VA={N_VA} (4 RX x 4 TX after DDMA unfold)")
    print()
    print("Theory: 16-VA coherent integration adds +12 dB array gain")
    print("  for a coherent source vs incoherent noise.")
    print()

    # Range bins to inspect: rb=15..30 covers 40-79m hover region
    INSPECT_BINS = list(range(15, 30))
    n_rb = len(INSPECT_BINS)
    # Freq bins: 50-3000 Hz in per-VA Nyquist (limited to ~3810 Hz by DDMA)
    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)
    INSPECT_FREQS = list(range(bin_50, bin_3000))
    n_fb = len(INSPECT_FREQS)
    print(f"Scanning {n_rb} range bins x {n_fb} freq bins")
    print(f"Range range: {INSPECT_BINS[0]*RANGE_RES_M:.1f}..{INSPECT_BINS[-1]*RANGE_RES_M:.1f}m")
    print(f"Freq range: {INSPECT_FREQS[0]*BIN_HZ:.0f}..{INSPECT_FREQS[-1]*BIN_HZ:.0f} Hz")
    print()

    # Hover and pre-drone windows
    HOVER_STARTS = [(470, "33.9s"), (650, "46.9s"), (830, "59.9s"),
                     (1010, "72.9s"), (1190, "85.9s")]
    PRE_STARTS = [(50, "3.6s"), (150, "10.8s"), (250, "18.0s")]

    # Test multiple N values to see how coherence-spread evolves
    import os
    N = int(os.environ.get("MIMO_N", "24"))
    print(f"Integrating covariance over N={N} frames per window...")

    # Compute coherence scores for each window
    hov_scores = []
    for start, label in HOVER_STARTS:
        print(f"  hover {label}...")
        R = integrate_covariance(AIRBORNE_BIN, start, N, INSPECT_BINS, INSPECT_FREQS)
        score = coherence_score(R)
        hov_scores.append((label, score))

    pre_scores = []
    for start, label in PRE_STARTS:
        print(f"  pre-drone {label}...")
        R = integrate_covariance(AIRBORNE_BIN, start, N, INSPECT_BINS, INSPECT_FREQS)
        score = coherence_score(R)
        pre_scores.append((label, score))

    # Average pre-drone reference
    pre_avg = np.mean([s for _, s in pre_scores], axis=0)

    # For each hover window, find the (rb, freq) cell where hover_score
    # exceeds pre_score by the largest amount (in dB-of-spread).
    print()
    print("=" * 100)
    print("MIMO COHERENCE: hover spread vs pre-drone spread (in dB)")
    print("=" * 100)
    f0_consistency = {}
    for label, hov_score in hov_scores:
        # ratio in dB of spread
        delta_db = 10.0 * np.log10(hov_score / np.maximum(pre_avg, 1.0))
        # Find top cells
        flat_idx = np.argsort(-delta_db.flatten())
        print(f"\nHover at t={label}: top 8 (rb, freq) with biggest coherence advantage")
        print(f"  {'rb':>3} {'range_m':>7} {'freq_Hz':>8} {'hov_spread':>11} {'pre_spread':>11} {'delta_dB':>8}")
        for idx in flat_idx[:8]:
            ir, ifq = np.unravel_index(idx, delta_db.shape)
            rb = INSPECT_BINS[ir]
            fb = INSPECT_FREQS[ifq]
            freq = fb * BIN_HZ
            mark = " <-- coherent" if delta_db[ir, ifq] > 3.0 else ""
            print(f"  {rb:>3} {rb*RANGE_RES_M:>7.1f} {freq:>8.0f} "
                  f"{hov_score[ir, ifq]:>11.2f} {pre_avg[ir, ifq]:>11.2f} "
                  f"{delta_db[ir, ifq]:>8.1f}{mark}")
            if delta_db[ir, ifq] > 3.0:
                key = (rb, round(freq / 50) * 50)  # bin to nearest 50 Hz
                f0_consistency.setdefault(key, []).append(delta_db[ir, ifq])

    # Repeatability
    print()
    print("=" * 100)
    print("REPEATABILITY: (rb, freq~50Hz) cells with coherence advantage in N hover windows")
    print("=" * 100)
    print(f"{'count':>5} {'rb':>3} {'range_m':>7} {'freq~Hz':>8} {'mean_dB':>8}")
    sorted_keys = sorted(f0_consistency.items(), key=lambda kv: (-len(kv[1]), -np.mean(kv[1])))
    for (rb, fbin), vals in sorted_keys[:15]:
        if len(vals) < 2: continue
        print(f"{len(vals):>5} {rb:>3} {rb*RANGE_RES_M:>7.1f} {fbin:>8} {np.mean(vals):>8.1f}")


if __name__ == "__main__":
    sys.exit(main())

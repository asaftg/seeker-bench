"""mmHawkeye-style spectrum folding across the FULL spectrum, swept
over multiple integration windows N = 6, 12, 24, 48, 96 frames.

Question: at any N, does spectrum folding reveal a (range, f0) cell
where the hover window shows a stronger fold score than the pre-drone
window in the SAME recording?

If the answer is "yes at N=24" we know: PMM works, sensitivity = ~24 frames.
If "yes at N=96" we know: PMM works but is too insensitive for practical use.
If "no at any N": PMM physics doesn't work for this drone in this geometry.

Spectrum folding (mmHawkeye):
   For each candidate f0:
     bin_period = round(f0 / bin_hz)
     k = N_pos // bin_period   (number of folds)
     folded[i] = sum_j spec[j*bin_period + i]   for i in [0, bin_period)
     score(f0) = max(folded) / median(folded)   in dB
   The f0 with the highest score wins. This is the published mmHawkeye
   primitive applied to OUR data with NO drone-spec assumptions.

Range: 50 Hz to 5000 Hz, log-spaced 60 candidates.
At Nyquist 15239 Hz, that's a fold count from 305 (at 50 Hz) down to
3 (at 5000 Hz). Below 50 Hz: MTI residual. Above 5000 Hz: too few folds.

Run:
    py -3.11 tools/full_spectrum_fold_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_FFT_SLOW = 4096
BIN_HZ = PRF_HZ / N_FFT_SLOW  # 7.44 Hz
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
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def per_frame_pwr(path, frame_idx):
    cube = load_frame(path, frame_idx)
    windowed = cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2).astype(np.complex64)
    rfft_out[:, 1:-1, :] *= 2.0
    rfft_out -= rfft_out.mean(axis=0, keepdims=True)
    rxsum = rfft_out.sum(axis=2)
    win = np.hanning(N_CHIRPS).astype(np.float32)
    spec = np.fft.fft(rxsum * win[:, None], n=N_FFT_SLOW, axis=0)
    pwr = (spec.real ** 2 + spec.imag ** 2)
    return pwr[: N_FFT_SLOW // 2, :].astype(np.float64)


def integrate_n(path, start, n):
    acc = None
    for i in range(n):
        try:
            spec = per_frame_pwr(path, start + i)
        except EOFError:
            break
        if acc is None:
            acc = spec.copy()
        else:
            acc += spec
    return acc


# Mask out DC + low-freq leakage (don't fold MTI residual)
def make_masked_spec(spec_int):
    spec_masked = spec_int.copy()
    spec_masked[:8, :] = 0
    return spec_masked


def fold_score_at_f0(pwr_pos, f0_hz):
    """mmHawkeye-style folding score at fundamental f0_hz.
    Returns (score_db, k_folds)."""
    bin_period = int(round(f0_hz / BIN_HZ))
    if bin_period < 4:
        return -np.inf, 0
    n = pwr_pos.shape[0]
    k = n // bin_period
    if k < 3:
        return -np.inf, 0
    truncated = pwr_pos[: k * bin_period]
    folded = truncated.reshape(k, bin_period).sum(axis=0)
    peak = float(folded.max())
    floor = float(np.median(folded))
    if floor <= 0:
        return -np.inf, k
    return 10.0 * np.log10(peak / floor), k


def best_fold_per_range(spec_int, f0_grid):
    """For each range bin, find the best fold score across f0_grid.
    Returns (n_range,) of (best_score_db, best_f0)."""
    n_freq, n_range = spec_int.shape
    spec_masked = make_masked_spec(spec_int)
    out_score = np.full(n_range, -np.inf)
    out_f0 = np.zeros(n_range)
    for rb in range(n_range):
        spec = spec_masked[:, rb]
        if spec.max() <= 0: continue
        for f0 in f0_grid:
            score, k = fold_score_at_f0(spec, f0)
            if score > out_score[rb]:
                out_score[rb] = score
                out_f0[rb] = f0
    return out_score, out_f0


def main():
    print(f"PRF={PRF_HZ:.0f}, Nyquist={PRF_HZ/2:.0f}, FFT={N_FFT_SLOW}, bin_hz={BIN_HZ:.2f}")
    print(f"Spectrum-folding mmHawkeye-style across full spectrum 50-5000 Hz")
    print()

    # f0 grid: log-spaced 50 to 5000 Hz, 60 points
    f0_grid = np.logspace(np.log10(50.0), np.log10(5000.0), 60)
    print(f"f0 grid: {f0_grid[0]:.0f} - {f0_grid[-1]:.0f} Hz, {len(f0_grid)} points")

    # Hover starts (we want to integrate forward from each)
    HOVER_START = 470   # t~33.9s — clean hover begins
    PRE_DRONE_START = 50  # well before drone takeoff at t=24s

    INTEGRATION_NS = [6, 12, 24, 48, 96]

    # For each N, do hover-vs-pre-drone fold-score comparison.
    print()
    print("=" * 100)
    print("Best fold score per range bin: HOVER (red) vs PRE-DRONE (blue), per N frames")
    print("=" * 100)

    for N in INTEGRATION_NS:
        print(f"\n=== N = {N} frames ({N*0.072:.2f} s integration) ===")
        # Integrate hover and pre-drone with same N
        hov = integrate_n(AIRBORNE_BIN, HOVER_START, N)
        pre = integrate_n(AIRBORNE_BIN, PRE_DRONE_START, N)
        if hov is None or pre is None:
            print("  load failed")
            continue
        hov_score, hov_f0 = best_fold_per_range(hov, f0_grid)
        pre_score, pre_f0 = best_fold_per_range(pre, f0_grid)

        # For each range bin show: hover score, pre-drone score, delta
        # Sort by hover-pre delta
        n_range = len(hov_score)
        rows = []
        for rb in range(1, min(30, n_range)):  # focus on near-range
            delta = hov_score[rb] - pre_score[rb]
            rows.append((rb, hov_score[rb], hov_f0[rb], pre_score[rb], pre_f0[rb], delta))
        rows.sort(key=lambda r: -r[5])

        print(f"  {'rb':>3} {'range_m':>7} {'hov_dB':>7} {'hov_f0':>7} "
              f"{'pre_dB':>7} {'pre_f0':>7} {'delta':>6}")
        for rb, h, hf, p, pf, d in rows[:12]:
            mark = " <-- drone-like" if d > 3.0 else ""
            print(f"  {rb:>3} {rb*RANGE_RES_M:>7.1f} {h:>7.1f} {hf:>7.0f} "
                  f"{p:>7.1f} {pf:>7.0f} {d:>6.1f}{mark}")

        # Also report: globally, what's the BEST (rb, f0) across all range bins for hover?
        rb_max = int(np.argmax(hov_score))
        print(f"  >>> HOVER  best: rb={rb_max} ({rb_max*RANGE_RES_M:.1f}m) "
              f"score={hov_score[rb_max]:.1f}dB f0={hov_f0[rb_max]:.0f}Hz")
        rb_max_pre = int(np.argmax(pre_score))
        print(f"  >>> PRE    best: rb={rb_max_pre} ({rb_max_pre*RANGE_RES_M:.1f}m) "
              f"score={pre_score[rb_max_pre]:.1f}dB f0={pre_f0[rb_max_pre]:.0f}Hz")

    # Final summary: at what N does ANY range bin show consistent excess?
    print()
    print("=" * 100)
    print("CONCLUSION: Looking for N where hover shows clear advantage over pre-drone")
    print("=" * 100)


if __name__ == "__main__":
    sys.exit(main())

"""Fine-grained f0 sweep at the rb=22-26 hover range bins.

Per user: drone hovered BEYOND 30m. Earlier full-spectrum sweep flagged
rb=22 (58m) at f0~3956Hz as the strongest hover-vs-pre-drone signature.
Now zoom in: fine f0 grid across the FULL spectrum (50-7000 Hz), N=24
integration, find the exact f0 and confirm the (rb, f0) is repeatable
across multiple hover sub-windows.

Run:
    py -3.11 tools/fine_f0_at_hover_range.py
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
BIN_HZ = PRF_HZ / N_FFT_SLOW
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
        try: spec = per_frame_pwr(path, start + i)
        except EOFError: break
        if acc is None: acc = spec.copy()
        else: acc += spec
    return acc


def fold_score(pwr_pos, f0_hz):
    bin_period = int(round(f0_hz / BIN_HZ))
    if bin_period < 4: return -np.inf, 0
    n = pwr_pos.shape[0]
    k = n // bin_period
    if k < 3: return -np.inf, 0
    truncated = pwr_pos[: k * bin_period]
    folded = truncated.reshape(k, bin_period).sum(axis=0)
    peak = float(folded.max())
    floor = float(np.median(folded))
    if floor <= 0: return -np.inf, k
    return 10.0 * np.log10(peak / floor), k


def main():
    print(f"PRF={PRF_HZ:.0f}, Nyquist={PRF_HZ/2:.0f}, bin_hz={BIN_HZ:.2f}")
    print(f"Fine f0 sweep across 50-7000 Hz, no drone-spec assumptions")
    print()

    # Multiple hover sub-windows to confirm repeatability
    HOVER_WINDOWS = [
        (470, "33.9s"),
        (560, "40.4s"),
        (650, "46.9s"),
        (740, "53.4s"),
        (830, "59.9s"),
        (920, "66.4s"),
        (1010, "72.9s"),
        (1100, "79.4s"),
        (1190, "85.9s"),
        (1280, "92.4s"),
    ]
    PRE_DRONE_WINDOWS = [
        (50, "3.6s"),
        (150, "10.8s"),
        (250, "18.0s"),
    ]

    # Fine f0 grid
    f0_grid = np.arange(50, 7000, 30.0)  # 30 Hz step, 232 candidates
    print(f"f0 grid: {f0_grid[0]:.0f}..{f0_grid[-1]:.0f} Hz, {len(f0_grid)} candidates")

    N = 24
    INSPECT_BINS = list(range(15, 30))  # 40-77m

    # Collect fold scores per (rb, f0) per window
    print(f"\nIntegrating {N} frames per window...")
    hov_scores = []
    for start, label in HOVER_WINDOWS:
        spec = integrate_n(AIRBORNE_BIN, start, N)
        if spec is None: continue
        spec_masked = spec.copy()
        spec_masked[:8, :] = 0
        rb_scores = {}
        for rb in INSPECT_BINS:
            best_score = -np.inf
            best_f0 = 0
            scores_at_f0 = []
            for f0 in f0_grid:
                s, _ = fold_score(spec_masked[:, rb], f0)
                scores_at_f0.append(s)
                if s > best_score:
                    best_score = s
                    best_f0 = f0
            rb_scores[rb] = (best_score, best_f0, scores_at_f0)
        hov_scores.append((label, rb_scores))

    pre_scores = []
    for start, label in PRE_DRONE_WINDOWS:
        spec = integrate_n(AIRBORNE_BIN, start, N)
        if spec is None: continue
        spec_masked = spec.copy()
        spec_masked[:8, :] = 0
        rb_scores = {}
        for rb in INSPECT_BINS:
            best_score = -np.inf
            best_f0 = 0
            scores_at_f0 = []
            for f0 in f0_grid:
                s, _ = fold_score(spec_masked[:, rb], f0)
                scores_at_f0.append(s)
                if s > best_score:
                    best_score = s
                    best_f0 = f0
            rb_scores[rb] = (best_score, best_f0, scores_at_f0)
        pre_scores.append((label, rb_scores))

    # Average pre-drone scores per (rb, f0) for stable reference
    pre_avg = {}
    for rb in INSPECT_BINS:
        if not pre_scores: continue
        per_f0 = np.zeros(len(f0_grid))
        for label, rb_scores in pre_scores:
            per_f0 += np.array(rb_scores[rb][2])
        per_f0 /= len(pre_scores)
        pre_avg[rb] = per_f0

    # For each hover window, find (rb, f0) with biggest hover-vs-pre advantage
    print()
    print("=" * 100)
    print("Per hover window: range bin where hover-vs-pre is largest, with exact f0")
    print("=" * 100)
    f0_consistency = {rb: [] for rb in INSPECT_BINS}
    for label, rb_scores in hov_scores:
        rows = []
        for rb in INSPECT_BINS:
            hov_f0_scores = np.array(rb_scores[rb][2])
            pre_f0_scores = pre_avg.get(rb)
            if pre_f0_scores is None: continue
            delta = hov_f0_scores - pre_f0_scores
            best_idx = int(np.argmax(delta))
            best_f0 = f0_grid[best_idx]
            best_delta = float(delta[best_idx])
            rows.append((rb, best_f0, best_delta))
            if best_delta > 2.0:
                f0_consistency[rb].append(best_f0)
        rows.sort(key=lambda r: -r[2])
        print(f"\nHover at t={label}:")
        for rb, f0, d in rows[:5]:
            mark = " <-- positive" if d > 2.0 else ""
            print(f"  rb={rb} ({rb*RANGE_RES_M:.1f}m) f0={f0:.0f}Hz delta={d:.1f}dB{mark}")

    # Repeatability: which (rb, f0) cells fire across multiple hover windows?
    print()
    print("=" * 100)
    print("REPEATABILITY: f0 values that recurred across hover windows per range bin")
    print("=" * 100)
    print(f"{'rb':>3} {'range_m':>7} {'n_hits':>6} {'f0_mean':>8} {'f0_std':>7} {'f0_values':<60}")
    for rb in INSPECT_BINS:
        if not f0_consistency[rb]: continue
        f0s = np.array(f0_consistency[rb])
        print(f"{rb:>3} {rb*RANGE_RES_M:>7.1f} {len(f0s):>6} "
              f"{f0s.mean():>8.0f} {f0s.std():>7.1f} "
              f"{', '.join(f'{x:.0f}' for x in f0s)}")


if __name__ == "__main__":
    sys.exit(main())

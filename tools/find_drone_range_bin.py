"""Find which range bin (if any) shows the drone signature.

Strategy: compute total in-band energy per range bin per frame, then
compare airborne frames vs ground frames at EACH range bin. The bin
where airborne > ground by the largest margin is where the drone is.

Run:
    py -3.11 tools/find_drone_range_bin.py
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
EFF_PRF = PRF_HZ / N_TX
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

BIN_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-05_21-14-35_radar.bin"
)


def load_frame(idx: int) -> np.ndarray:
    with open(BIN_PATH, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
           .transpose(0, 2, 1)
           .astype(np.float32)
    )
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube: np.ndarray) -> np.ndarray:
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def notch(rc: np.ndarray) -> None:
    n_range = rc.shape[1]
    period = N_SAMPLES // 8
    step = max(period // 2, 1)
    radius = 5
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        rc[:, lo:hi, :] = 0


def main():
    n_frames = BIN_PATH.stat().st_size // BYTES_PER_FRAME
    sample_idxs = list(range(0, n_frames, 5))  # every 5th frame for richer stats
    n_range = N_SAMPLES // 2 + 1

    # Compute, per (frame, rb), total slow-time variance (raw, no FFT).
    # Variance is sensitive to ANY motion: rotor blades, body Doppler,
    # operator motion. After MTI, variance reflects non-stationary energy.
    print(f"Sampling {len(sample_idxs)} frames out of {n_frames}, "
          f"computing slow-time variance per range bin...")

    # Also compare TWO approaches:
    # A) After DDMA unfold + RX-VA average (current detector path)
    # B) Raw RX-summed slow-time per range bin (what the OLD detector saw)
    var_unfold = np.zeros((len(sample_idxs), n_range), dtype=np.float64)
    var_raw = np.zeros((len(sample_idxs), n_range), dtype=np.float64)

    for i, fi in enumerate(sample_idxs):
        try:
            cube = load_frame(fi)
        except Exception:
            continue
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        notch(rc)

        # Approach B: raw RX-summed
        slow_raw = rc.sum(axis=2)  # (n_chirps, n_range)
        var_raw[i] = np.var(slow_raw, axis=0).real

        # Approach A: DDMA unfold + average across VAs
        virtual = ddma_unfold(rc)  # (n_per_tx, n_range, n_rx, n_tx)
        # Variance per range bin: average across VAs
        n_per_tx = virtual.shape[0]
        # Reshape to (n_per_tx, n_range, n_va)
        v2d = virtual.reshape(n_per_tx, n_range, N_RX * N_TX)
        var_unfold[i] = np.mean(np.var(v2d, axis=0), axis=1)

    # Ground truth: airborne ~ frames 188-295 (per visual)
    # ground = frames < 187 OR > 430 (drone landed)
    is_airborne = np.array([(s >= 187 and s <= 295) for s in sample_idxs])
    is_ground = np.array([(s < 187 or s >= 432) for s in sample_idxs])

    print(f"  airborne frames sampled: {is_airborne.sum()}")
    print(f"  ground   frames sampled: {is_ground.sum()}")
    print()

    print(f"=== APPROACH A: DDMA-unfolded VA-averaged variance per range bin ===")
    print(f"{'rb':>4} {'range_m':>8} {'ground_med':>14} {'airborne_med':>14} {'ratio':>8} {'delta':>14}")
    for rb in range(n_range):
        if not is_ground.any() or not is_airborne.any():
            break
        g = float(np.median(var_unfold[is_ground, rb]))
        a = float(np.median(var_unfold[is_airborne, rb]))
        if g <= 0:
            continue
        ratio = a / g
        if ratio > 1.20 or ratio < 0.83:  # only print bins where airborne != ground meaningfully
            print(f"{rb:>4} {rb*RANGE_RES_M:>8.1f} {g:>14.3e} {a:>14.3e} {ratio:>8.3f} {a-g:>14.3e}")

    print()
    print(f"=== APPROACH B: Raw RX-summed slow-time variance per range bin ===")
    print(f"{'rb':>4} {'range_m':>8} {'ground_med':>14} {'airborne_med':>14} {'ratio':>8} {'delta':>14}")
    for rb in range(n_range):
        g = float(np.median(var_raw[is_ground, rb]))
        a = float(np.median(var_raw[is_airborne, rb]))
        if g <= 0:
            continue
        ratio = a / g
        if ratio > 1.20 or ratio < 0.83:
            print(f"{rb:>4} {rb*RANGE_RES_M:>8.1f} {g:>14.3e} {a:>14.3e} {ratio:>8.3f} {a-g:>14.3e}")

    # Also: per-frame total variance across all range bins (1-90m, excluding chip-artifact bins)
    print()
    print("=== Per-frame total variance trend (rb 2-30, post-notch) ===")
    print("Looking for an envelope that rises during airborne window")
    for kind, var_arr in (("UNFOLD", var_unfold), ("RAW", var_raw)):
        per_frame_total = var_arr[:, 2:30].sum(axis=1)
        # Print first 50 frames worth
        print(f"\n{kind} approach — per-frame total var (rb 2-30):")
        print(f"{'frame':>5} {'phase':<9} {'var_total':>12}")
        for i, fi in enumerate(sample_idxs):
            if fi % 20 != 0: continue
            ph = "AIRBORNE" if is_airborne[i] else ("ground" if is_ground[i] else "transition")
            print(f"{fi:>5} {ph:<9} {per_frame_total[i]:>12.3e}")


if __name__ == "__main__":
    sys.exit(main())

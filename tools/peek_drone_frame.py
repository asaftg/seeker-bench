"""Peek at the actual radar spectrum at a frame where chip CFAR fired
on the drone. Tells us what the REAL drone signature looks like vs what
my synthetic-test-tuned detector expects.

Usage:
    py -3.11 tools/peek_drone_frame.py
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
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)
# Mapping: aa_frame_id 9138 -> bin_idx 0 (per replay tool output)
BASE_AA_FID = 9138


def load_frame(idx):
    with open(BIN_PATH, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
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


def notch(rc):
    period = N_SAMPLES // 8
    step = max(period // 2, 1)
    radius = 5
    n_range = rc.shape[1]
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        rc[:, lo:hi, :] = 0


def dump_spectrum(frame_idx, label):
    print(f"\n========== {label} frame_idx={frame_idx} ==========")
    cube = load_frame(frame_idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)
    notch(rc)
    virtual = ddma_unfold(rc)  # (n_per_tx, n_range, n_rx, n_tx)
    n_per_tx = virtual.shape[0]

    # Chip CFAR fired at ranges 5-13m (rb 2-5) and 36-67m (rb 14-26).
    # Dump spectrum at each.
    for rb in [2, 3, 4, 5, 14, 20, 25]:
        slow_per_va = virtual[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)
        win = np.hanning(n_per_tx).astype(np.float32)
        spec = np.fft.fft(slow_per_va * win[:, None], n=N_FFT, axis=0)
        pwr = (spec.real**2 + spec.imag**2).mean(axis=1)
        pwr_pos = pwr[: N_FFT // 2]
        floor = float(np.median(pwr_pos))
        peak_idx = int(np.argmax(pwr_pos))
        peak_db = 10.0 * np.log10(pwr_pos[peak_idx] / max(floor, 1e-30))

        # Also: power in 100-2000 Hz drone-rotor band
        b100 = int(100 / BIN_HZ)
        b2000 = int(2000 / BIN_HZ)
        band_pwr = float(pwr_pos[b100:b2000].sum())

        # Top 8 in-band peaks (excluding DC region)
        keep = np.ones(N_FFT // 2, dtype=bool)
        keep[:8] = False  # DC + leakage
        keep[336:346] = False  # known chip artifact at PRF/3
        masked = pwr_pos.copy()
        masked[~keep] = 0
        top_idx = np.argpartition(masked, -8)[-8:]
        top_idx = top_idx[np.argsort(-masked[top_idx])]
        top_str = ", ".join(
            f"{int(i)}@{i*BIN_HZ:.0f}Hz/{10*np.log10(masked[i]/max(floor,1e-30)):.1f}dB"
            for i in top_idx[:5]
        )

        print(f"  rb={rb:>3} ({rb*RANGE_RES_M:>5.1f}m): "
              f"floor={floor:.2e} peak_db={peak_db:.1f} band_pwr={band_pwr:.2e}")
        print(f"      top5 in-band: {top_str}")


def main():
    n_frames = BIN_PATH.stat().st_size // BYTES_PER_FRAME
    print(f"Total frames in bin: {n_frames}")
    print(f"PRF_eff={EFF_PRF:.0f} Hz, bin_hz={BIN_HZ:.2f} Hz, range_res={RANGE_RES_M} m")

    # FRAMES OF INTEREST (per chip CFAR data):
    # t=23.5 -> drone at 37m retreating fast (-9.8 m/s) -> bin_idx 470
    # t=26.4 -> drone at 2m approaching FAST (+29 m/s) -> bin_idx 528
    # t=26.5 -> drone at 5-10m, slow 1-1.4 m/s (could be transition to hover)
    # t=60   -> drone visible in EO at far range
    # Compare against frames where chip CFAR did NOT fire:
    # t=10  -> bin_idx 200, no chip detection
    # t=80  -> bin_idx 1600, drone visible but no chip det
    cases = [
        ("t=10s no-chip-det",  200),
        ("t=23.5s drone@37m fast retreat", 470),
        ("t=26.4s drone@2m FAST APPROACH +29m/s", 528),
        ("t=26.5s drone@5-10m slow", 530),
        ("t=30s drone visible mid-EO", 600),
        ("t=60s drone visible far-EO", 1200),
        ("t=80s drone airborne (no chip det?)", 1600),
        ("t=120s end of session", 2400 if n_frames > 2400 else n_frames - 5),
    ]
    for label, idx in cases:
        if idx >= n_frames:
            print(f"\nSkip {label}: idx {idx} >= n_frames {n_frames}")
            continue
        dump_spectrum(idx, label)


if __name__ == "__main__":
    sys.exit(main())

"""Direct comparison: drone vs chip artifact spectral breadth.

Method: at one specific frame in HOVER (t=43s), dump the per-VA
power-averaged spectrum at FIVE range bins:
  rb=20 (drone's known location at 53m)
  rb=24 (chip artifact bin at 63m)
  rb=10 (clean clutter at 26m)
  rb=40 (clean clutter at 105m)
  rb=72 (chip artifact at 189m)

For each: print the spectrum across full 50-3000 Hz at coarse Hz steps
so we can VISUALLY see if the drone signature is BROADBAND vs the chip
artifact's NARROWBAND tone.

Then quantify with three metrics:
  1. Number of freq bins above floor + N dB  (BREADTH)
  2. Sum of in-band power above floor          (TOTAL DRONE-LIKE ENERGY)
  3. Spectral flatness measure                  (BROADBAND vs TONAL)
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

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

AIRBORNE = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
                r"\recordings\seeker_2026-05-06_12-58-59_radar.bin")
BG = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
          r"\recordings\seeker_2026-05-06_12-54-23_radar.bin")


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


def integrate_pwr_per_va(path, start, n_frames):
    """Sum |spec|^2 per (freq, range, va) across n frames."""
    acc = None
    for i in range(n_frames):
        cube = load_frame(path, start + i)
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        virtual = ddma_unfold(rc)
        n_per_tx, n_range, _, _ = virtual.shape
        slow = virtual.reshape(n_per_tx, n_range, N_VA)
        win = np.hanning(n_per_tx).astype(np.float32)
        spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
        pwr = (spec.real**2 + spec.imag**2)[: N_FFT//2, :, :]
        if acc is None:
            acc = pwr.astype(np.float64)
        else:
            acc += pwr
    return acc  # (n_freq, n_range, n_va)


def metrics_per_rb(pwr_per_va, range_bins, label):
    """For each range bin: spectrum, breadth, total energy, SFM."""
    # Average across VAs incoherently for power spectrum
    pwr_avg = pwr_per_va.mean(axis=2)  # (n_freq, n_range)
    bin_50 = int(50 / BIN_HZ)
    bin_3000 = int(3000 / BIN_HZ)

    print(f"\n=== {label} ===")
    print(f"{'rb':>3} {'r_m':>6} {'floor':>10} {'#>10dB':>7} {'#>20dB':>7} "
          f"{'top_dB':>7} {'top_Hz':>7} {'totEdB':>7} {'SFM':>7} bar")

    for rb in range_bins:
        spec = pwr_avg[bin_50:bin_3000, rb]
        floor = float(np.median(spec))
        spec_db = 10.0 * np.log10(np.maximum(spec, 1e-30) / max(floor, 1e-30))
        n_above_10 = int((spec_db > 10).sum())
        n_above_20 = int((spec_db > 20).sum())
        peak_idx = int(np.argmax(spec_db))
        peak_db = float(spec_db[peak_idx])
        peak_freq = (bin_50 + peak_idx) * BIN_HZ
        # Total in-band energy excess (sum dB above floor)
        excess_dB_sum = float(np.maximum(spec_db, 0).sum())
        # SFM = geomean / arithmean. For tonal signal: SFM small. For broadband: SFM larger.
        # Use linear power, not dB. Add small epsilon.
        log_pwr = np.log(np.maximum(spec, 1e-30))
        sfm = float(np.exp(log_pwr.mean()) / max(spec.mean(), 1e-30))
        bar = "#" * min(60, n_above_10)
        print(f"{rb:>3} {rb*RANGE_RES_M:>6.1f} {floor:>10.2e} {n_above_10:>7} "
              f"{n_above_20:>7} {peak_db:>7.1f} {peak_freq:>7.0f} "
              f"{excess_dB_sum:>7.0f} {sfm:>7.3f} {bar}")


def main():
    # Hover frames in airborne1
    print("##" * 50)
    print("HOVER WINDOW (drone at rb~20, beyond CFAR range)")
    print("##" * 50)
    range_bins = list(range(4, 80, 2))   # every 2 bins from 10m to 200m

    for start, label in [(600, "AIRBORNE t=43.3s HOVER"),
                          (800, "AIRBORNE t=57.7s HOVER"),
                          (1000, "AIRBORNE t=72.1s HOVER"),
                          (1200, "AIRBORNE t=86.6s HOVER")]:
        pwr = integrate_pwr_per_va(AIRBORNE, start, 6)
        metrics_per_rb(pwr, range_bins, label)

    print("\n")
    print("##" * 50)
    print("FLY-AWAY WINDOW (drone moving, has Doppler — chip CFAR also sees this)")
    print("##" * 50)
    for start, label in [(380, "AIRBORNE t=27.4s FLY-AWAY"),
                          (430, "AIRBORNE t=31.0s FLY-AWAY")]:
        pwr = integrate_pwr_per_va(AIRBORNE, start, 6)
        metrics_per_rb(pwr, range_bins, label)

    print("\n")
    print("##" * 50)
    print("BACKGROUND (no drone)")
    print("##" * 50)
    for start, label in [(100, "BG t=7s"),
                          (300, "BG t=21s"),
                          (500, "BG t=36s")]:
        pwr = integrate_pwr_per_va(BG, start, 6)
        metrics_per_rb(pwr, range_bins, label)


if __name__ == "__main__":
    sys.exit(main())

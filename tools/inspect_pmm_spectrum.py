"""Diagnostic: dump the spectrum at rb=2 (drone's expected range bin)
across several frames so we can see whether there's a real harmonic
comb to detect, or just noise.

Run:
    py -3.11 tools/inspect_pmm_spectrum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

from radar_dca.ddma import ddma_unfold

N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)
N_TX = 4
EFF_PRF = PRF_HZ / N_TX  # 7619.5 Hz
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT  # 7.44 Hz/bin

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


def main():
    n_frames = BIN_PATH.stat().st_size // BYTES_PER_FRAME
    print(f"Frames: {n_frames}, eff_PRF={EFF_PRF:.0f} Hz, bin_hz={BIN_HZ:.2f} Hz")
    print()

    for fi in (5, 100, 250, 350, 450):
        print(f"=== FRAME {fi} ===")
        try:
            cube = load_frame(fi)
        except Exception as e:
            print(f"  load failed: {e}"); continue
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)  # MTI

        # DDMA unfold
        virtual = ddma_unfold(rc)  # (n_per_tx, n_range, n_rx, n_tx)
        n_per_tx = virtual.shape[0]

        for rb in (2, 3, 4, 5):
            slow_time_per_va = virtual[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)
            # Apply Hann window
            win = np.hanning(n_per_tx).astype(np.float32)
            windowed = slow_time_per_va * win[:, None]
            spec = np.fft.fft(windowed, n=N_FFT, axis=0)
            pwr = (spec.real ** 2 + spec.imag ** 2)
            pwr_avg = pwr.mean(axis=1)[: N_FFT // 2]

            floor = float(np.median(pwr_avg))
            peak_idx = int(np.argmax(pwr_avg))
            peak_freq = peak_idx * BIN_HZ
            peak_db = 10.0 * np.log10(pwr_avg[peak_idx] / max(floor, 1e-30))

            # Find top 6 peaks
            top6_idx = np.argpartition(pwr_avg, -6)[-6:]
            top6_idx.sort()
            top6 = [(int(i), float(i * BIN_HZ),
                     float(10.0 * np.log10(pwr_avg[i] / max(floor, 1e-30))))
                    for i in top6_idx]

            print(f"  rb={rb} (range {rb*RANGE_RES_M:.1f}m):")
            print(f"    floor power = {floor:.2e}")
            print(f"    peak: bin {peak_idx} (freq {peak_freq:.0f} Hz), {peak_db:.1f} dB above floor")
            print(f"    top 6 peaks (bin / Hz / dB-above-floor):")
            for i, f, db in top6:
                print(f"      bin={i:>3}  f={f:>7.1f} Hz  {db:>5.1f} dB")
        print()


if __name__ == "__main__":
    sys.exit(main())

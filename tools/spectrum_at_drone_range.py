"""For each frame, compute the slow-time spectrum at the drone range bin
and report the top 10 frequency peaks. Lets us see what's ACTUALLY
spinning at 5m, not just what the PMM band-detector says.

Drone reported at ~5m horizontal + 2-3m altitude → slant range ~5.5m.
Range res = 2.64 m/bin. So drone should be in range bin 2 (5.3m) or 3 (7.9m).
Operator also at 5m → both might overlap.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
RANGE_RES_M = 2.638466734211415
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)


def load_frame(path: Path, frame_idx: int) -> np.ndarray:
    off = frame_idx * BYTES_PER_FRAME
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(BYTES_PER_FRAME)
    raw = np.frombuffer(buf, dtype=np.int16)
    real_cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
           .transpose(0, 2, 1)
           .astype(np.float32)
    )
    real_cube -= real_cube.mean(axis=1, keepdims=True)
    return real_cube


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def slow_time_spectrum(range_cube_mti: np.ndarray, range_bin: int) -> np.ndarray:
    """Return |FFT|^2 (dB) for slow-time at one range bin."""
    chirp_x_range = range_cube_mti.sum(axis=2)            # (n_chirps, n_range)
    s = chirp_x_range[:, range_bin]                        # (n_chirps,)
    s = s - s.mean()
    s = s * np.hanning(len(s))
    n_fft = 4096
    X = sfft.fft(s, n=n_fft)
    mag2 = (X.real**2 + X.imag**2)
    return 10 * np.log10(mag2 + 1e-20)


def main():
    bin_path = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin")
    n_frames = bin_path.stat().st_size // BYTES_PER_FRAME
    print(f"Bin: {bin_path.name}, {n_frames} frames")
    print(f"Bin frequency resolution: {PRF_HZ/4096:.2f} Hz/bin")
    print(f"Drone range: 5m (rb 2, range 5.3m). Also checking rb 1, 3 (operator overlap zone).")

    # Sample 5 frames spread through recording
    sample_idxs = [50, 128, 200, 300, 450]
    for fi in sample_idxs:
        try:
            real_cube = load_frame(bin_path, fi)
        except Exception as e:
            print(f"frame {fi}: load failed: {e}"); continue
        rc = stage1_range_fft(real_cube)
        rc -= rc.mean(axis=0, keepdims=True)        # MTI

        for rb in (1, 2, 3):
            spec_db = slow_time_spectrum(rc, rb)
            n_fft = len(spec_db)
            # Only positive frequencies (drop negative half + DC + Nyquist guard)
            half = spec_db[1:n_fft//2]
            # Frequency for each bin: k * PRF/n_fft
            freqs = np.arange(1, n_fft//2) * PRF_HZ / n_fft
            # Top 8 peaks (by magnitude) — exclude DC bin
            top_idx = np.argpartition(half, -8)[-8:]
            top_idx = top_idx[np.argsort(-half[top_idx])]
            # Compute median of band as "noise floor"
            noise_floor = float(np.median(half))
            print(f"\n  Frame {fi} rb={rb} (range {rb*RANGE_RES_M:.1f}m) "
                  f"noise_floor={noise_floor:.1f} dB")
            print(f"    {'freq_Hz':>9} {'mag_dB':>8} {'snr_dB':>7}")
            for ti in top_idx:
                f_hz = freqs[ti]
                m_db = half[ti]
                snr_db = m_db - noise_floor
                print(f"    {f_hz:>9.1f} {m_db:>8.1f} {snr_db:>7.1f}")


if __name__ == "__main__":
    main()

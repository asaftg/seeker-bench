"""Offline range-Doppler + PMM analysis of a captured DCA1000 raw-ADC bin.

Reads the .bin produced by DataPortListener (concatenated frame buffers,
int16 little-endian, RX-major-per-chirp layout). For each sampled frame:
  1. Reshape -> (n_chirps, n_samples, n_rx) real
  2. Per-chirp-per-RX DC subtract + Hann + rfft along samples
  3. MTI: subtract slow-time mean per (range, rx)
  4. Sum coherently across RX -> slow-time grid (n_chirps, n_range)
  5. Range-Doppler magnitude map (slow-time FFT per range bin)
  6. Run scan_range_bins() with a sweep of (band, threshold) configs

Output: prints diagnostic summary to stdout. No plots (text-only).

Used to answer: was the drone signature present in the raw data, and
if yes, what slider config would have made the host PMM fire?
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np

# scipy is in the project; use its FFTs (fast)
import scipy.fft as sfft

# Import the actual production PMM detector so we test EXACTLY what
# the live pipeline runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_dca.pmm_detector import scan_range_bins  # type: ignore


# ── Frame dims from drone-fly meta.yaml ─────────────────────────────
N_CHIRPS = 768
N_RX     = 4
N_SAMPLES = 192
PRF_HZ   = 30478.51264858275
RANGE_RES_M = 2.638466734211415
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2  # int16
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)


def load_frame(path: Path, frame_idx: int) -> np.ndarray:
    """Read frame N from the bin file, return (n_chirps, n_samples, n_rx) float32."""
    off = frame_idx * BYTES_PER_FRAME
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError(f"short read at frame {frame_idx}: got {len(buf)} of {BYTES_PER_FRAME}")
    raw = np.frombuffer(buf, dtype=np.int16)
    real_cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
           .transpose(0, 2, 1)
           .astype(np.float32)
    )
    real_cube -= real_cube.mean(axis=1, keepdims=True)
    return real_cube


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    """Real cube (n_chirps, n_samples, n_rx) -> complex range cube (n_chirps, n_range, n_rx)."""
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def stage2_mti(range_cube: np.ndarray) -> np.ndarray:
    """Per-(range, rx) subtract slow-time mean. Kills static returns.
    Matches dca_pipeline.py:712."""
    mean = range_cube.mean(axis=0, keepdims=True)
    return range_cube - mean


def notch_harmonic_artifact(range_cube: np.ndarray) -> None:
    """In-place: zero range bins around chip-internal LO/ADC harmonics.
    Matches dca_pipeline.py:_notch_harmonic_artifact (lines 781-812).
    period = n_samples/8 = 24. step = period/2 = 12. radius = 5.
    Zeroes bins around 12, 24, 36, 48, 60, 72, 84, 96."""
    n_range = range_cube.shape[1]
    period = N_SAMPLES // 8
    if period <= 0: return
    step = max(period // 2, 1)
    radius = 5
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        range_cube[:, lo:hi, :] = 0


def slow_time_grid(range_cube_mti: np.ndarray) -> np.ndarray:
    """Sum across RX -> (n_range, n_chirps)."""
    chirp_x_range = range_cube_mti.sum(axis=2)            # (n_chirps, n_range)
    return chirp_x_range.T                                # (n_range, n_chirps)


def range_doppler_map(range_cube_mti: np.ndarray) -> np.ndarray:
    """Slow-time FFT per range bin -> log-magnitude (n_range, n_doppler)."""
    chirp_x_range = range_cube_mti.sum(axis=2)            # (n_chirps, n_range)
    x = chirp_x_range.T                                   # (n_range, n_chirps)
    x = x - x.mean(axis=1, keepdims=True)
    x = x * np.hanning(x.shape[1])
    n_fft = 1024
    X = sfft.fft(x, n=n_fft, axis=1, workers=2)
    return np.log10(np.abs(X) + 1e-9) * 20.0              # dB


def summarize_frame(frame_idx: int, real_cube: np.ndarray, apply_notch: bool) -> None:
    print(f"\n== FRAME {frame_idx} (notch={'ON' if apply_notch else 'OFF'}) ==")
    rc = stage1_range_fft(real_cube)
    rc_mti = stage2_mti(rc)
    if apply_notch:
        notch_harmonic_artifact(rc_mti)
    grid = slow_time_grid(rc_mti)        # (n_range, n_chirps)

    n_range, n_chirps = grid.shape
    # Range power profile post-MTI (sum over slow time)
    pwr = (grid.real**2 + grid.imag**2).sum(axis=1)
    top_n = 10
    top_idx = np.argpartition(pwr, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(-pwr[top_idx])]
    print(f"  Top {top_n} range bins post-MTI by power:")
    print(f"  {'rb':>4} {'range_m':>9} {'power':>14}")
    for rb in top_idx:
        print(f"  {rb:>4} {rb*RANGE_RES_M:>9.1f} {pwr[rb]:>14.3e}")

    # PMM scan with progressively looser thresholds + wider bands
    configs = [
        # (band_low, band_high, threshold_db, label)
        ( 50.0,  500.0, 18.0, "ORIGINAL settings (50-500 Hz, 18 dB)"),
        ( 50.0,  500.0, 10.0, "Same band, looser threshold (10 dB)"),
        ( 50.0,  500.0,  5.0, "Same band, very loose (5 dB)"),
        ( 20.0, 3000.0, 18.0, "Wide band, original threshold"),
        ( 20.0, 3000.0, 10.0, "Wide band, loose threshold"),
        ( 20.0, 3000.0,  5.0, "Wide band, very loose"),
        (100.0, 1000.0, 12.0, "DJI-class band (100-1000 Hz, mid threshold)"),
    ]
    print(f"\n  PMM scan results (sweep):")
    for low, high, thr, label in configs:
        try:
            hits = scan_range_bins(
                grid, prf_hz=PRF_HZ,
                band_low_hz=low, band_high_hz=high,
                threshold_db=thr, slow_time_win=N_CHIRPS,
            )
        except Exception as e:
            print(f"    {label}: SCAN FAILED ({e})")
            continue
        if not hits:
            print(f"    {label}: 0 hits")
        else:
            print(f"    {label}: {len(hits)} hits")
            for rb, res in hits[:5]:
                print(f"        rb={rb} range={rb*RANGE_RES_M:.1f}m  "
                      f"blade={res.blade_freq_hz:.1f}Hz  "
                      f"snr={res.band_snr_db:.1f}dB  conf={res.confidence:.2f}")

    # Doppler map: find the strongest non-DC bin per range, see if any
    # range bins have a strong Doppler peak (= moving target, drone or otherwise)
    rdm = range_doppler_map(rc_mti)
    n_range_rdm, n_dop = rdm.shape
    # Doppler axis: bin k corresponds to freq prf*(k - n_dop/2)/n_dop after fftshift,
    # but we kept unshifted. Just track raw bin indices.
    # Find the (range, dop) with max value, ignoring DC (bin 0)
    rdm[:, 0] = -np.inf
    flat_max_idx = np.argmax(rdm)
    rb_max, dop_max = np.unravel_index(flat_max_idx, rdm.shape)
    # Convert dop bin to frequency (signed)
    dop_freq_hz = (dop_max if dop_max <= n_dop / 2 else dop_max - n_dop) * PRF_HZ / n_dop
    print(f"\n  Strongest range-Doppler cell:")
    print(f"    range={rb_max * RANGE_RES_M:.1f}m  doppler={dop_freq_hz:+.1f}Hz  mag={rdm[rb_max, dop_max]:.1f} dB")


def main():
    bin_path = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin")
    if not bin_path.exists():
        print(f"BIN NOT FOUND: {bin_path}")
        return 1
    file_size = bin_path.stat().st_size
    n_frames = file_size // BYTES_PER_FRAME
    print(f"Bin: {bin_path.name}")
    print(f"Size: {file_size:,} bytes")
    print(f"Frames: {n_frames}")
    print(f"Frame layout: ({N_CHIRPS} chirps × {N_RX} rx × {N_SAMPLES} samples) × int16 = {BYTES_PER_FRAME:,} bytes/frame")
    print(f"PRF: {PRF_HZ:.1f} Hz, range res: {RANGE_RES_M:.2f} m")

    # Sample frames spread across the recording
    sample_idxs = [
        20,                         # early
        n_frames // 4,              # 25%
        n_frames // 2,              # mid
        (3 * n_frames) // 4,        # 75%
        n_frames - 20,              # late
    ]
    # Run twice: notch OFF (offline), notch ON (matches live pipeline).
    # If notch ON kills the hits → notch is the bug.
    for idx in sample_idxs:
        try:
            real_cube = load_frame(bin_path, idx)
        except Exception as e:
            print(f"frame {idx}: load failed: {e}")
            continue
        summarize_frame(idx, real_cube.copy(), apply_notch=False)
        summarize_frame(idx, real_cube.copy(), apply_notch=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())

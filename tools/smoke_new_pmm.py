"""Quick smoke test: run the NEW PMM detector against the existing
drone-fly recording and show what it finds per frame.

Run:
    py -3.11 tools/smoke_new_pmm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import scipy.fft as sfft

from radar_dca.pmm_detector import scan_range_bins

# Frame dims from cfg
N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
# Range res = c / (2 * BW_captured) where BW = freqSlope * adcCaptureTime
# = 8.883 MHz/us * (192 / 30 MHz) us = 56.85 MHz
# Range res = 3e8 / (2 * 56.85e6) = 2.638 m/bin
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

BIN_PATH = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-05_21-14-35_radar.bin"
)


def load_frame(idx: int) -> np.ndarray:
    """Returns float32 cube (n_chirps, n_samples, n_rx)."""
    with open(BIN_PATH, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError(f"short read at frame {idx}")
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (
        raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
           .transpose(0, 2, 1)
           .astype(np.float32)
    )
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    """(n_chirps, n_samples, n_rx) -> (n_chirps, n_range, n_rx) complex64."""
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def notch_harmonic_artifact(range_cube: np.ndarray) -> None:
    """Mirrors dca_pipeline._notch_harmonic_artifact: zero range bins around
    chip-internal LO/ADC harmonics. period=24, step=12, radius=5.
    Zeroes bins {7..17, 19..29, 31..41, 43..53, 55..65, 67..77, 79..89, 91..96}.
    """
    n_range = range_cube.shape[1]
    period = N_SAMPLES // 8  # 24
    if period <= 0:
        return
    step = max(period // 2, 1)  # 12
    radius = 5
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        range_cube[:, lo:hi, :] = 0


def main():
    if not BIN_PATH.exists():
        print(f"Bin file not found: {BIN_PATH}")
        return 1
    n_frames = BIN_PATH.stat().st_size // BYTES_PER_FRAME
    print(f"Bin file: {BIN_PATH.name}")
    print(f"Frames: {n_frames}")
    print(f"Drone airborne (per visual): t=14-22s ~= frames 280-440")
    print()

    # Sample frames across the recording
    sample_idxs = [
        5, 30, 60, 100, 150, 200, 250, 300, 350, 400, 450, 500, 510,
    ]
    sample_idxs = [i for i in sample_idxs if i < n_frames]

    print(f"{'frame':>5} {'phase':<13} {'#hits':>5}  hits (rb @ range / blade_Hz / SNR)")
    print("-" * 80)
    for fi in sample_idxs:
        if fi < 280:
            phase = "before"
        elif fi <= 440:
            phase = "AIRBORNE"
        else:
            phase = "after"

        try:
            real_cube = load_frame(fi)
        except Exception as e:
            print(f"{fi:>5}: load failed: {e}")
            continue

        rc = stage1_range_fft(real_cube)
        # MTI: subtract slow-time mean per (range, rx)
        rc -= rc.mean(axis=0, keepdims=True)
        # Apply chip-harmonic-artifact notch (matches live pipeline)
        notch_harmonic_artifact(rc)

        # Try multiple thresholds to find the operating point
        results_by_thr = {}
        for thr_db in (10.0, 6.0, 3.0, 0.0):
            try:
                hits = scan_range_bins(
                    rc,
                    prf_hz=PRF_HZ,
                    f0_min_hz=50.0,    # cover slow-rotor drones too
                    f0_max_hz=2500.0,  # cover FPV blade-pass
                    n_f0_candidates=120,
                    threshold_db=thr_db,
                    range_bin_min=2,   # 5.3m, just past operator at 5m
                )
                results_by_thr[thr_db] = hits
            except Exception as e:
                print(f"{fi:>5}: scan failed at thr={thr_db}: {e}")
                continue

        # Report at default threshold
        hits = results_by_thr.get(10.0, [])
        if hits:
            hits_str = "; ".join(
                f"rb{rb}@{rb*RANGE_RES_M:.1f}m/{r.blade_freq_hz:.0f}Hz/{r.band_snr_db:.1f}dB"
                for rb, r in hits[:5]
            )
            print(f"{fi:>5} {phase:<13} {len(hits):>5}  {hits_str}")
        else:
            # Fallback: show what the detector got at lower thresholds
            counts = "/".join(
                f"thr{int(t)}={len(results_by_thr.get(t, []))}"
                for t in (10.0, 6.0, 3.0, 0.0)
            )
            best_low_thr_hits = results_by_thr.get(0.0, [])
            if best_low_thr_hits:
                rb, r = best_low_thr_hits[0]
                detail = f" top: rb{rb}@{rb*RANGE_RES_M:.1f}m/{r.blade_freq_hz:.0f}Hz/{r.band_snr_db:.1f}dB"
            else:
                detail = ""
            print(f"{fi:>5} {phase:<13} {0:>5}  ({counts}){detail}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Validate the post-fold mmHawkeye signature detector.

Runs MIMOCoherenceDetector (now post-fold) with N=6 integration on:
  1. AIRBORNE1 hover window (drone at rb~20, t=33-100s)
  2. AIRBORNE1 fly-away (drone at close range, t=27-40s)
  3. BACKGROUND (no drone)

Per window: prints all hits with (range_bin, range_m, fold_score,
blade_freq_hz, n_harmonics, az_deg, el_deg).

GOAL:
  - hover: at least one hit per window at rb=18-22 (drone known location)
  - fly-away: hits at rb=10-20 (drone moving)
  - background: zero or near-zero hits
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_dca.mimo_coherence_detector import MIMOCoherenceDetector
from radar_dca.ddma import ddma_unfold

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
RANGE_RES_M = 2.638
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


def to_range_cube_post_mti(path, idx):
    """Mirrors the pipeline: range-FFT, MTI mean-subtract across slow-time."""
    cube = load_frame(path, idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)
    return rc


BASELINE_PATH = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
                     r"\recordings\chip_baseline_airborne1.npz")


def run_window(path, start_frame, n_frames, label,
               threshold_db=15.5, n_integration=14, scan_freq_lo=300.0):
    """Run detector through n_frames, report hits emitted during the run."""
    det = MIMOCoherenceDetector(
        prf_hz=PRF_HZ,
        n_chirps_per_tx=N_CHIRPS // N_TX,
        n_integration_frames=n_integration,
        threshold_db=threshold_db,
        scan_freq_lo_hz=scan_freq_lo,
        scan_freq_hi_hz=3000.0,
        range_bin_min=4,
        ac_thresh=0.20,
    )
    print(f"\n=== {label} (frames {start_frame}-{start_frame+n_frames-1}, "
          f"thr={threshold_db}) ===")
    print(f"{'frame':>5} {'rb':>3} {'r_m':>6} {'rawdB':>6} {'peak_Hz':>7} "
          f"{'chunks':>6} {'az':>5} {'el':>5}")
    total_hits = 0
    for i in range(n_frames):
        rc = to_range_cube_post_mti(path, start_frame + i)
        hits = det.process_frame(rc)
        for rb, res in hits:
            print(f"{start_frame+i:>5} {rb:>3} {rb*RANGE_RES_M:>6.1f} "
                  f"{res.band_snr_db:>6.1f} {res.blade_freq_hz:>6.0f} "
                  f"{res.n_folds:>3} {res.az_deg:>5.1f} {res.el_deg:>5.1f}")
            total_hits += 1
    print(f"  TOTAL hits in {n_frames} frames: {total_hits}")
    return total_hits


def main():
    # User wants this to work — let's validate honestly.
    print("#" * 70)
    print("POST-FOLD mmHawkeye SIGNATURE DETECTOR — VALIDATION")
    print("#" * 70)

    # AIRBORNE1 HOVER. Need >= N+8 frames per window so we get several
    # post-warmup detection opportunities.
    NWIN = 30
    hover_total = 0
    for start in [600, 800, 1000, 1200]:
        hover_total += run_window(AIRBORNE, start, NWIN,
                                   f"AIRBORNE1 HOVER frame={start}")

    # AIRBORNE1 FLY-AWAY (drone moving close range, t=27-40s)
    flyaway_total = 0
    for start in [380, 430]:
        flyaway_total += run_window(AIRBORNE, start, NWIN,
                                     f"AIRBORNE1 FLY-AWAY frame={start}")

    # BACKGROUND (no drone, must be near zero)
    bg_total = 0
    for start in [100, 300, 500]:
        bg_total += run_window(BG, start, NWIN, f"BACKGROUND frame={start}")

    print("\n" + "#" * 70)
    print("SUMMARY (N=14 integration, AC_THR=0.35, MAG_MARGIN=6 dB)")
    print("#" * 70)
    print(f"  HOVER     (4 windows × {NWIN} frames = {4*NWIN} frames): {hover_total} hits")
    print(f"  FLY-AWAY  (2 windows × {NWIN} frames = {2*NWIN} frames): {flyaway_total} hits")
    print(f"  BG        (3 windows × {NWIN} frames = {3*NWIN} frames): {bg_total} hits")
    if hover_total > 0 and bg_total <= 3:
        print("  >>> PASS: hover detected, background clean")
    elif hover_total == 0:
        print("  >>> FAIL: hover not detected (lower thresholds or raise N)")
    elif bg_total > 3:
        print(f"  >>> FAIL: too many background hits ({bg_total}) (raise thresholds)")


if __name__ == "__main__":
    sys.exit(main())

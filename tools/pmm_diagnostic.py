"""Instrumented PMM diagnostic for the 2026-05-05 21:14:35 DJI FPV recording.

Goal: explain WHY scan_range_bins returned 0 hits with the user's GUI
config (slow_time_win=512, band=20-3000 Hz, threshold=10 dB) on a
38-second recording where a drone was flown.

Strategy: replicate scan_range_bins step-by-step at sampled frames and
print every intermediate value so we can pin the failure to a specific
gate (power pre-filter / global-floor body gate / sideband threshold).

Run:
    py -3.11 tools/pmm_diagnostic.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_dca.pmm_detector import scan_range_bins  # type: ignore

# ── Frame dims ───────────────────────────────────────────────────────
N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
RANGE_RES_M = 2.638466734211415
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

# ── User's GUI config (the failure case) ─────────────────────────────
SLOW_TIME_WIN = 512
BAND_LOW_HZ = 20.0
BAND_HIGH_HZ = 3000.0
USER_THR_DB = 10.0
HANN_GUARD = 8       # matches scan_range_bins
GLOBAL_FLOOR_DB = 12.0  # matches scan_range_bins body gate

# Range bins to inspect (indices 1..6, ~2.6m through ~15.8m)
INSPECT_BINS = [1, 2, 3, 4, 5, 6]

BIN_PATH = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-05_21-14-35_radar.bin")


def load_frame(path: Path, frame_idx: int) -> np.ndarray:
    off = frame_idx * BYTES_PER_FRAME
    with open(path, "rb") as f:
        f.seek(off)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError(f"short read at frame {frame_idx}")
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


def stage2_mti(range_cube: np.ndarray) -> np.ndarray:
    mean = range_cube.mean(axis=0, keepdims=True)
    return range_cube - mean


def slow_time_grid_truncated(range_cube_mti: np.ndarray, n_use: int) -> np.ndarray:
    """Sum across RX, then truncate slow-time to n_use chirps -> (n_range, n_use)."""
    chirp_x_range = range_cube_mti.sum(axis=2)            # (n_chirps, n_range)
    grid = chirp_x_range.T                                # (n_range, n_chirps)
    return grid[:, :n_use]


def manual_pmm_for_bin(slow_time: np.ndarray, prf_hz: float,
                       band_low_hz: float, band_high_hz: float,
                       thr_db: float, *, label_prefix: str = "") -> dict:
    """Replicate scan_range_bins() per-bin logic and return EVERY intermediate."""
    n = slow_time.shape[0]
    s = slow_time - slow_time.mean()
    s = s * np.hanning(n)
    n_fft = 4 * (1 << int(np.ceil(np.log2(n))))
    X = sfft.fft(s, n=n_fft, workers=2)
    mag2 = (X.real * X.real + X.imag * X.imag).astype(np.float64)
    bin_hz = prf_hz / n_fft

    body_bin = int(np.argmax(mag2))
    body_pwr = float(mag2[body_bin])
    global_floor = float(np.median(mag2))
    body_db_over_floor = 10.0 * np.log10(body_pwr / max(global_floor, 1e-30))

    delta_min = max(HANN_GUARD, int(round(band_low_hz / bin_hz)))
    delta_max = min(n_fft // 2 - 1, int(round(band_high_hz / bin_hz)))

    deltas = np.arange(delta_min, delta_max + 1)
    plus_idx = (body_bin + deltas) % n_fft
    minus_idx = (body_bin - deltas) % n_fft
    inband = np.zeros(n_fft, dtype=bool)
    inband[plus_idx] = True
    inband[minus_idx] = True
    inband[body_bin] = True
    noise_floor = float(np.median(mag2[~inband]))
    if noise_floor <= 0:
        noise_floor = float(np.finfo(np.float64).tiny)

    plus_pwr = mag2[plus_idx]
    minus_pwr = mag2[minus_idx]
    sb_pwr = np.minimum(plus_pwr, minus_pwr)
    score_db_arr = 10.0 * np.log10(np.maximum(sb_pwr, 1e-30) / noise_floor)
    best_i = int(np.argmax(score_db_arr))
    best_score_db = float(score_db_arr[best_i])
    best_delta = int(deltas[best_i])
    best_blade_hz = best_delta * bin_hz

    # Also: top-3 inband spectral peaks (regardless of symmetry)
    inband_only = np.zeros_like(mag2)
    inband_only[plus_idx] = mag2[plus_idx]
    inband_only[minus_idx] = mag2[minus_idx]
    # Strongest single inband bins (could be asymmetric tones)
    top3_idx = np.argpartition(inband_only, -3)[-3:]
    top3_idx = top3_idx[np.argsort(-inband_only[top3_idx])]
    top3 = []
    for ti in top3_idx:
        offset_bins = ti - body_bin
        # Wrap to nearest signed offset
        if offset_bins > n_fft / 2:
            offset_bins -= n_fft
        elif offset_bins < -n_fft / 2:
            offset_bins += n_fft
        f_hz = abs(offset_bins) * bin_hz
        side = "+" if offset_bins > 0 else "-"
        snr_db = 10.0 * np.log10(max(inband_only[ti], 1e-30) / noise_floor)
        top3.append((side, f_hz, snr_db))

    # Symmetry residual at the BEST delta: |+pwr - -pwr| in dB
    plus_db = 10.0 * np.log10(max(plus_pwr[best_i], 1e-30) / noise_floor)
    minus_db = 10.0 * np.log10(max(minus_pwr[best_i], 1e-30) / noise_floor)
    sym_residual_db = abs(plus_db - minus_db)

    # Decide which gate fails
    if global_floor <= 0 or body_db_over_floor < GLOBAL_FLOOR_DB:
        gate = f"GLOBAL_FLOOR (body {body_db_over_floor:.1f} dB < {GLOBAL_FLOOR_DB:.0f} req)"
    elif best_score_db < thr_db:
        gate = f"SYMM_THRESHOLD (score {best_score_db:.1f} dB < {thr_db:.1f} req)"
    else:
        gate = "PASSED"

    return dict(
        n_fft=n_fft, bin_hz=bin_hz,
        body_bin=body_bin, body_pwr=body_pwr, global_floor=global_floor,
        body_db_over_floor=body_db_over_floor,
        delta_min=delta_min, delta_max=delta_max,
        noise_floor=noise_floor,
        best_score_db=best_score_db, best_delta=best_delta, best_blade_hz=best_blade_hz,
        plus_db=plus_db, minus_db=minus_db, sym_residual_db=sym_residual_db,
        top3=top3, gate=gate,
    )


def diagnose_frame(frame_idx: int, real_cube: np.ndarray) -> None:
    print(f"\n========== FRAME {frame_idx} ==========")
    rc = stage1_range_fft(real_cube)
    rc_mti = stage2_mti(rc)
    grid = slow_time_grid_truncated(rc_mti, SLOW_TIME_WIN)   # (n_range, 512)
    n_range, n_use = grid.shape

    # Pre-filter context (matches scan_range_bins step 1).
    pwr_per_bin = (grid.real * grid.real + grid.imag * grid.imag).sum(axis=1)
    median_pwr = float(np.median(pwr_per_bin))
    above4 = pwr_per_bin > (median_pwr * 4.0)
    n_candidates = int(above4.sum())
    print(f"  pre-filter: median_pwr={median_pwr:.2e}, "
          f"{n_candidates}/{n_range} bins pass >4x-median gate")

    print(f"  bin power for inspect-bins (1..6) and pre-filter status:")
    print(f"    {'rb':>3} {'range_m':>7} {'pwr':>11} {'pwr/med':>8} {'pre':>5}")
    for rb in INSPECT_BINS:
        ratio = pwr_per_bin[rb] / max(median_pwr, 1e-30)
        passes = "YES" if above4[rb] else "no"
        print(f"    {rb:>3} {rb*RANGE_RES_M:>7.1f} {pwr_per_bin[rb]:>11.2e} "
              f"{ratio:>8.2f} {passes:>5}")

    # PARITY CHECK: call the production scan_range_bins on this frame's grid
    # with the user's exact GUI config. If it returns 0 hits but our manual
    # replay says rb=1 should fire, the gap is in scan_range_bins itself
    # (e.g. truncated grid, batch FFT diff).
    # Note: 'grid' is already truncated to SLOW_TIME_WIN. Pass slow_time_win=None
    # so scan_range_bins doesn't double-truncate. This mirrors what the live
    # pipeline sees if it pre-truncates, OR — we ALSO test the case where the
    # full (n_range, 768) grid is passed and the slider truncates inside.
    prod_hits_pre = scan_range_bins(
        grid, prf_hz=PRF_HZ,
        band_low_hz=BAND_LOW_HZ, band_high_hz=BAND_HIGH_HZ,
        threshold_db=USER_THR_DB, slow_time_win=None,
    )
    print(f"  PROD scan(grid_already512, win=None, thr={USER_THR_DB:.0f}dB): "
          f"{len(prod_hits_pre)} hits"
          + ("  " + ", ".join(f"rb{rb}={r.band_snr_db:.1f}dB@{r.blade_freq_hz:.0f}Hz"
                              for rb, r in prod_hits_pre[:5]) if prod_hits_pre else ""))

    # Also call with the FULL 768-chirp grid + slow_time_win=512 (matches GUI exactly)
    full_grid = slow_time_grid_truncated(rc_mti, N_CHIRPS)  # full 768
    prod_hits_full = scan_range_bins(
        full_grid, prf_hz=PRF_HZ,
        band_low_hz=BAND_LOW_HZ, band_high_hz=BAND_HIGH_HZ,
        threshold_db=USER_THR_DB, slow_time_win=SLOW_TIME_WIN,
    )
    print(f"  PROD scan(full_grid_768, win={SLOW_TIME_WIN}, thr={USER_THR_DB:.0f}dB): "
          f"{len(prod_hits_full)} hits"
          + ("  " + ", ".join(f"rb{rb}={r.band_snr_db:.1f}dB@{r.blade_freq_hz:.0f}Hz"
                              for rb, r in prod_hits_full[:5]) if prod_hits_full else ""))

    # Per-bin manual replay for bins 1..6, at user's threshold + lowered ones
    for thr_db in (USER_THR_DB, 5.0, 0.0):
        print(f"\n  -- per-bin replay @ band [{BAND_LOW_HZ:.0f}-{BAND_HIGH_HZ:.0f}] Hz, "
              f"thr={thr_db:.0f} dB --")
        print(f"    {'rb':>3} {'body_db':>8} {'best_dB':>8} {'blade_Hz':>9} "
              f"{'+dB':>6} {'-dB':>6} {'symRes':>7} gate")
        for rb in INSPECT_BINS:
            d = manual_pmm_for_bin(grid[rb], PRF_HZ,
                                   BAND_LOW_HZ, BAND_HIGH_HZ, thr_db)
            print(f"    {rb:>3} {d['body_db_over_floor']:>8.1f} "
                  f"{d['best_score_db']:>8.1f} {d['best_blade_hz']:>9.1f} "
                  f"{d['plus_db']:>6.1f} {d['minus_db']:>6.1f} "
                  f"{d['sym_residual_db']:>7.1f} {d['gate']}")

    # Top-3 inband peaks at thr=user, only for bins 1..3 (drone candidates)
    print(f"\n  -- top-3 inband peaks (any symmetry) at bins 1..3 --")
    for rb in (1, 2, 3):
        d = manual_pmm_for_bin(grid[rb], PRF_HZ,
                               BAND_LOW_HZ, BAND_HIGH_HZ, USER_THR_DB)
        peaks_str = "  ".join(f"{s}{f:6.1f}Hz/{db:5.1f}dB" for s, f, db in d['top3'])
        print(f"    rb={rb}: body_bin={d['body_bin']} (dop {d['body_bin']*d['bin_hz']:+.1f}Hz)  {peaks_str}")


def comparison_row(real_early: np.ndarray, real_mid: np.ndarray) -> None:
    print("\n========== COMPARISON: frame 5 (early) vs frame 250 (mid) ==========")
    grids = []
    for label, rc_real in (("early(5)", real_early), ("mid(250)", real_mid)):
        rc = stage1_range_fft(rc_real)
        rc = stage2_mti(rc)
        grids.append((label, slow_time_grid_truncated(rc, SLOW_TIME_WIN)))

    print(f"  {'rb':>3} {'range_m':>7} | "
          f"{'early body_dB':>13} {'early best_dB':>13} {'early blade_Hz':>14} | "
          f"{'mid body_dB':>11} {'mid best_dB':>11} {'mid blade_Hz':>12}")
    for rb in INSPECT_BINS:
        de = manual_pmm_for_bin(grids[0][1][rb], PRF_HZ,
                                BAND_LOW_HZ, BAND_HIGH_HZ, USER_THR_DB)
        dm = manual_pmm_for_bin(grids[1][1][rb], PRF_HZ,
                                BAND_LOW_HZ, BAND_HIGH_HZ, USER_THR_DB)
        print(f"  {rb:>3} {rb*RANGE_RES_M:>7.1f} | "
              f"{de['body_db_over_floor']:>13.1f} {de['best_score_db']:>13.1f} "
              f"{de['best_blade_hz']:>14.1f} | "
              f"{dm['body_db_over_floor']:>11.1f} {dm['best_score_db']:>11.1f} "
              f"{dm['best_blade_hz']:>12.1f}")


def main():
    if not BIN_PATH.exists():
        print(f"BIN MISSING: {BIN_PATH}")
        return 1
    file_size = BIN_PATH.stat().st_size
    n_frames = file_size // BYTES_PER_FRAME
    print(f"Bin: {BIN_PATH.name}")
    print(f"Size: {file_size:,} bytes  |  Frames: {n_frames}")
    print(f"PRF={PRF_HZ:.1f} Hz  range_res={RANGE_RES_M:.3f} m")
    print(f"Config: slow_time_win={SLOW_TIME_WIN}  band={BAND_LOW_HZ:.0f}-{BAND_HIGH_HZ:.0f}Hz  "
          f"thr={USER_THR_DB:.0f}dB")
    bin_hz_512 = PRF_HZ / (4 * (1 << int(np.ceil(np.log2(SLOW_TIME_WIN)))))
    print(f"FFT bin_hz at slow_time_win={SLOW_TIME_WIN}: {bin_hz_512:.2f} Hz/bin  "
          f"(HANN_GUARD floor = {HANN_GUARD*bin_hz_512:.1f} Hz)")

    sample_idxs = [5, 30, 60, 100, 150, 200, 250, 300, 350, 400, 450]
    sample_idxs = [i for i in sample_idxs if i < n_frames]

    cubes = {}
    for idx in sample_idxs:
        try:
            cubes[idx] = load_frame(BIN_PATH, idx)
        except Exception as e:
            print(f"  frame {idx}: load failed: {e}")

    for idx in sample_idxs:
        if idx in cubes:
            diagnose_frame(idx, cubes[idx])

    if 5 in cubes and 250 in cubes:
        comparison_row(cubes[5], cubes[250])

    return 0


if __name__ == "__main__":
    sys.exit(main())

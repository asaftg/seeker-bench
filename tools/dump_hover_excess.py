"""Dump the actual spread_local_excess[freq, range] map at one hover
frame, focusing on the drone's known range (rb=18-22) and the chip
artifact range (rb=24).

Question: does the drone produce a comb (multiple freq bins lit
at multiples of some f0)? Or just a single peak at 1317 Hz?

If single peak: the post-fold idea is wrong for our drone — there's
no comb to fold. We need a different signature.
If comb: post-fold should work but f0 grid needs tuning.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.fft as sfft

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from radar_dca.ddma import ddma_unfold
from radar_dca.mimo_coherence_detector import MIMOCoherenceDetector

N_CHIRPS = 768; N_RX = 4; N_SAMPLES = 192
PRF_HZ = 30478.51264858275
N_TX = 4
EFF_PRF = PRF_HZ / N_TX
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

AIRBORNE = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
                r"\recordings\seeker_2026-05-06_12-58-59_radar.bin")


def load_frame(idx):
    with open(AIRBORNE, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
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


def feed_n_get_state(start_frame, n=6):
    """Build detector, feed N frames, return its (excess, spread_db) map
    at all candidate range bins."""
    det = MIMOCoherenceDetector(
        prf_hz=PRF_HZ,
        n_chirps_per_tx=N_CHIRPS // N_TX,
        n_integration_frames=n,
        threshold_db=5.0,
        scan_freq_lo_hz=50.0,
        scan_freq_hi_hz=3000.0,
        range_bin_min=4,
        max_candidate_bins=80,
    )

    # Run all but skip the last so we can intercept R_sum
    for i in range(n - 1):
        cube = load_frame(start_frame + i)
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        det.process_frame(rc)

    # On the last frame, do the same operations but capture intermediates
    cube = load_frame(start_frame + n - 1)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)
    # Mirror process_frame internals
    det.process_frame(rc)

    # Now manually rebuild the (range_bin, freq, excess) for ALL bins
    # using the same range_cube.
    rx_pwr = (rc.real * rc.real + rc.imag * rc.imag).sum(axis=(0, 2))
    median_pwr = float(np.median(rx_pwr))
    n_range = rx_pwr.size
    above = rx_pwr > (median_pwr * 4.0)
    cand = np.flatnonzero(above)
    cand = cand[(cand >= 4) & (cand <= n_range - 1)]
    if cand.size > 80:
        top = np.argpartition(rx_pwr[cand], -80)[-80:]
        cand = np.sort(cand[top])
    return det, cand, rc


def dump_spectrum(det, start_frame, target_rbs, label, n=6):
    """Print the spread_local_excess at target range bins around their peak."""
    # Need to access internals — we just rebuild from scratch
    # (simplest path).
    from radar_dca.ddma import ddma_unfold

    cubes_rc = []
    for i in range(n):
        cube = load_frame(start_frame + i)
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        cubes_rc.append(rc)

    # Common candidate set: union of bins above floor at last frame
    rc_last = cubes_rc[-1]
    rx_pwr = (rc_last.real * rc_last.real + rc_last.imag * rc_last.imag).sum(axis=(0, 2))
    median_pwr = float(np.median(rx_pwr))
    cand_pool = np.flatnonzero(rx_pwr > median_pwr * 4.0)
    cand_pool = cand_pool[cand_pool >= 4]
    # Force include target bins
    cand = np.unique(np.concatenate([cand_pool, np.array(target_rbs)]))
    cand = np.sort(cand)

    n_va = 16
    win = np.hanning(N_CHIRPS // N_TX).astype(np.float32)
    bin_lo = max(1, int(round(50.0 / BIN_HZ)))
    bin_hi = min(N_FFT // 2 - 1, int(round(3000.0 / BIN_HZ)))
    n_band = bin_hi - bin_lo + 1

    R_sum = None
    for rc in cubes_rc:
        virtual = ddma_unfold(rc)
        n_per_tx = virtual.shape[0]
        slow = virtual[:, cand, :, :].reshape(n_per_tx, cand.size, n_va)
        spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
        spec_band = spec[bin_lo:bin_hi+1, :, :]
        R_frame = np.einsum("fci,fcj->fcij", spec_band, spec_band.conj()).astype(np.complex64)
        R_sum = R_frame.copy() if R_sum is None else R_sum + R_frame

    eigvals = np.linalg.eigvalsh(R_sum)
    lam_max = eigvals[..., -1]
    other_mean = eigvals[..., :-1].mean(axis=-1)
    spread = np.where(other_mean > 0, lam_max / np.maximum(other_mean, 1e-30), 1.0)
    spread_db = 10.0 * np.log10(np.maximum(spread, 1e-12))

    # Apply spatial CFAR (fast vectorized version: rolling window)
    # For diagnostic purposes, use a simpler approximation:
    # local mean = mean over freq window, excluding guard.
    from radar_dca.mimo_coherence_detector import MIMOCoherenceDetector
    _det = MIMOCoherenceDetector(prf_hz=PRF_HZ, n_chirps_per_tx=N_CHIRPS//N_TX)
    excess = _det._spatial_cfar(spread_db)

    print(f"\n=== {label} (frames {start_frame}-{start_frame+n-1}) ===")
    print(f"  cand bins: {cand.tolist()[:15]}...")
    for rb in target_rbs:
        if rb not in cand:
            print(f"  rb={rb} NOT in candidate set")
            continue
        ci = int(np.where(cand == rb)[0][0])
        spec_excess = excess[:, ci]
        spec_db = spread_db[:, ci]
        # Find top-10 excess freqs
        top_idx = np.argsort(-spec_excess)[:10]
        top_idx_sorted = sorted(top_idx)
        print(f"\n  rb={rb} (range={rb*RANGE_RES_M:.1f}m), top-10 cells by EXCESS:")
        print(f"    {'freq_Hz':>8} {'excess':>7} {'raw_dB':>7}")
        for fi in top_idx_sorted:
            freq_hz = (bin_lo + fi) * BIN_HZ
            print(f"    {freq_hz:>8.0f} {spec_excess[fi]:>7.2f} {spec_db[fi]:>7.2f}")
        # Also count cells with excess > 1 dB
        n_lit_1 = int((spec_excess > 1.0).sum())
        n_lit_2 = int((spec_excess > 2.0).sum())
        n_lit_3 = int((spec_excess > 3.0).sum())
        print(f"    cells > 1 dB excess: {n_lit_1}  > 2 dB: {n_lit_2}  > 3 dB: {n_lit_3}")


def main():
    # Hover frames
    for start in [1200]:
        dump_spectrum(None, start, [20, 24], f"HOVER frame={start}")
    # BACKGROUND chip artifact range bin — must understand why chunks=5
    global AIRBORNE
    BG = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
              r"\recordings\seeker_2026-05-06_12-54-23_radar.bin")
    AIRBORNE = BG
    for start in [100, 300]:
        dump_spectrum(None, start, [12, 24, 36, 48, 60, 72], f"BG frame={start}")


if __name__ == "__main__":
    sys.exit(main())

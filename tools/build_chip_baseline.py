"""Build a per-chip-cfg coherence baseline from a NO-DRONE recording.

Method:
  - Run the same DDMA-unfold + per-VA FFT + N-frame R_sum + eigenvalue
    spread that the live detector uses
  - Average spread_db[freq, range] across many frames (no drone present)
  - Save to a .npy file the detector loads and SUBTRACTS at runtime

This characterizes the chip's intrinsic coherence pattern (chip-internal
spurs at every 12th range bin, low-freq MTI residual, etc.) so the
runtime detector can subtract it and only see TRUE excess from real
drones / vehicles / vibrating objects.

USAGE:
    py -3.11 tools/build_chip_baseline.py <bg_bin_path> <out_baseline.npy>

The baseline file is per (chip cfg + scene). Re-record when:
  - Chip cfg changes (PRF / chirps / MIMO mode)
  - Sensor moves to a fundamentally different scene
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
N_VA = 16
RANGE_BIN_MIN = 4
SCAN_FREQ_LO_HZ = 50.0
SCAN_FREQ_HI_HZ = 3000.0
N_INT = 6  # match the runtime detector
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)


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


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    bg_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    if not bg_path.exists():
        print(f"ERROR: bg recording not found: {bg_path}")
        return 1

    n_frames_total = bg_path.stat().st_size // BYTES_PER_FRAME
    print(f"Background recording has {n_frames_total} frames")

    bin_lo = max(1, int(round(SCAN_FREQ_LO_HZ / BIN_HZ)))
    bin_hi = min(N_FFT // 2 - 1, int(round(SCAN_FREQ_HI_HZ / BIN_HZ)))
    n_band = bin_hi - bin_lo + 1
    win = np.hanning(N_CHIRPS // N_TX).astype(np.float32)

    # We need n_range from the first frame to size the cand axis
    cube0 = load_frame(bg_path, 0)
    rc0 = stage1(cube0)
    n_range = rc0.shape[1]
    cand = np.arange(RANGE_BIN_MIN, n_range, dtype=np.int64)
    n_cand = cand.size

    print(f"Band: {bin_lo}..{bin_hi} ({n_band} bins, {SCAN_FREQ_LO_HZ}..{SCAN_FREQ_HI_HZ} Hz)")
    print(f"Range bins: {RANGE_BIN_MIN}..{n_range-1} ({n_cand} cands)")
    print(f"Integration: N={N_INT} frames")

    # Sliding window of N=6 frames; emit one spread_db[freq, cand] per
    # window-end. Capture both MEAN and STD so the runtime can apply a
    # z-score test (cell is anomalous if spread_db > mean + N·std).
    sum_spread_db = np.zeros((n_band, n_cand), dtype=np.float64)
    sum_spread_db_sq = np.zeros((n_band, n_cand), dtype=np.float64)
    n_emitted = 0

    spec_ring: list = []

    # Skip first 6 frames (warm-up); use rest to build baseline
    for idx in range(0, n_frames_total):
        try:
            cube = load_frame(bg_path, idx)
        except EOFError:
            break
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        virtual = ddma_unfold(rc)
        n_per_tx = virtual.shape[0]
        slow = virtual[:, cand, :, :].reshape(n_per_tx, n_cand, N_VA)
        spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
        spec_band = spec[bin_lo:bin_hi+1, :, :].astype(np.complex64)

        spec_ring.append(spec_band)
        if len(spec_ring) > N_INT:
            spec_ring.pop(0)

        if len(spec_ring) == N_INT:
            stacked = np.stack(spec_ring, axis=2)
            R_sum = np.einsum(
                "fcti,fctj->fcij", stacked, stacked.conj()
            ).astype(np.complex64)
            eigvals = np.linalg.eigvalsh(R_sum)
            lam_max = eigvals[..., -1]
            other_mean = eigvals[..., :-1].mean(axis=-1)
            spread = np.where(other_mean > 0,
                              lam_max / np.maximum(other_mean, 1e-30), 1.0)
            spread_db = 10.0 * np.log10(np.maximum(spread, 1e-12))
            sum_spread_db += spread_db
            sum_spread_db_sq += spread_db * spread_db
            n_emitted += 1
            if (n_emitted % 50) == 0:
                print(f"  processed window {n_emitted}, frame {idx}")

    if n_emitted == 0:
        print("ERROR: no windows produced")
        return 1

    baseline = (sum_spread_db / n_emitted).astype(np.float32)
    # Variance from sum of squares; std clamped to a sane minimum
    # (0.5 dB) to avoid divide-by-tiny in z-score.
    var = (sum_spread_db_sq / n_emitted) - baseline.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
    std = np.maximum(std, np.float32(0.5))
    # Save with metadata
    np.savez(out_path,
             baseline_db=baseline,
             baseline_std=std,
             scan_freq_lo_hz=SCAN_FREQ_LO_HZ,
             scan_freq_hi_hz=SCAN_FREQ_HI_HZ,
             prf_hz=PRF_HZ,
             n_chirps_per_tx=N_CHIRPS // N_TX,
             range_bin_min=RANGE_BIN_MIN,
             n_range=n_range,
             n_int=N_INT,
             n_windows_averaged=n_emitted,
             source_recording=str(bg_path))
    print(f"\nSaved baseline ({n_band}x{n_cand}) averaged over {n_emitted} windows")
    print(f"  out: {out_path}")
    print(f"  baseline mean: {baseline.mean():.2f} dB,  max: {baseline.max():.2f} dB")
    print(f"  std        mean: {std.mean():.2f} dB,    max: {std.max():.2f} dB")
    # Show the most-elevated cells (chip artifact fingerprints)
    flat_idx = np.argsort(-baseline.flatten())[:10]
    print("  top-10 baseline cells (chip artifact fingerprints):")
    print("    freq_Hz  range_bin  baseline_dB")
    for fi in flat_idx:
        fb, cb = np.unravel_index(fi, baseline.shape)
        freq_hz = (bin_lo + fb) * BIN_HZ
        rb = cand[cb]
        print(f"    {freq_hz:>8.0f} {rb:>9} {baseline[fb,cb]:>12.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

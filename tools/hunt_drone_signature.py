"""Hunt the drone signature in the 'drone fly' recording.

Strategy: compute several spectral features per (frame, range_bin) and
compare ground-truth airborne frames vs ground-truth on-ground frames.
ANY feature that distinguishes the two is a candidate detector statistic.

Features per (frame, range_bin) on the post-MTI, DDMA-unfolded, RX-averaged
power spectrum:
  1. Total in-band energy (excluding DC and chip artifact bins)
  2. Spectral entropy (broadband signal -> high entropy)
  3. Variance / kurtosis of the spectrum
  4. Number of bins above N x median
  5. Peak-to-median ratio
  6. Slow-time signal variance (raw, no FFT)
  7. Spectral roll-off frequency

Ground truth (from earlier visual analysis):
  Frames 0-187: drone on ground (or pre-takeoff)
  Frames 188-295: drone airborne (climbing/hovering, ~14-22s)
  Frames 296+: drone descending or landed

Run:
    py -3.11 tools/hunt_drone_signature.py
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
N_TX = 4
EFF_PRF = PRF_HZ / N_TX  # 7619.5
N_FFT = 1024
BIN_HZ = EFF_PRF / N_FFT  # 7.44 Hz
RANGE_RES_M = 2.638
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

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


def notch(rc: np.ndarray) -> None:
    n_range = rc.shape[1]
    period = N_SAMPLES // 8
    step = max(period // 2, 1)
    radius = 5
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        rc[:, lo:hi, :] = 0


def features_at_rb(virtual: np.ndarray, rb: int) -> dict:
    """Compute several spectral features at one range bin."""
    n_per_tx = virtual.shape[0]
    slow_per_va = virtual[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)

    # raw slow-time variance (across chirps, averaged across VAs)
    raw_var = float(np.mean(np.var(slow_per_va, axis=0)))

    # Hann + FFT
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow_per_va * win[:, None], n=N_FFT, axis=0)
    pwr = (spec.real ** 2 + spec.imag ** 2).mean(axis=1).astype(np.float64)
    pwr_pos = pwr[: N_FFT // 2]

    # Mask out DC + chip-artifact bin (PRF_eff/3 ~ bin 341)
    keep = np.ones(len(pwr_pos), dtype=bool)
    keep[:8] = False     # DC + low-freq leakage
    keep[336:346] = False  # chip artifact at PRF_eff/3
    keep[166:176] = False  # PRF_eff/6 just in case
    pwr_in = pwr_pos[keep]

    floor = float(np.median(pwr_in))
    floor = max(floor, 1e-30)

    # Total energy in band
    total_energy = float(pwr_in.sum())

    # Peak-to-median
    peak = float(pwr_in.max())
    p2m_db = 10.0 * np.log10(peak / floor)

    # Bins above 2x median (broadband activity indicator)
    n_above_2x = int((pwr_in > 2.0 * floor).sum())

    # Spectral entropy (Shannon entropy of normalized spectrum)
    p_norm = pwr_in / pwr_in.sum()
    p_norm = p_norm[p_norm > 0]
    entropy = float(-(p_norm * np.log(p_norm)).sum())

    # Power in 100-2000 Hz sub-band (drone blade signature region)
    sub_lo = int(100 / BIN_HZ)
    sub_hi = int(2000 / BIN_HZ)
    sub_pwr = float(pwr_pos[sub_lo:sub_hi].sum())

    # Doppler-spread: weighted std-dev of frequency
    freqs = np.arange(len(pwr_in))[: len(pwr_in)] * BIN_HZ
    # Re-derive frequency for masked bins; just use the masked array for spread
    p_for_spread = pwr_in / pwr_in.sum()
    freqs_in = np.where(keep)[0] * BIN_HZ
    mean_freq = float((freqs_in * p_for_spread).sum())
    spread = float(np.sqrt(((freqs_in - mean_freq) ** 2 * p_for_spread).sum()))

    return dict(
        raw_var=raw_var,
        total_energy=total_energy,
        peak_to_median_db=p2m_db,
        n_above_2x=n_above_2x,
        entropy=entropy,
        sub_band_pwr=sub_pwr,
        doppler_spread_hz=spread,
        floor=floor,
    )


def main():
    n_frames = BIN_PATH.stat().st_size // BYTES_PER_FRAME
    print(f"Frames: {n_frames}")
    print(f"eff_PRF={EFF_PRF:.0f} Hz, bin_hz={BIN_HZ:.2f} Hz, n_fft={N_FFT}")
    print()

    # Sample every 10th frame across the recording
    sample_step = 10
    sample_idxs = list(range(0, n_frames, sample_step))

    # Range bins to inspect (drone could be at 5.3-13m slant range)
    inspect_bins = [2, 3, 4, 5]

    # Ground truth labels (per visual EO/thermal analysis)
    def label(fi: int) -> str:
        # Frame rate ~13.4 fps. drone airborne t=14-22s -> frames ~187-295
        # Plus descent t=22-32 -> frames 295-430
        if fi < 187:
            return "ground"
        elif fi <= 295:
            return "AIRBORNE"
        elif fi <= 430:
            return "descend"
        else:
            return "after"

    # Collect features per (frame, rb)
    print(f"{'frame':>5} {'phase':<8} ", end='')
    for rb in inspect_bins:
        print(f"  rb{rb}_{'subPwr':>9}", end='')
    print()
    print("-" * (15 + 18 * len(inspect_bins)))

    all_data = {rb: {"ground": [], "AIRBORNE": [], "descend": [], "after": []} for rb in inspect_bins}

    for fi in sample_idxs:
        try:
            cube = load_frame(fi)
        except Exception:
            continue
        rc = stage1(cube)
        rc -= rc.mean(axis=0, keepdims=True)
        notch(rc)
        virtual = ddma_unfold(rc)

        ph = label(fi)
        line = f"{fi:>5} {ph:<8} "
        for rb in inspect_bins:
            feat = features_at_rb(virtual, rb)
            line += f"  rb{rb}_{feat['sub_band_pwr']:>9.2e}"
            all_data[rb][ph].append(feat)
        print(line)

    print()
    print("=" * 80)
    print("FEATURE COMPARISON: airborne vs ground (median across frames)")
    print("=" * 80)

    feature_keys = ["raw_var", "total_energy", "peak_to_median_db",
                    "n_above_2x", "entropy", "sub_band_pwr",
                    "doppler_spread_hz", "floor"]

    for rb in inspect_bins:
        ground = all_data[rb]["ground"]
        airborne = all_data[rb]["AIRBORNE"]
        if not ground or not airborne:
            print(f"\nrb={rb}: insufficient samples (ground={len(ground)}, airborne={len(airborne)})")
            continue
        print(f"\nrb={rb} (range {rb*RANGE_RES_M:.1f}m):  ground n={len(ground)}, airborne n={len(airborne)}")
        print(f"  {'feature':<25} {'ground_med':>14} {'airborne_med':>14} {'ratio':>8} {'delta':>10}")
        for k in feature_keys:
            g = np.median([f[k] for f in ground])
            a = np.median([f[k] for f in airborne])
            ratio = a / g if g != 0 else float('inf')
            delta = a - g
            marker = " <<<" if abs(np.log(max(ratio, 1e-10))) > 0.3 else ""
            print(f"  {k:<25} {g:>14.3e} {a:>14.3e} {ratio:>8.2f} {delta:>10.3e}{marker}")


if __name__ == "__main__":
    sys.exit(main())

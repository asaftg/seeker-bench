"""6-frame SPECTRAL INTEGRATION test on airborne1 hover window.

Per the plan we agreed on but I shipped wrong:
  - Stack the per-frame slow-time POWER SPECTRA across N=6 consecutive
    frames at each range bin (sum |X|^2). This is the +sqrt(N) ~ 7.8 dB
    SNR gain step.
  - Run the harmonic-comb detector ON THE INTEGRATED SPECTRUM.
  - Compare hits in airborne1 hover window vs background.

I previously implemented only the persistence smoother (4-of-6 frames
must independently fire). This integration step is what makes the
weak signal pop above noise BEFORE per-frame scoring even runs.

Run:
    py -3.11 tools/integrate_6frames_test.py
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

AIRBORNE_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-58-59_radar.bin"
)
BG_BIN = Path(
    r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench"
    r"\recordings\seeker_2026-05-06_12-54-23_radar.bin"
)


def load_frame(path: Path, idx: int) -> np.ndarray:
    with open(path, "rb") as f:
        f.seek(idx * BYTES_PER_FRAME)
        buf = f.read(BYTES_PER_FRAME)
    if len(buf) != BYTES_PER_FRAME:
        raise EOFError
    raw = np.frombuffer(buf, dtype=np.int16)
    cube = (raw.reshape(N_CHIRPS, N_RX, N_SAMPLES)
              .transpose(0, 2, 1).astype(np.float32))
    cube -= cube.mean(axis=1, keepdims=True)
    return cube


def stage1(real_cube):
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def per_frame_pwr_spec(bin_path: Path, frame_idx: int) -> np.ndarray:
    """Returns power spectrum (n_fft//2, n_range) — incoherent VA-averaged."""
    cube = load_frame(bin_path, frame_idx)
    rc = stage1(cube)
    rc -= rc.mean(axis=0, keepdims=True)  # MTI; NO notch
    virtual = ddma_unfold(rc)  # (n_per_tx, n_range, n_rx, n_tx)
    n_per_tx, n_range, _, _ = virtual.shape
    slow = virtual.reshape(n_per_tx, n_range, N_RX * N_TX)
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None, None], n=N_FFT, axis=0)
    pwr = (spec.real ** 2 + spec.imag ** 2).mean(axis=2)  # (N_FFT, n_range)
    return pwr[: N_FFT // 2, :].astype(np.float64)  # (N_FFT/2, n_range)


def integrate_n_frames(bin_path: Path, start_idx: int, n: int) -> np.ndarray:
    """Sum |X|^2 across n frames per (range_bin, freq_bin)."""
    acc = None
    for i in range(n):
        try:
            spec = per_frame_pwr_spec(bin_path, start_idx + i)
        except EOFError:
            break
        if acc is None:
            acc = spec.copy()
        else:
            acc += spec
    return acc  # shape (N_FFT/2, n_range)


def find_drone_in_integrated(spec_int: np.ndarray, label: str) -> None:
    """For each range bin, find the strongest in-band peak above noise floor.
    Report if any range bin shows a high peak at a plausible drone-blade-pass freq.
    """
    n_freq, n_range = spec_int.shape
    bin_50hz = int(50 / BIN_HZ)
    bin_3000hz = int(3000 / BIN_HZ)

    print(f"\n=== {label} ===")
    print(f"shape: {spec_int.shape}, bin_hz: {BIN_HZ:.2f}")
    print(f"{'rb':>3} {'range_m':>7} {'floor':>10} {'peak_dB':>7} {'peak_freq_Hz':>12} "
          f"{'comb_count':>10}")

    drone_candidates = []
    for rb in range(n_range):
        spec = spec_int[:, rb]
        # Mask DC and known chip artifact bins (PRF_eff/3 ~ 2540 Hz = bin 341)
        keep = np.ones(n_freq, dtype=bool)
        keep[:8] = False
        keep[166:176] = False  # PRF_eff/6
        keep[336:346] = False  # PRF_eff/3
        spec_masked = spec.copy()
        spec_masked[~keep] = 0
        # Restrict to drone band 50-3000 Hz
        band = spec_masked[bin_50hz:bin_3000hz]
        if band.max() <= 0:
            continue
        floor = float(np.median(spec[keep]))
        peak_idx_in_band = int(np.argmax(band))
        peak_freq = (bin_50hz + peak_idx_in_band) * BIN_HZ
        peak_db = 10.0 * np.log10(band[peak_idx_in_band] / max(floor, 1e-30))

        # Count how many bins in 50-3000 Hz exceed floor + 5 dB (comb-like indicator)
        threshold_lin = floor * (10.0 ** 0.5)
        comb_count = int((band > threshold_lin).sum())

        if peak_db > 6.0:
            drone_candidates.append((rb, peak_freq, peak_db, comb_count))
            print(f"{rb:>3} {rb*RANGE_RES_M:>7.1f} {floor:>10.2e} {peak_db:>7.1f} "
                  f"{peak_freq:>12.0f} {comb_count:>10}")

    if not drone_candidates:
        print("  NO range bins show peak > 6 dB above floor in 50-3000 Hz band")
    else:
        print(f"  -> {len(drone_candidates)} range bin(s) show possible drone signature")


def main():
    print(f"PRF_eff per VA = {EFF_PRF:.0f} Hz, bin_hz = {BIN_HZ:.2f} Hz")
    print(f"FFT length = {N_FFT}, positive-freq bins = {N_FFT // 2}")
    print()

    # Hover window in airborne1 per visual ground truth: t=33-100s
    # Recording is 2148 frames over 155s -> 13.86 fps actual
    # Hover frames: roughly 463-1387
    # Sample several 6-frame integration windows IN the hover

    print("=" * 80)
    print("AIRBORNE1 HOVER WINDOW — 6-frame spectral integration")
    print("=" * 80)
    HOVER_STARTS = [470, 600, 750, 900, 1050, 1200, 1350]
    for start in HOVER_STARTS:
        spec_int = integrate_n_frames(AIRBORNE_BIN, start, 6)
        if spec_int is None:
            continue
        t_rel = start / 13.86
        find_drone_in_integrated(spec_int, f"Airborne hover frames {start}-{start+5} (t~{t_rel:.1f}s)")

    print()
    print("=" * 80)
    print("BACKGROUND — 6-frame spectral integration (FPR baseline)")
    print("=" * 80)
    BG_STARTS = [50, 200, 400, 600, 800]
    for start in BG_STARTS:
        try:
            spec_int = integrate_n_frames(BG_BIN, start, 6)
        except EOFError:
            continue
        find_drone_in_integrated(spec_int, f"Background frames {start}-{start+5}")

    print()
    print("=" * 80)
    print("AIRBORNE1 BEFORE DRONE TAKEOFF (t<24s, frames<330)")
    print("=" * 80)
    PRE_STARTS = [50, 150, 250]
    for start in PRE_STARTS:
        spec_int = integrate_n_frames(AIRBORNE_BIN, start, 6)
        if spec_int is None:
            continue
        find_drone_in_integrated(spec_int, f"Airborne pre-drone frames {start}-{start+5}")

    print()
    print("=" * 80)
    print("If hover frames show NEW peaks at consistent freq/range bin")
    print("that DON'T appear in background or pre-drone frames, the integration")
    print("approach works. Otherwise we need more frames or a different method.")
    print("=" * 80)


if __name__ == "__main__":
    sys.exit(main())

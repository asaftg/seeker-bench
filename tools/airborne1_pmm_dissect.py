"""Dissect why pmm_detector returns 0 hits on airborne1 hover frames.

For each selected frame:
  - Load raw int16 -> Stage 1 range FFT -> MTI (mean removal across slow time)
  - Compute four pipeline variants of the slow-time spectrum at the
    chip-detected range bin:
       (a) WITH notch  + DDMA unfold + per-VA averaging
       (b) WITHOUT notch + DDMA unfold + per-VA averaging
       (c) WITH notch  + per-RX coherent sum (NO DDMA unfold)
       (d) WITHOUT notch + per-RX coherent sum (NO DDMA unfold)
  - Top-20 spectral peaks (Hz, dB above floor) for each variant
  - Targeted check at expected DJI FPV harmonics:
       1250 / 2500 / 3750 Hz (blade-pass + harmonics) at full PRF
       For DDMA per-VA spectrum (eff_PRF=PRF/4=7619.6 Hz, Nyquist=3810 Hz)
       1250 / 2500 / 3750 Hz still in band but 3750 right at Nyquist
  - Run detect_pmm at threshold_db = 0/3/5/10
  - Step-by-step trace of detect_pmm for one frame
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import scipy.fft as sfft

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from radar_dca.pmm_detector import detect_pmm                    # noqa: E402
from radar_dca.ddma import ddma_unfold, DEFAULT_ANTENNA_ORDER    # noqa: E402

# ── Frame dims ───────────────────────────────────────────────────────
N_CHIRPS = 768
N_RX = 4
N_SAMPLES = 192
PRF_HZ = 30478.51264858275
RANGE_RES_M = 2.638466734211415
BYTES_PER_FRAME = N_CHIRPS * N_RX * N_SAMPLES * 2
N_TX = 4
EFF_PRF = PRF_HZ / N_TX  # 7619.6 Hz per-VA after DDMA unfold

BIN_PATH = Path(r"C:\Users\asaf.ruf.BLUERIVERTECH\Desktop\Seeker01\seeker_bench\recordings\seeker_2026-05-06_12-58-59_radar.bin")

HANN_FAST = np.hanning(N_SAMPLES).astype(np.float32)

# Hover-class frames + chip-detected range (m). Frame_idx is 0-based.
SAMPLES = [
    # frame_idx, chip_range_m, |v|<=2 m/s
    (505, 8.0),
    (529, 5.3),
    (535, 5.3),
    (591, 8.0),
    (1823, 42.4),
    (2110, 23.9),
    (2120, 21.2),
    (2280, 21.2),
    (2520, 5.3),
    (2581, 5.3),
]


def m_to_bin(m: float) -> int:
    return int(round(m / RANGE_RES_M))


def load_real_cube(path: Path, frame_idx: int) -> np.ndarray:
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
    """Match dca_pipeline._stage1_range_fft."""
    windowed = real_cube * HANN_FAST[np.newaxis, :, np.newaxis]
    rfft_out = sfft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def stage2_mti(range_cube: np.ndarray) -> np.ndarray:
    """Match dca_pipeline: mean-subtract along slow-time axis."""
    return range_cube - range_cube.mean(axis=0, keepdims=True)


def apply_notch(range_cube: np.ndarray) -> np.ndarray:
    """Match dca_pipeline._notch_harmonic_artifact."""
    out = range_cube.copy()
    n_range = out.shape[1]
    period = N_SAMPLES // 8     # 24
    step = max(period // 2, 1)  # 12
    radius = 5
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        out[:, lo:hi, :] = 0
    return out


def is_notched(rb: int) -> bool:
    """Check if a range bin lands inside any notch window."""
    period = N_SAMPLES // 8
    step = max(period // 2, 1)
    radius = 5
    for b in range(step, 200, step):
        if b - radius <= rb <= b + radius:
            return True
    return False


def top_peaks(power_spectrum: np.ndarray, bin_hz: float, k: int = 20) -> List[Tuple[float, float]]:
    """Return top-k peaks as (Hz, dB above median floor)."""
    floor = max(float(np.median(power_spectrum)), 1e-30)
    db = 10.0 * np.log10(np.maximum(power_spectrum, 1e-30) / floor)
    # local maxima only (exclude DC)
    peaks_idx: List[int] = []
    for i in range(2, len(power_spectrum) - 1):
        if power_spectrum[i] > power_spectrum[i - 1] and power_spectrum[i] >= power_spectrum[i + 1]:
            peaks_idx.append(i)
    peaks_idx.sort(key=lambda i: -power_spectrum[i])
    out = []
    for i in peaks_idx[:k]:
        out.append((i * bin_hz, float(db[i])))
    return out


def measure_at_target(power_spectrum: np.ndarray, bin_hz: float, target_hz: float, tol_bins: int = 3) -> Tuple[float, float]:
    """Return (peak_hz, dB above median) within +/-tol bins of target."""
    floor = max(float(np.median(power_spectrum)), 1e-30)
    center = int(round(target_hz / bin_hz))
    if center < 1 or center >= len(power_spectrum):
        return (float("nan"), float("-inf"))
    lo = max(center - tol_bins, 0)
    hi = min(center + tol_bins + 1, len(power_spectrum))
    sub = power_spectrum[lo:hi]
    j = int(np.argmax(sub))
    pk_idx = lo + j
    db = 10.0 * np.log10(max(float(power_spectrum[pk_idx]), 1e-30) / floor)
    return (pk_idx * bin_hz, db)


def spectrum_per_va_avg(range_cube: np.ndarray, rb: int, n_fft: int) -> Tuple[np.ndarray, float]:
    """DDMA unfold + per-VA Hann + |FFT|^2 + average across 16 VAs.

    Returns (positive-half power, bin_hz).
    """
    virtual = ddma_unfold(range_cube, antenna_order=DEFAULT_ANTENNA_ORDER)
    # virtual: (n_chirps_per_tx, n_range, n_rx, n_tx)
    n_per_tx = virtual.shape[0]
    slow = virtual[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)  # (192, 16)
    win = np.hanning(n_per_tx).astype(np.float32)
    spec = np.fft.fft(slow * win[:, None], n=n_fft, axis=0)
    pwr = (spec.real * spec.real + spec.imag * spec.imag).mean(axis=1).astype(np.float64)
    pos = pwr[: n_fft // 2]
    return pos, EFF_PRF / n_fft


def spectrum_no_ddma(range_cube: np.ndarray, rb: int, n_fft: int) -> Tuple[np.ndarray, float]:
    """No DDMA unfold: coherent sum across RX, then |FFT|^2 of slow-time.

    PRF here is the FULL 30478 Hz (no quadrant slicing).
    """
    slow_per_rx = range_cube[:, rb, :]  # (768, 4) complex
    coh = slow_per_rx.sum(axis=1)       # (768,) complex
    n = coh.shape[0]
    win = np.hanning(n).astype(np.float32)
    spec = np.fft.fft(coh * win, n=n_fft)
    pwr = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float64)
    pos = pwr[: n_fft // 2]
    return pos, PRF_HZ / n_fft


def fmt_peaks(peaks: List[Tuple[float, float]], n: int = 20) -> str:
    return ", ".join(f"{f:7.0f}Hz/{d:+5.1f}dB" for (f, d) in peaks[:n])


def run_one_frame(frame_idx: int, chip_range_m: float, do_trace: bool = False) -> None:
    rb = m_to_bin(chip_range_m)
    print(f"\n{'='*78}")
    print(f"frame {frame_idx}  chip_range={chip_range_m:.1f} m  -> rb={rb}  notched={is_notched(rb)}")
    print('='*78)

    real_cube = load_real_cube(BIN_PATH, frame_idx)
    range_cube = stage1_range_fft(real_cube)        # (768, 97, 4)
    range_cube = stage2_mti(range_cube)
    range_cube_notched = apply_notch(range_cube)

    n_fft = 4 * (1 << int(np.ceil(np.log2(N_CHIRPS // N_TX))))   # 1024 -> match detect_pmm

    # Variant (a): NOTCH + DDMA + per-VA avg
    pwr_a, bin_a = spectrum_per_va_avg(range_cube_notched, rb, n_fft)
    # Variant (b): NO NOTCH + DDMA + per-VA avg
    pwr_b, bin_b = spectrum_per_va_avg(range_cube, rb, n_fft)
    # Variant (c): NOTCH + NO DDMA (coherent RX sum, full PRF)
    n_fft_full = 4 * (1 << int(np.ceil(np.log2(N_CHIRPS))))      # 4096
    pwr_c, bin_c = spectrum_no_ddma(range_cube_notched, rb, n_fft_full)
    # Variant (d): NO NOTCH + NO DDMA
    pwr_d, bin_d = spectrum_no_ddma(range_cube, rb, n_fft_full)

    for label, pwr, bhz, prf in [
        ("(a) NOTCH + DDMA per-VA avg", pwr_a, bin_a, EFF_PRF),
        ("(b) NO-NOTCH + DDMA per-VA avg", pwr_b, bin_b, EFF_PRF),
        ("(c) NOTCH + RX-coh-sum (no DDMA)", pwr_c, bin_c, PRF_HZ),
        ("(d) NO-NOTCH + RX-coh-sum (no DDMA)", pwr_d, bin_d, PRF_HZ),
    ]:
        peaks = top_peaks(pwr, bhz, k=20)
        floor = float(np.median(pwr))
        print(f"\n{label}  | bin_hz={bhz:.2f}  Nyq={prf/2:.0f} Hz  med_floor={floor:.3e}")
        print(f"  top10: {fmt_peaks(peaks, 10)}")
        # targeted DJI FPV harmonics
        for tgt in (60.0, 1250.0, 2500.0, 2575.0, 3750.0):
            if tgt > prf / 2:
                continue
            f_pk, db = measure_at_target(pwr, bhz, tgt, tol_bins=3)
            print(f"  @{tgt:6.0f}Hz: peak={f_pk:7.1f}Hz  {db:+5.1f} dB above median")

    # ── Run detect_pmm at multiple thresholds ─────────────────────────
    virtual = ddma_unfold(range_cube_notched, antenna_order=DEFAULT_ANTENNA_ORDER)
    n_per_tx = virtual.shape[0]
    slow_per_va = virtual[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)
    print("\n  detect_pmm sweep (NOTCH path, default f0_min=80, f0_max=500):")
    for thr in (0.0, 3.0, 5.0, 10.0):
        r = detect_pmm(slow_per_va, eff_prf_hz=EFF_PRF, threshold_db=thr,
                       f0_min_hz=80.0, f0_max_hz=500.0)
        print(f"    thr={thr:4.1f}  detected={r.detected!s:5s}  score={r.band_snr_db:+6.2f}  "
              f"f0={r.blade_freq_hz:7.2f} Hz  n_strong={r.n_folds}")

    # Also try a wider band 50..2500 Hz (matches AA preset)
    print("  detect_pmm sweep (NOTCH path, AA-preset f0_min=50, f0_max=2500):")
    for thr in (0.0, 3.0, 5.0, 10.0):
        r = detect_pmm(slow_per_va, eff_prf_hz=EFF_PRF, threshold_db=thr,
                       f0_min_hz=50.0, f0_max_hz=2500.0)
        print(f"    thr={thr:4.1f}  detected={r.detected!s:5s}  score={r.band_snr_db:+6.2f}  "
              f"f0={r.blade_freq_hz:7.2f} Hz  n_strong={r.n_folds}")

    # NO-notch variant for detect_pmm
    virtual_nn = ddma_unfold(range_cube, antenna_order=DEFAULT_ANTENNA_ORDER)
    slow_per_va_nn = virtual_nn[:, rb, :, :].reshape(n_per_tx, N_RX * N_TX)
    print("  detect_pmm sweep (NO-NOTCH path, AA-preset 50..2500 Hz):")
    for thr in (0.0, 3.0, 5.0, 10.0):
        r = detect_pmm(slow_per_va_nn, eff_prf_hz=EFF_PRF, threshold_db=thr,
                       f0_min_hz=50.0, f0_max_hz=2500.0)
        print(f"    thr={thr:4.1f}  detected={r.detected!s:5s}  score={r.band_snr_db:+6.2f}  "
              f"f0={r.blade_freq_hz:7.2f} Hz  n_strong={r.n_folds}")

    if do_trace:
        trace_detect_pmm(slow_per_va, label="NOTCH+DDMA")


def trace_detect_pmm(slow_time_per_va: np.ndarray, label: str) -> None:
    """Step-by-step trace of detect_pmm internals on one frame."""
    print(f"\n  --- detect_pmm trace ({label}) ---")
    n_chirps_per_tx, n_va = slow_time_per_va.shape
    n_fft = 4 * (1 << int(np.ceil(np.log2(n_chirps_per_tx))))   # 1024
    win = np.hanning(n_chirps_per_tx).astype(np.float32)
    spec = np.fft.fft(slow_time_per_va * win[:, None], n=n_fft, axis=0)
    pwr_per_va = (spec.real * spec.real + spec.imag * spec.imag)
    pwr = pwr_per_va.mean(axis=1).astype(np.float64)
    pwr_pos = pwr[: n_fft // 2]
    bin_hz = EFF_PRF / n_fft
    floor = max(float(np.median(pwr_pos)), 1e-30)
    HARMONIC_TOL_BINS = 2
    HARMONIC_GATE_DB = 5.0
    K_MAX = 8
    floor_threshold = floor * 10.0 ** (HARMONIC_GATE_DB / 10.0)
    print(f"    n_fft={n_fft}  bin_hz={bin_hz:.3f}  floor={floor:.3e}  thr={floor_threshold:.3e} (+5dB)")

    # Pick a few candidate f0 values, including the published DJI FPV motor 417 Hz
    f0_candidates = [80, 108, 200, 300, 417, 1250, 1250/3.0]
    n_pos = pwr_pos.shape[0]
    for f0 in f0_candidates:
        h1 = int(round(f0 / bin_hz))
        if h1 >= n_pos:
            continue
        lo = max(h1 - HARMONIC_TOL_BINS, 0); hi = min(h1 + HARMONIC_TOL_BINS + 1, n_pos)
        h1_pwr = float(pwr_pos[lo:hi].max())
        h1_strong = h1_pwr >= floor_threshold
        h1_db = 10.0 * np.log10(h1_pwr / floor)
        line = f"    f0={f0:7.1f}Hz: h1={h1}({h1_pwr:.2e},{h1_db:+5.1f}dB,{ 'OK' if h1_strong else 'WEAK'})"
        if not h1_strong:
            print(line + "  -> REJECTED at h1 gate (this f0 cannot be fundamental)")
            continue
        n_strong = 1
        sum_db = h1_db
        details = []
        for h in range(2, K_MAX + 1):
            c = int(round(h * f0 / bin_hz))
            if c >= n_pos:
                break
            lo = max(c - HARMONIC_TOL_BINS, 0); hi = min(c + HARMONIC_TOL_BINS + 1, n_pos)
            hp = float(pwr_pos[lo:hi].max())
            db = 10.0 * np.log10(max(hp, 1e-30) / floor)
            ok = hp >= floor_threshold
            if ok:
                n_strong += 1
                sum_db += db
            details.append(f"h{h}({c},{db:+4.1f},{ 'Y' if ok else 'n'})")
        avg_db = sum_db / n_strong
        print(line + f"  n_strong={n_strong}  avg_db={avg_db:+5.2f}  {' '.join(details)}")


def main() -> None:
    print(f"PMM dissect on {BIN_PATH.name}")
    print(f"PRF={PRF_HZ:.2f} Hz  range_res={RANGE_RES_M:.3f} m/bin  EFF_PRF={EFF_PRF:.1f} Hz/Nyq={EFF_PRF/2:.0f}Hz")
    print(f"Notch period={N_SAMPLES//8} step={N_SAMPLES//16} radius=5  -> notched bins centred at 12, 24, 36, 48...")

    for frame_idx, rng_m in SAMPLES:
        do_trace = (frame_idx == 535)   # full trace for one canonical hover frame
        run_one_frame(frame_idx, rng_m, do_trace=do_trace)


if __name__ == "__main__":
    main()

"""HERM-line / blade-pass propeller detector — corrected layout, slot-0 decimation.

This is a clean re-implementation of the propeller detector specifically
adapted to the seeker-bench AWR2944P + DCA1000 recordings, after the
2026-05-07 discovery of the layout bug and the slot-0 decimation
strategy that bypasses DDMA fold contamination.

Background
----------
- The wire layout is sample-major RX-interleaved (NOT RX-major as the
  live `radar_dca/dca_pipeline.py` assumes). Only RX0 and RX1 carry
  data; RX2/RX3 are zero-padded by the firmware (2 LVDS lanes wired).
- The DDMA scheme uses 6 chirp slots per loop. Slot 0 has all 4 TX at
  phase 0 (coherent sum). Decimating to slot 0 only gives a clean
  per-VA slow-time signal at PRF/6 = 5080 Hz with no DDMA fold spurs.
- 128 chirps per VA per frame. Per-VA Nyquist = 2540 Hz, comfortably
  above the FPV blade-pass band of [300, 1500] Hz.

Empirical confirmation (drone_fly recording, hover at ~5 m):
- Peaks at 1374 Hz (+76.6 dB) and 1166 Hz (+72.5 dB) appear at range
  bins 1 and 2 only (2.6-5.3 m).
- Same peaks ABSENT in background recording at any bin.
- Same peaks ABSENT in drone_fly bins 4+ (away from drone).
- Two distinct blade-pass frequencies = 4-motor FPV with slightly
  different motor RPMs.

Detection rule
--------------
For each range bin's slot-0-decimated slow-time signal:
1. Hann window, zero-pad to 1024, magnitude FFT.
2. Mask zero-Doppler ±50 Hz (kill static return DC leakage).
3. Find peaks in [300, 2200] Hz prop-search band.
4. Compute local noise floor (median magnitude in [200, 2200] Hz
   excluding peak vicinities).
5. Score = peak power / noise floor in dB.
6. Track peak frequency across multiple frames; if it's stable
   within ±50 Hz across ≥80% of K frames, declare detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy import fft as scipy_fft


@dataclass
class HermResult:
    detected: bool
    peak_freq_hz: float
    peak_power_db: float
    noise_floor_db: float
    snr_db: float
    secondary_freq_hz: float
    secondary_power_db: float
    confidence: float


def detect_blade_pass(
    slow_time_slot0: np.ndarray,
    prf_per_va_hz: float = 5080.0,
    band_low_hz: float = 300.0,
    band_high_hz: float = 2200.0,
    fft_n: int = 1024,
    snr_threshold_db: float = 6.0,
    zero_doppler_guard_hz: float = 60.0,
    fft_already_done: bool = False,
) -> HermResult:
    """Detect blade-pass tone in a slot-0-decimated slow-time signal.

    Parameters
    ----------
    slow_time_slot0 : (N,) complex
        Slow-time samples taken at slot 0 of every loop. For our cfg
        N=128 (one frame).
    prf_per_va_hz : effective per-VA PRF (PRF/6 for our 6-slot DDMA).
    band_low_hz, band_high_hz : prop-band search range.
    snr_threshold_db : minimum peak/noise-floor in dB to declare detection.

    Returns HermResult.
    """
    n = len(slow_time_slot0)
    if not fft_already_done:
        win = np.hanning(n).astype(np.float32)
        x = slow_time_slot0 * win
        X = np.abs(scipy_fft.fft(x, n=fft_n))
    else:
        X = slow_time_slot0
        fft_n = len(X)
    freqs = np.fft.fftfreq(fft_n, d=1.0 / prf_per_va_hz)
    pos_mask = freqs >= 0
    Xp = X[pos_mask]
    fp = freqs[pos_mask]

    # Build search mask
    band_mask = (fp >= band_low_hz) & (fp <= band_high_hz)
    if zero_doppler_guard_hz > 0:
        band_mask &= np.abs(fp) > zero_doppler_guard_hz

    # Find peak
    Xb = np.where(band_mask, Xp, 0.0)
    peak_idx = int(np.argmax(Xb))
    peak_freq = float(fp[peak_idx])
    peak_pwr = float(Xp[peak_idx])

    # Noise floor: median in the band excluding peak ±5 bins
    Xb_clean = Xb.copy()
    Xb_clean[max(0, peak_idx - 5): peak_idx + 6] = 0
    noise_samples = Xb_clean[band_mask][Xb_clean[band_mask] > 0]
    if len(noise_samples) == 0:
        noise_floor = 1.0
    else:
        noise_floor = float(np.median(noise_samples))

    snr_db = 20 * np.log10(max(peak_pwr / max(noise_floor, 1e-6), 1e-6))
    peak_pwr_db = 20 * np.log10(peak_pwr + 1e-6)
    noise_floor_db = 20 * np.log10(noise_floor + 1e-6)

    # Find a 2nd peak away from the first (for multi-motor verification)
    Xb2 = Xb_clean.copy()
    Xb2[max(0, peak_idx - 20): peak_idx + 21] = 0
    second_idx = int(np.argmax(Xb2))
    second_freq = float(fp[second_idx])
    second_pwr_db = 20 * np.log10(Xp[second_idx] + 1e-6)

    # Confidence: smooth ramp from threshold to threshold+12
    if snr_db >= snr_threshold_db:
        confidence = 0.5 + min(0.49, (snr_db - snr_threshold_db) / 24.0)
    else:
        confidence = max(0.0, 0.5 * snr_db / max(snr_threshold_db, 1e-6))

    return HermResult(
        detected=(snr_db >= snr_threshold_db),
        peak_freq_hz=peak_freq,
        peak_power_db=peak_pwr_db,
        noise_floor_db=noise_floor_db,
        snr_db=snr_db,
        secondary_freq_hz=second_freq,
        secondary_power_db=second_pwr_db,
        confidence=confidence,
    )


def scan_range_bins_blade_pass(
    range_doppler_grid: np.ndarray,
    prf_per_va_hz: float = 5080.0,
    band_low_hz: float = 300.0,
    band_high_hz: float = 2200.0,
    snr_threshold_db: float = 6.0,
) -> List[tuple]:
    """Scan every range bin of a slot-0-decimated slow-time grid.

    range_doppler_grid : (n_chirps_per_va=128, n_range) complex
        Slot-0-decimated slow-time samples × range bins.

    Returns list of (range_bin, HermResult) for bins where a detection
    fires.
    """
    n_va_chirps, n_range = range_doppler_grid.shape
    hits = []
    for r in range(n_range):
        slow = range_doppler_grid[:, r]
        result = detect_blade_pass(
            slow,
            prf_per_va_hz=prf_per_va_hz,
            band_low_hz=band_low_hz,
            band_high_hz=band_high_hz,
            snr_threshold_db=snr_threshold_db,
        )
        if result.detected:
            hits.append((r, result))
    return hits

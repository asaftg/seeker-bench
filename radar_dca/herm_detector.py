"""HERM-line propeller detector — comb-of-harmonics rule (Phase 1).

Replaces the failed `herm_detector_v2.detect_blade_pass` (single-peak-in-band)
with the structural fix prescribed in the plan: require energy at integer
multiples of one rotor rate, not a single peak in a band.

Structural reasoning
--------------------
A single bright reflection (wall, body) windowed by a Hann produces a
power-symmetric main lobe. The previous detector (`herm_detector_v2`)
thresholded on the strongest peak in the [300, 2200] Hz band and so
fired on every windowed bright reflection in the scene — that's why
both drone-present and drone-absent recordings produced ~50–80 % flat
fire rates regardless of range bin (validated empirically 2026-05-08
with the bin_parser-fixed pipeline).

A propeller produces something a single tone cannot fake: a comb of
peaks at integer multiples of one fundamental (the rotor rate). With
a Hann window at the rotor rate's leakage scale (~6–10 Hz for our
128-chirp slow-time and 5080 Hz PRF), the leakage at 2× and 3× of the
fundamental decays smoothly to noise floor; it does not produce
secondary peaks at exact integer multiples of itself. So a comb of 3+
peaks at exact 1×, 2×, 3× of the same fundamental is structurally
incompatible with single-tone leakage and so distinguishes drone from
clutter.

Detection rule
--------------
For one slot-0-decimated slow-time spectrum (positive frequencies):

1. Sweep candidate rotor rate f0 from `band_low_hz` to `band_high_hz`
   in `step_hz` increments (default 200..1500 Hz, 5 Hz step).
2. For each f0, identify all harmonics k·f0 (k=1..4) that fall below
   Nyquist with a small guard band.
3. For each surviving harmonic, take the peak magnitude in a tight
   ±tolerance_hz window around k·f0 (compensates for ~5 Hz freq
   quantization and small RPM jitter).
4. Compute a comb score:
       score(f0) = sum over harmonics of  10·log10(peak_k / noise_floor)
   with a minimum-harmonics-active gate (`min_harmonics`, default 3 of 4):
   if fewer than that many harmonics are individually above
   `per_harmonic_floor_db` (default 3 dB) above the noise floor, the
   candidate is rejected — this enforces the "comb" requirement.
5. Best candidate: max-scoring f0 over the sweep that passed the gate.
6. Detection: best comb score > `comb_threshold_db`.

Output: best rotor rate, comb score, count of active harmonics, and a
single SNR-like field (the score divided by the number of active
harmonics) for compatibility with the existing replay machinery.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class CombResult:
    detected: bool
    rotor_freq_hz: float
    comb_score_db: float          # sum of per-harmonic SNR
    n_active_harmonics: int       # how many harmonics passed individual gate
    snr_db: float                 # comb_score / n_active (mean per-harmonic SNR)
    noise_floor_db: float
    # for debug/inspection
    peak_freqs_hz: Tuple[float, float, float, float]
    peak_powers_db: Tuple[float, float, float, float]


def _peak_in_window(spec_db: np.ndarray, freqs_hz: np.ndarray,
                    target_hz: float, tol_hz: float) -> Tuple[float, float]:
    """Return (peak_freq_hz, peak_db) within ±tol_hz of target_hz.
    If the window is empty (off-spectrum), returns (nan, -inf)."""
    lo = target_hz - tol_hz
    hi = target_hz + tol_hz
    mask = (freqs_hz >= lo) & (freqs_hz <= hi)
    if not mask.any():
        return float('nan'), float('-inf')
    sub = spec_db[mask]
    sub_f = freqs_hz[mask]
    k = int(np.argmax(sub))
    return float(sub_f[k]), float(sub[k])


def detect_comb(
    slow_time_spectrum_mag: np.ndarray,
    prf_per_va_hz: float,
    *,
    band_low_hz: float = 200.0,
    band_high_hz: float = 1500.0,
    step_hz: float = 5.0,
    n_harmonics: int = 4,
    min_harmonics: int = 3,
    per_harmonic_floor_db: float = 3.0,
    comb_threshold_db: float = 12.0,
    tolerance_hz: float = 8.0,
    nyquist_guard_hz: float = 50.0,
    dc_kill_hz: float = 50.0,
) -> CombResult:
    """Run the comb-of-harmonics detector on one range bin's slow-time
    magnitude spectrum.

    Parameters
    ----------
    slow_time_spectrum_mag : np.ndarray
        Magnitude of FFT of the slot-0 slow-time signal at one range
        bin. Length is the FFT length (e.g. 1024); only the positive
        frequency half is used.
    prf_per_va_hz : float
        Per-virtual-antenna PRF (5080 Hz for our cfg = 30478/6).
    band_low_hz, band_high_hz : float
        Search band for the rotor fundamental frequency.
    step_hz : float
        Sweep granularity for f0.
    n_harmonics : int
        How many harmonics (k=1..n_harmonics) to score.
    min_harmonics : int
        Minimum harmonics that must individually exceed
        `per_harmonic_floor_db` for the candidate to be considered.
    per_harmonic_floor_db : float
        Per-harmonic SNR floor (dB above local noise) to count as "active".
    comb_threshold_db : float
        Total comb score (sum of per-harmonic SNR in dB) above which we
        declare detection.
    tolerance_hz : float
        Half-width of the search window around each k·f0.
    nyquist_guard_hz : float
        Skip harmonics that fall within this distance of Nyquist
        (avoids alias contamination near the edge).
    dc_kill_hz : float
        Zero-out spectrum below this freq (kills static-return leakage).
    """
    fft_n = len(slow_time_spectrum_mag)
    # Frequency axis of the positive half of the FFT bins.
    freqs_full = np.fft.fftfreq(fft_n, d=1.0 / prf_per_va_hz)
    # Take only positive frequencies; the magnitude spectrum is symmetric.
    pos_mask = freqs_full > 0
    spec = np.asarray(slow_time_spectrum_mag, dtype=np.float64)[pos_mask].copy()
    freqs = freqs_full[pos_mask]

    # Kill DC leakage.
    spec[freqs < dc_kill_hz] = 0.0

    # Convert to dB; protect log10(0).
    spec_db = 20.0 * np.log10(spec + 1e-12)

    # Local noise floor: median in the search-and-harmonics-relevant range,
    # excluding the very top end where harmonics can land.
    noise_band = (freqs >= band_low_hz) & (freqs <= prf_per_va_hz / 2.0 - nyquist_guard_hz)
    if not noise_band.any():
        return CombResult(False, float('nan'), float('-inf'), 0, float('-inf'),
                          float('nan'), (float('nan'),) * 4, (float('-inf'),) * 4)
    noise_floor_db = float(np.median(spec_db[noise_band]))

    nyquist_hz = prf_per_va_hz / 2.0 - nyquist_guard_hz

    best_score = float('-inf')
    best_f0 = float('nan')
    best_n_active = 0
    best_peak_freqs: list[float] = [float('nan')] * 4
    best_peak_dbs: list[float] = [float('-inf')] * 4

    n_steps = int(round((band_high_hz - band_low_hz) / step_hz)) + 1
    for i in range(n_steps):
        f0 = band_low_hz + i * step_hz

        peak_freqs = [float('nan')] * 4
        peak_dbs   = [float('-inf')] * 4
        per_harm_snr = [float('-inf')] * 4
        n_active = 0
        score = 0.0

        for k in range(1, n_harmonics + 1):
            target = k * f0
            if target > nyquist_hz:
                continue
            f_pk, db_pk = _peak_in_window(spec_db, freqs, target, tolerance_hz)
            peak_freqs[k - 1] = f_pk
            peak_dbs[k - 1]   = db_pk
            snr_k = db_pk - noise_floor_db
            per_harm_snr[k - 1] = snr_k
            if snr_k > per_harmonic_floor_db:
                n_active += 1
                score += snr_k

        if n_active < min_harmonics:
            continue
        if score > best_score:
            best_score = score
            best_f0 = f0
            best_n_active = n_active
            best_peak_freqs = peak_freqs
            best_peak_dbs = peak_dbs

    detected = best_score > comb_threshold_db and best_n_active >= min_harmonics
    snr_per_harm = (best_score / best_n_active) if best_n_active else float('-inf')

    # Pad to exactly 4 entries for the dataclass tuple.
    pf = tuple((best_peak_freqs + [float('nan')] * 4)[:4])  # type: ignore[arg-type]
    pp = tuple((best_peak_dbs + [float('-inf')] * 4)[:4])    # type: ignore[arg-type]
    return CombResult(
        detected=detected,
        rotor_freq_hz=best_f0,
        comb_score_db=best_score,
        n_active_harmonics=best_n_active,
        snr_db=snr_per_harm,
        noise_floor_db=noise_floor_db,
        peak_freqs_hz=pf,                     # type: ignore[arg-type]
        peak_powers_db=pp,                    # type: ignore[arg-type]
    )

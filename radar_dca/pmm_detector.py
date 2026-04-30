"""PMM (Periodic Micro-Motion) detector for drone-vs-clutter discrimination.

Pure-physics, no training data required. Looks for the symmetric
sideband signature that propellers create in the slow-time radar
spectrum at a single range bin.

The physics (mmHawkeye, Tsinghua TOSN 2024):
  - A propellered drone reflects radar energy as:
        body_doppler_tone (the airframe)
      + sideband_pair at body_doppler ± blade_pass_freq
        (modulation from rotating blades)
  - The DEFINING signature is the SYMMETRY of the sidebands around
    the body Doppler. Birds, vegetation, vehicles — anything else
    in the scene — produce single tones, not symmetric pairs.
  - Per-blade signature is at the blade-pass frequency (RPS × num_blades).
    DJI Mavic ~6.5 kRPM × 2 blades → ~217 Hz; FPV racers ~30 kRPM × 5 blades → ~2500 Hz.

Algorithm — symmetric-sideband matched filter:
    1. Subtract DC, apply Hann window (suppresses spectral leakage
       that would otherwise fake sidebands on any single-tone clutter).
    2. FFT slow-time signal to get spectrum X(f) (×4 zero padding for
       finer bin resolution).
    3. Identify body-Doppler peak: ``body_bin = argmax |X(f)|``.
    4. For each candidate offset Δ ∈ [band_low, band_high] Hz:
         score(Δ) = min(|X[body+Δ]|², |X[body−Δ]|²) / noise_floor
       The MIN(plus, minus) is critical — both sidebands must be
       strong; either one alone could be a coincidence.
    5. Pick the Δ with the highest score. That Δ is the estimated
       blade-pass frequency. Detection = score ≥ threshold dB.

Why not autocorrelation or cepstrum?
  Autocorrelation theory says |R[τ]| = (1 + 2·side_amp·cos(2π·f_blade·τ))
  for a body+sidebands signal — but this OSCILLATES rather than
  showing a peak above a flat noise floor, so a peak-vs-noise SNR
  metric isn't well-defined. Cepstrum requires a comb of HARMONICS
  to give a sharp lag-domain peak; a single sideband pair (the
  fundamental case for our test signals) doesn't produce one.
  Real propellers DO produce harmonics (body ± n·blade for n=1,2,3…),
  so cepstrum may still be added later as a second discriminator.

Inputs:
  - ``slow_time`` : 1-D complex (preferred) array of length n_chirps.
    Per-chirp signal at one range bin. For DDM-MIMO, integrate
    across the virtual antenna array first.
  - ``prf_hz``    : pulse repetition frequency = 1 / chirp period.

Outputs (``PMMResult``):
  - ``detected``      : bool — score exceeds threshold
  - ``band_snr_db``   : sideband-pair power vs. noise floor (dB)
  - ``blade_freq_hz`` : estimated blade-pass frequency (Hz)
  - ``confidence``    : 0..1, derived from band_snr_db

The detector is range-bin-agnostic; ``scan_range_bins(slow_2d, prf)``
runs the algorithm on every range bin and returns the hits.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PMMResult:
    """Output of the PMM detector for one range bin."""
    detected: bool
    band_snr_db: float
    blade_freq_hz: float
    confidence: float


def detect_pmm(
    slow_time: np.ndarray,
    prf_hz: float,
    band_low_hz: float = 50.0,
    band_high_hz: float = 500.0,
    threshold_db: float = 18.0,
) -> PMMResult:
    """Run PMM detection on a 1-D slow-time slice at one range bin.

    Parameters
    ----------
    slow_time : array (n_chirps,)
        Complex (preferred) or real per-chirp signal at one range bin.
        For DDM-MIMO, integrate across the virtual antenna array first
        and pass the integrated complex slice here.
    prf_hz : float
        Pulse Repetition Frequency (chirps per second). 1 / chirp period.
    band_low_hz, band_high_hz : float
        Blade-rate band of interest. Defaults: 50–500 Hz, covers most
        consumer-drone blade-pass frequencies (FPV ~3–5 kHz blade rate
        when aliased into Nyquist; Shahed-class ~150–250 Hz un-aliased).
    threshold_db : float
        Peak-to-noise-floor ratio above which we declare detection.
        Default 18 dB. Field test 2026-04-29 (live drill rig)
        showed 12 dB still produced 40-65 false alarms per frame
        at delta=6 (52.7 Hz Hann main-lobe leakage). After bumping
        HANN_GUARD_BINS from 4 to 8 to suppress leakage entirely,
        the remaining floor is the pure-noise "best of ~340 random
        offsets" statistic which peaks around +12 dB; 18 dB gives
        a 6 dB margin. Real propeller signatures land at +25-40 dB.
        Tunable via the GUI slider in production.

    Returns
    -------
    PMMResult — see class docstring.
    """
    n = slow_time.shape[0]
    if n < 32:
        return PMMResult(detected=False, band_snr_db=float("-inf"),
                         blade_freq_hz=float("nan"), confidence=0.0)

    # 1. Subtract DC.
    s = slow_time - np.mean(slow_time)

    # 2. Apply a Hann window to suppress spectral leakage. Without it,
    #    a strong body-Doppler tone leaks into all bins as 1/k, which
    #    fakes a symmetric-sideband signature on any single-tone
    #    clutter (bird, wind-blown vegetation, vehicle in zero-Doppler).
    #    Hann's first sidelobe is ~31 dB down — well below the
    #    threshold we'd ever set.
    s = s * np.hanning(n)
    # FFT with ×4 zero padding for fine bin resolution. At PRF=10 kHz
    # with n=256, padded to 1024, bin width = 9.77 Hz — fine enough
    # to resolve a 200 Hz blade rate within ±5 Hz.
    n_fft = 4 * (1 << int(np.ceil(np.log2(n))))
    X = np.fft.fft(s, n=n_fft)
    mag = np.abs(X)
    bin_hz = prf_hz / n_fft

    # 3. Body-Doppler bin. For a complex signal X(f) is asymmetric;
    #    the body Doppler is wherever the power is highest. We
    #    use "fft-shifted" indexing so positive freqs are first half,
    #    negative freqs second half — but argmax handles that fine.
    body_bin = int(np.argmax(mag))

    # 4. Search for symmetric sideband pair at offset
    #    Δ ∈ [band_low_hz, band_high_hz]. Score is min(|X[+Δ]|, |X[-Δ]|)
    #    — both sidebands must be strong; either one can be a
    #    coincidence. The propeller signature is the symmetry.
    #
    # IMPORTANT — Hann main-lobe guard. The Hann window we applied at
    # step 2 has a main lobe ~2 FFT bins wide on each side of any
    # tone, with first sidelobe at -31 dB about 3 bins out. A strong
    # body-Doppler return SPILLS into delta ∈ {±1..±7} via main-lobe
    # leakage, producing phantom symmetric "sideband pairs" on every
    # bright range bin even when there's no propeller present.
    #
    # Field test 2026-04-29 (live drill rig): with guard=4 the
    # detector locked onto delta=6 (52.7 Hz) on basically every
    # range bin, regardless of whether the drill was on or off,
    # producing ~40-65 false alarms per frame. Bumping guard to 8
    # cuts off the leakage region completely.
    #
    # Cost: minimum detectable blade rate becomes 8 × bin_hz (~70 Hz
    # at our 8.78 Hz/bin = 768-chirp slow time). DJI Mavic-class
    # blade-pass is ~217 Hz, FPV racers ~2.5 kHz; 70 Hz floor is
    # well below any real-drone signature. Slow rotors (windmills,
    # ceiling fans) won't be detected — that's a feature, not a bug.
    HANN_GUARD_BINS = 8
    delta_min = max(HANN_GUARD_BINS, int(round(band_low_hz / bin_hz)))
    delta_max = min(n_fft // 2 - 1, int(round(band_high_hz / bin_hz)))
    if delta_min >= delta_max:
        return PMMResult(detected=False, band_snr_db=float("-inf"),
                         blade_freq_hz=float("nan"), confidence=0.0)

    # Compute noise floor BEFORE masking the body bin: median power
    # in bins outside the prop band around the body. This is the
    # spectrum's "elsewhere", roughly the AWGN level.
    half_band_max = delta_max
    inband_mask = np.zeros(n_fft, dtype=bool)
    for d in range(delta_min, delta_max + 1):
        inband_mask[(body_bin + d) % n_fft] = True
        inband_mask[(body_bin - d) % n_fft] = True
    inband_mask[body_bin] = True  # exclude body itself from noise estimate
    noise_floor = float(np.median(mag[~inband_mask] ** 2))
    if noise_floor <= 0:
        noise_floor = float(np.finfo(np.float64).tiny)

    # Score each candidate Δ: min of the two sideband powers, in dB
    # vs. noise floor. Best Δ wins.
    best_score_db = float("-inf")
    best_delta = 0
    for d in range(delta_min, delta_max + 1):
        plus_idx = (body_bin + d) % n_fft
        minus_idx = (body_bin - d) % n_fft
        sb_pwr = min(mag[plus_idx] ** 2, mag[minus_idx] ** 2)
        score_db = 10.0 * np.log10(max(sb_pwr, 1e-30) / noise_floor)
        if score_db > best_score_db:
            best_score_db = score_db
            best_delta = d

    band_snr_db = float(best_score_db)
    detected = band_snr_db >= threshold_db
    blade_freq = float(best_delta * bin_hz) if best_delta > 0 else float("nan")

    # Confidence: at threshold → 0.5; +12 dB above → 0.99. Below
    # threshold → smoothly down to 0 at SNR=0.
    if band_snr_db >= threshold_db:
        confidence = 0.5 + min(0.49, (band_snr_db - threshold_db) / 24.0)
    else:
        confidence = max(0.0, 0.5 * band_snr_db / threshold_db)

    return PMMResult(
        detected=detected,
        band_snr_db=float(band_snr_db),
        blade_freq_hz=float(blade_freq),
        confidence=float(confidence),
    )


def scan_range_bins(
    slow_time_2d: np.ndarray,
    prf_hz: float,
    *,
    band_low_hz: float = 50.0,
    band_high_hz: float = 500.0,
    threshold_db: float = 18.0,
) -> List[Tuple[int, PMMResult]]:
    """Run ``detect_pmm`` on every range bin of a (n_range, n_chirps)
    slow-time matrix. Returns ``[(range_bin_idx, PMMResult), ...]``
    for bins that exceeded the threshold.

    The result list IS the drone-candidate output for the PMM-detection
    code path (alongside the standard CFAR detection list).
    """
    if slow_time_2d.ndim != 2:
        raise ValueError("expected 2-D (n_range, n_chirps), got "
                         f"shape {slow_time_2d.shape}")
    n_range = slow_time_2d.shape[0]
    out: List[Tuple[int, PMMResult]] = []
    for i in range(n_range):
        r = detect_pmm(slow_time_2d[i], prf_hz,
                       band_low_hz=band_low_hz,
                       band_high_hz=band_high_hz,
                       threshold_db=threshold_db)
        if r.detected:
            out.append((i, r))
    return out

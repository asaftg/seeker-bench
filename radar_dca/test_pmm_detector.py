"""Unit tests for radar_dca.pmm_detector.

Tests use synthetic slow-time signals — no chip required. The
acceptance criterion is "the detector reliably tells a noisy
signal-with-prop-tone apart from a noisy signal-without-prop-tone
across plausible SNR + blade-rate combinations."

Run:
    cd seeker_bench_phase3
    python -m pytest radar_dca/test_pmm_detector.py -v

Or as a smoke without pytest:
    python radar_dca/test_pmm_detector.py
"""
from __future__ import annotations

import math

import numpy as np

from radar_dca.pmm_detector import detect_pmm, scan_range_bins


def _synth_drone_slow_time(
    n_chirps: int,
    prf_hz: float,
    body_doppler_hz: float = 30.0,
    blade_freq_hz: float = 200.0,
    blade_amp_rel_body: float = 0.3,
    snr_db: float = 20.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a complex slow-time sequence with body Doppler +
    propeller sidebands + AWGN.

    The signal model:
        s[n] = A_body * exp(j 2π f_body n / PRF)
             + A_blade * exp(j 2π (f_body + f_blade) n / PRF)
             + A_blade * exp(j 2π (f_body - f_blade) n / PRF)
             + noise

    Body amplitude is fixed at 1.0 + 0j; blade sidebands are
    blade_amp_rel_body × that. SNR is set by the noise variance.
    """
    rng = np.random.default_rng(seed)
    n = np.arange(n_chirps)
    body = np.exp(1j * 2 * math.pi * body_doppler_hz * n / prf_hz)
    upper = blade_amp_rel_body * np.exp(
        1j * 2 * math.pi * (body_doppler_hz + blade_freq_hz) * n / prf_hz)
    lower = blade_amp_rel_body * np.exp(
        1j * 2 * math.pi * (body_doppler_hz - blade_freq_hz) * n / prf_hz)
    sig = body + upper + lower
    # SNR is total signal power / total noise power. We have power
    # ≈ 1 + 2*blade_amp^2 in the signal.
    sig_power = 1.0 + 2.0 * blade_amp_rel_body ** 2
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (rng.standard_normal(n_chirps) + 1j * rng.standard_normal(n_chirps))
    noise *= math.sqrt(noise_power / 2.0)
    return sig + noise


def _synth_clutter_slow_time(
    n_chirps: int,
    prf_hz: float,
    clutter_freq_hz: float = 3.0,
    snr_db: float = 20.0,
    seed: int = 0,
) -> np.ndarray:
    """Synthesize a 'bird/wind/vegetation' signal: low-frequency
    periodic component + AWGN. No prop sidebands. Detector should
    NOT fire on this."""
    rng = np.random.default_rng(seed)
    n = np.arange(n_chirps)
    sig = np.exp(1j * 2 * math.pi * clutter_freq_hz * n / prf_hz)
    sig_power = 1.0
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = (rng.standard_normal(n_chirps) + 1j * rng.standard_normal(n_chirps))
    noise *= math.sqrt(noise_power / 2.0)
    return sig + noise


# ─────────────────────── tests ──────────────────────────────────────────


def test_detector_fires_on_clean_drone_signature():
    """High-SNR FPV-like signature → detect, with confidence > 0.7."""
    s = _synth_drone_slow_time(
        n_chirps=256, prf_hz=10_000.0,
        body_doppler_hz=80.0, blade_freq_hz=200.0,
        blade_amp_rel_body=0.5, snr_db=25.0, seed=42,
    )
    r = detect_pmm(s, prf_hz=10_000.0)
    assert r.detected, f"expected detection, got SNR={r.band_snr_db:.1f} dB"
    assert r.confidence > 0.7
    # Peak should be within ±10 Hz of the true blade freq
    assert abs(r.blade_freq_hz - 200.0) < 10.0, f"blade_freq off: {r.blade_freq_hz}"


def test_detector_silent_on_clutter():
    """Bird/wind clutter (sub-15 Hz) → no detection."""
    s = _synth_clutter_slow_time(
        n_chirps=256, prf_hz=10_000.0,
        clutter_freq_hz=3.0, snr_db=20.0, seed=43,
    )
    r = detect_pmm(s, prf_hz=10_000.0)
    assert not r.detected, f"false alarm on clutter, SNR={r.band_snr_db:.1f} dB"


def test_detector_silent_on_pure_noise():
    """White noise → no detection."""
    rng = np.random.default_rng(7)
    n_chirps = 256
    s = (rng.standard_normal(n_chirps) + 1j * rng.standard_normal(n_chirps))
    r = detect_pmm(s, prf_hz=10_000.0)
    assert not r.detected, f"false alarm on noise, SNR={r.band_snr_db:.1f} dB"


def test_detector_finds_low_blade_rate():
    """Larger drones (Shahed-class) at ~150 Hz blade rate."""
    s = _synth_drone_slow_time(
        n_chirps=256, prf_hz=10_000.0,
        body_doppler_hz=120.0, blade_freq_hz=150.0,
        blade_amp_rel_body=0.4, snr_db=22.0, seed=99,
    )
    r = detect_pmm(s, prf_hz=10_000.0)
    assert r.detected, f"missed Shahed-class signature, SNR={r.band_snr_db:.1f} dB"
    assert abs(r.blade_freq_hz - 150.0) < 10.0


def test_detector_threshold_obeyed():
    """A 4 dB band-SNR signal should NOT fire when threshold=8 dB."""
    # Build a marginal signal: small blade sidebands buried in noise
    s = _synth_drone_slow_time(
        n_chirps=256, prf_hz=10_000.0,
        body_doppler_hz=50.0, blade_freq_hz=200.0,
        blade_amp_rel_body=0.05, snr_db=10.0, seed=11,
    )
    r = detect_pmm(s, prf_hz=10_000.0, threshold_db=8.0)
    # A weak signature at low blade-amplitude should be near or below threshold
    if r.band_snr_db < 8.0:
        assert not r.detected
    # And lowering the threshold should pick it up
    r2 = detect_pmm(s, prf_hz=10_000.0, threshold_db=2.0)
    if r.band_snr_db >= 2.0:
        assert r2.detected


def test_scan_range_bins_finds_drone_bin():
    """Build a (n_range, n_chirps) matrix where ONE bin has the prop
    signature. ``scan_range_bins`` should return exactly that bin."""
    n_range, n_chirps = 32, 256
    prf = 10_000.0
    rng = np.random.default_rng(123)
    grid = (rng.standard_normal((n_range, n_chirps))
            + 1j * rng.standard_normal((n_range, n_chirps))) * 0.3
    # Plant a drone in bin 17
    target = _synth_drone_slow_time(
        n_chirps=n_chirps, prf_hz=prf,
        body_doppler_hz=50.0, blade_freq_hz=250.0,
        blade_amp_rel_body=0.5, snr_db=25.0, seed=5,
    )
    grid[17] += target  # drone signature added on top of noise
    hits = scan_range_bins(grid, prf, threshold_db=8.0)
    bins_hit = sorted(b for b, _ in hits)
    assert 17 in bins_hit, f"missed drone bin; hits={bins_hit}"
    # Most other bins should not have fired
    assert len(bins_hit) <= 3, f"too many false alarms: {bins_hit}"


def test_short_signal_returns_no_detect():
    """Too-few-samples input should fail gracefully without exceptions."""
    s = np.zeros(8, dtype=np.complex128)
    r = detect_pmm(s, prf_hz=10_000.0)
    assert not r.detected
    assert math.isnan(r.blade_freq_hz)


# ─────────────────────── manual smoke runner ────────────────────────────

if __name__ == "__main__":
    # Run all tests as a smoke when pytest isn't available.
    import sys
    tests = [
        ("clean_drone", test_detector_fires_on_clean_drone_signature),
        ("clutter",     test_detector_silent_on_clutter),
        ("pure_noise",  test_detector_silent_on_pure_noise),
        ("low_blade",   test_detector_finds_low_blade_rate),
        ("threshold",   test_detector_threshold_obeyed),
        ("scan_bins",   test_scan_range_bins_finds_drone_bin),
        ("short_input", test_short_signal_returns_no_detect),
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failures += 1
            print(f"  FAIL  {name}: {e}")
        except Exception as e:
            failures += 1
            print(f"  ERR   {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests)-failures}/{len(tests)} tests passed.")
    sys.exit(1 if failures else 0)

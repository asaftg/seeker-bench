"""Unit tests for gimbal.bus_servo_calibration."""
from __future__ import annotations

import pytest

from gimbal.bus_servo_calibration import (
    BusServoCalibration,
    UNITS_PER_DEG,
    UNITS_PER_REV,
)


def _tilt_default() -> BusServoCalibration:
    """Tilt mounted so system 0° (horizon) = servo physical 90° = raw 1024.

    System range -15° → +90° maps to raw 853 → 2048.
    """
    return BusServoCalibration(
        servo_id=1, zero_raw=1024, invert=False,
        raw_min=853, raw_max=2048,
    )


def _pan_default() -> BusServoCalibration:
    """Pan mounted so system 0° = servo mid-travel = raw 2048.

    System ±60° maps to raw 1365 → 2731.
    """
    return BusServoCalibration(
        servo_id=2, zero_raw=2048, invert=False,
        raw_min=1365, raw_max=2731,
    )


# ── tilt ─────────────────────────────────────────────────────────

def test_tilt_zero_to_units():
    assert _tilt_default().angle_to_units(0.0) == 1024


def test_tilt_plus_90_to_units():
    assert _tilt_default().angle_to_units(90.0) == 2048


def test_tilt_minus_15_to_units():
    # -15° * 11.378 = -170.67 → round → -171 → 1024 - 171 = 853
    assert _tilt_default().angle_to_units(-15.0) == 853


def test_tilt_clamps_above_max():
    # +120° request → raw way past raw_max → clamp at raw_max
    assert _tilt_default().angle_to_units(120.0) == 2048


def test_tilt_clamps_below_min():
    # -45° request → clamp at raw_min
    assert _tilt_default().angle_to_units(-45.0) == 853


def test_tilt_units_to_angle_at_zero_raw():
    assert _tilt_default().units_to_angle(1024) == pytest.approx(0.0, abs=1e-6)


def test_tilt_units_to_angle_at_max_raw():
    assert _tilt_default().units_to_angle(2048) == pytest.approx(90.0, abs=0.1)


# ── pan ──────────────────────────────────────────────────────────

def test_pan_zero():
    assert _pan_default().angle_to_units(0.0) == 2048


def test_pan_plus_60():
    # +60° * 11.378 = 682.67 → round → 683 → 2048 + 683 = 2731
    assert _pan_default().angle_to_units(60.0) == 2731


def test_pan_minus_60():
    assert _pan_default().angle_to_units(-60.0) == 1365


def test_pan_clamps_above():
    assert _pan_default().angle_to_units(120.0) == 2731


def test_pan_clamps_below():
    assert _pan_default().angle_to_units(-120.0) == 1365


# ── invert flag ──────────────────────────────────────────────────

def test_invert_flips_direction():
    cal = BusServoCalibration(
        servo_id=3, zero_raw=2048, invert=True,
        raw_min=0, raw_max=UNITS_PER_REV - 1,
    )
    assert cal.angle_to_units(0.0) == 2048
    assert cal.angle_to_units(60.0) < 2048    # negative direction in raw
    assert cal.angle_to_units(-60.0) > 2048


def test_invert_round_trip():
    cal = BusServoCalibration(
        servo_id=3, zero_raw=2048, invert=True,
        raw_min=0, raw_max=UNITS_PER_REV - 1,
    )
    for d in (-60.0, -15.0, 0.0, 15.0, 60.0):
        assert cal.units_to_angle(cal.angle_to_units(d)) == pytest.approx(d, abs=0.1)


# ── round-trip ───────────────────────────────────────────────────

@pytest.mark.parametrize("deg", [-15.0, -10.0, -5.0, 0.0, 1.0, 22.5, 45.0, 75.0, 90.0])
def test_tilt_round_trip(deg):
    cal = _tilt_default()
    assert cal.units_to_angle(cal.angle_to_units(deg)) == pytest.approx(deg, abs=0.1)


@pytest.mark.parametrize("deg", [-60.0, -30.0, -1.0, 0.0, 1.0, 30.0, 60.0])
def test_pan_round_trip(deg):
    cal = _pan_default()
    assert cal.units_to_angle(cal.angle_to_units(deg)) == pytest.approx(deg, abs=0.1)


# ── unit conversion sanity ───────────────────────────────────────

def test_units_per_deg_value():
    assert UNITS_PER_DEG == pytest.approx(11.3778, abs=1e-3)


def test_360_deg_is_one_revolution():
    assert int(round(360.0 * UNITS_PER_DEG)) == UNITS_PER_REV

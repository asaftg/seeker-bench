"""Unit tests for gimbal/gimbal_controller.

Pure-math module: servo calibration, clamping, slew-rate limiting.
No serial / Maestro / hardware required.
"""
from __future__ import annotations

import pytest

from gimbal.gimbal_controller import (
    GimbalController,
    GimbalLimits,
    ServoCalibration,
)


def _pan_cal() -> ServoCalibration:
    return ServoCalibration(
        channel=0,
        min_deg=-90.0,
        max_deg=90.0,
        us_at_min_deg=500.0,
        us_at_max_deg=2500.0,
    )


def _tilt_cal(invert: bool = False) -> ServoCalibration:
    return ServoCalibration(
        channel=1,
        min_deg=0.0,
        max_deg=22.0,
        us_at_min_deg=1000.0,
        us_at_max_deg=2000.0,
        invert=invert,
    )


# ── ServoCalibration ─────────────────────────────────────────────

def test_angle_to_us_endpoints():
    cal = _pan_cal()
    assert cal.angle_to_us(-90.0) == pytest.approx(500.0)
    assert cal.angle_to_us(90.0) == pytest.approx(2500.0)


def test_angle_to_us_midpoint_is_linear():
    cal = _pan_cal()
    assert cal.angle_to_us(0.0) == pytest.approx(1500.0)


def test_angle_to_us_clamps_out_of_range():
    cal = _pan_cal()
    assert cal.angle_to_us(-9999.0) == pytest.approx(500.0)
    assert cal.angle_to_us(9999.0) == pytest.approx(2500.0)


def test_angle_to_us_invert_flips_direction():
    cal = _tilt_cal(invert=True)
    # With invert, min_deg maps to us_at_max_deg and vice versa.
    assert cal.angle_to_us(0.0) == pytest.approx(2000.0)
    assert cal.angle_to_us(22.0) == pytest.approx(1000.0)


def test_us_to_angle_inverts_angle_to_us():
    """Round-trip via the new us_to_angle reverse. Used to recover the
    SERVO'S actual-written pose from MaestroDriver.get_last_written_us
    so gimbal_manager can publish what the camera ACTUALLY sees, not
    the controller's commanded setpoint (which advances even when the
    PWM gate eats the command — see 'multiple bbs.jsonl' diagnosis)."""
    cal = _pan_cal()
    for ang in (-90.0, -45.0, 0.0, 30.0, 90.0):
        round_trip = cal.us_to_angle(cal.angle_to_us(ang))
        assert round_trip == pytest.approx(ang, abs=1e-6)


def test_us_to_angle_invert_flips_direction():
    cal = _tilt_cal(invert=True)
    # angle 0 -> us 2000 -> angle 0 (round-trip survives invert).
    for ang in (0.0, 5.5, 11.0, 22.0):
        round_trip = cal.us_to_angle(cal.angle_to_us(ang))
        assert round_trip == pytest.approx(ang, abs=1e-6)


def test_us_to_angle_clamps_to_software_limits():
    cal = _tilt_cal()
    # PWM well above us_at_max_deg should clamp to max angle.
    huge_us = 99999.0
    assert cal.us_to_angle(huge_us) == pytest.approx(cal.max_deg)
    tiny_us = -99999.0
    assert cal.us_to_angle(tiny_us) == pytest.approx(cal.min_deg)


# ── GimbalController: clamp + slew ───────────────────────────────

def _ctrl(slew_fast: bool = True) -> GimbalController:
    limits = GimbalLimits(
        pan_min_deg=-90.0, pan_max_deg=90.0,
        tilt_min_deg=0.0,  tilt_max_deg=22.0,
        pan_slew_deg_per_s=1000.0 if slew_fast else 10.0,
        tilt_slew_deg_per_s=1000.0 if slew_fast else 10.0,
    )
    return GimbalController(_pan_cal(), _tilt_cal(), limits,
                            home_pan_deg=0.0, home_tilt_deg=11.0)


def test_step_clamps_to_software_limits():
    c = _ctrl(slew_fast=True)
    # One big step that should be clamped to limits after slew allows it.
    c.reset_to(0.0, 11.0)
    # Advance time by 1s — fast slew (1000 deg/s) allows any move.
    p, t = c.step(9999.0, -9999.0, now=c._last_t + 1.0)
    assert p == pytest.approx(90.0)
    assert t == pytest.approx(0.0)


def test_step_rate_limits_big_jumps():
    c = _ctrl(slew_fast=False)  # 10 deg/s
    c.reset_to(0.0, 11.0)
    # Reset time
    p0, t0 = c.current
    # Pretend 0.1s elapsed → max 1 deg of pan movement
    # Controller stores _last_t at construction; feed now+0.1s.
    # Easiest: call step with a now relative to its internal _last_t.
    p, t = c.step(setpoint_pan=45.0, setpoint_tilt=11.0,
                  now=c._last_t + 0.1)  # 0.1 * 10 = 1 deg cap
    assert abs(p - p0) == pytest.approx(1.0, abs=0.01)
    # tilt unchanged (setpoint == current)
    assert t == pytest.approx(11.0)


def test_step_repeated_calls_converge_on_setpoint():
    c = _ctrl(slew_fast=False)
    c.reset_to(0.0, 11.0)
    t0 = c._last_t
    target = 5.0
    # 100 ticks of 0.1s = 10s total; slew 10 deg/s → can reach 100 deg.
    for i in range(1, 101):
        c.step(target, 11.0, now=t0 + 0.1 * i)
    p, _ = c.current
    assert p == pytest.approx(target, abs=0.1)


def test_angles_to_us_round_trip():
    c = _ctrl()
    us_pan, us_tilt = c.angles_to_us(0.0, 11.0)
    # Pan midpoint should land at 1500us per default calibration.
    assert us_pan == pytest.approx(1500.0)
    # Tilt: 11 is halfway between 0..22 on cal mapping 1000..2000 -> 1500us.
    assert us_tilt == pytest.approx(1500.0)


def test_reset_to_clamps():
    c = _ctrl()
    c.reset_to(999.0, -999.0)
    p, t = c.current
    assert p == pytest.approx(90.0)
    assert t == pytest.approx(0.0)

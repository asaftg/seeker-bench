"""Angle ↔ µs conversion, software limits, slew-rate limiting.

Stateless from the driver's point of view: ``step()`` takes the
current wall time, a target (pan, tilt) setpoint, and returns the
(pan, tilt) that should be commanded this tick after clamping to
software limits and rate-limiting against the previous command.

Why rate-limit in software instead of trusting Maestro speed/accel
registers? Because when we're auto-tracking a fused target, the
setpoint jumps around every tick — a software slew limit keeps the
gimbal smooth without having to touch Maestro config. It also
protects the servo if the config file ever gets silly values.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ServoCalibration:
    """Per-servo mapping between angle and pulse width.

    ``us_at_min_deg`` is the µs value that parks the gimbal at the
    software ``min_deg`` angle; ``us_at_max_deg`` likewise for
    ``max_deg``. These two points linearize the servo for us.

    Defaults match a generic 25 kg hobby servo (500–2500 µs) mapped
    linearly to ±90°. Real numbers should come from
    ``scripts/gimbal_calibrate.py``.
    """
    channel: int
    min_deg: float
    max_deg: float
    us_at_min_deg: float
    us_at_max_deg: float
    invert: bool = False       # flip direction if the servo is installed backwards

    def angle_to_us(self, angle_deg: float) -> float:
        a = max(self.min_deg, min(self.max_deg, float(angle_deg)))
        if self.invert:
            a = self.min_deg + (self.max_deg - a)
        span = (self.max_deg - self.min_deg) or 1.0
        t = (a - self.min_deg) / span
        return self.us_at_min_deg + t * (self.us_at_max_deg - self.us_at_min_deg)

    def us_to_angle(self, microseconds: float) -> float:
        """Inverse of ``angle_to_us``: convert a PWM value back to the
        software angle. Used to recover the servo's actual-commanded
        pose from MaestroDriver.get_last_written_us — the value the
        SERVO actually saw, not the controller's commanded setpoint.
        """
        us_span = (self.us_at_max_deg - self.us_at_min_deg) or 1.0
        t = (float(microseconds) - self.us_at_min_deg) / us_span
        a = self.min_deg + t * (self.max_deg - self.min_deg)
        if self.invert:
            a = self.min_deg + (self.max_deg - a)
        return max(self.min_deg, min(self.max_deg, a))


@dataclass
class GimbalLimits:
    pan_min_deg: float = -90.0
    pan_max_deg: float =  90.0
    tilt_min_deg: float =   0.0     # mechanical: 0 = horizon
    tilt_max_deg: float =  22.0     # mechanical: 22° = max up (user-specified)
    pan_slew_deg_per_s: float = 120.0
    tilt_slew_deg_per_s: float = 60.0


class GimbalController:
    """Convert desired (pan, tilt) into rate-limited, clamped angles.

    Does NOT talk to the driver — the manager does that. This class
    is pure math so it's trivially unit-testable.
    """

    def __init__(
        self,
        pan_cal: ServoCalibration,
        tilt_cal: ServoCalibration,
        limits: GimbalLimits,
        home_pan_deg: float = 0.0,
        home_tilt_deg: float = 11.0,
    ) -> None:
        self.pan_cal = pan_cal
        self.tilt_cal = tilt_cal
        self.limits = limits
        self._last_pan  = home_pan_deg
        self._last_tilt = home_tilt_deg
        self._last_t    = time.time()

    @property
    def current(self) -> Tuple[float, float]:
        return (self._last_pan, self._last_tilt)

    def reset_to(self, pan_deg: float, tilt_deg: float) -> None:
        self._last_pan  = self._clamp_pan(pan_deg)
        self._last_tilt = self._clamp_tilt(tilt_deg)
        self._last_t    = time.time()

    def _clamp_pan(self, v: float) -> float:
        return max(self.limits.pan_min_deg, min(self.limits.pan_max_deg, float(v)))

    def _clamp_tilt(self, v: float) -> float:
        return max(self.limits.tilt_min_deg, min(self.limits.tilt_max_deg, float(v)))

    def step(self, setpoint_pan: float, setpoint_tilt: float,
             now: float | None = None) -> Tuple[float, float]:
        """Advance one tick toward the setpoint under slew/limit rules.

        Returns the (pan_deg, tilt_deg) that should be commanded
        this tick (and caches them for next call).
        """
        if now is None:
            now = time.time()
        dt = max(1e-3, now - self._last_t)
        self._last_t = now

        tgt_pan  = self._clamp_pan(setpoint_pan)
        tgt_tilt = self._clamp_tilt(setpoint_tilt)

        max_dpan  = self.limits.pan_slew_deg_per_s  * dt
        max_dtilt = self.limits.tilt_slew_deg_per_s * dt

        dp = tgt_pan  - self._last_pan
        dt_deg = tgt_tilt - self._last_tilt
        if   dp >  max_dpan:  dp =  max_dpan
        elif dp < -max_dpan:  dp = -max_dpan
        if   dt_deg >  max_dtilt:  dt_deg =  max_dtilt
        elif dt_deg < -max_dtilt:  dt_deg = -max_dtilt

        self._last_pan  = self._clamp_pan(self._last_pan  + dp)
        self._last_tilt = self._clamp_tilt(self._last_tilt + dt_deg)
        return (self._last_pan, self._last_tilt)

    def angles_to_us(self, pan_deg: float, tilt_deg: float) -> Tuple[float, float]:
        return (self.pan_cal.angle_to_us(pan_deg),
                self.tilt_cal.angle_to_us(tilt_deg))

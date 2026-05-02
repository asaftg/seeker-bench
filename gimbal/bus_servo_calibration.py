"""Degree ↔ raw-units conversion for Waveshare ST-series serial-bus servos.

The ST3025 has a 12-bit absolute magnetic encoder over 360°, so positions
are integers in [0, 4095] where 1 unit ≈ 0.0879°. Each calibration
captures one servo's mounting:

    zero_raw      raw position that corresponds to system 0°. Encodes the
                  mounting offset — e.g. tilt is mounted so that system
                  tilt 0° (horizon) = servo at physical 90° = raw 1024.
                  Determined on-bench because the splined output coupler
                  can be re-clocked one tooth at install time.
    invert        flip direction if the servo turns the wrong way for
                  positive system angles.
    raw_min/max   hard clamps applied to every commanded raw position.
                  Defends against a bad config or a clamp miss in the
                  controller driving the servo into a mechanical stop.

Math is symmetric: angle_to_units and units_to_angle invert exactly
within rounding (±0.5 unit ≈ ±0.044°). Pure-Python, no numpy.
"""
from __future__ import annotations

from dataclasses import dataclass


UNITS_PER_REV = 4096
DEG_PER_REV = 360.0
UNITS_PER_DEG = UNITS_PER_REV / DEG_PER_REV  # 11.3777...


@dataclass
class BusServoCalibration:
    servo_id: int
    zero_raw: int = 2048
    invert: bool = False
    raw_min: int = 0
    raw_max: int = UNITS_PER_REV - 1

    def angle_to_units(self, system_deg: float) -> int:
        sign = -1.0 if self.invert else 1.0
        raw = self.zero_raw + int(round(sign * float(system_deg) * UNITS_PER_DEG))
        if raw < self.raw_min:
            return self.raw_min
        if raw > self.raw_max:
            return self.raw_max
        return raw

    def units_to_angle(self, raw: int) -> float:
        clamped = max(self.raw_min, min(self.raw_max, int(raw)))
        sign = -1.0 if self.invert else 1.0
        return sign * (clamped - self.zero_raw) / UNITS_PER_DEG

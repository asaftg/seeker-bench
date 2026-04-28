"""PWM-gating in MaestroDriver.set_target_us.

Stubs the underlying serial port so we can count actual writes to
verify that small commands get gated and only large-enough cumulative
changes reach the servo.

Why we gate: 60 Hz of tiny per-tick PWM updates (e.g. 0.5 µs each)
ride below the Yahboom servo's internal feedback deadband (~5-10 µs);
the servo treats them as noise and the camera doesn't physically
follow even though the controller's commanded position advances.
The gate lets accumulated change cross a threshold before issuing
one larger discrete step the servo can act on.
"""
from __future__ import annotations

import pytest

from gimbal.maestro_driver import MaestroDriver


class _StubSerial:
    """Just-enough drop-in for serial.Serial: counts writes."""
    def __init__(self):
        self.is_open = True
        self.writes = []

    def write(self, packet: bytes) -> int:
        self.writes.append(packet)
        return len(packet)


def _attach_stub(driver: MaestroDriver) -> _StubSerial:
    """Patch a stub serial into a driver that hasn't actually opened."""
    stub = _StubSerial()
    driver._ser = stub
    return stub


def test_legacy_no_gate_writes_every_command():
    d = MaestroDriver(min_us_step=0.0)
    s = _attach_stub(d)
    for us in (1500.0, 1500.5, 1501.0, 1501.5, 1502.0):
        d.set_target_us(0, us)
    assert len(s.writes) == 5


def test_gate_skips_below_threshold():
    d = MaestroDriver(min_us_step=5.0)
    s = _attach_stub(d)
    # First write always goes (no prior).
    d.set_target_us(0, 1500.0)
    assert len(s.writes) == 1
    # All within 5 µs of 1500.0 — gated.
    for us in (1501.0, 1502.0, 1503.0, 1504.5):
        d.set_target_us(0, us)
    assert len(s.writes) == 1, "small per-tick changes should be gated"


def test_gate_writes_when_cumulative_exceeds_threshold():
    d = MaestroDriver(min_us_step=5.0)
    s = _attach_stub(d)
    d.set_target_us(0, 1500.0)             # write 1
    d.set_target_us(0, 1502.0)             # gated (Δ=2 < 5)
    d.set_target_us(0, 1506.0)             # writes (Δ=6 from last write)
    assert len(s.writes) == 2
    d.set_target_us(0, 1507.0)             # gated (Δ=1 from 1506)
    assert len(s.writes) == 2
    d.set_target_us(0, 1512.0)             # writes (Δ=6 from 1506)
    assert len(s.writes) == 3


def test_gate_per_channel_independent():
    d = MaestroDriver(min_us_step=5.0)
    s = _attach_stub(d)
    d.set_target_us(0, 1500.0)             # ch0 first write
    d.set_target_us(1, 1500.0)             # ch1 first write
    d.set_target_us(0, 1503.0)             # ch0 gated
    d.set_target_us(1, 1510.0)             # ch1 writes
    assert len(s.writes) == 3


def test_gate_release_always_passes_and_resets_cache():
    d = MaestroDriver(min_us_step=5.0)
    s = _attach_stub(d)
    d.set_target_us(0, 1500.0)             # write
    d.set_target_us(0, 0.0)                # release — always passes
    # After release, the cache is cleared, so the next nonzero command
    # should write even if it's close to the pre-release value.
    d.set_target_us(0, 1500.5)             # writes (cache cleared)
    assert len(s.writes) == 3


def test_gate_zero_threshold_writes_every_call():
    d = MaestroDriver(min_us_step=0.0)
    s = _attach_stub(d)
    d.set_target_us(0, 1500.0)
    d.set_target_us(0, 1500.0001)
    assert len(s.writes) == 2  # not gated when threshold == 0

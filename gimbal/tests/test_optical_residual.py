"""Unit tests for the OpticalResidualTracker.

Synthetic scenes — no rig, no recordings. Verifies that LK + the
pixel-to-angle math produce the expected residual on:
    1. Matched-cmd case (commanded delta == actual scene shift) -> ~0 residual
    2. Undershoot case  (cmd +1deg tilt, no scene motion) -> +1.0 residual
    3. Chained measurements over multiple ticks accumulate correctly
    4. Shape mismatch (resolution/zoom changed) is detected
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from gimbal.optical_residual import OpticalResidualTracker


@pytest.fixture
def textured_frame() -> np.ndarray:
    rng = np.random.default_rng(42)
    h, w = 512, 640
    canvas = cv2.GaussianBlur(
        rng.integers(0, 255, (h, w), dtype=np.uint8), (5, 5), 0)
    for _ in range(40):
        cv2.circle(
            canvas,
            (int(rng.integers(50, w - 50)), int(rng.integers(50, h - 50))),
            int(rng.integers(8, 30)),
            int(rng.integers(0, 255)),
            -1,
        )
    return canvas


def _shift(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_matched_cmd_zero_residual(textured_frame):
    t = OpticalResidualTracker()
    assert t.set_anchor(textured_frame, hfov_deg=18.75, vfov_deg=15.0,
                        cur_pan=10.0, cur_tilt=5.0, t=0.0)
    # Shift scene 30 px right, 20 px up. At 18.75deg/640px and 15deg/512px:
    #   actual camera pan = -(30/640)*18.75 = -0.879 deg (panned LEFT)
    #   actual camera tilt = +(-20/512)*15  = -0.586 deg (tilted DOWN)
    shifted = _shift(textured_frame, 30, -20)
    m = t.measure(shifted, cur_pan=10.0 - 0.879, cur_tilt=5.0 - 0.586,
                  hfov_deg=18.75, vfov_deg=15.0)
    assert m.valid
    assert abs(m.dx_px - 30.0) < 0.5
    assert abs(m.dy_px - (-20.0)) < 0.5
    assert abs(m.daz_residual_deg) < 0.05
    assert abs(m.del_residual_deg) < 0.05


def test_undershoot_yields_residual(textured_frame):
    """Camera commanded +1 deg tilt up but scene didn't move: residual is +1."""
    t = OpticalResidualTracker()
    t.set_anchor(textured_frame, 18.75, 15.0, 10.0, 5.0, 0.0)
    m = t.measure(textured_frame.copy(), cur_pan=10.0, cur_tilt=6.0,
                  hfov_deg=18.75, vfov_deg=15.0)
    assert m.valid
    assert abs(m.del_residual_deg - 1.0) < 0.05


def test_chained_measurement_accumulates(textured_frame):
    t = OpticalResidualTracker()
    t.set_anchor(textured_frame, 18.75, 15.0, 0.0, 0.0, 0.0)
    for k in range(1, 4):
        m = t.measure(_shift(textured_frame, 10 * k, 0),
                      cur_pan=0.0, cur_tilt=0.0,
                      hfov_deg=18.75, vfov_deg=15.0)
        assert m.valid
        assert abs(m.dx_px - 10 * k) < 0.5


def test_shape_mismatch_invalid(textured_frame):
    t = OpticalResidualTracker()
    t.set_anchor(textured_frame, 18.75, 15.0, 0.0, 0.0, 0.0)
    small = cv2.resize(textured_frame, (320, 256))
    m = t.measure(small, 0.0, 0.0)
    assert not m.valid
    assert m.note == "shape_mismatch"


def test_no_anchor_returns_invalid(textured_frame):
    t = OpticalResidualTracker()
    m = t.measure(textured_frame, 0.0, 0.0)
    assert not m.valid
    assert m.note == "no_anchor"

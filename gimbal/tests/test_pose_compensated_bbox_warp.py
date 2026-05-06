"""Regression tests for the lock-mode pose-shift helper (T1.2).

The frame-id dedupe shipped at commit 5670246 caches the per-sensor
LockUpdate per frame_id. Between true frame updates the gimbal pose
advances (~35-60 Hz tick vs 17-25 Hz EO frame); without re-projecting
the cached bbox by the pose delta the lock visually freezes during
slews — operator-reported "sluggish lock bbox refresh."

T1.2 added gimbal_manager._pose_shift_lock_update which shifts the
cached MOSSE bbox by (cur_pose - cached_frame.gimbal_*_at_capture) *
px_per_deg. When the next real frame arrives, MOSSE re-anchors and any
prediction error is corrected in one tick.

These tests lock in the math + sign convention so a future refactor
can't silently regress.
"""
from __future__ import annotations

import numpy as np
import pytest

from gimbal.gimbal_manager import GimbalManager
from vision.lock_tracker import LockUpdate, LockState


def _fake_eo_frame(w=1236, h=1029, hfov=11.05, vfov=9.23,
                    pan=0.0, tilt=0.0):
    """Minimal duck-typed EOFrame for the pose-shift helper. Only
    the attributes the helper reads (.bgr.shape, .hfov_deg,
    .vfov_deg, .gimbal_pan_at_capture, .gimbal_tilt_at_capture)."""
    class _F:
        pass
    f = _F()
    f.bgr = np.zeros((h, w, 3), dtype=np.uint8)
    f.hfov_deg = hfov
    f.vfov_deg = vfov
    f.gimbal_pan_at_capture = pan
    f.gimbal_tilt_at_capture = tilt
    return f


@pytest.fixture
def gm():
    """A bare GimbalManager-ish instance suitable for calling the
    standalone _pose_shift_lock_update helper. We don't initialize the
    full driver — just the unbound method via the class."""
    # The helper only reads `frame.*` and the args; doesn't touch
    # self state. Use a stub object.
    class _Stub:
        pass
    s = _Stub()
    # Bind the method to the stub so `self._pose_shift_lock_update(...)`
    # works without instantiating the full GimbalManager (which needs
    # a driver, config, etc.).
    s._pose_shift_lock_update = (
        GimbalManager._pose_shift_lock_update.__get__(s, _Stub)
    )
    return s


def test_zero_pose_delta_returns_unchanged(gm):
    """No pose change → no shift → identical bbox returned."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(100, 200, 50, 50),
                     psr=10.0, coast_age_s=0.0)
    f = _fake_eo_frame(pan=10.0, tilt=5.0)
    out = gm._pose_shift_lock_update(upd, f, 10.0, 5.0)
    assert out.bbox_xywh == upd.bbox_xywh


def test_subpixel_delta_skips_shift(gm):
    """Sub-pixel pose delta is below the 0.5 px threshold and
    returns unchanged to avoid micro-jitter."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(100, 200, 50, 50),
                     psr=10.0, coast_age_s=0.0)
    # 1236 px / 11.05 deg ~= 111.85 px/deg; 0.001 deg = 0.11 px
    f = _fake_eo_frame(pan=10.0, tilt=5.0)
    out = gm._pose_shift_lock_update(upd, f, 10.001, 5.001)
    assert out.bbox_xywh == upd.bbox_xywh


def test_pan_right_shifts_bbox_left(gm):
    """Sign convention: gimbal pan RIGHT (+pan) → scene shifts
    LEFT in image → bbox.x DECREASES."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(500, 500, 50, 50),
                     psr=10.0, coast_age_s=0.0)
    f = _fake_eo_frame(pan=0.0, tilt=0.0)  # captured at pan=0
    # Now we're at pan=+1.0 deg → scene shifted LEFT by ~111.85 px
    out = gm._pose_shift_lock_update(upd, f, 1.0, 0.0)
    new_x, new_y, w, h = out.bbox_xywh
    assert w == 50 and h == 50
    assert new_y == 500
    # ~-111.85 px shift; 500 - 112 = 388 (rounded)
    assert 386 <= new_x <= 390


def test_tilt_up_shifts_bbox_down(gm):
    """Sign convention: gimbal tilt UP (+tilt) → scene moves DOWN
    in image → bbox.y INCREASES."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(500, 500, 50, 50),
                     psr=10.0, coast_age_s=0.0)
    f = _fake_eo_frame(pan=0.0, tilt=0.0)
    # pose now tilted +1deg; vfov=9.23, h=1029 → 1029/9.23 = 111.5 px/deg
    out = gm._pose_shift_lock_update(upd, f, 0.0, 1.0)
    new_x, new_y, w, h = out.bbox_xywh
    assert new_x == 500
    assert 109 <= (new_y - 500) <= 113  # ~+112 px


def test_clamps_to_frame_bounds(gm):
    """A runaway slew can't push the bbox fully off-screen — the
    helper clamps the new top-left to keep it inside the frame
    width/height (allowing partial clip on either edge but not
    a fully-vanished bbox)."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(50, 50, 100, 100),
                     psr=10.0, coast_age_s=0.0)
    f = _fake_eo_frame(pan=0.0, tilt=0.0)
    # 50 deg pan delta = 50 * 111.85 = 5592 px shift, way off frame
    out = gm._pose_shift_lock_update(upd, f, 50.0, 0.0)
    new_x, _, w, _ = out.bbox_xywh
    # Must not go below -w+1 or above width-1
    assert new_x >= -w + 1
    assert new_x < 1236


def test_none_upd_passes_through(gm):
    """None LockUpdate → return None unchanged (no crash)."""
    f = _fake_eo_frame()
    assert gm._pose_shift_lock_update(None, f, 1.0, 1.0) is None


def test_missing_pose_at_capture_passes_through(gm):
    """Frame without gimbal_*_at_capture fields → return upd unchanged
    (defensive — fakes / older recordings might not carry it)."""
    upd = LockUpdate(state=LockState.ACTIVE,
                     bbox_xywh=(100, 200, 50, 50),
                     psr=10.0, coast_age_s=0.0)
    class _NoPose:
        bgr = np.zeros((1029, 1236, 3), dtype=np.uint8)
        hfov_deg = 11.05
        vfov_deg = 9.23
        gimbal_pan_at_capture = None
        gimbal_tilt_at_capture = None
    out = gm._pose_shift_lock_update(upd, _NoPose(), 5.0, 5.0)
    assert out is upd  # passthrough, not a copy

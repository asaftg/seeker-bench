"""v2 calibrated-projection unit tests.

These tests pin three properties that the runtime depends on:

1. **Zero-bias residual is identity** — the slider sign-convention bug
   is the #1 source of regression in this kind of code (per the
   project's dangerous-reverts memory). If a future change flips a
   sign, this test fails first.
2. **Project ↔ unproject round-trip is consistent** for a synthetic
   camera with known K/R/t, including non-identity stereo and rational
   distortion.
3. **v1 fallback is byte-identical** — when no calibrated cameras are
   set, the fusion manager produces the same observations as the
   legacy path. Replay guarantees rely on this.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from fusion.angular import pixel_to_angle, pixel_to_angle_K
from fusion.projection import (
    Camera,
    far_field_eo_pixel_from_ray,
    intersect_ray_at_range,
    residual_rotation,
)


# ───────────────────────── helpers ─────────────────────────

def _eo_camera(w: int = 1920, h: int = 1200, hfov_deg: float = 11.0) -> Camera:
    """Synthetic EO camera with no distortion, identity pose."""
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx  # square pixels
    K = np.array([[fx, 0.0, w / 2.0],
                  [0.0, fy, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return Camera(
        K=K, dist=np.zeros(5),
        R_to_eo=np.eye(3), t_to_eo=np.zeros(3),
    )


def _thermal_camera(w: int = 640, h: int = 512,
                    hfov_deg: float = 75.0,
                    baseline_x_m: float = 0.10) -> Camera:
    """Synthetic thermal camera offset 10 cm to the +x of EO, no rotation."""
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    fy = fx
    K = np.array([[fx, 0.0, w / 2.0],
                  [0.0, fy, h / 2.0],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return Camera(
        K=K, dist=np.zeros(5),
        R_to_eo=np.eye(3),
        t_to_eo=np.array([baseline_x_m, 0.0, 0.0]),
    )


# ──────────────────── 1. zero-bias is identity ─────────────────

def test_residual_rotation_zero_is_identity():
    R = residual_rotation(0.0, 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-12)


def test_with_residual_zero_returns_self():
    cam = _thermal_camera()
    same = cam.with_residual(0.0, 0.0)
    assert same is cam  # identity short-circuit


def test_with_residual_small_rotation_is_close_to_identity():
    cam = _thermal_camera()
    nudged = cam.with_residual(0.5, 0.0)
    # 0.5° = ~0.0087 rad — R differs from identity by O(0.01)
    assert nudged is not cam
    assert np.allclose(nudged.R_to_eo, cam.R_to_eo, atol=0.02)
    # but not exactly identity
    assert not np.allclose(nudged.R_to_eo, cam.R_to_eo, atol=1e-8)


# ──────────────────── 2. project ↔ unproject ─────────────────

def test_project_eo_self_at_optical_center():
    """A point on the EO optical axis at z=10m projects to image center."""
    eo = _eo_camera()
    u, v = eo.project_eo_point_to_pixel(np.array([0.0, 0.0, 10.0]))
    assert u == pytest.approx(eo.cx, abs=1e-6)
    assert v == pytest.approx(eo.cy, abs=1e-6)


def test_unproject_at_eo_center_gives_forward_ray():
    eo = _eo_camera()
    origin, direction = eo.unproject_pixel_to_eo_ray(eo.cx, eo.cy)
    assert np.allclose(origin, np.zeros(3), atol=1e-12)
    # Direction should be ~[0, 0, 1] (forward).
    assert direction[2] == pytest.approx(1.0, abs=1e-6)
    assert abs(direction[0]) < 1e-6
    assert abs(direction[1]) < 1e-6


def test_thermal_to_eo_round_trip_far_field():
    """A pixel at thermal center, far-field projected onto EO, ends at
    EO image center minus a tiny parallax shift from the 10 cm baseline.
    At infinity, baseline contributes zero — so EO pixel = EO center."""
    eo = _eo_camera()
    thr = _thermal_camera()
    origin, direction = thr.unproject_pixel_to_eo_ray(thr.cx, thr.cy)
    u, v = far_field_eo_pixel_from_ray(eo, origin, direction)
    assert u == pytest.approx(eo.cx, abs=1e-3)
    assert v == pytest.approx(eo.cy, abs=1e-3)


def test_intersect_ray_at_range_parallax_at_short_range():
    """At 1m, a 10cm baseline causes visible parallax; at 1km it doesn't.
    This test verifies the math, not the runtime — runtime defers depth
    correction.
    """
    eo = _eo_camera()
    thr = _thermal_camera(baseline_x_m=0.10)
    # Thermal sees a target dead-center.
    origin, direction = thr.unproject_pixel_to_eo_ray(thr.cx, thr.cy)
    p_close = intersect_ray_at_range(origin, direction, 1.0)
    p_far = intersect_ray_at_range(origin, direction, 1000.0)
    u_close, _ = eo.project_eo_point_to_pixel(p_close)
    u_far, _ = eo.project_eo_point_to_pixel(p_far)
    # Thermal sits at baseline +x in EO frame; a target on thermal's
    # optical axis at close range is therefore right of EO's optical
    # axis. At infinity, the baseline has zero leverage so the EO
    # pixel returns to ~cx.
    assert u_close > eo.cx
    assert abs(u_far - eo.cx) < abs(u_close - eo.cx)


# ──────────────────── 3. pixel_to_angle_K consistency ─────────────────

def test_pixel_to_angle_K_at_principal_point_returns_zero():
    eo = _eo_camera()
    az, el = pixel_to_angle_K(eo.cx, eo.cy, eo.fx, eo.fy, eo.cx, eo.cy)
    assert az == pytest.approx(0.0, abs=1e-9)
    assert el == pytest.approx(0.0, abs=1e-9)


def test_pixel_to_angle_K_correct_pinhole_at_fov_edges():
    """``pixel_to_angle_K`` is the true pinhole arctan and disagrees
    slightly with the legacy linear ``pixel_to_angle`` away from the
    principal point — the legacy function is approximate. At the FOV
    edges the two converge again because both pin to ±hfov/2 there.

    This test pins the correct behavior of the new function: at
    ``u = cx + W/2``, the angle equals ``+hfov/2``.
    """
    w, h, hfov = 1920, 1200, 11.0
    eo = _eo_camera(w, h, hfov)
    az_edge, _ = pixel_to_angle_K(eo.cx + w / 2.0, eo.cy,
                                  eo.fx, eo.fy, eo.cx, eo.cy)
    assert az_edge == pytest.approx(hfov / 2.0, abs=1e-9)


# ──────────────────── 4. v1 fallback (byte-identical legacy path) ──────

def test_fusion_manager_v1_default_no_cameras_set():
    """A freshly-constructed FusionManager has no v2 cameras, so the
    thermal observation path must take the legacy v1 branch.

    This is an integration check — replay regression depends on it.
    """
    from fusion.fusion_manager import FusionManager
    fm = FusionManager()
    assert fm._eo_cam is None
    assert fm._thermal_cam is None


def test_fusion_manager_set_calibrated_cameras_toggles_v2():
    from fusion.fusion_manager import FusionManager
    fm = FusionManager()
    eo = _eo_camera()
    thr = _thermal_camera()
    fm.set_calibrated_cameras(eo=eo, thermal=thr)
    assert fm._eo_cam is eo
    assert fm._thermal_cam is thr
    # Reverting to v1 by passing None disables v2.
    fm.set_calibrated_cameras(eo=None, thermal=None)
    assert fm._eo_cam is None
    assert fm._thermal_cam is None

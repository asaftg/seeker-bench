"""Calibrated multi-sensor projection.

This module replaces the FOV-only pinhole math in ``fusion.angular`` for
the v2 calibration path. ``angular.py`` is left untouched — it remains
the v1 path and the cross-sensor association layer reads angular form
from us either way.

Frames and conventions
----------------------

- **EO is the reference frame.** ``R_to_eo``, ``t_to_eo`` map a point
  expressed in this camera's local frame to the EO frame:

      P_eo = R_to_eo @ P_self + t_to_eo

  For the EO ``Camera`` itself, R = I, t = 0.

- **Camera frame is OpenCV-standard.** x = right (image x), y = down
  (image y), z = forward (boresight). All distortion/projection math
  in cv2 uses this convention.

- **Distance units are metres.** Translations stored that way.

- **Residual rotation (slider) composition:**

      R_effective_to_eo = R_to_eo @ Rodrigues([0, az_rad, 0])
                                  @ Rodrigues([el_rad, 0, 0])

  i.e. residual is applied in the sensor's local frame, az first about
  the camera-down axis, then el about the camera-right axis. ``az_bias``
  and ``el_bias`` are degrees. Sign convention: positive az rotates the
  boresight to the camera's right (matches the existing angular slider
  intuition); positive el rotates it up. Future-you who is debugging
  why the slider feels backward: the issue is almost certainly here,
  not in the GUI. The unit test asserts that zero bias is identity, so
  any sign flip is a deliberate code change.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Small-rotation Rodrigues without pulling cv2 just for this.

    rvec is a 3-vector whose direction is the axis and magnitude is the
    angle in radians. Returns the 3x3 rotation matrix.
    """
    theta = float(np.linalg.norm(rvec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = (rvec / theta).reshape(3)
    K = np.array([
        [0.0, -k[2], k[1]],
        [k[2], 0.0, -k[0]],
        [-k[1], k[0], 0.0],
    ], dtype=np.float64)
    return (np.eye(3) + math.sin(theta) * K
            + (1.0 - math.cos(theta)) * (K @ K))


def residual_rotation(az_bias_deg: float, el_bias_deg: float) -> np.ndarray:
    """Build the small-angle residual rotation in the sensor-local frame.

    See module docstring for sign convention. Returns identity for
    (0, 0). This function exists separately so the unit test can pin
    the convention without depending on cv2.
    """
    az = math.radians(float(az_bias_deg))
    el = math.radians(float(el_bias_deg))
    R_az = _rodrigues(np.array([0.0, az, 0.0]))   # about camera-down
    R_el = _rodrigues(np.array([el, 0.0, 0.0]))   # about camera-right
    return R_az @ R_el


@dataclass
class Camera:
    """Calibrated pinhole camera with distortion + 6-DoF pose to EO frame.

    Construct one per sensor at startup from ``calibration.json`` v2.
    The fusion manager stores live instances and re-applies residual
    sliders by calling :meth:`with_residual`.
    """

    K: np.ndarray            # 3x3
    dist: np.ndarray         # length-5 (standard) or length-8 (rational)
    R_to_eo: np.ndarray      # 3x3
    t_to_eo: np.ndarray      # length-3, metres

    # Cached inverse rotation; populated lazily (R^T == R^-1 for
    # rotation matrices).
    _R_eo_to_self: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(self.dist, dtype=np.float64).reshape(-1)
        self.R_to_eo = np.asarray(self.R_to_eo, dtype=np.float64).reshape(3, 3)
        self.t_to_eo = np.asarray(self.t_to_eo, dtype=np.float64).reshape(3)

    # ------- accessors -------

    @property
    def fx(self) -> float: return float(self.K[0, 0])
    @property
    def fy(self) -> float: return float(self.K[1, 1])
    @property
    def cx(self) -> float: return float(self.K[0, 2])
    @property
    def cy(self) -> float: return float(self.K[1, 2])

    def hfov_deg(self, frame_w: int) -> float:
        """Horizontal FOV implied by K and frame width. Used to feed the
        legacy ``angular_to_bbox`` call that still wants degrees."""
        return 2.0 * math.degrees(math.atan2(frame_w / 2.0, self.fx))

    def vfov_deg(self, frame_h: int) -> float:
        return 2.0 * math.degrees(math.atan2(frame_h / 2.0, self.fy))

    # ------- residual slider composition -------

    def with_residual(self, az_bias_deg: float, el_bias_deg: float) -> "Camera":
        """Return a copy whose R_to_eo includes the slider residual.

        Calibrated R is the baseline; sliders nudge in sensor-local frame.
        Zero bias yields a Camera identical to ``self`` (verified by tests).
        """
        if az_bias_deg == 0.0 and el_bias_deg == 0.0:
            return self
        R_res = residual_rotation(az_bias_deg, el_bias_deg)
        return Camera(
            K=self.K.copy(),
            dist=self.dist.copy(),
            R_to_eo=self.R_to_eo @ R_res,
            t_to_eo=self.t_to_eo.copy(),
        )

    # ------- projection -------

    def project_eo_point_to_pixel(self, P_eo: np.ndarray) -> Tuple[float, float]:
        """Project a 3D point given in EO frame onto this camera's image.

        Brings P_eo into this camera's local frame via the inverse of
        R_to_eo, t_to_eo, then applies K + distortion. Uses cv2 for the
        distortion model (handles 5-param and 8-param rational alike).
        """
        import cv2  # local import — keeps test_projection importable without cv2
        P_eo = np.asarray(P_eo, dtype=np.float64).reshape(3)
        if self._R_eo_to_self is None:
            self._R_eo_to_self = self.R_to_eo.T
        P_self = self._R_eo_to_self @ (P_eo - self.t_to_eo)
        # cv2.projectPoints expects rvec/tvec for the world-to-camera
        # transform; we already did it, so pass zeros.
        pts, _ = cv2.projectPoints(
            P_self.reshape(1, 1, 3),
            np.zeros(3), np.zeros(3),
            self.K, self.dist,
        )
        u, v = pts.reshape(2)
        return float(u), float(v)

    def unproject_pixel_to_eo_ray(self, u: float, v: float) -> Tuple[np.ndarray, np.ndarray]:
        """Unproject a pixel to a 3D ray expressed in EO frame.

        Returns ``(origin_eo, direction_eo)``. Origin is this camera's
        optical center in EO frame (= t_to_eo). Direction is unit-length.

        The caller intersects this ray at a known depth (e.g. a fused
        radar range) to get a world point, then re-projects via
        ``Camera.project_eo_point_to_pixel`` of the destination camera.
        """
        import cv2
        # Undistort to normalized image coords (z=1 plane in this camera).
        pts = np.array([[[u, v]]], dtype=np.float64)
        norm = cv2.undistortPoints(pts, self.K, self.dist).reshape(2)
        ray_self = np.array([norm[0], norm[1], 1.0], dtype=np.float64)
        ray_eo = self.R_to_eo @ ray_self
        ray_eo /= np.linalg.norm(ray_eo)
        return self.t_to_eo.copy(), ray_eo


def far_field_eo_pixel_from_ray(
    eo_camera: Camera,
    origin_eo: np.ndarray,
    direction_eo: np.ndarray,
) -> Tuple[float, float]:
    """Project a ray onto the EO image at the far-field limit.

    When no depth is known (no fused radar range), translation between
    cameras has zero leverage: a ray at infinity has the same image
    coordinates regardless of the camera optical center. We pick a
    very large depth along the ray and project.

    This is mathematically equivalent to the homography-at-infinity
    that the legacy angular path implicitly uses, so v1→v2 with no
    radar range produces a result very close to v1 (modulo distortion
    correction, which is the whole point of v2).
    """
    P_far = origin_eo + 1.0e6 * direction_eo
    return eo_camera.project_eo_point_to_pixel(P_far)


def intersect_ray_at_range(
    origin_eo: np.ndarray,
    direction_eo: np.ndarray,
    range_m: float,
) -> np.ndarray:
    """Point along the ray at the given range from the EO origin.

    Used when a fused track has a radar range — gives parallax-correct
    re-projection. Direction must be unit-length (Camera produces it
    that way).
    """
    return origin_eo + float(range_m) * direction_eo

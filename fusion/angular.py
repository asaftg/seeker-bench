"""Pixel ↔ angle conversions for a pinhole-like sensor.

We model each sensor as an ideal rectilinear camera with a known
horizontal and vertical FOV. The boresight is the frame center;
az/el are angles relative to that center.

Convention:
    az_deg positive → right
    el_deg positive → up
    pixel (0, 0) is top-left
"""
from __future__ import annotations

from typing import Tuple


def pixel_to_angle(
    cx_px: float, cy_px: float,
    frame_w: int, frame_h: int,
    hfov_deg: float, vfov_deg: float,
) -> Tuple[float, float]:
    """Map a pixel coordinate to (az_deg, el_deg) relative to boresight."""
    if frame_w <= 0 or frame_h <= 0:
        return 0.0, 0.0
    az = ((cx_px / frame_w) - 0.5) * hfov_deg
    el = (0.5 - (cy_px / frame_h)) * vfov_deg
    return az, el


def pixel_to_angle_K(
    cx_px: float, cy_px: float,
    fx: float, fy: float, cx: float, cy: float,
) -> Tuple[float, float]:
    """Map a pixel coordinate to (az_deg, el_deg) using camera intrinsics.

    Used by the v2 projection path: after we project a world point to the
    EO image plane via cv2.projectPoints, we convert the resulting pixel
    back to angular form so the existing angular_iou-based association
    layer is unchanged.

    Convention matches pixel_to_angle: az positive = right, el positive
    = up. Sign on el flips because pixel y grows downward.

    fx, fy in pixels; cx, cy = principal point in pixels (NOT image
    center — use the calibrated value).
    """
    import math
    if fx <= 0 or fy <= 0:
        return 0.0, 0.0
    az = math.degrees(math.atan2(cx_px - cx, fx))
    el = -math.degrees(math.atan2(cy_px - cy, fy))
    return az, el


def bbox_to_angular(
    bx: float, by: float, bw: float, bh: float,
    frame_w: int, frame_h: int,
    hfov_deg: float, vfov_deg: float,
) -> Tuple[float, float, float, float]:
    """Return (az_center, el_center, angular_width, angular_height)."""
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    az, el = pixel_to_angle(cx, cy, frame_w, frame_h, hfov_deg, vfov_deg)
    ang_w = (bw / max(1, frame_w)) * hfov_deg
    ang_h = (bh / max(1, frame_h)) * vfov_deg
    return az, el, ang_w, ang_h


def angular_to_bbox(
    az_deg: float, el_deg: float,
    ang_w_deg: float, ang_h_deg: float,
    frame_w: int, frame_h: int,
    hfov_deg: float, vfov_deg: float,
) -> Tuple[int, int, int, int]:
    """Project an angular bbox back into a target sensor's pixel grid.

    Returns (x, y, w, h) clamped to [0, frame_w/h]. The projected bbox
    may fall partly outside the target FOV — callers should check the
    returned size before rendering.
    """
    if frame_w <= 0 or frame_h <= 0 or hfov_deg <= 0 or vfov_deg <= 0:
        return 0, 0, 0, 0
    cx_norm = 0.5 + (az_deg / hfov_deg)
    cy_norm = 0.5 - (el_deg / vfov_deg)
    cx = cx_norm * frame_w
    cy = cy_norm * frame_h
    w = max(1.0, (ang_w_deg / hfov_deg) * frame_w)
    h = max(1.0, (ang_h_deg / vfov_deg) * frame_h)
    x = cx - w / 2.0
    y = cy - h / 2.0
    # Clamp (allow negative then clip so a partially-offscreen target
    # still shows its visible edge).
    x = max(0.0, min(float(frame_w), x))
    y = max(0.0, min(float(frame_h), y))
    w = max(0.0, min(float(frame_w) - x, w))
    h = max(0.0, min(float(frame_h) - y, h))
    return int(x), int(y), int(w), int(h)


def angular_iou(
    az_a: float, el_a: float, w_a: float, h_a: float,
    az_b: float, el_b: float, w_b: float, h_b: float,
) -> float:
    """Intersection-over-Union of two axis-aligned bboxes in (az, el) space.

    Treats each (az_center, el_center, ang_w, ang_h) as a rectangle in
    angular space. Useful as the association metric across sensors and
    across ticks: a small target inside a big target's bbox has IoU≈0
    (tiny intersection, huge union), so the big one doesn't false-merge
    distant same-class neighbors.
    """
    ax1 = az_a - w_a / 2.0
    ax2 = az_a + w_a / 2.0
    ay1 = el_a - h_a / 2.0
    ay2 = el_a + h_a / 2.0
    bx1 = az_b - w_b / 2.0
    bx2 = az_b + w_b / 2.0
    by1 = el_b - h_b / 2.0
    by2 = el_b + h_b / 2.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, w_a) * max(0.0, h_a)
    area_b = max(0.0, w_b) * max(0.0, h_b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def angular_bbox_visible(
    az_deg: float, el_deg: float,
    ang_w_deg: float, ang_h_deg: float,
    hfov_deg: float, vfov_deg: float,
) -> bool:
    """True if any part of the angular bbox lies inside the sensor FOV."""
    return (
        abs(az_deg) < (hfov_deg / 2.0 + ang_w_deg / 2.0) and
        abs(el_deg) < (vfov_deg / 2.0 + ang_h_deg / 2.0)
    )


def angular_iou_matrix(a_boxes, b_boxes):
    """Vectorized IoU between two sets of angular bboxes.

    Args:
        a_boxes: array-like of shape (N, 4) — rows are (az, el, w, h)
        b_boxes: array-like of shape (M, 4) — rows are (az, el, w, h)

    Returns:
        numpy array of shape (N, M) with IoU values in [0, 1].

    Used by `fusion.fusion_manager._update_tracks` (and dedup/merge)
    to compute the full candidate×track IoU table in one numpy call
    instead of a O(C·T) pure-Python loop. ~25x faster at T=30 (3-5 ms
    -> ~0.15 ms in benchmarks); scales cleanly to T=200+.

    Bit-equivalent to scalar `angular_iou(a_i, b_j)` for any cell —
    verified by `test_angular_iou_matrix_matches_scalar`.

    Notes:
      * Numpy arrays are created here even if N or M is small; the
        ~10 us overhead is negligible vs the per-pair Python-call cost.
      * Returns shape (N, M) — caller uses `.argmax(axis=1)` to pick
        each candidate's best track, or `[i, j]` to look up a
        specific pair.
    """
    import numpy as np
    a = np.asarray(a_boxes, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b_boxes, dtype=np.float64).reshape(-1, 4)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float64)
    # (N, 1) vs (1, M) broadcast
    ax1 = (a[:, 0] - a[:, 2] / 2.0)[:, None]
    ax2 = (a[:, 0] + a[:, 2] / 2.0)[:, None]
    ay1 = (a[:, 1] - a[:, 3] / 2.0)[:, None]
    ay2 = (a[:, 1] + a[:, 3] / 2.0)[:, None]
    bx1 = (b[:, 0] - b[:, 2] / 2.0)[None, :]
    bx2 = (b[:, 0] + b[:, 2] / 2.0)[None, :]
    by1 = (b[:, 1] - b[:, 3] / 2.0)[None, :]
    by2 = (b[:, 1] + b[:, 3] / 2.0)[None, :]
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (np.maximum(0.0, a[:, 2]) * np.maximum(0.0, a[:, 3]))[:, None]
    area_b = (np.maximum(0.0, b[:, 2]) * np.maximum(0.0, b[:, 3]))[None, :]
    union = area_a + area_b - inter
    out = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
    # Force exact 0 where inter==0 to match scalar's early-return.
    out[inter <= 0] = 0.0
    return out

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

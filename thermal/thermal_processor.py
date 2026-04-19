"""
Thermal frame processing primitives.

Pure functions — no state, no threads. Easy to unit-test with
synthetic 16-bit arrays. The `ThermalManager` thread composes
these into a pipeline.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────
# Automatic Gain Control (AGC)
# ───────────────────────────────────────────────────────────────

def apply_agc(
    frame_u16: np.ndarray,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """Percentile-clip linear stretch a 16-bit thermal frame to 8-bit.

    Robust to hot/cold outliers because we discard the extreme
    percentiles before stretching.
    """
    if frame_u16.ndim != 2:
        raise ValueError(f"apply_agc expects 2-D uint16, got shape {frame_u16.shape}")

    lo, hi = np.percentile(frame_u16, [low_percentile, high_percentile])
    if hi <= lo:
        # Flat frame — return zeros rather than dividing by 0
        return np.zeros_like(frame_u16, dtype=np.uint8)

    stretched = (frame_u16.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────
# Colormap
# ───────────────────────────────────────────────────────────────

_COLORMAP_LOOKUP = {
    "INFERNO":   cv2.COLORMAP_INFERNO,
    "IRONBOW":   cv2.COLORMAP_HOT,       # OpenCV has no ironbow; HOT is closest
    "WHITE_HOT": None,                   # no colormap; stay grayscale → BGR
    "JET":       cv2.COLORMAP_JET,
    "MAGMA":     cv2.COLORMAP_MAGMA,
}


def apply_colormap(frame_u8: np.ndarray, name: str = "INFERNO") -> np.ndarray:
    """Apply a colormap to an 8-bit thermal frame. Returns BGR."""
    if frame_u8.ndim != 2:
        raise ValueError(f"apply_colormap expects 2-D uint8, got shape {frame_u8.shape}")

    cmap = _COLORMAP_LOOKUP.get(name.upper())
    if cmap is None:
        # WHITE_HOT: grayscale → BGR
        return cv2.cvtColor(frame_u8, cv2.COLOR_GRAY2BGR)
    return cv2.applyColorMap(frame_u8, cmap)


# ───────────────────────────────────────────────────────────────
# Convenience: raw16 → display BGR in one shot
# ───────────────────────────────────────────────────────────────

def raw16_to_display(
    frame_u16: np.ndarray,
    colormap: str = "INFERNO",
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (agc_u8, display_bgr) from a raw 16-bit frame."""
    agc = apply_agc(frame_u16, low_percentile, high_percentile)
    bgr = apply_colormap(agc, colormap)
    return agc, bgr

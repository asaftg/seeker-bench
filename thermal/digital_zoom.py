"""
Digital zoom via center-crop.

The FLIR ADK we have is 75° HFOV. The production sensor is 12°.
We emulate narrower FOVs by cropping the center of the frame.
This does NOT increase the pixels-on-target (critical note from
the work instructions) — it just removes surrounding FOV for
display, which is useful to show the operator what the production
sensor would see.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass(frozen=True)
class ZoomPreset:
    name: str
    hfov_deg: float


# Named presets. Key is the preset identifier used in app_config.yaml.
PRESETS: Dict[str, ZoomPreset] = {
    "full":   ZoomPreset("full",   75.0),
    "wide":   ZoomPreset("wide",   37.5),
    "mid":    ZoomPreset("mid",    18.75),
    "narrow": ZoomPreset("narrow", 12.5),
}


def crop_fraction(full_hfov_deg: float, target_hfov_deg: float) -> float:
    """Return the fraction of the frame dimension to keep after cropping."""
    if target_hfov_deg >= full_hfov_deg:
        return 1.0
    if target_hfov_deg <= 0:
        raise ValueError("target_hfov_deg must be > 0")
    # Small-angle approximation is fine here; exact tan ratio for precision:
    return float(np.tan(np.radians(target_hfov_deg / 2)) /
                 np.tan(np.radians(full_hfov_deg / 2)))


def center_crop(
    frame: np.ndarray,
    full_hfov_deg: float,
    target_hfov_deg: float,
    upscale_to_original: bool = True,
) -> np.ndarray:
    """Center-crop `frame` to emulate a narrower FOV.

    If `upscale_to_original`, the crop is bilinear-upscaled back
    to the original dimensions so the GUI rendering size stays
    constant.
    """
    if frame.ndim not in (2, 3):
        raise ValueError(f"Unsupported frame shape {frame.shape}")

    frac = crop_fraction(full_hfov_deg, target_hfov_deg)
    h, w = frame.shape[:2]
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))

    x0 = (w - cw) // 2
    y0 = (h - ch) // 2
    cropped = frame[y0:y0 + ch, x0:x0 + cw]

    if upscale_to_original and (ch != h or cw != w):
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return cropped


def apply_preset(frame: np.ndarray, full_hfov_deg: float, preset: str) -> np.ndarray:
    """Apply a named preset from PRESETS."""
    if preset not in PRESETS:
        raise KeyError(f"Unknown zoom preset '{preset}'. Known: {list(PRESETS)}")
    return center_crop(frame, full_hfov_deg, PRESETS[preset].hfov_deg)

"""Parameterized post-processing pipeline the optimizer searches over.

Input: raw mono Y-plane uint8 (H, W) — the same buffer ``IMX568Capture``
hands EOManager (we strip the BGR replication first so we work with one
channel).

Output: post-processed uint8 (H, W) — what the operator will actually
see on the GUI.

Stages, in order:
    1. AGC (percentile-stretch with optional gamma)
    2. Denoise (NLM, bilateral, or none)
    3. Sharpen (unsharp mask, optional)
    4. CLAHE (optional local contrast enhancement)

All stages are skipped at zero / default values, so the identity
pipeline is "all defaults" and the optimizer can converge there if no
improvement helps.

The single sensor-side knob ``exposure_ext`` is included on the params
dataclass for completeness, but applying it is the optimizer's job
(spawns the 32-bit Leopard SDK helper); ``apply_pipeline`` only does
the post-capture software stages.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass
class CalibParams:
    # ── sensor (applied via Leopard SDK helper subprocess) ──
    exposure_ext: int = 1000           # LPCamera.ExposureExt — only working knob

    # ── AGC ──
    # Percentile clip points. (0.5, 99.5) keeps real outliers out without
    # crushing genuine shadow/highlight detail. (2.0, 98.0) is more
    # aggressive and posterizes — but on a flat scene it can pull more
    # contrast. The optimizer decides.
    agc_low_pct: float = 0.5
    agc_high_pct: float = 99.5
    # Tone curve gamma. 1.0 = linear; 0.7 = darker midtones (matches the
    # "dim Leopard look" the user referenced); 1.4 = brighter midtones.
    gamma: float = 1.0

    # ── denoise ──
    # 0 = disabled. Otherwise NLM strength (typical 3..15 for grain).
    nlm_h: float = 0.0
    nlm_template: int = 7      # template window
    nlm_search: int = 21       # search window
    # Bilateral filter (alternative): d=0 disables. d=5..9 typical.
    bilateral_d: int = 0
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 25.0

    # ── sharpen (unsharp mask) ──
    # amount=0 disables. amount=0.3..1.0 is a reasonable visible range.
    sharpen_amount: float = 0.0
    sharpen_radius: float = 1.0
    sharpen_threshold: int = 0

    # ── CLAHE ──
    clahe_clip: float = 0.0    # 0 disables. 1..4 typical for visible boost.
    clahe_tile: int = 8

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "CalibParams":
        """Inverse of ``to_vector``. Used by the optimizer."""
        return cls(
            exposure_ext=int(v[0]),
            agc_low_pct=float(v[1]),
            agc_high_pct=float(v[2]),
            gamma=float(v[3]),
            nlm_h=float(v[4]),
            bilateral_d=int(v[5]),
            bilateral_sigma_color=float(v[6]),
            sharpen_amount=float(v[7]),
            sharpen_radius=float(v[8]),
            clahe_clip=float(v[9]),
        )

    def to_vector(self) -> np.ndarray:
        return np.array([
            float(self.exposure_ext),
            float(self.agc_low_pct),
            float(self.agc_high_pct),
            float(self.gamma),
            float(self.nlm_h),
            float(self.bilateral_d),
            float(self.bilateral_sigma_color),
            float(self.sharpen_amount),
            float(self.sharpen_radius),
            float(self.clahe_clip),
        ], dtype=np.float64)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Search bounds for the optimizer. Order MATCHES ``from_vector`` /
# ``to_vector`` exactly. Any value outside these bounds is clipped by
# scipy's differential_evolution before evaluation.
PARAM_BOUNDS = [
    (200, 5000),     # exposure_ext  — only "alive" range, see eo_probe_exposure_ext_sweep
    (0.0, 5.0),      # agc_low_pct
    (95.0, 100.0),   # agc_high_pct  (keep > agc_low_pct; clamp at use)
    (0.5, 1.8),      # gamma
    (0.0, 15.0),     # nlm_h
    (0, 9),          # bilateral_d (0 disables; even values rejected at apply)
    (5.0, 80.0),     # bilateral_sigma_color
    (0.0, 1.5),      # sharpen_amount
    (0.5, 3.0),      # sharpen_radius
    (0.0, 4.0),      # clahe_clip
]


def _agc(y: np.ndarray, lo_pct: float, hi_pct: float, gamma: float) -> np.ndarray:
    """Percentile-stretch + optional gamma. Returns uint8."""
    if hi_pct <= lo_pct + 0.5:
        hi_pct = lo_pct + 0.5
    p_lo, p_hi = np.percentile(y, [lo_pct, hi_pct])
    if p_hi <= p_lo:
        return y.astype(np.uint8, copy=True)
    out = (y.astype(np.float32) - p_lo) * (255.0 / (p_hi - p_lo))
    np.clip(out, 0.0, 255.0, out=out)
    if abs(gamma - 1.0) > 1e-3:
        # gamma applied in [0, 1]; 1/gamma exponent = standard gamma curve
        out = np.power(out / 255.0, 1.0 / max(gamma, 0.05)) * 255.0
    return out.astype(np.uint8)


def _denoise(y: np.ndarray, p: CalibParams) -> np.ndarray:
    if p.nlm_h > 0.5:
        # NLM is slow on full-res frames. The optimizer should mostly
        # avoid this unless really needed; we cap its bounds tight.
        return cv2.fastNlMeansDenoising(
            y, None, h=float(p.nlm_h),
            templateWindowSize=int(max(3, p.nlm_template) | 1),  # odd
            searchWindowSize=int(max(7, p.nlm_search) | 1),
        )
    if p.bilateral_d >= 3:
        d = int(p.bilateral_d)
        if d % 2 == 0:
            d += 1
        return cv2.bilateralFilter(
            y, d=d,
            sigmaColor=float(p.bilateral_sigma_color),
            sigmaSpace=float(p.bilateral_sigma_space),
        )
    return y


def _sharpen(y: np.ndarray, amount: float, radius: float, threshold: int) -> np.ndarray:
    if amount < 0.05:
        return y
    ksize = int(max(1, round(radius * 2.0))) | 1
    blur = cv2.GaussianBlur(y, (ksize, ksize), float(radius))
    diff = y.astype(np.int16) - blur.astype(np.int16)
    if threshold > 0:
        mask = np.abs(diff) >= int(threshold)
        diff = diff * mask
    out = y.astype(np.int16) + (diff * float(amount)).astype(np.int16)
    np.clip(out, 0, 255, out=out)
    return out.astype(np.uint8)


def _clahe(y: np.ndarray, clip: float, tile: int) -> np.ndarray:
    if clip < 0.1:
        return y
    cl = cv2.createCLAHE(clipLimit=float(clip),
                         tileGridSize=(int(max(2, tile)), int(max(2, tile))))
    return cl.apply(y)


def apply_pipeline(raw_y: np.ndarray, params: CalibParams) -> np.ndarray:
    """Apply the full software pipeline — AGC → denoise → sharpen → CLAHE.

    ``raw_y`` is the raw mono Y plane the IMX568 capture path produces.
    Returned uint8 (H, W) is what the GUI shows / what the cost
    function evaluates.
    """
    if raw_y.ndim == 3:
        # BGR replicated — all channels equal, take channel 0.
        raw_y = raw_y[..., 0]
    y = _agc(raw_y, params.agc_low_pct, params.agc_high_pct, params.gamma)
    y = _denoise(y, params)
    y = _sharpen(y, params.sharpen_amount, params.sharpen_radius,
                 params.sharpen_threshold)
    y = _clahe(y, params.clahe_clip, params.clahe_tile)
    return y

"""Image-quality metrics used by the auto-calibrator cost function.

All inputs are uint8 mono (H, W) frames — the AGC-stretched output of
the pipeline being optimized. All metrics return small Python floats so
they can be combined into a scalar cost without numpy gymnastics in the
optimizer hot path.

Design rules:
  * Every metric is monotonic in the "better" direction. We negate the
    "higher is better" ones in the composite so cost is always
    minimized.
  * Every metric is bounded. Composite weights assume each term is
    roughly in [0, 1] after normalization — see ``normalize_*`` helpers.
  * No SciPy, no skimage — keep dependencies thin.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


# ──────────────────────────── basic stats ─────────────────────────────

def mean_abs_error(img: np.ndarray, target: float) -> float:
    """|mean(img) - target| / 255 — normalized to roughly [0, 1]."""
    return abs(float(img.mean()) - float(target)) / 255.0


def saturation_rate(img: np.ndarray, hi_threshold: int = 254) -> float:
    """Fraction of pixels at or above the saturation threshold."""
    return float((img >= hi_threshold).mean())


def black_clip_rate(img: np.ndarray, lo_threshold: int = 1) -> float:
    """Fraction of pixels at or below the black-clip threshold."""
    return float((img <= lo_threshold).mean())


# ───────────────────── histogram / dynamic range ──────────────────────

def histogram_entropy_norm(img: np.ndarray) -> float:
    """Shannon entropy of the 256-bin histogram, normalized to [0, 1].

    1.0 = uniform distribution across 0..255 (full dynamic-range usage).
    0.0 = degenerate single-value image.

    A common AE failure mode is a frame that's mostly clustered around
    one or two grays with no real distribution; this metric punishes
    that hard.
    """
    hist, _ = np.histogram(img, bins=256, range=(0, 256))
    p = hist.astype(np.float64)
    p_sum = p.sum()
    if p_sum <= 0:
        return 0.0
    p = p / p_sum
    nz = p[p > 0]
    h = float(-(nz * np.log2(nz)).sum())
    return h / 8.0  # log2(256) = 8 bits = max entropy


def histogram_chi2(img: np.ndarray, ref_hist: np.ndarray) -> float:
    """Chi-squared distance between img's histogram and ref_hist (256-bin).

    ref_hist is a probability distribution (sum=1). Returns a value
    roughly in [0, 2]; clamp to 1 by ``min(x, 1)`` in the composite if
    you want a strict [0, 1] term.
    """
    h, _ = np.histogram(img, bins=256, range=(0, 256))
    p = h.astype(np.float64)
    s = p.sum()
    if s <= 0:
        return 2.0
    p = p / s
    # Symmetric chi2: sum( (p - q)^2 / (p + q + eps) )
    eps = 1e-9
    return float(np.sum((p - ref_hist) ** 2 / (p + ref_hist + eps)))


def precompute_ref_hist(ref_img: np.ndarray) -> np.ndarray:
    """Return a 256-bin probability distribution from a reference image."""
    h, _ = np.histogram(ref_img, bins=256, range=(0, 256))
    p = h.astype(np.float64)
    s = p.sum()
    return (p / s) if s > 0 else p


# ───────────────────────── sharpness / MTF ────────────────────────────

def sharpness_laplacian_var(img: np.ndarray) -> float:
    """Variance of cv2.Laplacian — classic blur-detection metric.

    Higher is sharper. Normalized to roughly [0, 1] by dividing by 1000
    and clamping (typical sharp natural image hits 200-500, motion-
    blurred ~10-50, totally blurred ~1).
    """
    lap = cv2.Laplacian(img, cv2.CV_64F)
    v = float(lap.var())
    return min(v / 1000.0, 1.0)


def sharpness_sobel(img: np.ndarray) -> float:
    """Mean of |Sobel-x| + |Sobel-y|, normalized.

    Less spikey than Laplacian variance — averages edge response across
    the whole frame. Useful as a sanity-check companion metric.
    """
    sx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.abs(sx) + np.abs(sy)
    return min(float(mag.mean()) / 100.0, 1.0)


# ─────────────────────────────── SNR ──────────────────────────────────

def local_snr(img: np.ndarray,
              patch: int = 32,
              flat_pct: float = 25.0) -> float:
    """SNR proxy: median(mean / std) over the FLATTEST patches.

    Algorithm:
      1. Tile the image into non-overlapping ``patch x patch`` blocks
      2. Compute (mean, std) per block
      3. Sort by std ascending, take the bottom ``flat_pct``%
      4. Return median(mean / max(std, eps)) over those flat patches

    Why "flat patches": grain is most visible in regions that should
    look smooth (sky, walls, dryer panels). Bright textured regions
    have high std for legitimate reasons (texture, edges) — they don't
    tell us about noise. By picking the flattest tiles, we isolate
    sensor noise from real scene content.

    Returned in dB-ish units: 0 = degenerate, 30+ = professional, 50+
    is rare for low-end sensors. Normalize by /60 in the composite to
    get roughly [0, 1].
    """
    h, w = img.shape[:2]
    nh = h // patch
    nw = w // patch
    if nh < 4 or nw < 4:
        return 0.0
    img_f = img[:nh * patch, :nw * patch].astype(np.float32)
    blocks = img_f.reshape(nh, patch, nw, patch).transpose(0, 2, 1, 3)
    blocks = blocks.reshape(nh * nw, patch, patch)
    means = blocks.mean(axis=(1, 2))
    stds = blocks.std(axis=(1, 2))
    # Pick the flattest fraction by std
    n = max(4, int(len(stds) * (flat_pct / 100.0)))
    flat_idx = np.argsort(stds)[:n]
    flat_means = means[flat_idx]
    flat_stds = stds[flat_idx]
    eps = 1e-3
    snr_db_like = 20.0 * np.log10(flat_means / np.maximum(flat_stds, eps) + eps)
    return float(np.median(snr_db_like))


def local_snr_norm(img: np.ndarray, patch: int = 32) -> float:
    """SNR-in-flat-patches normalized to [0, 1] (0=garbage, 1=clean)."""
    return max(0.0, min(local_snr(img, patch) / 60.0, 1.0))


# ───────────────────────── reference-match ────────────────────────────

def reference_pixel_match(img: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute pixel-wise difference, normalized to [0, 1].

    Both inputs are mono uint8. Reference is resized to img's shape
    if they differ. A small constant blur is applied to both before
    comparison so the metric isn't dominated by sub-pixel-shift noise.
    """
    if ref.shape != img.shape:
        ref = cv2.resize(ref, (img.shape[1], img.shape[0]),
                         interpolation=cv2.INTER_AREA)
    a = cv2.GaussianBlur(img, (3, 3), 0)
    b = cv2.GaussianBlur(ref, (3, 3), 0)
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean()) / 255.0


# ────────────────────────── composite cost ────────────────────────────

@dataclass
class CostWeights:
    """All weights are non-negative. Cost is sum of weight * normalized term.
    Higher cost = worse image. Optimizer minimizes."""
    target_mean: float = 1.0
    saturation: float = 5.0          # cliff function — saturation is bad
    black_clip: float = 5.0
    histogram_entropy: float = 1.0   # NEGATED (we want higher entropy)
    sharpness: float = 1.0           # NEGATED (we want higher)
    snr: float = 2.0                 # NEGATED
    histogram_match: float = 0.0     # set > 0 only if reference provided
    pixel_match: float = 0.0         # set > 0 only if reference provided


@dataclass
class CostBreakdown:
    """Per-term cost contributions for transparent logging."""
    total: float
    target_mean: float
    saturation: float
    black_clip: float
    histogram_entropy: float
    sharpness: float
    snr: float
    histogram_match: float
    pixel_match: float


def composite_cost(
    img: np.ndarray,
    weights: CostWeights,
    target_mean: float = 100.0,
    ref_img: Optional[np.ndarray] = None,
    ref_hist: Optional[np.ndarray] = None,
) -> CostBreakdown:
    """Compute the full composite cost and return the breakdown.

    All 'higher is better' metrics are inverted to '1 - x' so the sum
    is uniformly something the optimizer minimizes.
    """
    cm = weights.target_mean * mean_abs_error(img, target_mean)
    cs = weights.saturation * saturation_rate(img)
    cb = weights.black_clip * black_clip_rate(img)
    ce = weights.histogram_entropy * (1.0 - histogram_entropy_norm(img))
    cp = weights.sharpness * (1.0 - sharpness_laplacian_var(img))
    cn = weights.snr * (1.0 - local_snr_norm(img))
    ch = 0.0
    if weights.histogram_match > 0 and ref_hist is not None:
        ch = weights.histogram_match * min(histogram_chi2(img, ref_hist), 2.0)
    cpx = 0.0
    if weights.pixel_match > 0 and ref_img is not None:
        cpx = weights.pixel_match * reference_pixel_match(img, ref_img)
    total = cm + cs + cb + ce + cp + cn + ch + cpx
    return CostBreakdown(
        total=float(total),
        target_mean=float(cm),
        saturation=float(cs),
        black_clip=float(cb),
        histogram_entropy=float(ce),
        sharpness=float(cp),
        snr=float(cn),
        histogram_match=float(ch),
        pixel_match=float(cpx),
    )

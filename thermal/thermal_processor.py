"""
Thermal frame processing primitives.

Pure functions — no state, no threads. Easy to unit-test with
synthetic 16-bit arrays. The `ThermalManager` thread composes
these into a pipeline.

Pipeline order when all stages are enabled:

    raw16 → dead-pixel median → AGC (percentile stretch) → CLAHE? →
    gamma → bilateral denoise → unsharp mask → colormap → display

Detection always runs on the raw16 (pre-AGC) frame in ThermalManager,
so toggling any of the display-side stages cannot shift detection
statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np


# ───────────────────────────────────────────────────────────────
# Dead-pixel suppression (raw16, pre-AGC)
# ───────────────────────────────────────────────────────────────

def apply_dead_pixel_median(frame_u16: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Median-filter a 16-bit thermal frame to suppress single-pixel defects.

    Boson 640 sensors typically ship with FFC handling most bad pixels,
    but a few survive and appear as bright dots in the colormapped
    display. A 3×3 median knocks them out without softening real
    structure (a single pixel is below the kernel's plurality).
    """
    if frame_u16.ndim != 2:
        raise ValueError(f"apply_dead_pixel_median expects 2-D, got shape {frame_u16.shape}")
    if frame_u16.dtype != np.uint16:
        raise ValueError(f"apply_dead_pixel_median expects uint16, got {frame_u16.dtype}")
    k = int(ksize)
    if k < 3 or k % 2 == 0:
        raise ValueError(f"ksize must be odd and >= 3, got {ksize}")
    return cv2.medianBlur(frame_u16, k)


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


def apply_roi_agc(
    frame_u16: np.ndarray,
    roi_top_frac: float = 0.4,
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> np.ndarray:
    """ROI percentile AGC — compute percentiles over the bottom region
    of the frame ONLY, then apply the linear stretch globally.

    Use case: operator's targets sit in a known band of the frame
    (vehicles on the road = bottom ~60%). Hot trees in the upper 40%
    would otherwise force the AGC to allocate display range to them,
    squishing the road into mid-gray. By computing percentiles only
    over the ROI, the full 0-255 range gets allocated to the band the
    operator cares about. Out-of-ROI pixels may saturate to 0 or 255 —
    that's the intended trade.

    `roi_top_frac` = 0.4 means percentile is computed over the bottom
    60% of the frame. 0.0 reduces to global percentile (whole frame).
    """
    if frame_u16.ndim != 2:
        raise ValueError(f"apply_roi_agc expects 2-D uint16, got shape {frame_u16.shape}")
    h = frame_u16.shape[0]
    top = max(0, min(h - 1, int(h * float(roi_top_frac))))
    roi = frame_u16[top:, :]
    lo, hi = np.percentile(roi, [low_percentile, high_percentile])
    if hi <= lo:
        return np.zeros(frame_u16.shape, dtype=np.uint8)
    stretched = (frame_u16.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_clahe_y16(
    frame_u16: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """CLAHE on raw 16-bit thermal data, output 8-bit.

    Local-contrast equalization at full 16-bit precision before
    quantizing to 8-bit. This is the closest pure-software
    approximation of what the Boson's onboard AGC does (it has
    hardware DDE — Digital Detail Enhancement — that pumps edges
    locally). Result is typically sharper than linear stretch but
    can amplify noise on a low-contrast scene.

    Returns u8. CLAHE on uint16 is supported by OpenCV; output is
    rescaled from the full u16 dynamic range to 0..255.
    """
    if frame_u16.ndim != 2:
        raise ValueError(f"apply_clahe_y16 expects 2-D, got shape {frame_u16.shape}")
    if frame_u16.dtype != np.uint16:
        raise ValueError(f"apply_clahe_y16 expects uint16, got {frame_u16.dtype}")
    g = max(2, int(tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(g, g))
    eq16 = clahe.apply(frame_u16)
    # Rescale the equalized 16-bit output to 0..255
    lo, hi = np.percentile(eq16, [2, 98])
    if hi <= lo:
        return np.zeros(eq16.shape, dtype=np.uint8)
    stretched = (eq16.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def apply_gates_agc(
    frame_u16: np.ndarray,
    cold_count: int,
    hot_count: int,
) -> np.ndarray:
    """Operator-gate AGC — fixed cold/hot raw-count thresholds.

    Linear stretch from `cold_count` (mapped to 0) to `hot_count`
    (mapped to 255). Any pixel < cold_count clips to black; > hot_count
    clips to white. No histogram dependence — output is stable
    frame-to-frame, no scene-driven "breathing".

    The operator picks the gates to bracket their working temperature
    range (e.g. 19500..22500 raw counts on the bench scene to capture
    the road + vehicles + warm targets, while clipping cool sky to
    black and hot tree-tops to white).

    On a Boson 640 in radiometric Y16 mode, raw counts roughly
    correlate to scene radiance; on this rig the typical scene spans
    18500..23000 counts. cold_count and hot_count are full uint16
    values (0..65535).
    """
    if frame_u16.ndim != 2:
        raise ValueError(f"apply_gates_agc expects 2-D uint16, got shape {frame_u16.shape}")
    cold = float(cold_count)
    hot = float(hot_count)
    if hot <= cold:
        # Degenerate — return flat mid-gray rather than blow up
        return np.full(frame_u16.shape, 128, dtype=np.uint8)
    stretched = (frame_u16.astype(np.float32) - cold) * (255.0 / (hot - cold))
    return np.clip(stretched, 0, 255).astype(np.uint8)


# ───────────────────────────────────────────────────────────────
# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ───────────────────────────────────────────────────────────────

def apply_clahe(
    frame_u8: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: int = 8,
) -> np.ndarray:
    """Local-contrast equalization on an 8-bit thermal frame.

    Useful when the AGC'd image is technically full-range but visually
    flat — long-range scenes at thermal equilibrium often produce a
    histogram that fills 0-255 yet renders the whole scene as varying
    shades of one tone.
    """
    if frame_u8.ndim != 2:
        raise ValueError(f"apply_clahe expects 2-D uint8, got shape {frame_u8.shape}")
    if frame_u8.dtype != np.uint8:
        raise ValueError(f"apply_clahe expects uint8, got {frame_u8.dtype}")
    g = max(2, int(tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(g, g))
    return clahe.apply(frame_u8)


# ───────────────────────────────────────────────────────────────
# Gamma (LUT)
# ───────────────────────────────────────────────────────────────

def _gamma_lut(gamma: float) -> np.ndarray:
    g = max(1e-6, float(gamma))
    return np.clip(((np.arange(256, dtype=np.float32) / 255.0) ** g) * 255.0,
                   0, 255).astype(np.uint8)


def apply_gamma(frame_u8: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Apply a gamma curve via 256-entry LUT. gamma=1.0 is identity.

    Convention: ``out = (in/255)**gamma * 255`` — gamma<1 lifts midtones,
    gamma>1 sinks them. Matches the EO `enhance.gamma` semantics.
    """
    if frame_u8.dtype != np.uint8:
        raise ValueError(f"apply_gamma expects uint8, got {frame_u8.dtype}")
    if abs(float(gamma) - 1.0) < 1e-3:
        return frame_u8  # cheap identity
    return cv2.LUT(frame_u8, _gamma_lut(gamma))


# ───────────────────────────────────────────────────────────────
# Bilateral denoise
# ───────────────────────────────────────────────────────────────

def apply_bilateral_denoise(
    frame_u8: np.ndarray,
    d: int = 5,
    sigma_color: float = 15.0,
    sigma_space: float = 15.0,
) -> np.ndarray:
    """Edge-preserving denoise on an 8-bit thermal display image.

    Knocks out fine grain ("snow") without softening target silhouettes.
    Kept conservative by default — large sigmas turn vehicles into
    Photoshop poster art.
    """
    if frame_u8.ndim != 2:
        raise ValueError(f"apply_bilateral_denoise expects 2-D uint8, got shape {frame_u8.shape}")
    if frame_u8.dtype != np.uint8:
        raise ValueError(f"apply_bilateral_denoise expects uint8, got {frame_u8.dtype}")
    return cv2.bilateralFilter(
        frame_u8, int(d), float(sigma_color), float(sigma_space)
    )


# ───────────────────────────────────────────────────────────────
# Unsharp mask
# ───────────────────────────────────────────────────────────────

def apply_unsharp_mask(
    frame_u8: np.ndarray,
    amount: float = 0.30,
    radius: float = 1.0,
) -> np.ndarray:
    """Subtle edge-sharpen via Gaussian-blur subtraction.

    out = frame + amount * (frame - blur(frame))

    amount=0 is identity (returns input). radius is the Gaussian sigma.
    """
    if frame_u8.dtype != np.uint8:
        raise ValueError(f"apply_unsharp_mask expects uint8, got {frame_u8.dtype}")
    a = float(amount)
    if abs(a) < 1e-4:
        return frame_u8  # cheap identity
    r = max(0.1, float(radius))
    # OpenCV requires odd ksize; pick from radius.
    ksize = max(3, int(round(r * 3.0)) | 1)
    blurred = cv2.GaussianBlur(frame_u8, (ksize, ksize), r)
    sharp = cv2.addWeighted(frame_u8, 1.0 + a, blurred, -a, 0)
    return sharp  # already uint8 (addWeighted preserves dtype with clip)


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
# Parameterized pipeline
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ThermalEnhanceParams:
    """All knobs for the raw16→display chain in one place.

    Defaults match the legacy behaviour (only AGC + colormap apply,
    every enhancement stage off) so passing `ThermalEnhanceParams()`
    reproduces pre-2026-04-27 output exactly.
    """
    # AGC mode dispatch:
    #   "global"    — percentile over whole frame (LEGACY)
    #   "roi"       — percentile over the bottom (1-roi_top_frac) of the
    #                 frame; out-of-ROI pixels may saturate
    #   "gates"     — fixed cold/hot raw u16 thresholds; histogram-free
    #   "clahe_y16" — CLAHE on raw 16-bit before bit reduction. Closest
    #                 software approximation of the camera's onboard
    #                 DDE/AGC; gave the highest sharpness in 4-up tests
    mode: str = "global"
    # Used by mode="global" and mode="roi"
    low_percentile: float = 2.0
    high_percentile: float = 98.0
    # Used by mode="roi"
    roi_top_frac: float = 0.4
    # Used by mode="gates"
    cold_count: int = 0
    hot_count: int = 65535
    # Used by mode="clahe_y16"
    clahe_y16_clip_limit: float = 2.0
    clahe_y16_tile_grid: int = 8
    colormap: str = "INFERNO"
    # Dead-pixel median (raw16 stage)
    dead_pixel_median_enabled: bool = False
    dead_pixel_median_ksize: int = 3
    # CLAHE (post-AGC)
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: int = 8
    # Gamma
    gamma: float = 1.0
    # Bilateral
    bilateral_enabled: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: float = 15.0
    bilateral_sigma_space: float = 15.0
    # Unsharp
    unsharp_enabled: bool = False
    unsharp_amount: float = 0.30
    unsharp_radius: float = 1.0


def apply_agc_mode(frame_u16: np.ndarray, p: "ThermalEnhanceParams") -> np.ndarray:
    """Dispatch AGC by ``ThermalEnhanceParams.mode``.

    Returns a uint8 grayscale frame in all modes. Legacy default
    (mode="global") routes to the existing :func:`apply_agc` so
    behaviour is byte-equivalent to pre-2026-04-27 when YAML doesn't
    set ``thermal.agc.mode``.
    """
    if p.mode == "gates":
        return apply_gates_agc(frame_u16, p.cold_count, p.hot_count)
    if p.mode == "roi":
        return apply_roi_agc(frame_u16, p.roi_top_frac,
                             p.low_percentile, p.high_percentile)
    if p.mode == "clahe_y16":
        return apply_clahe_y16(frame_u16, p.clahe_y16_clip_limit,
                               p.clahe_y16_tile_grid)
    # Default: legacy global percentile
    return apply_agc(frame_u16, p.low_percentile, p.high_percentile)


def from_config(thermal_cfg: dict) -> ThermalEnhanceParams:
    """Build ThermalEnhanceParams from a parsed `thermal:` config block.

    Tolerant to missing keys — every field falls back to the dataclass
    default. This is the only adapter between YAML and the pipeline;
    `ThermalManager` should call this once at init.
    """
    cfg = thermal_cfg or {}
    agc = cfg.get("agc", {}) or {}
    clahe = (agc.get("clahe", {}) or {})
    enh = cfg.get("enhance", {}) or {}
    dpm = (enh.get("dead_pixel_median", {}) or {})
    bil = (enh.get("bilateral_denoise", {}) or {})
    usm = (enh.get("unsharp_mask", {}) or {})

    gates = (agc.get("gates", {}) or {})
    roi = (agc.get("roi", {}) or {})
    clahe_y16_cfg = (agc.get("clahe_y16", {}) or {})

    # Mode default = "global" preserves legacy behaviour exactly when
    # YAML omits the key.
    mode = str(agc.get("mode", "global")).lower()
    if mode not in ("global", "roi", "gates", "clahe_y16"):
        mode = "global"

    return ThermalEnhanceParams(
        mode=mode,
        low_percentile=float(agc.get("low_percentile", 2.0)),
        high_percentile=float(agc.get("high_percentile", 98.0)),
        roi_top_frac=float(roi.get("top_frac", 0.4)),
        cold_count=int(gates.get("cold_count", 0)),
        hot_count=int(gates.get("hot_count", 65535)),
        clahe_y16_clip_limit=float(clahe_y16_cfg.get("clip_limit", 2.0)),
        clahe_y16_tile_grid=int(clahe_y16_cfg.get("tile_grid", 8)),
        colormap=str(agc.get("colormap", "INFERNO")),
        dead_pixel_median_enabled=bool(dpm.get("enabled", False)),
        dead_pixel_median_ksize=int(dpm.get("ksize", 3)),
        clahe_enabled=bool(clahe.get("enabled", False)),
        clahe_clip_limit=float(clahe.get("clip_limit", 2.0)),
        clahe_tile_grid=int(clahe.get("tile_grid", 8)),
        gamma=float(enh.get("gamma", 1.0)),
        bilateral_enabled=bool(bil.get("enabled", False)),
        bilateral_d=int(bil.get("d", 5)),
        bilateral_sigma_color=float(bil.get("sigma_color", 15.0)),
        bilateral_sigma_space=float(bil.get("sigma_space", 15.0)),
        unsharp_enabled=bool(usm.get("enabled", False)),
        unsharp_amount=float(usm.get("amount", 0.30)),
        unsharp_radius=float(usm.get("radius", 1.0)),
    )


def enhance_post_agc(frame_u8: np.ndarray, p: ThermalEnhanceParams) -> np.ndarray:
    """Apply the post-AGC enhancement chain to an 8-bit frame.

    Order: CLAHE → gamma → bilateral → unsharp. Each stage is a no-op
    when its `_enabled` flag is False (or, for gamma, when value≈1.0
    and for unsharp, when amount≈0). Pure function — same input, same
    output.

    Exposed separately so the offline A/B tool (which only has the
    already-AGC'd 8-bit JPEG from JSONL recordings) can apply just
    this segment of the chain.
    """
    out = frame_u8
    if p.clahe_enabled:
        out = apply_clahe(out, p.clahe_clip_limit, p.clahe_tile_grid)
    out = apply_gamma(out, p.gamma)
    if p.bilateral_enabled:
        out = apply_bilateral_denoise(
            out, p.bilateral_d, p.bilateral_sigma_color, p.bilateral_sigma_space
        )
    if p.unsharp_enabled:
        out = apply_unsharp_mask(out, p.unsharp_amount, p.unsharp_radius)
    return out


# ───────────────────────────────────────────────────────────────
# Convenience: raw16 → display BGR in one shot
# ───────────────────────────────────────────────────────────────

def raw16_to_display(
    frame_u16: np.ndarray,
    colormap: str = "INFERNO",
    low_percentile: float = 2.0,
    high_percentile: float = 98.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (agc_u8, display_bgr) from a raw 16-bit frame.

    Legacy entry point — kept for back-compat. New callers should use
    `raw16_to_display_with_params` for the full parameterized chain.
    """
    agc = apply_agc(frame_u16, low_percentile, high_percentile)
    bgr = apply_colormap(agc, colormap)
    return agc, bgr


def raw16_to_display_with_params(
    frame_u16: np.ndarray,
    p: ThermalEnhanceParams,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full raw16→display pipeline driven by `ThermalEnhanceParams`.

    Returns `(enhanced_u8, display_bgr)`:

    * `enhanced_u8` is the 8-bit grayscale frame after every enabled
      enhancement stage (this is what the heat tracker / OF bridge
      should consume — they live in display space and want a clean
      grayscale view).
    * `display_bgr` is the same image after the configured colormap.
    """
    f = frame_u16
    if p.dead_pixel_median_enabled:
        f = apply_dead_pixel_median(f, p.dead_pixel_median_ksize)
    # Mode dispatch — at default mode="global" this calls the legacy
    # apply_agc with the same parameters, so output is byte-equivalent
    # to pre-mode-dispatch behaviour. mode="roi" or "gates" route to
    # the new primitives.
    agc = apply_agc_mode(f, p)
    enhanced = enhance_post_agc(agc, p)
    bgr = apply_colormap(enhanced, p.colormap)
    return enhanced, bgr

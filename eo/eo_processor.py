"""EO frame processor — luma enhancement chain.

The IMX568 + FX3 bridge hands us 8-bit BGR via DirectShow's YUY2 decode
on a 12-bit mono sensor. That's a 4-bit precision tax we can't undo,
so the displayed image only looks "like a proper day sensor" if we
process the 8 bits we DO have correctly. Pipeline:

    luma → median denoise → percentile clip → gamma → CLAHE → unsharp → BGR

Default config runs ONLY median denoise + percentile clip + gentle
gamma. CLAHE and unsharp are off by default — they're great on a
clean daylight frame and disastrous on a noisy low-light frame
(seen 2026-04-24, 7 pm indoor: clip+gamma+CLAHE+unsharp turned
underexposed grain into an oil-painting torture render). They are
config-toggleable so you can enable them once exposure is good.

Stage rationale:

  * Median denoise (3×3) ─ the single biggest "low-light frame stops
    looking horrible" stage. Removes salt-and-pepper sensor noise
    that would otherwise be amplified by every later stage. Edge-
    preserving (median > mean for grain). ~1 ms at 1236×1029.

  * Percentile clip [0.5, 99.5] → 0..255 ─ pushes outlier pixels to
    the rails so a single sun-disk or hot pixel can't squeeze the
    rest of the histogram into a 30-step grey blob.

  * Gamma correction ─ the sensor is roughly linear; the eye is
    logarithmic. Default γ=0.85 = mild midtone lift. γ < 1 brightens,
    γ = 1 is no change, γ > 1 darkens (rarely useful here).

  * CLAHE (off by default) ─ local contrast on 8×8 tiles. Wonderful
    on clean daytime frames, brutal on grain — a small clip_limit
    (1.0–1.5) is the safe range when re-enabled. The 2.0 default
    that shipped earlier amplified noise into ridge artifacts.

  * Unsharp mask (off by default) ─ restores sharpness lost to the
    2472→1236 downscale. ALSO amplifies noise; only re-enable once
    the source frame is clean. amount ≤ 0.4, radius ≤ 1.0 if you do.

Order matters. Denoise FIRST so later contrast stages don't have
noise to amplify. Percentile clip before gamma so gamma operates on
the full 0..255 range. Gamma before CLAHE so midtones are already
lifted when CLAHE looks for local detail. Unsharp last because
sharpening before CLAHE gets re-balanced.

Performance budget on this laptop at 1236×1029, default config:
  median 3×3                     :  ~0.8 ms
  percentile clip + LUT stretch  :  ~1.5 ms
  gamma LUT                      :  ~0.8 ms
  total (default)                :  ~3   ms — leaves ~47 ms/frame for
                                              YOLO + JPEG + WS at 20 Hz.
With CLAHE+unsharp on, add ~5 ms.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# ─── small helpers ──────────────────────────────────────────────────────


def _looks_like_yuy2_zero_chroma(bgr: np.ndarray) -> bool:
    """True iff the frame matches the FX3 bridge's mono-as-YUY2 pattern.

    When the bridge ships a mono Y in YUY2 with U=V=0, DirectShow's
    auto-decode produces B ≈ Y-227 (clipped to 0), G ≈ Y+135 (saturated
    at 255 above Y=120), R ≈ Y-179 (clipped to 0 below Y=179). On any
    real-world scene the means come out roughly:

        mean(B) ≈ 0..few         (only sun-bright pixels lift it)
        mean(R) ≈ 0..tens        (only Y > 179 contributes)
        mean(G) ≈ 200+           (heavily saturated by Y > 120)

    Verified on the bench laptop 2026-04-24 against a real bridge dump:
    B mean=0.6, R mean=5.7, G mean=221.3. We test on means (not maxes)
    so a single sun-bright pixel can't false-negative the detection.

    A real color webcam has all three channels in the same ballpark and
    their means within a factor of ~3, so the heuristic g_mean >> r_mean
    AND g_mean >> b_mean cannot misfire.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        return False
    sub = bgr[::8, ::8]
    b_mean = float(sub[..., 0].mean())
    g_mean = float(sub[..., 1].mean())
    r_mean = float(sub[..., 2].mean())
    if g_mean < 100.0:
        return False
    # G must dwarf B and R. 5x is conservative — real color frames keep
    # channel means well within 2x of each other on natural scenes.
    return g_mean > 5.0 * max(1.0, b_mean) and g_mean > 5.0 * max(1.0, r_mean)


def _recover_y_from_yuy2_bgr(bgr: np.ndarray) -> np.ndarray:
    """Reconstruct a mono Y plane from a YUY2-zero-chroma BGR decode.

    Two estimators with confidence weights:

      * y_from_g = G - 135   trusted while G < 240 (unsaturated)
      * y_from_r = R + 179   trusted while R > 0  (lifted off floor)

    Confidence ramps:
        g_weight = clip((255 - G) / 15, 0, 1)
        r_weight = clip( R       / 15, 0, 1)

    **Soft-confidence blending** (the whole reason this function got
    rewritten 2026-04-24 after a screenshot-off vs. Google Meet showed
    Meet's raw green frame looking visibly cleaner than our recovery):

    The old version had a hard cutoff — if total confidence was below
    0.01, dump the pixel to a flat DEAD_ZONE_FILL=150. That produced
    *speckle* all along the transition band: one pixel with R=0 got
    filled with 150, its neighbor with R=1 got y=(179+1)=180 from the
    R-only estimator. Adjacent pixels jumped 30 levels. Salt-and-pepper.

    New formula blends the pixel estimate with a *spatially smooth* fill
    (large-kernel Gaussian of the trusted neighbors) in proportion to
    confidence:

        alpha  = clip(total_w, 0, 1)        # 1 = fully trusted
        fill   = Gauss41(y*trust) / Gauss41(trust)   # neighbor interp
        out    = alpha * y_blend + (1 - alpha) * fill

    So dead-zone pixels get their local neighborhood's recovered value,
    and confidence-0.3 pixels get 30% of their own estimate + 70% of the
    neighborhood — gradual, no jump discontinuities, no speckle.

    Y∈[120,179] is still hardware-side information loss in the bridge
    firmware — we cannot manufacture data that isn't there. But the
    visible *artifacts* from that loss are gone.
    """
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)

    # Trust ramp width: 30 codes (was 15). A narrower ramp produced
    # visible discontinuity stripes at the trust-transition boundaries
    # — the "shattered glass on a leather chair" artifact in the live
    # GUI (2026-04-24). 30 codes overlaps both estimators across a wide
    # transition so neighboring pixels' alpha blends drift smoothly.
    RAMP = 30.0
    g_weight = np.clip((255.0 - g) / RAMP, 0.0, 1.0)
    r_weight = np.clip(r / RAMP, 0.0, 1.0)
    y_from_g = np.clip(g - 135.0, 0.0, 255.0)
    y_from_r = np.clip(r + 179.0, 0.0, 255.0)

    total_w = g_weight + r_weight
    safe_w = np.maximum(total_w, 1e-3)
    y_blend = (y_from_g * g_weight + y_from_r * r_weight) / safe_w

    # Spatially smooth fill for low-trust pixels.
    # Two-scale fill: a wide kernel (81 px) carries broad illumination
    # and a tight kernel (15 px) preserves nearer-neighbor structure.
    # Combine 50/50 — the wide one alone produced posterized dead-zone
    # fills (the leather-chair artifact); the tight one alone bled
    # untrusted pixels into trusted ones.
    trust = np.clip(total_w, 0.0, 1.0)
    weighted = (y_blend * trust).astype(np.float32)
    trust_f = trust.astype(np.float32)
    num_wide = cv2.GaussianBlur(weighted, (81, 81), 0)
    den_wide = cv2.GaussianBlur(trust_f, (81, 81), 0)
    num_tight = cv2.GaussianBlur(weighted, (15, 15), 0)
    den_tight = cv2.GaussianBlur(trust_f, (15, 15), 0)
    fill_wide = num_wide / np.maximum(den_wide, 1e-3)
    fill_tight = num_tight / np.maximum(den_tight, 1e-3)
    fill = 0.5 * fill_wide + 0.5 * fill_tight

    alpha = trust
    out = alpha * y_blend + (1.0 - alpha) * fill
    return np.clip(out, 0, 255).astype(np.uint8)


def _to_luma(bgr: np.ndarray) -> np.ndarray:
    """Collapse a BGR frame to single-channel luma.

    Two modes, autodetected per frame:

    1. Real BGR (true color webcam, FakeEOSource, etc.) — use the
       standard ITU-R BT.601 formula via cvtColor(BGR2GRAY) =
       0.114·B + 0.587·G + 0.299·R. Reliable for any normal source.

    2. FX3 bridge YUY2-zero-chroma decode — the IMX568 sensor is mono
       but the bridge firmware ships Y in YUY2 with U=V=0. DirectShow's
       auto-decode then loses the Y∈[120,179] range. Detect the
       all-green pattern and stitch luma from G/R directly to recover
       most of the dynamic range. This is the path that produced the
       "oil painting mid-tone patches" before — defense in depth in
       case CAP_PROP_CONVERT_RGB=0 in imx568_capture.py is not honored.

    ``bgr[:, :, 0]`` is never safe on this camera: B can be flat-zero
    while G carries the whole signal.
    """
    if bgr.ndim == 2:
        return bgr
    if _looks_like_yuy2_zero_chroma(bgr):
        return _recover_y_from_yuy2_bgr(bgr)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def scene_mean(bgr: np.ndarray) -> float:
    """Mean brightness of the luma channel on a 4× subsample.

    Used by ProfileSelector to decide which exposure profile to run —
    profile selection is a coarse heuristic, full-res precision wastes
    cycles. ~0.3 ms at native resolution.
    """
    if bgr is None:
        return 0.0
    y = _to_luma(bgr)
    return float(y[::4, ::4].mean())


# ─── stage implementations (each operates on uint8 luma) ────────────────


def _percentile_stretch(y: np.ndarray, low_pct: float, high_pct: float) -> np.ndarray:
    """Linear stretch: clip [low_pct, high_pct] of pixels, expand to 0..255.

    Falls back to absolute min/max if percentiles are degenerate (lens
    cap on, dead sensor). Ultimate fallback: pass through unchanged so
    the operator can SEE the failure (a flat image at the actual sensor
    value) instead of an ambiguous middle-grey blob.
    """
    sample = y[::4, ::4]
    lo, hi = np.percentile(sample, [low_pct, high_pct])
    if hi <= lo:
        lo = float(sample.min())
        hi = float(sample.max())
        if hi <= lo:
            return y  # truly flat — pass through
    scale = 255.0 / (hi - lo)
    return np.clip((y.astype(np.float32) - lo) * scale, 0, 255).astype(np.uint8)


# Gamma LUT cache. Building a 256-entry power LUT is cheap (~50 µs) but
# we re-enter enhance() at frame rate; cache by rounded gamma so we
# only rebuild when the user actually moves the knob.
_GAMMA_LUT_CACHE: dict[int, np.ndarray] = {}


def _gamma_lut(gamma: float) -> np.ndarray:
    """Return a uint8 LUT for output = (input/255) ^ (1/gamma) * 255."""
    key = int(round(gamma * 100))  # 0.01 resolution is plenty
    cached = _GAMMA_LUT_CACHE.get(key)
    if cached is not None:
        return cached
    inv = 1.0 / max(1e-3, float(gamma))
    table = np.array(
        [((i / 255.0) ** inv) * 255.0 for i in range(256)],
        dtype=np.uint8,
    )
    _GAMMA_LUT_CACHE[key] = table
    return table


def _apply_gamma(y: np.ndarray, gamma: float) -> np.ndarray:
    if gamma is None or abs(gamma - 1.0) < 1e-3:
        return y
    return cv2.LUT(y, _gamma_lut(gamma))


# CLAHE objects are stateful (they cache tile lookup tables). Construct
# once per (clip, grid) pair to avoid the per-frame allocation cost
# (small, but free is free).
_CLAHE_CACHE: dict[tuple[float, int], "cv2.CLAHE"] = {}


def _apply_clahe(y: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
    if clip_limit <= 0 or tile_grid <= 0:
        return y
    key = (round(float(clip_limit), 3), int(tile_grid))
    clahe = _CLAHE_CACHE.get(key)
    if clahe is None:
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=(int(tile_grid), int(tile_grid)),
        )
        _CLAHE_CACHE[key] = clahe
    return clahe.apply(y)


def _apply_unsharp(y: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Classic unsharp mask: out = y + amount * (y - blur(y))."""
    if amount <= 0 or radius <= 0:
        return y
    # ksize=(0,0) → OpenCV picks ksize from sigma, the way you want it.
    blur = cv2.GaussianBlur(y, (0, 0), float(radius))
    # cv2.addWeighted clamps to uint8 range, so we don't have to clip.
    return cv2.addWeighted(y, 1.0 + float(amount), blur, -float(amount), 0)


# ─── public API ─────────────────────────────────────────────────────────


def passthrough(bgr: np.ndarray) -> np.ndarray:
    """Minimum-touch luma view of the camera frame.

    Pipeline: BGR → luma extraction (YUY2-recovery-aware) → 3×3 median
    denoise → adaptive percentile stretch → mild gamma → back to BGR.

    Why the stretch + gamma aren't "optional enhancement" anymore (they
    were, until 2026-04-24): after clamping the bridge exposure down in
    ``imx568_capture.py`` so G doesn't saturate, the recovered Y plane
    sits in a narrow bright-biased band (Y≈60..140 for a typical scene)
    and looks TOO DARK on screen unless we remap it to 0..255. A flat
    percentile stretch + γ=0.85 is the same "this always helps, never
    hurts" territory as the median. Without it, the image quality is
    technically correct (all the sensor detail survives) but subjectively
    awful — dim, low-contrast, looks worse than Google Meet's raw green
    decode did.

    Stages:
      * luma extraction ─ mandatory because DirectShow's YUY2→BGR
        decode on this mono sensor produces broken 3-channel output;
        ``_to_luma`` recovers Y from G and R (soft-confidence blend) and
        falls back to BT.601 gray on real color input.
      * 3×3 median ─ edge-preserving salt-and-pepper filter, ~1 ms at
        1236×1029. Kills residual bridge-gain speckle.
      * percentile [1, 99] stretch ─ reclaims the ~0..255 display range
        from the narrow-band recovery. Sampling on a 4× subsample.
      * γ=0.85 LUT ─ gentle midtone lift for the human eye's response.

    Use ``enhance()`` if you want the heavier CLAHE/unsharp stack on top.
    """
    if bgr is None:
        return bgr
    y = _to_luma(bgr)
    # That's it. No stretch, no gamma, no bilateral filter, no median.
    #
    # Explicit anti-history note 2026-04-24: every stage we used to do
    # here was either fighting the broken DSHOW YUY2->BGR decode (now
    # bypassed by the PyAV raw path in imx568_capture.py) or amplifying
    # contrast on a dim scene. On an indoor frame with mean ≈33 the
    # percentile+gamma chain stretched the histogram ~4× and turned
    # JPEG quantization into visible wave/ripple artefacts on flat
    # surfaces (washer doors, walls). Side-by-side vs Leopard's
    # CameraTool the unprocessed Y plane was indistinguishable from
    # the manufacturer's render — Leopard does NOT apply AGC stretch,
    # they show what the sensor sees and let bridge AE handle exposure.
    #
    # If the user wants brightness or contrast knobs, those belong in
    # ``enhance()`` (opt-in via config), not in passthrough. Passthrough
    # is reserved for "show me what the sensor actually delivered".
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def enhance(
    bgr: np.ndarray,
    *,
    denoise_ksize: int = 0,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    gamma: float = 1.0,
    clahe_clip: float = 0.0,
    clahe_grid: int = 8,
    unsharp_amount: float = 0.0,
    unsharp_radius: float = 1.0,
) -> np.ndarray:
    """Optional enhancement chain. Returns a contiguous BGR uint8 frame.

    Default args = "all stages disabled" → equivalent to passthrough().
    The caller (EOManager) wires every knob from YAML so the user can
    enable individual stages without touching code.

    Stage order:
        luma → median denoise → percentile clip → gamma → CLAHE → unsharp → BGR
    """
    if bgr is None:
        return bgr
    y = _to_luma(bgr)
    if denoise_ksize and denoise_ksize >= 3:
        # medianBlur ksize must be odd; force it.
        k = int(denoise_ksize)
        if k % 2 == 0:
            k += 1
        y = cv2.medianBlur(y, k)
    # Percentile stretch is part of the chain only when CLAHE/gamma/etc
    # are enabled — i.e. when the user has opted into processing. Skip
    # if all later stages are no-ops (saves ~1.5 ms on the bypass path).
    if (gamma is not None and abs(gamma - 1.0) > 1e-3) \
       or clahe_clip > 0 or unsharp_amount > 0:
        y = _percentile_stretch(y, low_pct, high_pct)
    y = _apply_gamma(y, gamma)
    y = _apply_clahe(y, clahe_clip, clahe_grid)
    y = _apply_unsharp(y, unsharp_amount, unsharp_radius)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def apply_agc(
    bgr: np.ndarray,
    low_pct: float = 0.5,
    high_pct: float = 99.5,
    out_min: int = 0,
    out_max: int = 255,
) -> np.ndarray:
    """Backwards-compatible percentile-only AGC.

    Kept so any code path or test that imported the old function still
    works. New callers should use ``enhance()``. ``out_min``/``out_max``
    are accepted for signature compatibility but no longer plumbed
    through — historical use of non-default values was always 0..255.
    """
    if bgr is None:
        return bgr
    y = _to_luma(bgr)
    y = _percentile_stretch(y, low_pct, high_pct)
    return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)


def luma(bgr: np.ndarray) -> np.ndarray:
    """Single-channel mono view of a BGR frame. Zero-copy on mono input."""
    if bgr.ndim == 2:
        return bgr
    return bgr[:, :, 0]

"""Iterate Y-recovery algorithms against the saved raw_bridge.png.

This runs entirely OFFLINE — no camera, no user involvement. We have
``01_raw_bridge.png`` on disk from the last diagnostic capture, and
that's the worst-case real-world input we need to beat Leopard's
CameraTool on.

The point: I (Claude) run this, Read the PNG outputs directly, see
which algorithm produces the cleanest mono image, and copy the winner
into ``eo_processor._recover_y_from_yuy2_bgr``. The operator does
nothing.

Candidates (all produce uint8 mono of the same shape as the input):

  A. current       — confidence-weighted blend + flat DEAD_ZONE_FILL=150
  B. inpaint       — same blend, but dead-zone pixels filled by
                     cv2.inpaint(..., cv2.INPAINT_TELEA) from neighbors
  C. smooth_fill   — same blend, dead-zone filled by a large-kernel
                     blur of the trusted pixels (cheaper than inpaint)
  D. median_repair — run a 5×5 median ONLY over dead-zone pixels,
                     leaving trusted pixels untouched
  E. bilateral     — whole-frame bilateral filter on the weighted-blend
                     output (edge-preserving denoise of the speckle)
  F. combo         — C (smooth_fill for dead zone) + 3×3 median on the
                     final frame (current pipeline's trailing step)

Outputs land in ``scripts/eo_snapshots/diagnostic/`` with names
``30_recA_current.png`` .. ``35_recF_combo.png`` + ``iter_report.json``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"


def _weighted_blend(bgr: np.ndarray):
    """Return (y_blend, trusted_mask, dead_mask).

    y_blend is the confidence-weighted G-135 / R+179 reconstruction
    (0..255 float). trusted_mask is where total confidence > 0.01
    (i.e. at least one estimator was usable). dead_mask is the complement.
    """
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    g_weight = np.clip((255.0 - g) / 15.0, 0.0, 1.0)
    r_weight = np.clip(r / 15.0, 0.0, 1.0)
    y_from_g = np.clip(g - 135.0, 0.0, 255.0)
    y_from_r = np.clip(r + 179.0, 0.0, 255.0)
    total_w = g_weight + r_weight
    safe_w = np.where(total_w > 0.01, total_w, 1.0)
    y_blend = (y_from_g * g_weight + y_from_r * r_weight) / safe_w
    trusted = total_w > 0.01
    return y_blend, trusted, ~trusted


def recA_current(bgr):
    """Flat DEAD_ZONE_FILL=150 — the current eo_processor behavior."""
    y, trusted, dead = _weighted_blend(bgr)
    y = np.where(trusted, y, 150.0)
    return np.clip(y, 0, 255).astype(np.uint8)


def recB_inpaint(bgr):
    """Fill dead-zone pixels with cv2.inpaint(TELEA) from neighbors."""
    y, trusted, dead = _weighted_blend(bgr)
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    # cv2.inpaint wants a uint8 mask: 255 where to inpaint, 0 where to keep.
    mask = (dead.astype(np.uint8)) * 255
    if mask.sum() == 0:
        return y8
    return cv2.inpaint(y8, mask, 3, cv2.INPAINT_TELEA)


def recC_smooth_fill(bgr):
    """Dead-zone filled by a large-kernel blur of the trusted pixels.

    Cheaper than inpaint, nearly as good for small dead-zone regions.
    Strategy: blur (trusted*y) / blur(trusted) is an interpolation of
    only the trusted values, which we then stamp into the dead cells.
    """
    y, trusted, dead = _weighted_blend(bgr)
    t = trusted.astype(np.float32)
    y_masked = y * t
    # 31×31 Gaussian — big enough to bridge typical dead-zone clumps
    # without washing out real structure (we only stamp into dead cells).
    k = 31
    num = cv2.GaussianBlur(y_masked, (k, k), 0)
    den = cv2.GaussianBlur(t, (k, k), 0)
    fill = num / np.maximum(den, 1e-3)
    out = np.where(trusted, y, fill)
    return np.clip(out, 0, 255).astype(np.uint8)


def recD_median_repair(bgr):
    """Apply a 5×5 median ONLY to dead-zone pixels, untrusted regions
    get neighborhood median; trusted pixels pass through untouched.
    """
    y, trusted, dead = _weighted_blend(bgr)
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    # Full-frame 5×5 median
    med = cv2.medianBlur(y8, 5)
    # Keep trusted pixels; replace dead with median. trusted is bool HxW.
    out = np.where(trusted, y8, med)
    return out.astype(np.uint8)


def recE_bilateral(bgr):
    """Whole-frame bilateral filter on the weighted-blend output."""
    y, trusted, dead = _weighted_blend(bgr)
    y = np.where(trusted, y, 150.0)  # fill first so bilateral doesn't seed with NaN
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    # d=7, sigmaColor=25, sigmaSpace=7 — preserves text edges, kills mid-tone grain
    return cv2.bilateralFilter(y8, 7, 25, 7)


def recF_combo(bgr):
    """Smooth-fill dead zone, 3×3 median on final. This is the
    candidate production pipeline.
    """
    y = recC_smooth_fill(bgr).astype(np.uint8)
    return cv2.medianBlur(y, 3)


def recG_combo_bilateral(bgr):
    """Smooth-fill + bilateral instead of median — preserves fine edges
    better than median at the cost of ~3× CPU.
    """
    y = recC_smooth_fill(bgr)
    return cv2.bilateralFilter(y, 5, 20, 5)


def recH_combo_median_stretch(bgr):
    """Smooth-fill + 3×3 median + mild percentile stretch (1..99)."""
    y = recF_combo(bgr)
    lo, hi = np.percentile(y, [1, 99])
    if hi > lo:
        y = np.clip((y.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                    0, 255).astype(np.uint8)
    return y


def recI_nlm(bgr):
    """Non-local means denoise on the weighted-blend output.

    NLM is the gold standard for photographic noise (Gaussian +
    salt-and-pepper mix). Slow at full-res — budgeted ~80-150 ms/frame
    at 1236×1029. Try with h=10 (aggressive) and h=6 (moderate).
    """
    y = recC_smooth_fill(bgr)
    return cv2.fastNlMeansDenoising(y, h=10, templateWindowSize=7,
                                    searchWindowSize=21)


def recJ_nlm_mild(bgr):
    """Mild NLM (h=6) + 3×3 median — preserves more texture."""
    y = recC_smooth_fill(bgr)
    y = cv2.fastNlMeansDenoising(y, h=6, templateWindowSize=7,
                                 searchWindowSize=15)
    return cv2.medianBlur(y, 3)


def recK_strong_bilateral(bgr):
    """Heavy bilateral on fill+median output."""
    y = recF_combo(bgr)
    return cv2.bilateralFilter(y, 9, 45, 9)


def recL_full_chain(bgr):
    """Smooth-fill → NLM → bilateral → percentile stretch.

    The "full polish" candidate — matches Leopard-style clean mono.
    """
    y = recC_smooth_fill(bgr)
    y = cv2.fastNlMeansDenoising(y, h=7, templateWindowSize=7,
                                 searchWindowSize=15)
    y = cv2.bilateralFilter(y, 5, 25, 5)
    lo, hi = np.percentile(y, [1, 99])
    if hi > lo:
        y = np.clip((y.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                    0, 255).astype(np.uint8)
    return y


def recN_soft_confidence(bgr):
    """Fix the speckle at its source: instead of a hard dead-zone
    threshold (trusted/not trusted), blend the weighted estimate with
    the smooth-fill value in proportion to confidence. Adjacent pixels
    with slightly different R values no longer jump 30 levels.

    Formula:
        alpha = clip(total_w, 0, 1)     # 0=fully dead, 1=fully trusted
        out   = alpha * y_blend + (1 - alpha) * smooth_fill
    """
    g = bgr[..., 1].astype(np.float32)
    r = bgr[..., 2].astype(np.float32)
    g_weight = np.clip((255.0 - g) / 15.0, 0.0, 1.0)
    r_weight = np.clip(r / 15.0, 0.0, 1.0)
    y_from_g = np.clip(g - 135.0, 0.0, 255.0)
    y_from_r = np.clip(r + 179.0, 0.0, 255.0)
    total_w = g_weight + r_weight
    safe_w = np.maximum(total_w, 1e-3)
    y_blend = (y_from_g * g_weight + y_from_r * r_weight) / safe_w

    # Smooth-fill source: Gaussian-interpolate using only trusted pixels.
    trust = np.clip(total_w, 0.0, 1.0)
    weighted_y = y_blend * trust
    k = 41  # a bit wider than the transition band is thick
    num = cv2.GaussianBlur(weighted_y, (k, k), 0)
    den = cv2.GaussianBlur(trust, (k, k), 0)
    fill = num / np.maximum(den, 1e-3)

    alpha = trust  # already 0..1
    out = alpha * y_blend + (1.0 - alpha) * fill
    return np.clip(out, 0, 255).astype(np.uint8)


def recO_soft_plus_median(bgr):
    """Soft-confidence blend + 3×3 median + mild bilateral."""
    y = recN_soft_confidence(bgr)
    y = cv2.medianBlur(y, 3)
    return cv2.bilateralFilter(y, 5, 20, 5)


def recP_soft_plus_stretch(bgr):
    """Soft-confidence + 3×3 median + percentile stretch 1..99."""
    y = recN_soft_confidence(bgr)
    y = cv2.medianBlur(y, 3)
    lo, hi = np.percentile(y, [1, 99])
    if hi > lo:
        y = np.clip((y.astype(np.float32) - lo) * (255.0 / (hi - lo)),
                    0, 255).astype(np.uint8)
    return y


def recM_downscaled_nlm(bgr):
    """Downscale to GUI size FIRST (1236 wide), then NLM.

    NLM at half-res is 4× faster AND the downscale itself averages out
    a lot of the speckle. This is the production-realistic candidate —
    the GUI shows 1236px anyway.
    """
    y = recC_smooth_fill(bgr)
    if y.shape[1] > 1236:
        scale = 1236 / y.shape[1]
        y = cv2.resize(y, (1236, int(round(y.shape[0] * scale))),
                       interpolation=cv2.INTER_AREA)
    y = cv2.fastNlMeansDenoising(y, h=8, templateWindowSize=7,
                                 searchWindowSize=17)
    return y


def _stats(name, y):
    flat = y.reshape(-1)
    p1, p50, p99 = np.percentile(flat, [1, 50, 99]).tolist()
    return {
        "name": name,
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": round(float(flat.mean()), 2),
        "std": round(float(flat.std()), 2),
        "p1": round(p1, 2),
        "p50": round(p50, 2),
        "p99": round(p99, 2),
    }


def main() -> int:
    raw_path = OUT / "01_raw_bridge.png"
    if not raw_path.exists():
        print(f"missing {raw_path}", file=sys.stderr)
        return 2
    bgr = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"could not read {raw_path}", file=sys.stderr)
        return 3

    # What fraction of pixels are in the dead zone? Drives expectations.
    _, trusted, dead = _weighted_blend(bgr)
    dead_pct = 100.0 * float(dead.sum()) / float(dead.size)
    print(f"dead-zone pixel fraction: {dead_pct:.2f}%")

    variants = [
        ("30_recA_current.png",          recA_current),
        ("31_recB_inpaint.png",          recB_inpaint),
        ("32_recC_smooth_fill.png",      recC_smooth_fill),
        ("33_recD_median_repair.png",    recD_median_repair),
        ("34_recE_bilateral.png",        recE_bilateral),
        ("35_recF_combo.png",            recF_combo),
        ("36_recG_combo_bilat.png",      recG_combo_bilateral),
        ("37_recH_combo_stretch.png",    recH_combo_median_stretch),
        ("38_recI_nlm.png",              recI_nlm),
        ("39_recJ_nlm_mild.png",         recJ_nlm_mild),
        ("40_recK_strong_bilat.png",     recK_strong_bilateral),
        ("41_recL_full_chain.png",       recL_full_chain),
        ("42_recM_downscale_nlm.png",    recM_downscaled_nlm),
    ]

    report = {"dead_zone_pct": round(dead_pct, 2), "variants": []}
    for name, fn in variants:
        t0 = time.perf_counter()
        y = fn(bgr)
        dt = (time.perf_counter() - t0) * 1000.0
        cv2.imwrite(str(OUT / name), cv2.cvtColor(y, cv2.COLOR_GRAY2BGR))
        st = _stats(name, y)
        st["ms"] = round(dt, 2)
        report["variants"].append(st)
        print(f"  {name:38s} mean={st['mean']:6.1f} std={st['std']:5.1f} "
              f"p1={st['p1']:5.1f} p99={st['p99']:5.1f}  {dt:5.1f} ms")

    (OUT / "iter_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(variants)} variants to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
Live thermal A/B comparison — captures direct from the Boson.

For each raw16 frame, runs:

    BEFORE  =  the actual pre-2026-04-27 baseline:
                 2/98 percentile AGC, INFERNO colormap, NO enhancement
                 chain, BILINEAR digital-zoom upscale.
    AFTER   =  the current YAML config exactly as the live seeker runs.

Side-by-side PNG + MP4 land in recordings/thermal_compare/live_*.

Optional digital zoom via --hfov (default 75 = full sensor; pass 18 for
the mid preset, 12.5 for narrow). The zoom is applied with each
pipeline's native interpolation — bilinear for BEFORE (matches the
hardcoded INTER_LINEAR in pre-fix thermal_manager.py), config-driven
(cubic by default) for AFTER. So the zoom comparison itself reflects
the upscale-quality improvement, not just the AGC/enhance work.

The seeker MUST be stopped before running so the camera handle is free.

    python scripts/thermal_live_compare.py
    python scripts/thermal_live_compare.py --hfov 18
    python scripts/thermal_live_compare.py --hfov 18 --frames 120
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.config import load_config  # noqa: E402
from thermal.boson_capture import BosonCapture  # noqa: E402
from thermal.digital_zoom import crop_fraction  # noqa: E402
from thermal.thermal_processor import (  # noqa: E402
    apply_colormap,
    enhance_post_agc,
    from_config as enhance_from_config,
    raw16_to_display,
    raw16_to_display_with_params,
)


OUT_DIR = os.path.join("recordings", "thermal_compare")
FULL_HFOV_DEG = 75.0


def _apply_zoom(bgr: np.ndarray, hfov_deg: float, interp: int) -> np.ndarray:
    """Center-crop a BGR display image to emulate `hfov_deg` and upscale.

    Mirrors the manager's zoom path (see thermal_manager._process_and_publish):
    crop the ALREADY-DISPLAY image, resize back to original dims with the
    configured interpolation. Pre-fix, that interpolation was hardcoded
    to INTER_LINEAR; post-fix it's config-driven (cubic by default).
    """
    if hfov_deg >= FULL_HFOV_DEG:
        return bgr
    h, w = bgr.shape[:2]
    frac = crop_fraction(FULL_HFOV_DEG, hfov_deg)
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    cropped = bgr[cy0:cy0 + ch, cx0:cx0 + cw]
    return cv2.resize(cropped, (w, h), interpolation=interp)


def _label(width: int, text: str, *, height: int = 28) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _laplacian_var(u8_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(u8_gray, cv2.CV_64F).var())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60, help="Frames to capture (default 60 ≈ 1s)")
    ap.add_argument("--warmup", type=int, default=10, help="Throw away this many initial frames")
    ap.add_argument("--hfov", type=float, default=FULL_HFOV_DEG,
                    help="Digital zoom: target HFOV in degrees (default 75 = full). "
                         "Boson presets: full=75, wide=37.5, mid=18.75, narrow=12.5.")
    ap.add_argument("--out-tag", type=str, default=None, help="Filename suffix; default = timestamp")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_zoom{int(round(args.hfov))}deg" if args.hfov < FULL_HFOV_DEG else ""
    out_png = os.path.join(OUT_DIR, f"live_{tag}{suffix}.png")
    out_mp4 = os.path.join(OUT_DIR, f"live_{tag}{suffix}.mp4")

    cfg = load_config()
    p_after = enhance_from_config(cfg.get("thermal", {}))
    print(f"[live] AFTER params: {p_after}")
    print(f"[live] hfov requested: {args.hfov:.2f} deg "
          f"({'no zoom' if args.hfov >= FULL_HFOV_DEG else f'crop_fraction={crop_fraction(FULL_HFOV_DEG, args.hfov):.3f}'})")

    cap = BosonCapture()
    print("[live] opening Boson...")
    cap.start()
    if not cap.raw16_available:
        print("[live] WARNING: Boson did not negotiate Y16; "
              "the legacy pipeline can't run on AGC8-only output.",
              file=sys.stderr)
        cap.stop()
        return 2

    for _ in range(args.warmup):
        cap.grab()

    print(f"[live] capturing {args.frames} frames...")
    frames_u16 = []
    t0 = time.time()
    while len(frames_u16) < args.frames:
        f = cap.grab()
        if f is None:
            print("[live] grab returned None — aborting", file=sys.stderr)
            break
        if f.ndim != 2:
            print("[live] frame is not raw16 — aborting", file=sys.stderr)
            break
        frames_u16.append(f.copy())
    cap.stop()
    if not frames_u16:
        return 3
    print(f"[live] captured {len(frames_u16)} frames in {time.time()-t0:.2f}s")

    h, w = frames_u16[0].shape
    pairs = []
    sum_before = 0.0
    sum_after = 0.0

    for fu16 in frames_u16:
        # BEFORE — true pre-2026-04-27 baseline:
        #   percentile AGC 2/98 -> INFERNO, no enhance,
        #   then digital zoom with hardcoded INTER_LINEAR upscale.
        agc_legacy, bgr_before_full = raw16_to_display(
            fu16, colormap="INFERNO", low_percentile=2.0, high_percentile=98.0
        )
        bgr_before = _apply_zoom(bgr_before_full, args.hfov, cv2.INTER_LINEAR)

        # AFTER — exactly what the live seeker runs:
        #   current YAML enhance chain -> configured colormap,
        #   then digital zoom with the configured interp (default cubic).
        zoom_interp_name = str(
            (cfg.get("thermal", {}).get("digital_zoom", {}) or {}).get("interpolation", "cubic")
        ).lower()
        zoom_interp = {
            "linear": cv2.INTER_LINEAR, "cubic": cv2.INTER_CUBIC,
            "lanczos4": cv2.INTER_LANCZOS4, "area": cv2.INTER_AREA,
        }.get(zoom_interp_name, cv2.INTER_CUBIC)

        enh_after, bgr_after_full = raw16_to_display_with_params(fu16, p_after)
        bgr_after = _apply_zoom(bgr_after_full, args.hfov, zoom_interp)

        pairs.append((bgr_before, bgr_after))
        # Sharpness measured on the underlying gray, post-zoom, so the
        # metric reflects what the user actually sees (zoom magnifies
        # any softness the upscale introduces).
        sum_before += _laplacian_var(cv2.cvtColor(bgr_before, cv2.COLOR_BGR2GRAY))
        sum_after  += _laplacian_var(cv2.cvtColor(bgr_after,  cv2.COLOR_BGR2GRAY))

    n = len(pairs)
    delta_pct = 100.0 * (sum_after - sum_before) / max(1e-6, sum_before)
    print(f"[live] mean post-zoom sharpness  BEFORE {sum_before/n:.1f}  "
          f"AFTER {sum_after/n:.1f}  ({delta_pct:+.0f}%)")

    gap = 8
    panel_h, panel_w = h, w
    canvas_w = panel_w * 2 + gap
    canvas_h = panel_h + 28 + 28
    hfov_str = f"{args.hfov:.1f}°" if args.hfov < FULL_HFOV_DEG else "full 75°"
    metrics = (
        f"sharpness  BEFORE {sum_before/n:5.1f}    "
        f"AFTER {sum_after/n:5.1f}    "
        f"d {delta_pct:+5.1f}%    "
        f"hfov={hfov_str}   {n} frames"
    )
    labels = [
        f"BEFORE  legacy 2/98 + INFERNO + bilinear zoom  (no enhance)",
        f"AFTER   current YAML  ({p_after.colormap} + enhance + cubic zoom)",
    ]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, 30.0, (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"[live] could not open writer at {out_mp4}", file=sys.stderr)
        return 4

    mid_idx = len(pairs) // 2
    for i, (b, a) in enumerate(pairs):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x = 0
        for img, lbl in zip((b, a), labels):
            canvas[:panel_h, x:x + panel_w] = img
            canvas[panel_h:panel_h + 28, x:x + panel_w] = _label(panel_w, lbl)
            x += panel_w + gap
        canvas[panel_h + 28:, :] = _label(canvas_w, metrics)
        writer.write(canvas)
        if i == mid_idx:
            cv2.imwrite(out_png, canvas)
    writer.release()

    print(f"[live] still: {out_png}")
    print(f"[live] video: {out_mp4}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

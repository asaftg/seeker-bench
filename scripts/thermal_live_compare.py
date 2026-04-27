"""
Live thermal A/B (and colormap-3way) comparison.

Captures N frames directly from the Boson, processes each three ways,
saves a side-by-side image + MP4 to recordings/thermal_compare/live_*:

    BEFORE  =  legacy pipeline (2/98 percentile AGC, INFERNO, no enhance)
    AFTER   =  current YAML config (1/99 + dead-pixel + gamma + bilateral
               + unsharp, INFERNO)
    GRAY    =  current YAML config but with colormap forced to WHITE_HOT
               (true grayscale)

Use this to (a) see the live improvement and (b) decide if grayscale is
preferable to the false-color palette. Runs against the real Boson
hardware — run AFTER plugging the camera in.

    python scripts/thermal_live_compare.py
    python scripts/thermal_live_compare.py --frames 90
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
from thermal.thermal_processor import (  # noqa: E402
    ThermalEnhanceParams,
    apply_colormap,
    enhance_post_agc,
    from_config as enhance_from_config,
    raw16_to_display,
    raw16_to_display_with_params,
)


OUT_DIR = os.path.join("recordings", "thermal_compare")


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
    ap.add_argument("--warmup", type=int, default=10, help="Throw away this many initial frames (default 10)")
    ap.add_argument("--out-tag", type=str, default=None, help="Filename suffix; default = unix timestamp")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(OUT_DIR, f"live_{tag}.png")
    out_mp4 = os.path.join(OUT_DIR, f"live_{tag}.mp4")

    cfg = load_config()
    p_after = enhance_from_config(cfg.get("thermal", {}))
    # Same params, grayscale colormap.
    p_after_gray = ThermalEnhanceParams(**{**p_after.__dict__, "colormap": "WHITE_HOT"})
    print(f"[live] AFTER params: {p_after}")

    cap = BosonCapture()
    print("[live] opening Boson...")
    cap.start()
    if not cap.raw16_available:
        print("[live] WARNING: Boson did not negotiate Y16; "
              "the legacy pipeline can't run on AGC8-only output.",
              file=sys.stderr)
        cap.stop()
        return 2

    # Discard warmup frames — Boson FFC and AE settle in the first few.
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
            # Boson fell back to AGC8 mid-stream.
            print("[live] frame is not raw16 — aborting", file=sys.stderr)
            break
        frames_u16.append(f.copy())
    cap.stop()
    if not frames_u16:
        return 3
    print(f"[live] captured {len(frames_u16)} frames in {time.time()-t0:.2f}s")

    # Process each frame three ways.
    h, w = frames_u16[0].shape
    triples = []
    sums = {"before": 0.0, "after": 0.0, "after_gray": 0.0}
    for fu16 in frames_u16:
        # BEFORE — legacy: 2/98 percentile + INFERNO, no enhance.
        agc_legacy, bgr_before = raw16_to_display(
            fu16, colormap="INFERNO", low_percentile=2.0, high_percentile=98.0
        )
        # AFTER (current YAML, INFERNO)
        enh_after, bgr_after = raw16_to_display_with_params(fu16, p_after)
        # AFTER (current YAML, WHITE_HOT)
        enh_gray, bgr_gray = raw16_to_display_with_params(fu16, p_after_gray)
        triples.append((bgr_before, bgr_after, bgr_gray))
        # Sharpness on the underlying gray (luma) — fair across colormaps.
        sums["before"]     += _laplacian_var(agc_legacy)
        sums["after"]      += _laplacian_var(enh_after)
        sums["after_gray"] += _laplacian_var(enh_gray)

    n = len(triples)
    print(f"[live] mean Laplacian-variance sharpness:  "
          f"BEFORE {sums['before']/n:.1f}  "
          f"AFTER  {sums['after']/n:.1f}  "
          f"GRAY   {sums['after_gray']/n:.1f}")

    # Compose the canvas: 3-up horizontal with header + per-panel labels.
    gap = 8
    panel_h = h
    panel_w = w
    canvas_w = panel_w * 3 + gap * 2
    canvas_h = panel_h + 28 + 28
    metrics = (
        f"sharpness  BEFORE {sums['before']/n:5.1f}    "
        f"AFTER {sums['after']/n:5.1f}    "
        f"GRAY {sums['after_gray']/n:5.1f}    "
        f"({n} frames)"
    )
    labels = ["BEFORE  legacy 2/98 + INFERNO  (no enhance)",
              "AFTER   current YAML + INFERNO",
              "GRAY    current YAML + WHITE_HOT  (true grayscale)"]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, 30.0, (canvas_w, canvas_h))
    if not writer.isOpened():
        print(f"[live] could not open writer at {out_mp4}", file=sys.stderr)
        return 4

    mid_idx = len(triples) // 2
    for i, (b, a, g) in enumerate(triples):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        x = 0
        for img, lbl in zip((b, a, g), labels):
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

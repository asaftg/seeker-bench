"""
PROOF TOOL — capture N frames in CURRENT live-seeker mode for staged
validation (phases 0/1/2/3 of the thermal optimization plan).

Standalone, does not touch any seeker code. Captures the Boson in
whichever mode the legacy `boson_capture.py` property-set order
negotiates on this laptop (=AGC8 fallback, BGR), then renders each
frame the same way the live seeker does:
    BGR -> bgr2gray -> WHITE_HOT colormap -> software zoom

Usage:
    python scripts/_phase_capture.py --tag 00_baseline
    python scripts/_phase_capture.py --tag 01_patch1 --frames 60 --hfov 38
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

OUT_DIR = os.path.join("recordings", "thermal_compare")
FULL_HFOV = 75.0


def _crop_zoom(bgr: np.ndarray, hfov: float, interp: int = cv2.INTER_LINEAR) -> np.ndarray:
    if hfov >= FULL_HFOV:
        return bgr
    h, w = bgr.shape[:2]
    frac = float(np.tan(np.radians(hfov / 2)) / np.tan(np.radians(FULL_HFOV / 2)))
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    return cv2.resize(bgr[cy0:cy0 + ch, cx0:cx0 + cw], (w, h), interpolation=interp)


def _open_boson_legacy_order(idx_hint: int = 1):
    """Open Boson with the LEGACY property-set order (the order currently
    in boson_capture.py). On this laptop's DSHOW that falls through to
    8-bit BGR (AGC8 fallback path) — exactly what the live seeker uses.
    """
    for idx in [idx_hint, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # Legacy order: FOURCC -> CONVERT_RGB -> WIDTH/HEIGHT.
        # Silently downconverts to 8-bit BGR on this laptop.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        ok, test = cap.read()
        if ok and test is not None and test.ndim == 3 and test.shape == (512, 640, 3):
            return cap, idx
        cap.release()
    return None, None


def _label(width: int, text: str, *, height: int = 28) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _laplacian_var(gray_u8: np.ndarray) -> float:
    return float(cv2.Laplacian(gray_u8, cv2.CV_64F).var())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", required=True,
                    help="Filename tag, e.g. 00_baseline / 01_patch1 / 03_patch1+2")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--hfov", type=float, default=38.0,
                    help="Software zoom HFOV (default 38° matches your live preset)")
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_png = os.path.join(OUT_DIR, f"{args.tag}.png")
    out_mp4 = os.path.join(OUT_DIR, f"{args.tag}.mp4")

    print(f"[capture {args.tag}] opening Boson (legacy AGC8 path)...")
    cap, idx = _open_boson_legacy_order()
    if cap is None:
        print("[capture] could not open Boson (is seeker still running?)", file=sys.stderr)
        return 2
    print(f"[capture {args.tag}] opened on index {idx}, mode=AGC8 BGR")

    for _ in range(args.warmup):
        cap.read()

    print(f"[capture {args.tag}] grabbing {args.frames} frames...")
    bgrs = []
    t0 = time.time()
    while len(bgrs) < args.frames:
        ok, f = cap.read()
        if ok and f is not None and f.ndim == 3:
            bgrs.append(f.copy())
    elapsed = time.time() - t0
    cap.release()
    if not bgrs:
        return 3
    print(f"[capture {args.tag}] got {len(bgrs)} frames in {elapsed:.2f}s "
          f"(camera-side fps {len(bgrs)/elapsed:.1f})")

    # Render each frame the same way the live seeker would: BGR -> gray -> WHITE_HOT -> zoom
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = 512, 640
    canvas_h = h + 28
    writer = cv2.VideoWriter(out_mp4, fourcc, 20.0, (w, canvas_h))
    sum_sharp = 0.0
    for i, bgr in enumerate(bgrs):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # WHITE_HOT = pure grayscale (R=G=B). Just stack.
        whitehot = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        zoomed = _crop_zoom(whitehot, args.hfov, cv2.INTER_LINEAR)
        sum_sharp += _laplacian_var(cv2.cvtColor(zoomed, cv2.COLOR_BGR2GRAY))
        canvas = np.zeros((canvas_h, w, 3), dtype=np.uint8)
        canvas[:h, :] = zoomed
        canvas[h:, :] = _label(
            w, f"{args.tag}  AGC8+WHITE_HOT  hfov={args.hfov:.1f}  frame {i+1}/{len(bgrs)}")
        writer.write(canvas)
        if i == len(bgrs) // 2:
            cv2.imwrite(out_png, canvas)
    writer.release()

    print(f"[capture {args.tag}] still: {out_png}")
    print(f"[capture {args.tag}] video: {out_mp4}")
    print(f"[capture {args.tag}] mean Laplacian sharpness: {sum_sharp/len(bgrs):.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

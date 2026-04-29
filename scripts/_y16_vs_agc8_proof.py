"""
PROOF TOOL — Y16 vs AGC8 side-by-side capture.

Standalone. Does NOT touch any seeker code path. Writes a side-by-side
PNG + MP4 to recordings/thermal_compare/y16_vs_agc8_<tag>.{png,mp4}.

Captures the SAME camera (Boson on whichever index it lives on),
toggling between two property-set orders that select either:
  - AGC8 path:  current live behavior — camera does its own AGC,
                returns 8-bit BGR. Seeker's `bgr2gray -> WHITE_HOT
                colormap` pass renders this as grayscale.
  - Y16 path:   raw 16-bit thermal counts. We render with the SAME
                conservative percentile AGC the YAML uses (2/98)
                followed by WHITE_HOT colormap — no enhancement chain,
                no CLAHE, no unsharp.

Both panels then go through the SAME software digital zoom (the
operator's current 38° preset) so the comparison is fair.

The seeker MUST be stopped before running so the camera handle is free.

Usage:
    python scripts/_y16_vs_agc8_proof.py --hfov 38
    python scripts/_y16_vs_agc8_proof.py --hfov 38 --frames 90
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

OUT_DIR = os.path.join("recordings", "thermal_compare")
FULL_HFOV = 75.0


def _label(width: int, text: str, *, height: int = 32) -> np.ndarray:
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 1, cv2.LINE_AA)
    return strip


def _crop_zoom(bgr: np.ndarray, hfov: float, interp: int) -> np.ndarray:
    """Center-crop to emulate a narrower HFOV, upscale back to original."""
    if hfov >= FULL_HFOV:
        return bgr
    h, w = bgr.shape[:2]
    frac = float(np.tan(np.radians(hfov / 2)) / np.tan(np.radians(FULL_HFOV / 2)))
    cw = max(1, int(round(w * frac)))
    ch = max(1, int(round(h * frac)))
    cx0 = (w - cw) // 2
    cy0 = (h - ch) // 2
    return cv2.resize(bgr[cy0:cy0 + ch, cx0:cx0 + cw], (w, h), interpolation=interp)


def _agc_percentile(u16: np.ndarray, lo_pct: float = 2.0, hi_pct: float = 98.0) -> np.ndarray:
    lo, hi = np.percentile(u16, [lo_pct, hi_pct])
    if hi <= lo:
        return np.zeros(u16.shape, dtype=np.uint8)
    return np.clip((u16.astype(np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _open_cam(prefer_y16: bool, idx_hint: int = 1):
    """Open Boson at the right index in the requested mode.

    Returns (cap, mode_str) where mode_str is 'Y16' or 'AGC8'.
    Tries multiple indices because the Boson's index can shuffle.
    """
    for idx in [idx_hint, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if prefer_y16:
            # Y16 path: WIDTH/HEIGHT before FOURCC (the order that
            # actually negotiates Y16 on this laptop's DSHOW).
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
            ok, test = cap.read()
            if ok and test is not None and test.ndim == 2 and test.shape == (512, 640):
                # Guard: aspect 1.25 = Boson, anything else = webcam.
                return cap, "Y16"
            cap.release()
        else:
            # AGC8 path: legacy property order (FOURCC first, which on
            # this laptop falls through to 8-bit BGR).
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
            ok, test = cap.read()
            if ok and test is not None and test.ndim == 3 and test.shape == (512, 640, 3):
                return cap, "AGC8"
            cap.release()
    return None, None


def _bgr_to_gray_then_whitehot(bgr: np.ndarray) -> np.ndarray:
    """AGC8 path render — same as live seeker with colormap=WHITE_HOT."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _y16_to_whitehot(u16: np.ndarray) -> np.ndarray:
    """Y16 path render — percentile AGC then WHITE_HOT (matches YAML 2/98)."""
    agc8 = _agc_percentile(u16, 2.0, 98.0)
    return cv2.cvtColor(agc8, cv2.COLOR_GRAY2BGR)


def _laplacian_var(u8_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(u8_gray, cv2.CV_64F).var())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=60, help="Frames per mode")
    ap.add_argument("--warmup", type=int, default=15)
    ap.add_argument("--hfov", type=float, default=38.0,
                    help="Software zoom HFOV (default 38° = your current operator preset)")
    ap.add_argument("--out-tag", type=str, default=None)
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.out_tag or time.strftime("%Y%m%d_%H%M%S")
    suffix = f"_zoom{int(round(args.hfov))}deg"
    out_png = os.path.join(OUT_DIR, f"y16_vs_agc8_{tag}{suffix}.png")
    out_mp4 = os.path.join(OUT_DIR, f"y16_vs_agc8_{tag}{suffix}.mp4")
    out_npz = os.path.join(OUT_DIR, f"y16_vs_agc8_{tag}{suffix}.npz")

    # ── Capture pass 1: AGC8 (current live behavior) ──────────────
    print("[proof] opening Boson in AGC8 mode (current live behavior)...")
    cap_agc8, mode_agc8 = _open_cam(prefer_y16=False)
    if cap_agc8 is None or mode_agc8 != "AGC8":
        print("[proof] could not negotiate AGC8 path", file=sys.stderr)
        return 2
    for _ in range(args.warmup):
        cap_agc8.read()
    print(f"[proof] capturing {args.frames} AGC8 frames...")
    agc8_frames = []
    for _ in range(args.frames):
        ok, f = cap_agc8.read()
        if ok and f is not None and f.ndim == 3:
            agc8_frames.append(f.copy())
    cap_agc8.release()
    print(f"[proof]  got {len(agc8_frames)} AGC8 frames")

    # ── Capture pass 2: Y16 (proposed) ────────────────────────────
    time.sleep(0.5)  # let DirectShow release the handle cleanly
    print("[proof] opening Boson in Y16 mode (proposed path)...")
    cap_y16, mode_y16 = _open_cam(prefer_y16=True)
    if cap_y16 is None or mode_y16 != "Y16":
        print("[proof] could not negotiate Y16 path", file=sys.stderr)
        return 3
    for _ in range(args.warmup):
        cap_y16.read()
    print(f"[proof] capturing {args.frames} Y16 frames...")
    y16_frames = []
    for _ in range(args.frames):
        ok, f = cap_y16.read()
        if ok and f is not None and f.ndim == 2 and f.dtype == np.uint16:
            y16_frames.append(f.copy())
    cap_y16.release()
    print(f"[proof]  got {len(y16_frames)} Y16 frames  "
          f"(value range across capture: min={min(f.min() for f in y16_frames)}, "
          f"max={max(f.max() for f in y16_frames)})")

    if not agc8_frames or not y16_frames:
        print("[proof] missing frames — abort", file=sys.stderr)
        return 4

    # ── Render side-by-side ───────────────────────────────────────
    n = min(len(agc8_frames), len(y16_frames))
    h, w = 512, 640
    gap = 8
    panel_w = w
    panel_h = h
    canvas_w = panel_w * 2 + gap
    canvas_h = panel_h + 32 + 32

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, 20.0, (canvas_w, canvas_h))
    sum_left = 0.0
    sum_right = 0.0

    for i in range(n):
        agc8_bgr = _bgr_to_gray_then_whitehot(agc8_frames[i])
        y16_bgr = _y16_to_whitehot(y16_frames[i])
        # Apply the same software zoom both panels use
        agc8_bgr = _crop_zoom(agc8_bgr, args.hfov, cv2.INTER_LINEAR)
        y16_bgr = _crop_zoom(y16_bgr, args.hfov, cv2.INTER_LINEAR)

        sum_left += _laplacian_var(cv2.cvtColor(agc8_bgr, cv2.COLOR_BGR2GRAY))
        sum_right += _laplacian_var(cv2.cvtColor(y16_bgr, cv2.COLOR_BGR2GRAY))

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        canvas[:panel_h, :panel_w] = agc8_bgr
        canvas[:panel_h, panel_w + gap:] = y16_bgr
        canvas[panel_h:panel_h + 32, :panel_w] = _label(
            panel_w, "CURRENT  AGC8 fallback (8-bit, camera AGC) -> WHITE_HOT")
        canvas[panel_h:panel_h + 32, panel_w + gap:] = _label(
            panel_w, "PROPOSED  Y16 raw (16-bit) -> 2/98 percentile AGC -> WHITE_HOT")
        sharp_l = sum_left / (i + 1)
        sharp_r = sum_right / (i + 1)
        delta = 100.0 * (sharp_r - sharp_l) / max(1e-6, sharp_l)
        metrics = (
            f"sharpness  CURRENT {sharp_l:6.1f}    "
            f"PROPOSED {sharp_r:6.1f}    "
            f"d {delta:+5.1f}%   "
            f"hfov={args.hfov:.1f}   frame {i+1}/{n}"
        )
        canvas[panel_h + 32:, :] = _label(canvas_w, metrics)
        writer.write(canvas)
        if i == n // 2:
            cv2.imwrite(out_png, canvas)

    writer.release()

    # Persist raw Y16 + AGC8 frames so the calibration script
    # (_phase2_calibrate.py) can run the heat detector against the
    # actual scene without needing the sensor live.
    np.savez_compressed(
        out_npz,
        y16=np.stack(y16_frames[:n], axis=0),
        agc8=np.stack(agc8_frames[:n], axis=0),
        hfov_deg=np.float32(args.hfov),
    )

    print(f"[proof] still: {out_png}")
    print(f"[proof] video: {out_mp4}")
    print(f"[proof] frames: {out_npz}  ({n} frames each mode)")
    print(f"[proof] mean sharpness  CURRENT {sum_left/n:.1f}  "
          f"PROPOSED {sum_right/n:.1f}  "
          f"({100.0*(sum_right-sum_left)/max(1e-6, sum_left):+.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

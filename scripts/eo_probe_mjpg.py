"""Probe LI-IMX568 (or any UVC) with explicit MJPG FOURCC at several resolutions.

Leopard Imaging USB3 UVC cameras usually default to YUY2 640x480 at
~1 FPS. Switching the FOURCC to MJPG unlocks the sensor's real modes
(1920x1080 @ 30, 2464x2056 @ 20, etc.).
"""
from __future__ import annotations

import argparse
import time

import cv2


MODES = [
    (1920, 1080),
    (2464, 2056),
    (2560, 1440),
    (3840, 2160),
    (1280, 720),
    (640, 480),
]


def fourcc_to_str(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def try_mode(idx: int, w: int, h: int, duration_s: float = 2.0) -> None:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  {w}x{h}  index {idx} failed to open")
        return
    try:
        # Set FOURCC *before* W/H — some drivers lock the format list
        # on the first set, so order matters.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        ok, f0 = cap.read()
        if not ok or f0 is None:
            print(f"  req {w}x{h}  got {aw}x{ah} {fc}  -- first read NOTHING")
            return
        frames = 1
        t0 = time.time()
        while time.time() - t0 < duration_s:
            ok, f = cap.read()
            if ok and f is not None:
                frames += 1
        elapsed = time.time() - t0
        fps = frames / elapsed if elapsed > 0 else 0.0
        hh, ww = f0.shape[:2]
        print(f"  req {w}x{h}  got {ww}x{hh} {fc}  {fps:.1f} FPS  ({frames} in {elapsed:.1f}s)")
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1, help="Camera index (default 1 = IMX568)")
    ap.add_argument("--duration", type=float, default=2.0)
    args = ap.parse_args()
    print(f"-- Testing MJPG modes on index {args.index} --")
    for w, h in MODES:
        try_mode(args.index, w, h, args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

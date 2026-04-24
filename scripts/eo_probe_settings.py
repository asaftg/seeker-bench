"""Headless exposure/resolution sweep for LI-IMX568.

Measures FPS and mean brightness under several camera settings, saves
one snapshot per setting to scripts/eo_snapshots/. No GUI.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
import numpy as np


OUT_DIR = Path(__file__).resolve().parent / "eo_snapshots"
OUT_DIR.mkdir(exist_ok=True)


def fourcc(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def measure(idx: int, w: int, h: int, auto_exp: bool, exposure: float,
            gain: float | None, label: str, warmup_s: float = 0.5,
            measure_s: float = 2.0) -> None:
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  {label:<36}  OPEN FAILED")
        return
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        if auto_exp:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DSHOW: 0.75 = auto
        else:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual
            cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, gain)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fc = fourcc(cap.get(cv2.CAP_PROP_FOURCC))

        # Warmup
        t0 = time.time()
        while time.time() - t0 < warmup_s:
            cap.read()

        frames = 0
        last_frame = None
        t0 = time.time()
        while time.time() - t0 < measure_s:
            ok, f = cap.read()
            if ok and f is not None:
                frames += 1
                last_frame = f
        elapsed = time.time() - t0
        fps = frames / elapsed if elapsed > 0 else 0.0
        mean = float(last_frame.mean()) if last_frame is not None else -1
        print(f"  {label:<36}  {aw}x{ah} {fc}  {fps:5.1f} FPS  mean={mean:5.1f}/255")

        if last_frame is not None:
            safe = label.replace(" ", "_").replace("=", "").replace("/", "_")
            p = OUT_DIR / f"{safe}.jpg"
            cv2.imwrite(str(p), last_frame)
    finally:
        cap.release()


def main() -> int:
    idx = 1
    print(f"Saving snapshots to: {OUT_DIR}\n")

    print("== 1280x720 sweep ==")
    measure(idx, 1280, 720, auto_exp=True,  exposure=0,    gain=None, label="auto-exp 1280x720")
    for e in (-4, -6, -8, -10, -12):
        measure(idx, 1280, 720, auto_exp=False, exposure=float(e), gain=None,
                label=f"manual-exp{e} 1280x720")

    print("\n== 640x480 sweep ==")
    measure(idx, 640, 480, auto_exp=True, exposure=0, gain=None, label="auto-exp 640x480")
    for e in (-6, -10):
        measure(idx, 640, 480, auto_exp=False, exposure=float(e), gain=None,
                label=f"manual-exp{e} 640x480")

    print("\n== 1920x1080 sweep ==")
    measure(idx, 1920, 1080, auto_exp=True, exposure=0, gain=None, label="auto-exp 1920x1080")

    print(f"\nDone. Open {OUT_DIR} and eyeball the .jpg files.")
    print("Best setting = highest FPS with an image that isn't black or blown-out.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

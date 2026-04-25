"""Ask DirectShow what pixel formats the IMX568 actually advertises.

OpenCV's cv2.VideoCapture doesn't expose enumeration directly, but we can
probe by trying each common FOURCC and seeing which ones negotiate without
falling back. If YUY2 is the ONLY one that works, the FX3 is emulating it
and the flat-frame problem is baked in. If RAW8/Y8/GREY/BA81 works, we may
get actual sensor data that way.
"""
from __future__ import annotations

import cv2
import numpy as np

INDEX = 0
W, H = 2472, 2064

CANDIDATES = [
    "YUY2", "MJPG", "NV12", "YV12", "I420",
    "GREY", "Y800", "Y8  ", "Y16 ",
    "BA81", "GRBG", "RGGB", "BGGR", "GBRG",  # Bayer hints
    "RAW ", "BGR3", "RGB3",
]


def try_fourcc(tag: str) -> None:
    cap = cv2.VideoCapture(INDEX, cv2.CAP_DSHOW)
    try:
        fourcc = cv2.VideoWriter_fourcc(*tag.ljust(4)[:4])
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)

        got = int(cap.get(cv2.CAP_PROP_FOURCC))
        got_tag = "".join(chr((got >> (8 * i)) & 0xFF) for i in range(4))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, f = cap.read()
        if not ok or f is None:
            print(f"  {tag!r:8s} -> negotiated {got_tag!r} {w}x{h}  FRAME=None")
            return
        stats = (
            f"shape={f.shape} dtype={f.dtype} "
            f"min={int(f.min())} max={int(f.max())} "
            f"mean={float(f.mean()):.1f} std={float(f.std()):.1f}"
        )
        # Per-channel std tells us if it's a truly flat frame or has real content
        if f.ndim == 3:
            ch_std = [float(f[:, :, c].std()) for c in range(f.shape[2])]
            stats += f"  per-ch std={[round(s, 2) for s in ch_std]}"
        print(f"  {tag!r:8s} -> {got_tag!r} {w}x{h}  {stats}")
    finally:
        cap.release()


if __name__ == "__main__":
    print(f"Probing index {INDEX} for usable formats at {W}x{H}")
    print("(per-ch std ~= 0 means flat frame, nonzero means real scene content)")
    for tag in CANDIDATES:
        try:
            try_fourcc(tag)
        except Exception as e:
            print(f"  {tag!r:8s} -> EXCEPTION: {e}")

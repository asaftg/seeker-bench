"""Probe LI-IMX568 via MSMF with many resolutions.

Also dumps the advertised UVC format list using pygrabber (DSHOW
graph inspector) and PyShot Media Foundation enumerators if available.
"""
from __future__ import annotations

import argparse
import time
import cv2


MODES = [
    (640, 480),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
    (2048, 1536),
    (2464, 2056),
    (3840, 2160),
]


def fourcc_to_str(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def try_mode(idx: int, backend_name: str, backend_flag: int,
             w: int, h: int, force_mjpg: bool, duration_s: float = 1.5) -> None:
    cap = cv2.VideoCapture(idx, backend_flag)
    if not cap.isOpened():
        print(f"  [{backend_name}] {w}x{h}  open failed")
        return
    try:
        if force_mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        fc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        try:
            ok, f0 = cap.read()
        except Exception as e:
            print(f"  [{backend_name}] req {w}x{h}  got {aw}x{ah} {fc}  read threw {e!r}")
            return
        if not ok or f0 is None:
            print(f"  [{backend_name}] req {w}x{h}  got {aw}x{ah} {fc}  first read NOTHING")
            return
        frames = 1
        t0 = time.time()
        while time.time() - t0 < duration_s:
            ok, _ = cap.read()
            if ok:
                frames += 1
        elapsed = time.time() - t0
        fps = frames / elapsed if elapsed > 0 else 0.0
        hh, ww = f0.shape[:2]
        tag = " MJPG-req" if force_mjpg else ""
        print(f"  [{backend_name}]{tag} req {w}x{h}  got {ww}x{hh} {fc}  {fps:.1f} FPS")
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1)
    args = ap.parse_args()

    for bname, bflag in [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)]:
        print(f"\n== {bname} on index {args.index} (no FOURCC request) ==")
        for w, h in MODES:
            try_mode(args.index, bname, bflag, w, h, force_mjpg=False)
        print(f"\n== {bname} on index {args.index} (force MJPG) ==")
        for w, h in MODES:
            try_mode(args.index, bname, bflag, w, h, force_mjpg=True)

    print("\n== DSHOW advertised format list (pygrabber) ==")
    try:
        from pygrabber.dshow_graph import FilterGraph  # type: ignore
        g = FilterGraph()
        devs = g.get_input_devices()
        for i, name in enumerate(devs):
            print(f"  [{i}] {name}")
    except Exception as e:
        print(f"  pygrabber not available: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

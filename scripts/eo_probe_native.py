"""Probe the LI-IMX568 using ONLY the modes it actually advertises.

From eo_probe_formats output:
    YUY2 1280x720 @ 15 fps
    YUY2  800x460 @ 30 fps
    YUY2  640x480 @ 30 fps
    YUY2 1280x960 @ 20 fps

If any of these matches reality (measured_fps close to advertised,
mean > 0), the camera is fine and our earlier probe just wasn't
negotiating the format correctly.
"""
from __future__ import annotations

import time
import cv2


MODES = [
    (640, 480, 30.0),
    (800, 460, 30.0),
    (1280, 720, 15.0),
    (1280, 960, 20.0),
]


def fourcc(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def try_mode(w: int, h: int, expected_fps: float) -> None:
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"  {w}x{h} @ {expected_fps:.0f}  OPEN FAILED")
        return
    try:
        # Set FOURCC FIRST (DSHOW picks a matching media type)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, expected_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        afps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fc = fourcc(cap.get(cv2.CAP_PROP_FOURCC))

        # Warmup 0.5s
        t0 = time.time()
        while time.time() - t0 < 0.5:
            cap.read()

        # Measure 3s
        frames = 0
        nonzero_frames = 0
        last_mean = 0.0
        t0 = time.time()
        while time.time() - t0 < 3.0:
            ok, f = cap.read()
            if ok and f is not None:
                frames += 1
                m = float(f.mean())
                last_mean = m
                if m > 1.0:
                    nonzero_frames += 1
        elapsed = time.time() - t0
        measured_fps = frames / elapsed if elapsed > 0 else 0

        status = "OK" if (nonzero_frames > 0 and measured_fps >= expected_fps * 0.5) else "DEAD"
        print(f"  req {w}x{h}@{expected_fps:>4.0f}  got {aw}x{ah} {fc} @ {afps:>4.1f}  "
              f"meas={measured_fps:5.1f} FPS  mean={last_mean:5.1f}  "
              f"nonzero={nonzero_frames}/{frames}  [{status}]")
    finally:
        cap.release()


def main() -> int:
    print("-- Native-mode probe (exact formats from device descriptor) --")
    for w, h, fps in MODES:
        try_mode(w, h, fps)
    print()
    print("Any row with nonzero>0 means the sensor is producing real")
    print("pixel data in that mode. If ALL rows are dead, the sensor")
    print("head is not talking to the EVA bridge -- check the coax.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

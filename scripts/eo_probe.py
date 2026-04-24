"""EO camera diagnostic probe.

Tries every cv2.VideoCapture index on DSHOW and MSMF backends WITHOUT
setting a resolution hint (some UVC cameras, including Leopard Imaging
IMX568 kits, fail the first .read() if the host asks for a format they
don't natively expose). Reports index, backend, native WxH, FOURCC,
and measured FPS for each that returns a frame.

    python -m scripts.eo_probe
    python -m scripts.eo_probe --index 1
"""
from __future__ import annotations

import argparse
import time

import cv2


BACKENDS = [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]


def fourcc_to_str(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def probe_index(idx: int, backend_name: str, backend_flag: int,
                duration_s: float = 1.5) -> None:
    try:
        cap = cv2.VideoCapture(idx, backend_flag)
    except Exception as e:
        print(f"  idx={idx:<2} {backend_name:<5}  ctor threw: {e!r}")
        return
    if not cap.isOpened():
        cap.release()
        return
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fc = fourcc_to_str(cap.get(cv2.CAP_PROP_FOURCC))
        fps_reported = cap.get(cv2.CAP_PROP_FPS) or 0.0

        try:
            ok, f0 = cap.read()
        except Exception as e:
            print(f"  idx={idx:<2} {backend_name:<5}  opened {w}x{h} {fc} — first read threw: {e!r}")
            return
        if not ok or f0 is None:
            print(f"  idx={idx:<2} {backend_name:<5}  opened {w}x{h} {fc} — first read returned nothing")
            return

        # Measure real FPS over duration_s
        frames = 0
        t0 = time.time()
        while time.time() - t0 < duration_s:
            ok, f = cap.read()
            if ok and f is not None:
                frames += 1
        elapsed = time.time() - t0
        fps_meas = frames / elapsed if elapsed > 0 else 0.0
        h_actual, w_actual = f0.shape[:2]
        print(f"  idx={idx:<2} {backend_name:<5}  OK  {w_actual}x{h_actual} {fc} "
              f"reported_fps={fps_reported:.1f} measured_fps={fps_meas:.1f} "
              f"({frames} frames in {elapsed:.1f}s)")
    finally:
        cap.release()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None,
                    help="Only probe this index. Default: 0..5.")
    ap.add_argument("--duration", type=float, default=1.5,
                    help="Seconds of frames to measure per successful open.")
    args = ap.parse_args()

    indices = [args.index] if args.index is not None else list(range(6))

    print("-- Probing cv2.VideoCapture indices (no resolution hint) --")
    for bname, bflag in BACKENDS:
        print(f"\n[{bname}]")
        for idx in indices:
            probe_index(idx, bname, bflag, duration_s=args.duration)

    print()
    print("Tip: the LI-IMX568 should show up as a non-640x512 entry")
    print("     (thermal is 640x512). Whichever index reports roughly")
    print("     2464x2056 (or a scaled-down 1920x1080) is the EO camera.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

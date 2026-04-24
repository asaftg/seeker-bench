"""Live EO preview with exposure / auto-exposure / gain tweaks.

Opens the LI-IMX568 at a chosen resolution and streams to an OpenCV
window. Shows measured FPS, brightness stats, and lets you toggle
auto-exposure via keys so we can find a setting that actually runs
at a sane framerate.

Keys (while the window has focus):
    q           quit
    a           toggle CAP_PROP_AUTO_EXPOSURE between 0.25 (manual)
                and 0.75 (auto) -- DirectShow convention
    [ / ]       exposure down / up (log steps)
    - / +       gain down / up
    1 / 2 / 3   switch resolution 640x480 / 1280x720 / 1920x1080
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


def fourcc(v: float) -> str:
    n = int(v)
    if n <= 0:
        return "----"
    return "".join(chr((n >> (8 * i)) & 0xFF) for i in range(4))


def apply_mode(cap, w: int, h: int) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--auto", action="store_true",
                    help="Start in auto-exposure mode (default: manual, short exposure)")
    ap.add_argument("--exposure", type=float, default=-6.0,
                    help="DSHOW manual exposure log2 value. -6 = ~1/64s (fast), "
                         "-4 = ~1/16s, 0 = 1s. Ignored if --auto.")
    ap.add_argument("--gain", type=float, default=None)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Failed to open camera index {args.index}")
        return 1

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    apply_mode(cap, args.width, args.height)

    # Exposure control
    if args.auto:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DSHOW: 0.75=auto
        print("Auto-exposure ON")
    else:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DSHOW: 0.25=manual
        cap.set(cv2.CAP_PROP_EXPOSURE, args.exposure)
        print(f"Manual exposure={args.exposure} (log2 s)")

    if args.gain is not None:
        cap.set(cv2.CAP_PROP_GAIN, args.gain)

    aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fc = fourcc(cap.get(cv2.CAP_PROP_FOURCC))
    print(f"Opened: {aw}x{ah} {fc}")
    print("Window keys: q=quit  a=toggle auto  [=exp-  ]=exp+  -=gain-  +=gain+  1/2/3=res")

    frames = 0
    t_last = time.time()
    fps_ema = 0.0
    cur_exp = args.exposure
    cur_gain = 0.0
    auto_on = args.auto

    win = "IMX568 probe"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("read() returned nothing; retrying...")
                time.sleep(0.05)
                continue
            frames += 1
            now = time.time()
            dt = now - t_last
            if dt > 0:
                inst = 1.0 / dt
                fps_ema = 0.9 * fps_ema + 0.1 * inst if fps_ema else inst
            t_last = now

            mean = float(frame.mean())
            txt = (f"{frame.shape[1]}x{frame.shape[0]} "
                   f"fps={fps_ema:5.1f} "
                   f"mean={mean:5.1f}/255 "
                   f"exp={cur_exp:+.1f} gain={cur_gain:+.1f} "
                   f"auto={'ON' if auto_on else 'OFF'}")
            disp = frame
            if max(disp.shape[:2]) > 1200:
                sc = 1200 / max(disp.shape[:2])
                disp = cv2.resize(disp, (int(disp.shape[1] * sc), int(disp.shape[0] * sc)))
            cv2.putText(disp, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(win, disp)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('a'):
                auto_on = not auto_on
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75 if auto_on else 0.25)
                if not auto_on:
                    cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp)
                print(f"auto={auto_on}")
            elif k == ord('['):
                cur_exp -= 1
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp)
                auto_on = False
                print(f"exposure={cur_exp}")
            elif k == ord(']'):
                cur_exp += 1
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                cap.set(cv2.CAP_PROP_EXPOSURE, cur_exp)
                auto_on = False
                print(f"exposure={cur_exp}")
            elif k == ord('-'):
                cur_gain -= 1
                cap.set(cv2.CAP_PROP_GAIN, cur_gain)
                print(f"gain={cur_gain}")
            elif k in (ord('+'), ord('=')):
                cur_gain += 1
                cap.set(cv2.CAP_PROP_GAIN, cur_gain)
                print(f"gain={cur_gain}")
            elif k == ord('1'):
                apply_mode(cap, 640, 480)
            elif k == ord('2'):
                apply_mode(cap, 1280, 720)
            elif k == ord('3'):
                apply_mode(cap, 1920, 1080)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

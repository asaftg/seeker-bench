"""
Record frames from the live thermal feed into a YOLO dataset.

Usage:

    python -m thermal.training.record_for_training \\
        --duration 60 --out datasets/hand_v1 --fps 3 --fake

Every `1/fps` seconds we grab the current ThermalFrame (from BUS if
an in-process manager is running, otherwise we spin up our own source)
and write:

    <out>/images/unlabeled/<timestamp>.png      AGC display image
    <out>/images/unlabeled/<timestamp>.npy      raw 16-bit frame (if available)

The operator then labels the PNGs with labelImg / Roboflow. Once
each image has a paired .txt next to it, `dataset.split_unlabeled()`
moves them into train/val.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from common.logging_setup import configure, get_logger
from thermal.fake_thermal_source import FakeThermalSource
from thermal.thermal_processor import apply_agc, apply_colormap
from thermal.training import dataset as ds

log = get_logger(__name__)


def _open_source(fake: bool, device: int | str):
    if fake:
        src = FakeThermalSource(width=640, height=512, fps=30)
        src.start()
        return src
    from thermal.boson_capture import BosonCapture
    cap = BosonCapture(device_index=device)
    cap.start()
    return cap


def _to_display(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (display_bgr, raw16_or_None)."""
    if frame.ndim == 2:
        raw16 = frame.astype(np.uint16, copy=False)
        agc = apply_agc(raw16, 2, 98)
        display = apply_colormap(agc, "INFERNO")
        return display, raw16
    # already BGR — camera 8-bit path
    return frame, None


def main() -> int:
    p = argparse.ArgumentParser(description="Record thermal frames for training.")
    p.add_argument("--out", required=True, help="Dataset root (e.g. datasets/hand_v1)")
    p.add_argument("--duration", type=float, default=30.0, help="Seconds to record")
    p.add_argument("--fps", type=float, default=3.0, help="Sample rate in Hz")
    p.add_argument("--fake", action="store_true", help="Use synthetic source")
    p.add_argument("--device", default="auto", help="Camera device index")
    p.add_argument("--name", default="seeker_thermal", help="Dataset name written into data.yaml")
    args = p.parse_args()

    configure(level="INFO")

    layout = ds.create(Path(args.out), name=args.name)
    log.info("Dataset ready at %s", layout.root.resolve())

    period = 1.0 / max(0.1, args.fps)
    src = _open_source(args.fake, args.device)
    t_end = time.time() + args.duration
    n = 0
    try:
        while time.time() < t_end:
            frame = src.grab()
            if frame is None:
                time.sleep(0.05)
                continue
            display, raw16 = _to_display(frame)
            stem = f"{int(time.time() * 1000)}_{n:05d}"
            img_path = layout.images_unlabeled / f"{stem}.png"
            cv2.imwrite(str(img_path), display)
            if raw16 is not None:
                np.save(layout.images_unlabeled / f"{stem}.npy", raw16)
            n += 1
            time.sleep(period)
    finally:
        try:
            src.stop()
        except Exception:
            pass

    log.info("Recorded %d frames to %s", n, layout.images_unlabeled)
    log.info("Next: label them with labelImg/Roboflow, then call dataset.split_unlabeled()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

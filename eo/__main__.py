"""Standalone EO viewer.

    python -m eo                      # auto-pick a USB webcam
    python -m eo --fake               # synthetic source
    python -m eo --device 1           # force index
    python -m eo --no-classify        # raw feed, no YOLO overlays
    python -m eo --enumerate          # list camera indices and exit

Same pattern as `python -m thermal`: opens an OpenCV window, no
FastAPI, no frontend — the "is my EO pipeline alive?" debugging tool.
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from common.logging_setup import configure, get_logger

log = get_logger(__name__)


# Class-aware colors (BGR) matching the GUI legend
_COLOR_PERSON  = (236, 169, 175)   # purple   #AFA9EC (BGR swap)
_COLOR_VEHICLE = (74, 75, 226)     # red      #E24B4A
_COLOR_DRONE   = (221, 138, 55)    # blue     #378ADD
_COLOR_OTHER   = (53, 107, 255)    # orange   #ff6b35


def _color_for(class_name: str) -> tuple:
    c = (class_name or "").lower()
    if c == "person":  return _COLOR_PERSON
    if c == "vehicle": return _COLOR_VEHICLE
    if c == "drone":   return _COLOR_DRONE
    return _COLOR_OTHER


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone EO viewer")
    parser.add_argument("--fake", action="store_true", help="Synthetic source")
    parser.add_argument("--no-classify", action="store_true", help="Disable YOLO overlays")
    parser.add_argument("--device", default="auto", help="Camera index (auto|0|1|...)")
    parser.add_argument("--enumerate", action="store_true",
                        help="List working camera indices and exit")
    args = parser.parse_args()

    configure(level="INFO")

    if args.enumerate:
        from eo.webcam_capture import enumerate_cameras
        for c in enumerate_cameras():
            print(f"index {c['index']}: {c['width']}x{c['height']}")
        return 0

    if args.fake:
        from eo.fake_eo_source import FakeEOSource
        source = FakeEOSource()
        source.start()
        grab = source.grab
    else:
        from eo.webcam_capture import WebcamCapture
        try:
            cap = WebcamCapture(device_index=args.device)
            cap.start()
        except RuntimeError as e:
            log.error("%s", e)
            return 2
        grab = cap.grab

    classifier = None
    if not args.no_classify:
        try:
            from eo.eo_classifier import EOClassifier
            classifier = EOClassifier()
            if not classifier.active:
                log.warning("EOClassifier inactive — running feed-only")
                classifier = None
        except Exception as e:
            log.warning("Classifier unavailable, running feed-only: %s", e)

    window = "Seeker-01 EO (q = quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_id = 0
    t0 = time.time()
    try:
        while True:
            frame = grab()
            if frame is None:
                log.warning("No frame — camera disconnected?")
                time.sleep(0.1)
                continue

            if classifier is not None:
                dets = classifier.detect(frame)
                for d in dets:
                    x, y, w, h = d["bbox"]
                    color = _color_for(str(d["class"]))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = f"{str(d['class']).upper()} {d['conf']:.0%}"
                    cv2.putText(frame, label,
                                (x, max(14, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 1, cv2.LINE_AA)

            frame_id += 1
            fps = frame_id / max(1e-3, time.time() - t0)
            cv2.putText(frame, f"Frame {frame_id}  {fps:.1f} FPS",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
    finally:
        cv2.destroyAllWindows()
        if args.fake:
            source.stop()
        else:
            cap.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())

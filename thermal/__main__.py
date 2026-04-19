"""
Standalone thermal viewer — runs the whole thermal pipeline in
an OpenCV window, no GUI, no FastAPI. This is the "is my camera
even working?" debugging tool.

    python -m thermal                 # real camera
    python -m thermal --fake          # synthetic source
    python -m thermal --no-detect     # skip heat detector overlay
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from common.logging_setup import configure, get_logger

log = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone thermal viewer")
    parser.add_argument("--fake", action="store_true", help="Use synthetic source")
    parser.add_argument("--no-detect", action="store_true", help="Disable heat detector overlay")
    parser.add_argument("--no-classify", action="store_true", help="Disable classifier overlay")
    parser.add_argument("--device", default="auto", help="Camera device index (auto|0|1|...)")
    args = parser.parse_args()

    configure(level="INFO")

    # Lazy imports so the __main__ runner works even if some downstream
    # module (e.g. classifier) blows up — we still want the raw viewer.
    if args.fake:
        from thermal.fake_thermal_source import FakeThermalSource
        source = FakeThermalSource()
        source.start()
        grab = source.grab
        raw16_available = True
    else:
        from thermal.boson_capture import BosonCapture
        try:
            cap = BosonCapture(device_index=args.device)
            cap.start()
        except RuntimeError as e:
            log.error("%s", e)
            return 2
        grab = cap.grab
        raw16_available = cap.raw16_available

    from thermal.thermal_processor import apply_agc, apply_colormap

    detector = None
    if not args.no_detect:
        from thermal.heat_detector import HeatDetector
        detector = HeatDetector()

    classifier = None
    if not args.no_classify:
        try:
            from thermal.drone_classifier import Classifier
            classifier = Classifier()
        except Exception as e:
            log.warning("Classifier unavailable, running detection-only: %s", e)

    window = "Seeker-01 Thermal (q = quit)"
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

            # Normalize to 16-bit for processing if we got 8-bit
            if frame.ndim == 3:
                raw16 = None
                display = frame.copy()
            else:
                raw16 = frame
                agc = apply_agc(frame)
                display = apply_colormap(agc)

            detections = []
            if detector is not None and raw16 is not None:
                detections = detector.detect(raw16)
                for det in detections:
                    x, y, w, h = det.bbox.as_tuple()
                    color = (53, 107, 255)  # orange BGR
                    cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        display, f"HEAT {det.contrast:.0f}",
                        (x, max(0, y - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.4, color, 1, cv2.LINE_AA,
                    )

            if classifier is not None and detections:
                results = classifier.classify(display, detections)
                for det, result in zip(detections, results):
                    if result is None:
                        continue
                    x, y, w, h = det.bbox.as_tuple()
                    label = f"{result.target_class.value.upper()} {result.confidence:.0%}"
                    cv2.putText(
                        display, label,
                        (x, y + h + 14), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (143, 232, 0), 1, cv2.LINE_AA,
                    )

            # HUD
            frame_id += 1
            fps = frame_id / max(1e-3, (time.time() - t0))
            cv2.putText(
                display, f"Frame {frame_id}  {fps:.1f} FPS  raw16={raw16_available}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )

            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # q or Esc
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

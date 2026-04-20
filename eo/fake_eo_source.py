"""Synthetic EO source.

Generates a neutral-gray scene with a couple of moving colored blobs
that stand in for a person and a vehicle. Used for GUI dev and CI
when no webcam is plugged in.

Matches WebcamCapture.start / grab / stop so EOManager can consume
either.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np


class FakeEOSource:
    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.raw16_available = False  # API parity with WebcamCapture

        self._rng = np.random.default_rng(seed)
        self._t0: Optional[float] = None
        self._running = False
        self._frame_id = 0

    def start(self) -> None:
        self._t0 = time.time()
        self._running = True

    def stop(self) -> None:
        self._running = False

    def is_open(self) -> bool:
        return self._running

    def grab(self) -> Optional[np.ndarray]:
        if not self._running:
            return None

        # Throttle to `fps`
        target_dt = 1.0 / self.fps
        expected_elapsed = self._frame_id * target_dt
        actual_elapsed = time.time() - (self._t0 or 0)
        sleep_for = expected_elapsed - actual_elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        t = time.time() - (self._t0 or 0)

        # Neutral sky background, slight gradient top→bottom
        bg_top = np.array([130, 140, 150], dtype=np.float32)   # BGR
        bg_bot = np.array([100, 110, 120], dtype=np.float32)
        row_weights = np.linspace(0, 1, self.height)[:, None]
        rows = (1 - row_weights) * bg_top + row_weights * bg_bot
        frame = np.tile(rows[:, None, :], (1, self.width, 1)).astype(np.float32)
        # A little noise so YOLO doesn't see a pathological flat image
        frame += self._rng.normal(0, 4, size=frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # "Person" — tall upright dark blob moving horizontally. Person-shaped
        # aspect ratio plus a head makes YOLO far more likely to trigger than
        # a plain rectangle, so the GUI actually shows a HUMAN label in fake
        # mode.
        px = int(self.width * (0.3 + 0.2 * math.sin(0.4 * t)))
        py = int(self.height * 0.55)
        cv2.rectangle(frame, (px - 12, py - 40), (px + 12, py + 40),
                      (40, 40, 40), -1)
        cv2.circle(frame, (px, py - 48), 10, (30, 30, 30), -1)

        # "Vehicle" — wide horizontal blob on the lower part of the frame.
        vx = int(self.width * (0.7 + 0.15 * math.sin(0.25 * t + 1.0)))
        vy = int(self.height * 0.72)
        cv2.rectangle(frame, (vx - 55, vy - 20), (vx + 55, vy + 20),
                      (20, 40, 80), -1)   # dark red car
        cv2.rectangle(frame, (vx - 35, vy - 32), (vx + 20, vy - 20),
                      (30, 50, 90), -1)   # cabin

        self._frame_id += 1
        return frame


if __name__ == "__main__":
    src = FakeEOSource()
    src.start()
    try:
        while True:
            f = src.grab()
            if f is None:
                break
            cv2.imshow("FakeEOSource (q = quit)", f)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()

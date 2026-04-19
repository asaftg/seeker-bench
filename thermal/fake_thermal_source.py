"""
Synthetic thermal source.

Generates a cold-noise background with one or two moving hot
blobs. Used when:
    - The FLIR ADK is not plugged in (GUI dev on a plane).
    - Unit tests need a drop-in replacement for BosonCapture.
    - CI runs the smoke test end-to-end.

It exposes the same `start() / grab() / stop()` interface as
`BosonCapture` so `ThermalManager` can consume either.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np


class FakeThermalSource:
    def __init__(
        self,
        width: int = 640,
        height: int = 512,
        fps: float = 30.0,
        num_blobs: int = 1,
        seed: int = 42,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.num_blobs = num_blobs
        self.raw16_available = True  # match BosonCapture attribute

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
        """Return the next synthetic 16-bit thermal frame."""
        if not self._running:
            return None

        # Throttle to `fps` so the GUI doesn't get overwhelmed
        target_dt = 1.0 / self.fps
        expected_elapsed = self._frame_id * target_dt
        actual_elapsed = time.time() - (self._t0 or 0)
        sleep_for = expected_elapsed - actual_elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

        # Cold, low-noise background — sky-like
        base = self._rng.normal(loc=3000, scale=25, size=(self.height, self.width))
        frame = np.clip(base, 0, 65535).astype(np.uint16)

        t = time.time() - (self._t0 or 0)

        # Inject moving hot blob(s). Positions trace sine curves.
        for i in range(self.num_blobs):
            phase = i * math.pi / 2
            cx = int(self.width * (0.5 + 0.35 * math.sin(0.5 * t + phase)))
            cy = int(self.height * (0.5 + 0.25 * math.sin(0.3 * t + phase)))
            radius = 6 + i * 2
            cv2.circle(frame, (cx, cy), radius, 14000 + 1500 * i, -1)

        self._frame_id += 1
        return frame


# Allow `python -m thermal.fake_thermal_source` for quick viewing
if __name__ == "__main__":
    from thermal.thermal_processor import apply_agc, apply_colormap

    src = FakeThermalSource()
    src.start()
    try:
        while True:
            f = src.grab()
            if f is None:
                break
            disp = apply_colormap(apply_agc(f))
            cv2.imshow("FakeThermalSource (q = quit)", disp)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    finally:
        src.stop()
        cv2.destroyAllWindows()

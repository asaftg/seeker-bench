"""
FLIR ADK Boson 640 USB capture via OpenCV (DirectShow UVC).

The ADK exposes as a standard webcam on Windows — no FLIR SDK
or driver install required. We request Y16 pixel format to get
16-bit raw thermal counts. If Y16 isn't available (some laptops
only expose the 8-bit YUY2 path), we gracefully fall back to
8-bit AGC mode and flag `raw16_available = False`.

Standalone usage for debugging:

    python -m thermal              # opens an OpenCV window with live feed
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)


class BosonCapture:
    """Thin wrapper over cv2.VideoCapture with Boson-friendly defaults."""

    def __init__(
        self,
        device_index: int | str = "auto",
        width: int = 640,
        height: int = 512,
        prefer_raw16: bool = True,
    ) -> None:
        self.requested_index = device_index
        self.width = width
        self.height = height
        self.prefer_raw16 = prefer_raw16

        self.device_index: Optional[int] = None
        self.raw16_available: bool = False
        self._cap: Optional[cv2.VideoCapture] = None

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        """Open the capture device. Raises RuntimeError if nothing works."""
        if self._cap is not None:
            return

        candidates = self._candidate_indices()
        last_err: Optional[str] = None
        for idx in candidates:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                last_err = f"index {idx} failed to open"
                continue

            # Keep the driver's internal queue at 1 so every .read()
            # returns the freshest frame, not one buffered from 200ms ago.
            # Without this, slow downstream processing causes latency
            # buildup that looks like "the camera froze".
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            # Request Y16 first for raw 16-bit thermal
            raw16_ok = False
            if self.prefer_raw16:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, test = cap.read()
                # OpenCV silently ignores Y16 on some laptops and still
                # returns a BGR frame — the only reliable signal that we
                # actually got raw 16-bit data is a single-channel array.
                if ok and test is not None and test.ndim == 2:
                    raw16_ok = True

            if not raw16_ok:
                # Fall back to default 8-bit pipeline
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                ok, test = cap.read()
                if not ok or test is None:
                    cap.release()
                    last_err = f"index {idx} opened but produces no frames"
                    continue

            self._cap = cap
            self.device_index = idx
            self.raw16_available = raw16_ok
            log.info(
                "Boson capture opened on index %d (raw16=%s, shape=%s)",
                idx, raw16_ok, test.shape,
            )
            return

        raise RuntimeError(
            f"Could not open any camera device (tried {candidates}). "
            f"Last error: {last_err}. "
            f"Is the FLIR ADK plugged in? Check Device Manager > Cameras."
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "BosonCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ───────────────────────── capture ─────────────────────────

    def grab(self) -> Optional[np.ndarray]:
        """Grab the next frame.

        Returns:
            uint16 (H, W) array if raw16 mode is active,
            uint8 (H, W, 3) BGR array otherwise,
            or None on read failure (caller should treat as disconnect).
        """
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        return frame

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ───────────────────────── helpers ─────────────────────────

    def _candidate_indices(self) -> list[int]:
        if isinstance(self.requested_index, int):
            return [self.requested_index]
        if self.requested_index == "auto" or self.requested_index is None:
            return [0, 1, 2, 3]
        try:
            return [int(self.requested_index)]
        except (TypeError, ValueError):
            return [0, 1, 2, 3]


# ───────────────────────────────────────────────────────────────
# Self-test: quick frame-rate probe
# ───────────────────────────────────────────────────────────────

def probe(duration_s: float = 2.0) -> Tuple[int, float]:
    """Capture for `duration_s` and return (frames_read, fps)."""
    cap = BosonCapture()
    cap.start()
    frames = 0
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            f = cap.grab()
            if f is not None:
                frames += 1
    finally:
        cap.stop()
    elapsed = time.time() - t0
    return frames, frames / elapsed if elapsed > 0 else 0.0

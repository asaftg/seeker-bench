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
        exclude_indices: Optional[list[int]] = None,
    ) -> None:
        self.requested_index = device_index
        self.width = width
        self.height = height
        self.prefer_raw16 = prefer_raw16
        # Indices to skip during auto-probe — used so the thermal probe
        # doesn't steal the IMX568's handle. Without this, opening+setting
        # FOURCC on the IMX568's active index knocks its stream offline.
        self.exclude_indices = list(exclude_indices or [])

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
            if idx in self.exclude_indices:
                # Another manager already owns this index (typically the
                # IMX568 on the EO side). Probing it would disrupt the
                # active stream — skip silently.
                continue
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

            # Request Y16 first for raw 16-bit thermal.
            #
            # Property-set ORDER MATTERS on this laptop's DSHOW (verified
            # 2026-04-27 by direct probe). FOURCC-before-WIDTH/HEIGHT
            # silently drops Y16 and downconverts to 8-bit BGR; setting
            # WIDTH/HEIGHT then CONVERT_RGB=0 then FOURCC negotiates Y16
            # correctly and returns a uint16 single-channel array.
            #
            # 2026-04-27 re-enable: paired with thermal.agc.mode=clahe_y16
            # in YAML so the wider dynamic range gets put to use through
            # CLAHE on the raw 16-bit data (closest software equivalent
            # of the camera's onboard DDE/AGC). Heat detector
            # threshold_k slider extended to max 100 in the GUI for
            # operator suppression headroom.
            raw16_ok = False
            if self.prefer_raw16:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("Y", "1", "6", " "))
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

            # Reject non-Boson devices. The ADK is 640×512 (5:4 aspect, ~1.25).
            # Any webcam is 16:9 at 720p+/1080p (~1.78). If we're in auto-probe
            # mode and got a webcam-shaped frame, skip to the next index so
            # the Insta360 can't impersonate the thermal camera.
            if isinstance(self.requested_index, str) and self.requested_index == "auto":
                fh, fw = test.shape[:2]
                aspect = fw / max(1, fh)
                # Boson 640 = 640×512 (1.25); Boson 320 = 320×256 (1.25).
                # Everything ≥ 1.5 is a webcam. Also reject anything wider
                # than 800 px — Boson never exceeds 640.
                if aspect >= 1.5 or fw > 800:
                    cap.release()
                    last_err = (f"index {idx} is a webcam ({fw}x{fh}, aspect "
                                f"{aspect:.2f}) not a FLIR Boson — skipping")
                    log.info(last_err)
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

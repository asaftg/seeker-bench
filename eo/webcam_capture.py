"""USB webcam capture for the EO pipeline.

A thin wrapper around ``cv2.VideoCapture`` with DirectShow backend on
Windows. Designed to stand in for the real IMX568 hardware until it
arrives — the API (``start / grab / stop``) matches ``BosonCapture``
so ``EOManager`` can consume either.

Standalone::

    python -m eo.webcam_capture            # auto-pick a working index
    python -m eo.webcam_capture --device 1 # force index
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)


def enumerate_cameras(max_index: int = 5) -> list[dict]:
    """Probe indices 0..max_index-1 and return a list of working cameras.

    Called by the GUI to populate the device-selector dropdowns so the
    user can assign which physical camera is Thermal vs EO.

    Returns a list of dicts::

        {"index": int, "ok": bool, "width": int, "height": int}

    Opens and immediately releases each candidate so we don't lock a
    device another module is already using.
    """
    out: list[dict] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        try:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            out.append({"index": idx, "ok": True, "width": w, "height": h})
        finally:
            cap.release()
    return out


class WebcamCapture:
    """UVC USB webcam wrapper, BGR uint8 output."""

    def __init__(
        self,
        device_index: int | str = "auto",
        width: int = 1920,
        height: int = 1080,
        exclude_indices: Optional[list[int]] = None,
    ) -> None:
        self.requested_index = device_index
        self.req_width = width
        self.req_height = height
        # Indices to skip during auto-probe — used to keep the thermal
        # camera's index out of the EO auto-pick pool, and vice versa.
        self.exclude_indices = list(exclude_indices or [])

        self.device_index: Optional[int] = None
        self.raw16_available: bool = False  # API parity with BosonCapture
        self.actual_width: int = 0
        self.actual_height: int = 0
        self._cap: Optional[cv2.VideoCapture] = None
        # Try DirectShow first (faster, honors resolution hints). Fall
        # back to MSMF + "any backend" — some UVC webcams only surface
        # on MSMF, others only on the default backend. This is the
        # single biggest source of "camera not found" bugs on Windows.
        self._backends: list[tuple[str, int]] = [
            ("DSHOW", cv2.CAP_DSHOW),
            ("MSMF",  cv2.CAP_MSMF),
            ("ANY",   cv2.CAP_ANY),
        ]

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        """Open the capture device. Raises RuntimeError if nothing works."""
        if self._cap is not None:
            return

        candidates = self._candidate_indices()
        last_err: Optional[str] = None
        # Every cv2 call on a flaky device can throw "Unknown C++
        # exception" — not just .set()/.read() but .release() and even
        # .isOpened(). Wrap them all. Treat any exception as "try the
        # next backend/index".
        def _safe(fn, *args, default=None):
            try:
                return fn(*args)
            except Exception:
                return default

        for backend_name, backend_flag in self._backends:
            for idx in candidates:
                if idx in self.exclude_indices:
                    continue
                try:
                    cap = cv2.VideoCapture(idx, backend_flag)
                except Exception as e:
                    last_err = f"[{backend_name}] idx {idx} VideoCapture ctor threw {e!r}"
                    continue

                if not _safe(cap.isOpened, default=False):
                    _safe(cap.release)
                    last_err = f"[{backend_name}] index {idx} failed to open"
                    continue

                _safe(cap.set, cv2.CAP_PROP_BUFFERSIZE, 1)
                # CAP_PROP_FRAME_* can raise "Unknown C++ exception" on
                # Windows when the device is in a weird state — e.g.
                # right after a failed DirectShow probe on another index
                # left the bus stale. Silently fall back to the device's
                # native resolution; the probe below still rejects if
                # the frame is unusable.
                if _safe(cap.set, cv2.CAP_PROP_FRAME_WIDTH,  self.req_width)  is None:
                    log.info("[%s] idx %d CAP_PROP_FRAME_WIDTH set threw — using native",
                             backend_name, idx)
                if _safe(cap.set, cv2.CAP_PROP_FRAME_HEIGHT, self.req_height) is None:
                    log.info("[%s] idx %d CAP_PROP_FRAME_HEIGHT set threw — using native",
                             backend_name, idx)

                try:
                    ok, test = cap.read()
                except Exception as e:
                    _safe(cap.release)
                    last_err = f"[{backend_name}] idx {idx} read threw {e!r}"
                    log.info("%s — skipping", last_err)
                    continue
                if not ok or test is None:
                    _safe(cap.release)
                    last_err = f"[{backend_name}] index {idx} opened but produces no frames"
                    continue

                # Guard against DirectShow handing back a colliding handle
                # to the thermal camera: if the frame matches thermal's
                # 640x512, treat it as a mis-identification and keep probing.
                h_test, w_test = (test.shape[0], test.shape[1]) if test.ndim == 3 else (0, 0)
                if (w_test, h_test) == (640, 512) and idx in self.exclude_indices:
                    _safe(cap.release)
                    last_err = f"[{backend_name}] index {idx} returned thermal-shaped frame, skipping"
                    continue

                self._cap = cap
                self.device_index = idx
                self.actual_height, self.actual_width = h_test, w_test
                log.info(
                    "WebcamCapture opened on index %d via %s (requested %dx%d, got %dx%d)",
                    idx, backend_name, self.req_width, self.req_height,
                    self.actual_width, self.actual_height,
                )
                return

        raise RuntimeError(
            f"Could not open any EO camera (tried {candidates} "
            f"across DSHOW/MSMF/ANY, excluded {self.exclude_indices}). "
            f"Last error: {last_err}."
        )

    def stop(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ───────────────────────── capture ─────────────────────────

    def grab(self) -> Optional[np.ndarray]:
        """Return the next BGR uint8 frame, or None on read failure."""
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
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        try:
            return [int(self.requested_index)]
        except (TypeError, ValueError):
            return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


# ───────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────
def probe(duration_s: float = 2.0) -> Tuple[int, float]:
    cap = WebcamCapture()
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


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--enumerate", action="store_true",
                    help="List working cv2.VideoCapture indices and exit")
    args = ap.parse_args()

    if args.enumerate:
        for c in enumerate_cameras():
            print(f"index {c['index']}: {c['width']}x{c['height']}")
    else:
        cap = WebcamCapture(device_index=args.device)
        cap.start()
        print(f"Opened index {cap.device_index} at {cap.actual_width}x{cap.actual_height}")
        print(f"Probing 2s...")
        f, fps = probe(2.0)
        print(f"  {f} frames / {fps:.1f} FPS")

"""EO focus assist — web edition.

MJPEG stream in browser with Tenengrad + Laplacian sharpness overlay.
Same IMX568 capture + sharpness math as the Tkinter version, but served
as JPEG frames over HTTP — works remotely with no X11.

Stop the seeker before running (camera is exclusive-open).

Run:
    cd ~/seeker-bench
    .venv/bin/python scripts/eo_focus_web.py
Then open http://<jetson-ip>:8090 in a browser.
"""
from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from aiohttp import web

from eo.imx568_capture import IMX568Capture

DEFAULT_EXPOSURE_MS = 5.0
DEFAULT_ROI_FRAC = 0.3
PREVIEW_W = 900
JPEG_QUALITY = 70
PORT = 8090


def _tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def _laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


class FocusEngine:
    """Grabs frames, computes sharpness, renders annotated JPEG."""

    def __init__(self, device_index, exposure_ms, roi_frac, auto_exp):
        self.cap = IMX568Capture(device_index=device_index)
        self.cap.start()
        if not self.cap.is_open():
            raise RuntimeError("Could not open IMX568")

        self.aw = int(getattr(self.cap, "actual_width", 0) or 0)
        self.ah = int(getattr(self.cap, "actual_height", 0) or 0)
        print(f"Opened: {self.aw}x{self.ah}", flush=True)

        self.roi_frac = roi_frac
        self.best_t = 0.0
        self.best_l = 0.0
        self._fps_ema = 0.0
        self._t_last = time.time()

        if auto_exp:
            self.cap.set_auto_exposure(True)
            self._exp_label = "AUTO"
        else:
            self.cap.set_exposure_ms(exposure_ms)
            self._exp_label = f"{exposure_ms:.1f}ms"

        self._lock = threading.Lock()
        self._jpeg: bytes = b""
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while self._running:
            frame = self.cap.grab()
            if frame is None:
                time.sleep(0.05)
                continue

            now = time.time()
            dt = now - self._t_last
            self._t_last = now
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = (0.85 * self._fps_ema + 0.15 * inst
                                 if self._fps_ema else inst)

            h, w = frame.shape[:2]
            rw = max(8, int(w * self.roi_frac))
            rh = max(8, int(h * self.roi_frac))
            x0 = (w - rw) // 2
            y0 = (h - rh) // 2
            x1 = x0 + rw
            y1 = y0 + rh

            roi = frame[y0:y1, x0:x1]
            gray = (cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    if roi.ndim == 3 else roi)

            ten = _tenengrad(gray)
            lap = _laplacian_var(gray)
            if ten > self.best_t:
                self.best_t = ten
            if lap > self.best_l:
                self.best_l = lap

            rt = ten / self.best_t * 100 if self.best_t else 0
            rl = lap / self.best_l * 100 if self.best_l else 0

            # Downscale for preview
            sc = PREVIEW_W / w if w > PREVIEW_W else 1.0
            disp = cv2.resize(frame, (int(w * sc), int(h * sc)))

            # Draw ROI box
            cv2.rectangle(disp,
                          (int(x0 * sc), int(y0 * sc)),
                          (int(x1 * sc), int(y1 * sc)),
                          (0, 255, 0), 2)

            # Overlay text
            font = cv2.FONT_HERSHEY_SIMPLEX
            lines = [
                f"{self.aw}x{self.ah} @ {self._fps_ema:.1f} fps   exp={self._exp_label}",
                f"Tenengrad: {ten:.0f}  (best {self.best_t:.0f} = {rt:.0f}%)",
                f"Laplacian: {lap:.0f}  (best {self.best_l:.0f} = {rl:.0f}%)",
            ]
            y_text = int(self.ah * sc) - 10
            for line in reversed(lines):
                # Black outline + white text
                cv2.putText(disp, line, (10, y_text), font, 0.65,
                            (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(disp, line, (10, y_text), font, 0.65,
                            (255, 255, 255), 1, cv2.LINE_AA)
                y_text -= 28

            ok, buf = cv2.imencode(".jpg", disp,
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()

    def get_jpeg(self) -> bytes:
        with self._lock:
            return self._jpeg

    def stop(self):
        self._running = False
        try:
            self.cap.stop()
        except Exception:
            pass


HTML = """\
<!DOCTYPE html>
<html><head><title>EO Focus Assist</title>
<style>
  body { background: #111; margin: 0; display: flex;
         justify-content: center; align-items: center; height: 100vh; }
  img  { max-width: 100vw; max-height: 100vh; }
</style></head>
<body><img src="/stream"></body></html>
"""


async def index(request):
    return web.Response(text=HTML, content_type="text/html")


async def stream(request):
    engine: FocusEngine = request.app["engine"]
    resp = web.StreamResponse()
    resp.content_type = "multipart/x-mixed-replace; boundary=frame"
    await resp.prepare(request)
    try:
        while True:
            jpeg = engine.get_jpeg()
            if jpeg:
                await resp.write(
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            import asyncio
            await asyncio.sleep(0.03)  # ~30 Hz cap
    except (ConnectionResetError, ConnectionAbortedError):
        pass
    return resp


def main():
    ap = argparse.ArgumentParser(description="EO focus assist (web)")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--auto", action="store_true")
    ap.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS)
    ap.add_argument("--roi-frac", type=float, default=DEFAULT_ROI_FRAC)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    device_index = args.index if args.index is not None else "auto"
    print(f"Opening IMX568 (device_index={device_index})...", flush=True)

    engine = FocusEngine(device_index, args.exposure_ms,
                         args.roi_frac, args.auto)

    app = web.Application()
    app["engine"] = engine
    app.router.add_get("/", index)
    app.router.add_get("/stream", stream)

    print(f"\n  Focus assist running on http://0.0.0.0:{args.port}/\n",
          flush=True)
    try:
        web.run_app(app, host="0.0.0.0", port=args.port, print=None)
    finally:
        engine.stop()


if __name__ == "__main__":
    main()

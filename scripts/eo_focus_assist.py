"""EO focus assist — live sharpness metric for IMX568 lens focusing.

Small Tkinter GUI: live preview from the IMX568 plus Tenengrad and Laplacian
sharpness scores computed over a center ROI, with a "best so far" tracker.
Use it when changing the lens or adjusting the focus ring on the bench.

Stop the seeker before running — the IMX568 is exclusive-open under
DirectShow, so only one process can hold it at a time.

Why this routes through eo.imx568_capture.IMX568Capture instead of a plain
cv2.VideoCapture: the LI-IMX568-GMSL2 bridge advertises YUY2 over UVC but
DirectShow's auto-decode mangles the colors (frame comes back uniform
green). The proper driver pulls raw YUY2 and slices the Y plane — that's
what gives us a usable mono image to compute sharpness on.

Run:
    python -m scripts.eo_focus_assist
"""
from __future__ import annotations

import argparse
import collections
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

# Allow running as `python eo_focus_assist.py` from inside scripts/, or
# `python -m scripts.eo_focus_assist` from the project root, or via a
# shortcut that doesn't bother setting the working directory.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from eo.imx568_capture import IMX568Capture


DEFAULT_EXPOSURE_MS = 5.0
DEFAULT_ROI_FRAC = 0.3
PREVIEW_MAX_W = 900
HISTORY_LEN = 200
TICK_MS = 50          # ~20 Hz
GRAPH_W = 880
GRAPH_H = 140
BEEP_THRESHOLD = 0.98
BEEP_HOLD_S = 0.5
EXPOSURE_MIN_MS = 0.1
EXPOSURE_MAX_MS = 33.0


def _tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return float((gx * gx + gy * gy).mean())


def _laplacian_var(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


class FocusAssist:
    def __init__(self, args):
        self.args = args
        device_index = args.index if args.index is not None else "auto"
        print(f"Opening IMX568 (device_index={device_index})… "
              f"this can take a few seconds.", flush=True)
        self.cap = IMX568Capture(device_index=device_index)
        self.cap.start()
        if not self.cap.is_open():
            raise SystemExit("IMX568Capture.start() did not yield an open device.")
        self.aw = int(getattr(self.cap, "actual_width", 0) or 0)
        self.ah = int(getattr(self.cap, "actual_height", 0) or 0)
        print(f"Opened: {self.aw}x{self.ah}  "
              f"fourcc={getattr(self.cap, '_fourcc', '?')}  "
              f"raw_yuy2={'yes' if getattr(self.cap, '_raw_yuy2_mode', False) else 'no'}",
              flush=True)

        self.auto_exp = bool(args.auto)
        self.exposure_ms = float(args.exposure_ms)
        if self.auto_exp:
            self.cap.set_auto_exposure(True)
        else:
            self.cap.set_exposure_ms(self.exposure_ms)

        self.roi_frac = args.roi_frac
        self.full_frame = False
        self.beep_enabled = bool(args.beep)

        self.hist_t: collections.deque[float] = collections.deque(maxlen=HISTORY_LEN)
        self.hist_l: collections.deque[float] = collections.deque(maxlen=HISTORY_LEN)
        self.best_t = 0.0
        self.best_l = 0.0
        self._near_peak_since: float | None = None
        self._fps_ema = 0.0
        self._t_last = time.time()
        self._photo = None

        self.root = tk.Tk()
        self.root.title(f"EO focus assist — {self.aw}x{self.ah}")
        self.root.bind("<KeyPress>", self._on_key)

        self.preview_lbl = tk.Label(self.root, bg="black")
        self.preview_lbl.pack(padx=8, pady=(8, 4))

        self.status_var = tk.StringVar(value="…")
        ttk.Label(self.root, textvariable=self.status_var,
                  font=("Consolas", 10)).pack(anchor="w", padx=8)

        self.score_var = tk.StringVar(value="…")
        ttk.Label(self.root, textvariable=self.score_var,
                  font=("Consolas", 13), justify="left").pack(
            anchor="w", padx=8, pady=(4, 4))

        self.graph = tk.Canvas(self.root, width=GRAPH_W, height=GRAPH_H,
                               bg="#101010", highlightthickness=0)
        self.graph.pack(padx=8, pady=4)

        btns = ttk.Frame(self.root)
        btns.pack(pady=(4, 8))
        ttk.Button(btns, text="Reset best (R)",
                   command=self._reset_best).pack(side="left", padx=4)
        ttk.Button(btns, text="Toggle ROI (F)",
                   command=self._toggle_roi).pack(side="left", padx=4)
        ttk.Button(btns, text="Auto exp. (A)",
                   command=self._toggle_auto).pack(side="left", padx=4)
        ttk.Button(btns, text="Quit (Q)",
                   command=self._quit).pack(side="left", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.after(TICK_MS, self._tick)

    def _on_key(self, ev):
        k = ev.keysym.lower()
        if k in ("q", "escape"):
            self._quit()
        elif k == "r":
            self._reset_best()
        elif k == "f":
            self._toggle_roi()
        elif k == "a":
            self._toggle_auto()
        elif k == "bracketleft":
            self._step_exposure(0.5)
        elif k == "bracketright":
            self._step_exposure(2.0)

    def _step_exposure(self, factor: float):
        new_ms = max(EXPOSURE_MIN_MS,
                     min(EXPOSURE_MAX_MS, self.exposure_ms * factor))
        self.exposure_ms = new_ms
        self.auto_exp = False
        self.cap.set_exposure_ms(new_ms)
        self._reset_best()

    def _toggle_auto(self):
        self.auto_exp = not self.auto_exp
        if self.auto_exp:
            self.cap.set_auto_exposure(True)
        else:
            self.cap.set_exposure_ms(self.exposure_ms)
        self._reset_best()

    def _reset_best(self):
        self.best_t = 0.0
        self.best_l = 0.0
        self._near_peak_since = None

    def _toggle_roi(self):
        self.full_frame = not self.full_frame
        self._reset_best()

    def _quit(self):
        try:
            self.cap.stop()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _tick(self):
        frame = self.cap.grab()
        if frame is None:
            self.status_var.set("grab() returned None — retrying…")
            self.root.after(TICK_MS, self._tick)
            return

        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst if self._fps_ema else inst

        h, w = frame.shape[:2]
        if self.full_frame:
            x0, y0, x1, y1 = 0, 0, w, h
        else:
            rw = max(8, int(w * self.roi_frac))
            rh = max(8, int(h * self.roi_frac))
            x0 = (w - rw) // 2
            y0 = (h - rh) // 2
            x1 = x0 + rw
            y1 = y0 + rh

        roi = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi

        ten = _tenengrad(gray)
        lap = _laplacian_var(gray)

        self.hist_t.append(ten)
        self.hist_l.append(lap)
        if ten > self.best_t:
            self.best_t = ten
        if lap > self.best_l:
            self.best_l = lap

        if self.beep_enabled and self.best_t > 0:
            ratio = ten / self.best_t
            if ratio >= BEEP_THRESHOLD:
                if self._near_peak_since is None:
                    self._near_peak_since = now
                elif now - self._near_peak_since >= BEEP_HOLD_S:
                    try:
                        import winsound
                        winsound.Beep(1200, 80)
                    except Exception:
                        pass
                    self._near_peak_since = now + 1e9
            else:
                self._near_peak_since = None

        disp = frame
        sc = 1.0
        if w > PREVIEW_MAX_W:
            sc = PREVIEW_MAX_W / w
            disp = cv2.resize(disp, (int(w * sc), int(h * sc)))
        cv2.rectangle(disp,
                      (int(x0 * sc), int(y0 * sc)),
                      (int(x1 * sc), int(y1 * sc)),
                      (0, 255, 0), 2)
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_lbl.configure(image=self._photo)

        roi_w = x1 - x0
        roi_h = y1 - y0
        roi_pct = roi_w / w * 100.0
        if self.auto_exp:
            exp_lbl = "AUTO exp"
        else:
            exp_lbl = f"exp={self.exposure_ms:5.2f} ms"
        self.status_var.set(
            f"{w}x{h} @ {self._fps_ema:5.1f} fps   "
            f"{exp_lbl}   "
            f"ROI {roi_w}x{roi_h} ({roi_pct:.0f}%)   "
            f"{'FULL' if self.full_frame else 'CENTER'}"
        )

        rt = (ten / self.best_t * 100.0) if self.best_t > 0 else 0.0
        rl = (lap / self.best_l * 100.0) if self.best_l > 0 else 0.0
        self.score_var.set(
            f"Tenengrad : {ten:10.1f}   (best {self.best_t:10.1f}  →  {rt:5.1f}% of peak)\n"
            f"Laplacian : {lap:10.1f}   (best {self.best_l:10.1f}  →  {rl:5.1f}% of peak)"
        )

        self._draw_graph()
        self.root.after(TICK_MS, self._tick)

    def _draw_graph(self):
        c = self.graph
        c.delete("all")
        if not self.hist_t:
            return
        hi = max(max(self.hist_t), self.best_t)
        if hi <= 0:
            return

        def y_for(v: float) -> float:
            return GRAPH_H - 4 - (v / hi) * (GRAPH_H - 8)

        def x_for(i: int, n: int) -> float:
            if n <= 1:
                return GRAPH_W - 4
            return 4 + (i / (n - 1)) * (GRAPH_W - 8)

        by = y_for(self.best_t)
        c.create_line(4, by, GRAPH_W - 4, by, fill="#666666", dash=(2, 4))

        n = len(self.hist_t)
        pts_t: list[float] = []
        for i, v in enumerate(self.hist_t):
            pts_t.extend([x_for(i, n), y_for(v)])
        if len(pts_t) >= 4:
            c.create_line(*pts_t, fill="#ffcc33", width=2)

        if self.best_l > 0:
            scale = self.best_t / self.best_l
            pts_l: list[float] = []
            for i, v in enumerate(self.hist_l):
                pts_l.extend([x_for(i, n), y_for(v * scale)])
            if len(pts_l) >= 4:
                c.create_line(*pts_l, fill="#33ccff", width=1, dash=(3, 3))

    def run(self):
        try:
            self.root.mainloop()
        finally:
            try:
                self.cap.stop()
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(
        description="EO focus assist (Tenengrad/Laplacian on a live IMX568 stream).")
    ap.add_argument("--index", type=int, default=None,
                    help="Force a cv2.VideoCapture index. Default: auto-probe "
                         "0..4 (matches the seeker's IMX568Capture default).")
    ap.add_argument("--auto", action="store_true",
                    help="Start with software auto-exposure on. Press 'a' to "
                         "toggle. NOTE: AE makes the sharpness metric noisier "
                         "as it re-meters; for the actual focus sweep prefer "
                         "manual exposure.")
    ap.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
                    help=f"Manual exposure in ms (default {DEFAULT_EXPOSURE_MS}). "
                         f"Range {EXPOSURE_MIN_MS}..{EXPOSURE_MAX_MS}. "
                         "Use [ / ] to halve / double live.")
    ap.add_argument("--roi-frac", type=float, default=DEFAULT_ROI_FRAC,
                    help="Center-ROI side length as fraction of frame (0.05..1.0).")
    ap.add_argument("--beep", action="store_true",
                    help="Beep on Windows when within 98%% of peak (held 0.5 s).")
    args = ap.parse_args()

    if not 0.05 <= args.roi_frac <= 1.0:
        print("--roi-frac must be in [0.05, 1.0]", file=sys.stderr)
        return 2

    app = FocusAssist(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

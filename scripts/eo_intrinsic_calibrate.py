"""EO intrinsic calibration — ChArUco-based, writes config/eo_intrinsics.yaml.

Tkinter GUI for IMX568 intrinsic calibration. Live preview shows the
detected ChArUco corners; you capture 15-20 views from different angles /
distances, hit Compute to run cv2.calibrateCamera, then Save to write the
result to config/eo_intrinsics.yaml plus a full session record under
recordings/<timestamp>_eo_intrinsics/.

The output is consumed by the parallel 6-DOF extrinsic calibration tool —
that's the canonical handoff path.

Stop the seeker before running; the IMX568 is exclusive-open under DSHOW.

Run:
    python -m scripts.eo_intrinsic_calibrate
    python -m scripts.eo_intrinsic_calibrate --cols 8 --rows 11 \
        --square-mm 15 --marker-mm 11 --dict DICT_4X4_50

NOTE: --cols/--rows/--dict MUST match what's printed on the board, or
detection finds nothing. The square/marker mm only affect rvecs/tvecs
scale, not the intrinsics K/dist — leave them at the defaults if unsure.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk

# Allow running as `python eo_intrinsic_calibrate.py` from inside scripts/,
# or `python -m scripts.eo_intrinsic_calibrate` from the project root, or
# via a shortcut that doesn't set the working directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageTk

from eo.imx568_capture import IMX568Capture

RECORDINGS_DIR = REPO_ROOT / "recordings"
INTRINSICS_OUT = REPO_ROOT / "config" / "eo_intrinsics.yaml"

DEFAULT_COLS = 8
DEFAULT_ROWS = 11
DEFAULT_SQUARE_MM = 15.0
DEFAULT_MARKER_MM = 11.0
DEFAULT_DICT = "DICT_4X4_50"
DEFAULT_TARGET_VIEWS = 20
DEFAULT_EXPOSURE_MS = 5.0

PREVIEW_MAX_W = 900
TICK_MS = 50
COVERAGE_GRID_X = 4
COVERAGE_GRID_Y = 3


def _resolve_dict(name: str):
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    code = getattr(cv2.aruco, name, None)
    if code is None:
        raise SystemExit(f"Unknown ArUco dictionary: {name}. "
                         f"Try DICT_4X4_50, DICT_5X5_100, DICT_6X6_250, etc.")
    return cv2.aruco.getPredefinedDictionary(code)


class IntrinsicCalibrator:
    def __init__(self, args):
        self.args = args

        self.dict_name = args.dict
        self.dictionary = _resolve_dict(args.dict)
        self.cols = int(args.cols)
        self.rows = int(args.rows)
        self.square_mm = float(args.square_mm)
        self.marker_mm = float(args.marker_mm)
        self.board = cv2.aruco.CharucoBoard(
            (self.cols, self.rows),
            self.square_mm / 1000.0,
            self.marker_mm / 1000.0,
            self.dictionary,
        )
        self.detector = cv2.aruco.CharucoDetector(self.board)

        self.target_views = int(args.target_views)

        device_index = args.index if args.index is not None else "auto"
        print(f"Opening IMX568 (device_index={device_index})…", flush=True)
        self.cap = IMX568Capture(device_index=device_index)
        self.cap.start()
        if not self.cap.is_open():
            raise SystemExit("IMX568Capture.start() did not yield an open device.")
        self.image_w = int(getattr(self.cap, "actual_width", 0) or 0)
        self.image_h = int(getattr(self.cap, "actual_height", 0) or 0)
        print(f"Opened: {self.image_w}x{self.image_h}", flush=True)

        self.exposure_ms = float(args.exposure_ms)
        self.cap.set_exposure_ms(self.exposure_ms)

        self.saved_charuco_corners: list[np.ndarray] = []
        self.saved_charuco_ids: list[np.ndarray] = []
        self.saved_frames: list[np.ndarray] = []
        self.coverage = np.zeros((COVERAGE_GRID_Y, COVERAGE_GRID_X), dtype=bool)

        self.last_charuco_corners = None
        self.last_charuco_ids = None
        self.last_marker_count = 0
        self.last_corner_count = 0

        self.calib_result: dict | None = None
        self.session_dir: Path | None = None
        self._photo = None
        self._last_full_frame = None

        self._fps_ema = 0.0
        self._t_last = time.time()

        self.root = tk.Tk()
        self.root.title(f"EO intrinsic calibration — {self.image_w}x{self.image_h}")
        self.root.bind("<KeyPress>", self._on_key)

        self.preview_lbl = tk.Label(self.root, bg="black")
        self.preview_lbl.pack(padx=8, pady=(8, 4))

        self.status_var = tk.StringVar(value="…")
        ttk.Label(self.root, textvariable=self.status_var,
                  font=("Consolas", 10)).pack(anchor="w", padx=8)

        self.detect_var = tk.StringVar(value="…")
        ttk.Label(self.root, textvariable=self.detect_var,
                  font=("Consolas", 11), justify="left").pack(
            anchor="w", padx=8, pady=(2, 2))

        self.captured_var = tk.StringVar(
            value=f"Captured 0 / {self.target_views}    coverage: 0 / "
                  f"{COVERAGE_GRID_X * COVERAGE_GRID_Y} cells")
        ttk.Label(self.root, textvariable=self.captured_var,
                  font=("Consolas", 11), justify="left").pack(
            anchor="w", padx=8, pady=(2, 2))

        self.result_var = tk.StringVar(value="(no calibration computed yet)")
        ttk.Label(self.root, textvariable=self.result_var,
                  font=("Consolas", 11), justify="left").pack(
            anchor="w", padx=8, pady=(4, 4))

        btns = ttk.Frame(self.root)
        btns.pack(pady=(4, 8))
        self.btn_capture = ttk.Button(btns, text="Capture (Space)",
                                       command=self._capture)
        self.btn_capture.pack(side="left", padx=4)
        self.btn_compute = ttk.Button(btns, text="Compute (C)",
                                       command=self._compute, state="disabled")
        self.btn_compute.pack(side="left", padx=4)
        self.btn_save = ttk.Button(btns, text="Save && emit YAML (S)",
                                    command=self._save, state="disabled")
        self.btn_save.pack(side="left", padx=4)
        ttk.Button(btns, text="Discard last (U)",
                   command=self._undo).pack(side="left", padx=4)
        ttk.Button(btns, text="Quit (Q)",
                   command=self._quit).pack(side="left", padx=4)

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.after(TICK_MS, self._tick)

    def _on_key(self, ev):
        k = ev.keysym.lower()
        if k in ("q", "escape"):
            self._quit()
        elif k == "space":
            self._capture()
        elif k == "c":
            self._compute()
        elif k == "s":
            self._save()
        elif k == "u":
            self._undo()
        elif k == "bracketleft":
            self._step_exposure(0.5)
        elif k == "bracketright":
            self._step_exposure(2.0)

    def _step_exposure(self, factor: float):
        new_ms = max(0.1, min(33.0, self.exposure_ms * factor))
        self.exposure_ms = new_ms
        self.cap.set_exposure_ms(new_ms)

    def _capture(self):
        if self.last_charuco_corners is None or self.last_corner_count < 6:
            self.detect_var.set(f"  cannot capture — need >=6 corners, have "
                                f"{self.last_corner_count}")
            return
        if self._last_full_frame is None:
            return
        self.saved_charuco_corners.append(self.last_charuco_corners.copy())
        self.saved_charuco_ids.append(self.last_charuco_ids.copy())
        self.saved_frames.append(self._last_full_frame.copy())
        self._update_coverage(self.last_charuco_corners)
        self._refresh_counts()
        if len(self.saved_charuco_corners) >= 8:
            self.btn_compute.configure(state="normal")

    def _undo(self):
        if not self.saved_charuco_corners:
            return
        self.saved_charuco_corners.pop()
        self.saved_charuco_ids.pop()
        self.saved_frames.pop()
        self.coverage[:] = False
        for cc in self.saved_charuco_corners:
            self._update_coverage(cc)
        self._refresh_counts()
        if len(self.saved_charuco_corners) < 8:
            self.btn_compute.configure(state="disabled")
            self.btn_save.configure(state="disabled")
            self.calib_result = None
            self.result_var.set("(no calibration computed yet)")

    def _update_coverage(self, charuco_corners: np.ndarray):
        pts = charuco_corners.reshape(-1, 2)
        cell_w = self.image_w / COVERAGE_GRID_X
        cell_h = self.image_h / COVERAGE_GRID_Y
        for x, y in pts:
            cx = int(min(COVERAGE_GRID_X - 1, max(0, x // cell_w)))
            cy = int(min(COVERAGE_GRID_Y - 1, max(0, y // cell_h)))
            self.coverage[cy, cx] = True

    def _refresh_counts(self):
        n = len(self.saved_charuco_corners)
        cells = int(self.coverage.sum())
        total_cells = COVERAGE_GRID_X * COVERAGE_GRID_Y
        self.captured_var.set(
            f"Captured {n} / {self.target_views}    "
            f"coverage: {cells} / {total_cells} cells"
        )

    def _compute(self):
        if len(self.saved_charuco_corners) < 8:
            return
        all_obj: list[np.ndarray] = []
        all_img: list[np.ndarray] = []
        for cc, cid in zip(self.saved_charuco_corners, self.saved_charuco_ids):
            try:
                obj_pts, img_pts = self.board.matchImagePoints(cc, cid)
            except Exception as e:
                print(f"matchImagePoints failed on a view: {e}")
                continue
            if obj_pts is None or img_pts is None:
                continue
            if len(obj_pts) < 4:
                continue
            all_obj.append(obj_pts)
            all_img.append(img_pts)

        if len(all_obj) < 6:
            self.result_var.set(
                f"  not enough usable views ({len(all_obj)}); "
                "capture more, with the board rotated / tilted")
            return

        try:
            ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                all_obj, all_img, (self.image_w, self.image_h), None, None
            )
        except cv2.error as e:
            self.result_var.set(f"  calibrateCamera failed: {e}")
            return

        per_view_rms = []
        for obj_pts, img_pts, rvec, tvec in zip(all_obj, all_img, rvecs, tvecs):
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, dist)
            err = float(np.sqrt(np.mean((proj.reshape(-1, 2) -
                                          img_pts.reshape(-1, 2)) ** 2)))
            per_view_rms.append(err)

        self.calib_result = {
            "rms": float(ret),
            "K": K,
            "dist": dist,
            "n_views": len(all_obj),
            "per_view_rms": per_view_rms,
        }

        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        hfov_deg = float(np.degrees(2.0 * np.arctan(self.image_w / (2.0 * fx))))
        self.result_var.set(
            f"  RMS reproj err = {ret:.3f} px   over {len(all_obj)} views\n"
            f"  fx={fx:.1f}  fy={fy:.1f}  cx={cx:.1f}  cy={cy:.1f}\n"
            f"  hFOV (computed) = {hfov_deg:.2f}°   (expected ~11°)"
        )
        self.btn_save.configure(state="normal")

    def _save(self):
        if not self.calib_result:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sess = RECORDINGS_DIR / f"{ts}_eo_intrinsics"
        (sess / "frames").mkdir(parents=True, exist_ok=True)
        for i, frm in enumerate(self.saved_frames):
            cv2.imwrite(str(sess / "frames" / f"charuco_{i:02d}.png"), frm)

        K = self.calib_result["K"]
        dist = self.calib_result["dist"].reshape(-1).tolist()
        fx = float(K[0, 0]); fy = float(K[1, 1])
        cx = float(K[0, 2]); cy = float(K[1, 2])
        hfov_deg = float(np.degrees(2.0 * np.arctan(self.image_w / (2.0 * fx))))
        vfov_deg = float(np.degrees(2.0 * np.arctan(self.image_h / (2.0 * fy))))

        result_record = {
            "image_width": self.image_w,
            "image_height": self.image_h,
            "fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "camera_matrix": K.tolist(),
            "dist_coeffs": dist,
            "reprojection_error_px": float(self.calib_result["rms"]),
            "per_view_rms_px": self.calib_result["per_view_rms"],
            "hfov_deg_computed": hfov_deg,
            "vfov_deg_computed": vfov_deg,
            "n_views": int(self.calib_result["n_views"]),
            "calibrated_at": datetime.now().isoformat(timespec="seconds"),
            "board": {
                "type": "charuco",
                "dict": self.dict_name,
                "squares_x": self.cols,
                "squares_y": self.rows,
                "square_length_mm": self.square_mm,
                "marker_length_mm": self.marker_mm,
            },
        }

        with open(sess / "calibration.json", "w", encoding="utf-8") as f:
            json.dump(result_record, f, indent=2)

        yaml_text = self._format_yaml(result_record)
        INTRINSICS_OUT.parent.mkdir(parents=True, exist_ok=True)
        INTRINSICS_OUT.write_text(yaml_text, encoding="utf-8")
        (sess / "eo_intrinsics.yaml").write_text(yaml_text, encoding="utf-8")

        self.session_dir = sess
        self.result_var.set(
            f"  RMS = {result_record['reprojection_error_px']:.3f} px   "
            f"hFOV={hfov_deg:.2f}°  vFOV={vfov_deg:.2f}°\n"
            f"  WROTE {INTRINSICS_OUT}\n"
            f"  session: {sess.relative_to(REPO_ROOT)}"
        )
        print(f"\n=== Wrote {INTRINSICS_OUT} ===")
        print(yaml_text)
        print(f"=== Session dir: {sess} ===\n")

    def _format_yaml(self, r: dict) -> str:
        K = r["camera_matrix"]
        dist = r["dist_coeffs"]
        lines = [
            "# EO intrinsics — IMX568 + 35mm lens",
            f"# Generated by scripts/eo_intrinsic_calibrate.py at {r['calibrated_at']}",
            f"# Board: ChArUco {r['board']['squares_x']}x{r['board']['squares_y']}, "
            f"{r['board']['dict']}, square={r['board']['square_length_mm']}mm",
            f"# Reprojection RMS = {r['reprojection_error_px']:.4f} px over "
            f"{r['n_views']} views",
            "",
            "eo_intrinsics:",
            f"  image_width: {r['image_width']}",
            f"  image_height: {r['image_height']}",
            f"  fx: {r['fx']:.6f}",
            f"  fy: {r['fy']:.6f}",
            f"  cx: {r['cx']:.6f}",
            f"  cy: {r['cy']:.6f}",
            "  camera_matrix:",
            f"    - [{K[0][0]: .8f}, {K[0][1]: .8f}, {K[0][2]: .8f}]",
            f"    - [{K[1][0]: .8f}, {K[1][1]: .8f}, {K[1][2]: .8f}]",
            f"    - [{K[2][0]: .8f}, {K[2][1]: .8f}, {K[2][2]: .8f}]",
            "  dist_coeffs: ["
            + ", ".join(f"{x:.8f}" for x in dist) + "]",
            f"  reprojection_error_px: {r['reprojection_error_px']:.6f}",
            f"  hfov_deg_computed: {r['hfov_deg_computed']:.4f}",
            f"  vfov_deg_computed: {r['vfov_deg_computed']:.4f}",
            f"  n_views: {r['n_views']}",
            f"  calibrated_at: \"{r['calibrated_at']}\"",
            "  board:",
            f"    type: {r['board']['type']}",
            f"    dict: {r['board']['dict']}",
            f"    squares_x: {r['board']['squares_x']}",
            f"    squares_y: {r['board']['squares_y']}",
            f"    square_length_mm: {r['board']['square_length_mm']}",
            f"    marker_length_mm: {r['board']['marker_length_mm']}",
            "",
        ]
        return "\n".join(lines)

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
        self._last_full_frame = frame

        now = time.time()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            inst = 1.0 / dt
            self._fps_ema = 0.9 * self._fps_ema + 0.1 * inst if self._fps_ema else inst

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self.detector.detectBoard(gray)
        except Exception:
            charuco_corners, charuco_ids = None, None
            marker_corners, marker_ids = None, None

        n_markers = 0 if marker_ids is None else int(len(marker_ids))
        n_corners = 0 if charuco_ids is None else int(len(charuco_ids))
        self.last_charuco_corners = charuco_corners
        self.last_charuco_ids = charuco_ids
        self.last_marker_count = n_markers
        self.last_corner_count = n_corners

        disp = frame.copy()
        sc = 1.0
        if w > PREVIEW_MAX_W:
            sc = PREVIEW_MAX_W / w
            disp = cv2.resize(disp, (int(w * sc), int(h * sc)))
        if marker_corners is not None and len(marker_corners) > 0:
            scaled = [c * sc for c in marker_corners]
            cv2.aruco.drawDetectedMarkers(disp, scaled, marker_ids,
                                          borderColor=(0, 255, 255))
        if charuco_corners is not None and n_corners > 0:
            for p in charuco_corners.reshape(-1, 2):
                cv2.circle(disp, (int(p[0] * sc), int(p[1] * sc)), 4,
                           (0, 255, 0), 2)

        self._draw_coverage(disp)

        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.preview_lbl.configure(image=self._photo)

        self.status_var.set(
            f"{w}x{h} @ {self._fps_ema:5.1f} fps   exp={self.exposure_ms:5.2f} ms   "
            f"board: {self.cols}x{self.rows} {self.dict_name}"
        )
        ok_capture = "✓ ready" if n_corners >= 6 else "(need >=6 corners)"
        self.detect_var.set(
            f"  detected: {n_markers} markers   {n_corners} ChArUco corners   "
            f"{ok_capture}"
        )

        self.root.after(TICK_MS, self._tick)

    def _draw_coverage(self, disp: np.ndarray):
        pad = 8
        cell = 18
        gw = COVERAGE_GRID_X * cell
        gh = COVERAGE_GRID_Y * cell
        x0 = disp.shape[1] - gw - pad
        y0 = pad
        cv2.rectangle(disp, (x0 - 2, y0 - 2),
                      (x0 + gw + 2, y0 + gh + 2),
                      (200, 200, 200), 1)
        for cy in range(COVERAGE_GRID_Y):
            for cx in range(COVERAGE_GRID_X):
                px = x0 + cx * cell
                py = y0 + cy * cell
                color = (0, 200, 0) if self.coverage[cy, cx] else (60, 60, 60)
                cv2.rectangle(disp, (px + 1, py + 1),
                              (px + cell - 1, py + cell - 1),
                              color, -1)

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
        description="ChArUco intrinsic calibration for IMX568, "
                    "writes config/eo_intrinsics.yaml.")
    ap.add_argument("--index", type=int, default=None,
                    help="Force a cv2.VideoCapture index. Default: auto-probe.")
    ap.add_argument("--cols", type=int, default=DEFAULT_COLS,
                    help=f"ChArUco squares in X. Default {DEFAULT_COLS} "
                         "(matches calib.io 8x11 A4 board).")
    ap.add_argument("--rows", type=int, default=DEFAULT_ROWS,
                    help=f"ChArUco squares in Y. Default {DEFAULT_ROWS}.")
    ap.add_argument("--square-mm", type=float, default=DEFAULT_SQUARE_MM,
                    help=f"Square length in mm. Default {DEFAULT_SQUARE_MM}. "
                         "Doesn't affect K, only rvecs/tvecs scale.")
    ap.add_argument("--marker-mm", type=float, default=DEFAULT_MARKER_MM,
                    help=f"Aruco marker length in mm. Default {DEFAULT_MARKER_MM}.")
    ap.add_argument("--dict", type=str, default=DEFAULT_DICT,
                    help=f"Aruco dictionary name. Default {DEFAULT_DICT}. "
                         "Must match the board.")
    ap.add_argument("--target-views", type=int, default=DEFAULT_TARGET_VIEWS,
                    help=f"Suggested number of captured views. Default "
                         f"{DEFAULT_TARGET_VIEWS}.")
    ap.add_argument("--exposure-ms", type=float, default=DEFAULT_EXPOSURE_MS,
                    help=f"Initial manual exposure in ms (default "
                         f"{DEFAULT_EXPOSURE_MS}). [/ ] to halve / double live.")
    args = ap.parse_args()

    if args.cols < 3 or args.rows < 3:
        print("--cols and --rows must each be >= 3", file=sys.stderr)
        return 2
    if args.marker_mm >= args.square_mm:
        print("--marker-mm must be < --square-mm", file=sys.stderr)
        return 2

    app = IntrinsicCalibrator(args)
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

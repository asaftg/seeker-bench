"""Live capture GUI for thermal + EO↔thermal stereo calibration.

Opens both cameras, shows a side-by-side preview with live ChArUco
overlay, tracks coverage of the FOV with a 3×3 heat grid, and saves
shots on SPACEBAR. Two modes:

  --mode thermal   capture thermal stills (for calibrate_thermal_intrinsics)
  --mode paired    capture synchronized EO+thermal pairs (for stereo)

The ChArUco target is the dual-modal board described in the plan
(paper printout on aluminum foil/cookie-sheet backing, hair-dryer or
heating-pad pre-heated for thermal contrast).

Capture procedure
-----------------
For each shot:

  1. Hold the board still for ~200 ms (Boson thermal time constant +
     IMX568 rolling shutter both demand this). The GUI shows a
     red→green "MOVE/STILL" indicator based on inter-frame pixel
     difference, but trust your hands more than the indicator.
  2. Aim for ~25 shots covering all four image quadrants and tilts
     of ±30° in pitch and yaw. The coverage grid fills in green as
     each cell receives a shot.
  3. Vary distance such that the board fills 30–70% of the frame.
  4. SPACE captures. ESC/Q quits and prints the calibration command
     to run on the saved directory.

Why not video: each pose is one independent constraint on the solve;
adjacent video frames are nearly identical and add no information,
plus video encoding compresses chroma + edges (the very signal corner
detection lives on). Stills are correct.

Usage
-----
    # thermal-only intrinsic capture
    python -m scripts.calibration_capture --mode thermal

    # synchronized EO+thermal pair capture (for stereo extrinsic)
    python -m scripts.calibration_capture --mode paired

Keys
----
    SPACE   capture current frame(s)
    Q/ESC   quit
    G       toggle coverage grid overlay
    R       reset coverage grid
    H       toggle help overlay
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Reconfigure stdout/stderr to UTF-8 so --help (which prints the
# unicode-rich docstring) works on Windows cp1252 consoles. No-op on
# Linux/macOS.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

log = logging.getLogger("calibration_capture")


# ─────────────────────────── ChArUco helpers ───────────────────────────

def _build_charuco(squares_x: int, squares_y: int,
                   square_mm: float, marker_mm: float):
    """Build the board + detector. OpenCV 4.7+ API only — legacy free
    functions (detectMarkers, interpolateCornersCharuco,
    CharucoBoard_create) were removed.
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sq_m = square_mm * 1e-3
    mk_m = marker_mm * 1e-3
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_m, mk_m, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, aruco_dict, detector


def _detect_charuco(gray: np.ndarray, board, aruco_dict, detector):
    """Return (charuco_corners, charuco_ids, marker_corners) or
    (None, None, marker_corners) if not enough corners refined.

    OpenCV 4.7+ CharucoDetector API: detector.detectBoard(image)
    → (charuco_corners, charuco_ids, marker_corners, marker_ids).
    """
    ch_corners, ch_ids, m_corners, _m_ids = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None, m_corners
    return ch_corners, ch_ids, m_corners


# ─────────────────────────── coverage grid ───────────────────────────

class CoverageGrid:
    """3×3 grid that tracks where in the image plane shots have been
    placed. Helps the user spread captures across the FOV."""

    def __init__(self, frame_w: int, frame_h: int, n: int = 3) -> None:
        self.n = n
        self.cw = frame_w / n
        self.ch = frame_h / n
        self.counts = np.zeros((n, n), dtype=np.int32)

    def reset(self) -> None:
        self.counts[:] = 0

    def record(self, corners: np.ndarray) -> None:
        """Increment whichever cells the detected ChArUco corners cover."""
        cx = float(np.mean(corners[:, 0, 0]))
        cy = float(np.mean(corners[:, 0, 1]))
        ix = min(self.n - 1, int(cx / self.cw))
        iy = min(self.n - 1, int(cy / self.ch))
        self.counts[iy, ix] += 1

    def overlay(self, img: np.ndarray) -> None:
        """Draw the grid on `img` in-place."""
        h, w = img.shape[:2]
        cw = w / self.n
        ch = h / self.n
        for iy in range(self.n):
            for ix in range(self.n):
                x0, y0 = int(ix * cw), int(iy * ch)
                x1, y1 = int((ix + 1) * cw), int((iy + 1) * ch)
                count = int(self.counts[iy, ix])
                if count == 0:
                    color = (60, 60, 200)     # red-ish: needed
                elif count < 3:
                    color = (60, 200, 200)    # yellow: some
                else:
                    color = (60, 200, 60)     # green: well-covered
                cv2.rectangle(img, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2),
                              color, 1, cv2.LINE_AA)
                cv2.putText(img, str(count),
                            (x0 + 6, y0 + 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, color, 1, cv2.LINE_AA)


# ─────────────────────────── motion gate ───────────────────────────

class MotionGate:
    """Cheap inter-frame diff to flag whether the board is being held
    still. Captures during motion bake error into the calibration."""

    def __init__(self, threshold: float = 3.0) -> None:
        self.prev: np.ndarray | None = None
        self.threshold = threshold
        self.last_diff: float = 0.0

    def update(self, gray: np.ndarray) -> bool:
        """Return True if the frame is "still" (low motion)."""
        small = cv2.resize(gray, (160, 120))
        if self.prev is None:
            self.prev = small
            return False
        d = float(np.mean(cv2.absdiff(small, self.prev)))
        self.last_diff = d
        self.prev = small
        return d < self.threshold


# ─────────────────────────── frame conversion ─────────────────────────

def _thermal_to_display(frame: np.ndarray) -> np.ndarray:
    """Convert whatever the BosonCapture grab() returned into a 3-channel
    BGR uint8 image suitable for both display and ChArUco detection."""
    if frame is None:
        return np.zeros((512, 640, 3), dtype=np.uint8)
    if frame.dtype == np.uint16:
        # Y16: linear stretch to 8-bit. AGC8 isn't necessarily available
        # outside the seeker_bench process, so we DIY a simple percentile
        # stretch here.
        lo = np.percentile(frame, 2)
        hi = np.percentile(frame, 98)
        if hi - lo < 1:
            hi = lo + 1
        scaled = np.clip((frame.astype(np.float32) - lo) / (hi - lo) * 255.0,
                         0, 255).astype(np.uint8)
        return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


# ─────────────────────────── HUD overlay ───────────────────────────

def _draw_hud(img: np.ndarray, lines: list[str], origin=(10, 24)) -> None:
    """Draw a translucent box with HUD text in-place."""
    x, y = origin
    box_w = max(cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]
                for s in lines) + 16 if lines else 0
    box_h = 22 * len(lines) + 8
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 6, y - 18), (x - 6 + box_w, y - 18 + box_h),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (x, y + 22 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 1,
                    cv2.LINE_AA)


# ─────────────────────────── main loop ───────────────────────────

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=("thermal", "paired"), required=True,
                   help="thermal: thermal-only stills (intrinsic). "
                        "paired: synchronized EO+thermal (stereo).")
    p.add_argument("--out", default=None,
                   help="Output directory. Defaults to "
                        "recordings/calib_<mode>_<timestamp>/")
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=9)
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--marker-mm", type=float, default=22.0)
    p.add_argument("--motion-threshold", type=float, default=3.0,
                   help="Mean inter-frame pixel difference below which the "
                        "frame is considered 'still'. Lower = stricter.")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # Output directory
    if args.out:
        out_dir = Path(args.out)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = _ROOT / "recordings" / f"calib_{args.mode}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Saving to %s", out_dir)

    # ChArUco target
    board, aruco_dict, charuco_detector = _build_charuco(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm)

    # Open cameras
    log.info("Opening thermal camera (Boson)…")
    from thermal.boson_capture import BosonCapture
    thr = BosonCapture()
    thr.start()
    if not thr.is_open():
        log.error("Thermal camera failed to open")
        return 2

    eo = None
    if args.mode == "paired":
        log.info("Opening EO camera (IMX568)…")
        from eo.imx568_capture import IMX568Capture
        eo = IMX568Capture(exclude_indices=[thr.device_index]
                           if thr.device_index is not None else None)
        eo.start()

    # Tkinter window — works without cv2 GUI support so the user
    # doesn't have to muck with their pip environment.
    import tkinter as tk
    from PIL import Image, ImageTk

    state = {
        "shot_idx": 0,
        "show_grid": True,
        "show_help": True,
        "coverage_thr": None,
        "coverage_eo": None,
        "motion_thr": MotionGate(args.motion_threshold),
        "motion_eo": MotionGate(args.motion_threshold),
        "running": True,
        "last_thr_gray": None,
        "last_eo_gray": None,
        "last_t_corners": None,
        "last_e_corners": None,
        "last_ready": False,
    }

    root = tk.Tk()
    root.title(f"Calibration capture — {args.mode.upper()}  "
               f"[SPACE capture · Q quit · G grid · R reset · H help]")
    root.configure(bg="#101010")

    image_label = tk.Label(root, bg="#101010")
    image_label.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    status_var = tk.StringVar(value="opening cameras…")
    status_label = tk.Label(root, textvariable=status_var,
                            font=("Consolas", 11), fg="#dddddd",
                            bg="#181818", anchor="w", padx=8, pady=4)
    status_label.pack(fill=tk.X, side=tk.BOTTOM)

    def on_close():
        state["running"] = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)

    def do_quit(_ev=None):
        on_close()

    def do_toggle_grid(_ev=None):
        state["show_grid"] = not state["show_grid"]

    def do_toggle_help(_ev=None):
        state["show_help"] = not state["show_help"]

    def do_reset(_ev=None):
        if state["coverage_thr"] is not None:
            state["coverage_thr"].reset()
        if state["coverage_eo"] is not None:
            state["coverage_eo"].reset()
        log.info("Coverage grid reset")

    def do_capture(_ev=None):
        if not state["last_ready"]:
            log.warning("Not ready — corners missing or motion detected")
            return
        state["shot_idx"] += 1
        idx = state["shot_idx"]
        if args.mode == "thermal":
            out_path = out_dir / f"shot_{idx:03d}.png"
            cv2.imwrite(str(out_path), state["last_thr_gray"])
            state["coverage_thr"].record(state["last_t_corners"])
            log.info("Captured %s", out_path.name)
        else:
            shot_dir = out_dir / f"shot_{idx:03d}"
            shot_dir.mkdir(exist_ok=True)
            cv2.imwrite(str(shot_dir / "thermal.png"), state["last_thr_gray"])
            cv2.imwrite(str(shot_dir / "eo.png"), state["last_eo_gray"])
            state["coverage_thr"].record(state["last_t_corners"])
            state["coverage_eo"].record(state["last_e_corners"])
            log.info("Captured %s/", shot_dir.name)

    root.bind("<space>", do_capture)
    root.bind("q", do_quit)
    root.bind("Q", do_quit)
    root.bind("<Escape>", do_quit)
    root.bind("g", do_toggle_grid)
    root.bind("G", do_toggle_grid)
    root.bind("h", do_toggle_help)
    root.bind("H", do_toggle_help)
    root.bind("r", do_reset)
    root.bind("R", do_reset)

    log.info("Live preview running. SPACE to capture, Q/ESC to quit.")

    def tick():
        if not state["running"]:
            return
        try:
            thr_raw = thr.grab()
            if thr_raw is None:
                root.after(33, tick)
                return
            thr_bgr = _thermal_to_display(thr_raw)
            thr_gray = _to_gray(thr_bgr)
            if state["coverage_thr"] is None:
                state["coverage_thr"] = CoverageGrid(
                    thr_gray.shape[1], thr_gray.shape[0])

            eo_bgr = None
            eo_gray = None
            if eo is not None:
                eo_bgr = eo.grab()
                if eo_bgr is None:
                    root.after(33, tick)
                    return
                eo_gray = _to_gray(eo_bgr)
                if state["coverage_eo"] is None:
                    state["coverage_eo"] = CoverageGrid(
                        eo_gray.shape[1], eo_gray.shape[0])

            # Live ChArUco detection
            t_corners, t_ids, t_markers = _detect_charuco(
                thr_gray, board, aruco_dict, charuco_detector)
            if t_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(
                    thr_bgr, t_corners, t_ids, (0, 255, 0))
            elif t_markers is not None and len(t_markers) > 0:
                cv2.aruco.drawDetectedMarkers(
                    thr_bgr, t_markers, borderColor=(0, 180, 180))

            e_corners = e_ids = e_markers = None
            if eo_gray is not None:
                e_corners, e_ids, e_markers = _detect_charuco(
                    eo_gray, board, aruco_dict, charuco_detector)
                if e_corners is not None:
                    cv2.aruco.drawDetectedCornersCharuco(
                        eo_bgr, e_corners, e_ids, (0, 255, 0))
                elif e_markers is not None and len(e_markers) > 0:
                    cv2.aruco.drawDetectedMarkers(
                        eo_bgr, e_markers, borderColor=(0, 180, 180))

            still_thr = state["motion_thr"].update(thr_gray)
            still_eo = (state["motion_eo"].update(eo_gray)
                        if eo_gray is not None else True)

            if state["show_grid"]:
                state["coverage_thr"].overlay(thr_bgr)
                if state["coverage_eo"] is not None:
                    state["coverage_eo"].overlay(eo_bgr)

            thr_hud = [
                f"THERMAL  {thr_bgr.shape[1]}x{thr_bgr.shape[0]}",
                f"corners: {0 if t_corners is None else len(t_corners)}",
                f"motion : {state['motion_thr'].last_diff:5.2f}  "
                f"{'STILL' if still_thr else 'MOVE '}",
            ]
            _draw_hud(thr_bgr, thr_hud)

            if eo_bgr is not None:
                eo_hud = [
                    f"EO  {eo_bgr.shape[1]}x{eo_bgr.shape[0]}",
                    f"corners: {0 if e_corners is None else len(e_corners)}",
                    f"motion : {state['motion_eo'].last_diff:5.2f}  "
                    f"{'STILL' if still_eo else 'MOVE '}",
                ]
                _draw_hud(eo_bgr, eo_hud)

            if eo_bgr is not None:
                target_h = thr_bgr.shape[0]
                scale = target_h / eo_bgr.shape[0]
                eo_disp = cv2.resize(
                    eo_bgr,
                    (int(eo_bgr.shape[1] * scale), target_h),
                    interpolation=cv2.INTER_AREA,
                )
                composite = np.hstack([eo_disp, thr_bgr])
            else:
                composite = thr_bgr

            ready = False
            if args.mode == "thermal":
                ready = (t_corners is not None) and still_thr
            else:
                ready = (t_corners is not None and e_corners is not None
                         and still_thr and still_eo)

            badge = "READY" if ready else "WAIT"
            badge_color = (0, 220, 0) if ready else (0, 120, 220)
            cv2.rectangle(composite, (10, composite.shape[0] - 50),
                          (130, composite.shape[0] - 10), (0, 0, 0), -1)
            cv2.putText(composite, badge,
                        (22, composite.shape[0] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, badge_color, 2,
                        cv2.LINE_AA)

            shots_str = f"shots: {state['shot_idx']}"
            cv2.putText(composite, shots_str,
                        (composite.shape[1] - 160, composite.shape[0] - 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2,
                        cv2.LINE_AA)

            if state["show_help"]:
                _draw_hud(composite,
                          ["SPACE capture   Q quit   G grid   R reset   H help"],
                          origin=(10, 26))

            # Persist last-frame state for capture handler
            state["last_thr_gray"] = thr_gray
            state["last_eo_gray"] = eo_gray
            state["last_t_corners"] = t_corners
            state["last_e_corners"] = e_corners
            state["last_ready"] = ready

            # Tk wants RGB; cv2 produces BGR.
            rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
            # Cap the displayed width so a 2472-px EO doesn't overflow
            # the screen — preserves aspect ratio. Detection uses the
            # full-resolution image; this is display-only.
            max_w = 1600
            if rgb.shape[1] > max_w:
                scale = max_w / rgb.shape[1]
                rgb = cv2.resize(
                    rgb,
                    (max_w, int(rgb.shape[0] * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            pil = Image.fromarray(rgb)
            photo = ImageTk.PhotoImage(pil)
            image_label.configure(image=photo)
            image_label.image = photo  # keep reference

            status_var.set(
                f"mode={args.mode}  shots={state['shot_idx']}  "
                f"thermal_corners={0 if t_corners is None else len(t_corners)}  "
                f"eo_corners={'-' if e_corners is None else len(e_corners)}  "
                f"{'READY' if ready else 'WAIT'}"
            )
        except Exception as e:
            # Rate-limit: a tight 30Hz loop hitting the same error
            # would otherwise dump thousands of identical tracebacks
            # in seconds. Log the first occurrence with full trace,
            # then suppress further identical messages for 5 s.
            now = time.monotonic()
            sig = repr(e)
            last_sig = state.get("_last_err_sig")
            last_t = state.get("_last_err_t", 0.0)
            if sig != last_sig or (now - last_t) > 5.0:
                log.exception("tick failed: %s", e)
                state["_last_err_sig"] = sig
                state["_last_err_t"] = now
                state["_err_count_since"] = 0
            else:
                state["_err_count_since"] = state.get("_err_count_since", 0) + 1
            # Update status bar so the user can see something is wrong
            # even when we suppress the log line.
            try:
                status_var.set(f"ERROR: {e}  (suppressing repeats)")
            except Exception:
                pass

        if state["running"]:
            root.after(33, tick)

    root.after(50, tick)
    try:
        root.mainloop()
    finally:
        state["running"] = False
        try:
            thr.stop()
        except Exception:
            pass
        if eo is not None:
            try:
                eo.stop()
            except Exception:
                pass

    shot_idx = state["shot_idx"]

    # Print next-step command.
    log.info("Captured %d shot(s) → %s", shot_idx, out_dir)
    if args.mode == "thermal":
        log.info("Run intrinsic calibration:")
        log.info("    python -m scripts.calibrate_thermal_intrinsics --images %s",
                 out_dir)
    else:
        log.info("Run stereo calibration (EO intrinsics must already be saved):")
        log.info("    python -m scripts.calibrate_eo_thermal_stereo --captures %s",
                 out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

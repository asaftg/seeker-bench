"""EO ↔ thermal 6-DoF stereo calibration via shared ChArUco target.

Run AFTER both per-camera intrinsics have been written to
``config/calibration.json`` (i.e. EO calibration done by the parallel
"Lens calibration" pipeline; thermal calibration done by
``scripts/calibrate_thermal_intrinsics.py``).

Inputs
------
A directory of *paired* synchronized captures, one subdirectory per
shot. Layout:

    captures/
      shot_001/  eo.png      thermal.png
      shot_002/  eo.png      thermal.png
      …

The board must be **static for ≥200 ms** during each shot (Boson
thermal time constant + IMX568 rolling shutter both demand this).
Hardware sync isn't required — emissivity contrast is steady, no
motion to fit.

Outputs
-------
Writes ``R_to_eo`` + ``t_to_eo`` for the thermal sensor into
``config/calibration.json`` via
``calibration_store.save_extrinsic_6dof("thermal", ...)``. EO is the
reference frame so it has no extrinsic of its own.

Quality gate
------------
Stereo reprojection RMS < 1.5 px is the merge gate. Above that, the
script writes the result anyway and logs a WARN. Re-shoot if the
target wasn't centered across the FOV envelope.

Usage
-----
    python -m scripts.calibrate_eo_thermal_stereo \
        --captures recordings/calib_stereo_2026_05_01/ \
        --squares-x 7 --squares-y 9 --square-mm 30
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import calibration_store  # noqa: E402

log = logging.getLogger("calibrate_eo_thermal_stereo")


def _build_charuco_board(squares_x: int, squares_y: int,
                         square_mm: float, marker_mm: float):
    """Build the ChArUco board + detector (OpenCV 4.7+ API)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sq_m = square_mm * 1e-3
    mk_m = marker_mm * 1e-3
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_m, mk_m, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, aruco_dict, detector


def _detect(gray: np.ndarray, detector):
    ch_corners, ch_ids, _m_corners, _m_ids = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    return ch_corners, ch_ids


def _board_object_points(board, ids: np.ndarray) -> np.ndarray:
    """3D positions in board frame for the given ChArUco corner ids."""
    all_corners = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    return all_corners[ids.reshape(-1)].astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--captures", required=True,
                   help="Directory containing one shot_NNN/ subdir per pair")
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=9)
    p.add_argument("--square-mm", type=float, default=30.0)
    p.add_argument("--marker-mm", type=float, default=22.0)
    p.add_argument("--rms-gate", type=float, default=1.5)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cap_dir = Path(args.captures)
    if not cap_dir.is_dir():
        log.error("Not a directory: %s", cap_dir)
        return 2

    # Pull pre-computed intrinsics from calibration.json (EO from the
    # parallel pipeline, thermal from scripts/calibrate_thermal_intrinsics.py).
    cal = calibration_store.load()
    eo = cal.get("eo") or {}
    thr = cal.get("thermal") or {}
    if not eo.get("K") or not eo.get("dist"):
        log.error("EO intrinsics missing in calibration.json — run the "
                  "EO calibration pipeline first")
        return 2
    if not thr.get("K") or not thr.get("dist"):
        log.error("Thermal intrinsics missing in calibration.json — run "
                  "scripts/calibrate_thermal_intrinsics.py first")
        return 2
    K_eo = np.asarray(eo["K"], dtype=np.float64)
    dist_eo = np.asarray(eo["dist"], dtype=np.float64)
    K_thr = np.asarray(thr["K"], dtype=np.float64)
    dist_thr = np.asarray(thr["dist"], dtype=np.float64)

    board, aruco_dict, detector = _build_charuco_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm
    )

    # Walk shot_NNN directories, detect on both sides, intersect by ID.
    shots = sorted(d for d in cap_dir.iterdir() if d.is_dir())
    if not shots:
        log.error("No shot subdirectories under %s", cap_dir)
        return 2

    obj_points: list[np.ndarray] = []
    eo_image_points: list[np.ndarray] = []
    thr_image_points: list[np.ndarray] = []
    eo_size: tuple[int, int] | None = None
    thr_size: tuple[int, int] | None = None
    used = 0

    for shot in shots:
        eo_path = next((shot / n for n in ("eo.png", "eo.jpg", "eo.tif")
                        if (shot / n).exists()), None)
        thr_path = next((shot / n for n in ("thermal.png", "thermal.jpg", "thermal.tif")
                         if (shot / n).exists()), None)
        if eo_path is None or thr_path is None:
            log.warning("%s: missing eo/thermal file — skipping", shot.name)
            continue

        eo_img = cv2.imread(str(eo_path), cv2.IMREAD_GRAYSCALE)
        thr_img = cv2.imread(str(thr_path), cv2.IMREAD_GRAYSCALE)
        if eo_img is None or thr_img is None:
            log.warning("%s: unreadable image(s) — skipping", shot.name)
            continue

        if eo_size is None:
            eo_size = (eo_img.shape[1], eo_img.shape[0])
        if thr_size is None:
            thr_size = (thr_img.shape[1], thr_img.shape[0])

        eo_corners, eo_ids = _detect(eo_img, detector)
        thr_corners, thr_ids = _detect(thr_img, detector)
        if eo_corners is None or thr_corners is None:
            log.debug("%s: corners missing in one modality", shot.name)
            continue

        # Intersect by ChArUco corner ID — partial detection is fine,
        # we keep only the corners both sensors saw.
        eo_id_set = set(int(i) for i in eo_ids.reshape(-1))
        thr_id_set = set(int(i) for i in thr_ids.reshape(-1))
        common = sorted(eo_id_set & thr_id_set)
        if len(common) < 6:
            log.debug("%s: only %d shared corners — skipping", shot.name, len(common))
            continue
        common_arr = np.array(common, dtype=np.int32)

        eo_idx = {int(i): k for k, i in enumerate(eo_ids.reshape(-1))}
        thr_idx = {int(i): k for k, i in enumerate(thr_ids.reshape(-1))}

        eo_pts = np.array([eo_corners[eo_idx[i]].reshape(2)
                           for i in common], dtype=np.float32)
        thr_pts = np.array([thr_corners[thr_idx[i]].reshape(2)
                            for i in common], dtype=np.float32)
        obj_pts = _board_object_points(board, common_arr)

        obj_points.append(obj_pts)
        eo_image_points.append(eo_pts)
        thr_image_points.append(thr_pts)
        used += 1

    if used < 8:
        log.error("Only %d shared-corner pair(s); need ≥8", used)
        return 2

    log.info("Stereo-calibrating with %d pair(s)…", used)

    flags = cv2.CALIB_FIX_INTRINSIC  # K's are pinned; only solve R, t

    # OpenCV stereoCalibrate: image1=thermal, image2=eo. R, t map a
    # point from image1 frame INTO image2 frame, which is exactly
    # our R_thermal_to_eo, t_thermal_to_eo convention.
    ret = cv2.stereoCalibrate(
        obj_points,
        thr_image_points, eo_image_points,
        K_thr, dist_thr,
        K_eo, dist_eo,
        thr_size,
        flags=flags,
    )
    rms, _Kt, _Dt, _Ke, _De, R, t, *_ = ret

    log.info("Stereo RMS reprojection error: %.3f px", rms)
    log.info("R_thermal_to_eo =\n%s",
             np.array2string(R, precision=4, suppress_small=True))
    log.info("t_thermal_to_eo (m) = %s",
             np.array2string(t.reshape(-1), precision=4, suppress_small=True))

    if rms > args.rms_gate:
        log.warning("RMS %.3f > gate %.3f — stereo extrinsic is questionable",
                    rms, args.rms_gate)

    if args.dry_run:
        log.info("--dry-run: not writing calibration.json")
        return 0

    path = calibration_store.save_extrinsic_6dof(
        "thermal",
        R_to_eo=R.tolist(),
        t_to_eo=t.reshape(-1).tolist(),
        stereo_rms_px=float(rms),
    )
    log.info("Saved → %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

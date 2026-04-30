"""Thermal-camera intrinsic calibration via ChArUco target.

Run AFTER capturing thermal frames of the heated ChArUco target. Each
frame should show the board static (Boson has ~8 ms thermal time
constant — motion smears corners) at varied tilts, distances, and
positions in the FOV. Aim for 25 frames covering all four quadrants
of the image plane.

Inputs
------
A directory of single-channel thermal images (PNG/TIFF/JPG accepted).
The script auto-detects ChArUco corners in each one and runs OpenCV's
ChArUco-aware calibration.

Outputs
-------
Writes K + dist into ``config/calibration.json`` under the
``thermal`` key via ``calibration_store.save_intrinsic("thermal", ...)``.
The runtime picks this up next time ``apply_to_managers`` is called
(e.g. on the next start of the seeker_bench process).

Distortion model
----------------
Boson lenses have visible barrel; we use OpenCV's rational model
(8 distortion params) instead of the standard 5-param. The runtime
``cv2.projectPoints`` / ``cv2.undistortPoints`` calls handle either
length transparently.

Quality gate
------------
Reprojection RMS must be < 1.0 px to be considered usable. Above
that, the script writes the result anyway but logs a WARN — the
operator should recapture before trusting it.

Usage
-----
    python -m scripts.calibrate_thermal_intrinsics \
        --images recordings/calib_thermal_2026_05_01/ \
        --squares-x 7 --squares-y 9 --square-mm 30

The board geometry must match the printed pattern. Defaults assume
the 7×9 / 30mm pattern from the calibration plan.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow running as ``python scripts/calibrate_thermal_intrinsics.py``
# (no -m) by making the project root importable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from common import calibration_store  # noqa: E402

log = logging.getLogger("calibrate_thermal_intrinsics")


def _build_charuco_board(squares_x: int, squares_y: int,
                         square_mm: float, marker_mm: float):
    """Construct an OpenCV ChArUco board + detector (4.7+ API)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    sq_m = square_mm * 1e-3
    mk_m = marker_mm * 1e-3
    board = cv2.aruco.CharucoBoard((squares_x, squares_y), sq_m, mk_m, aruco_dict)
    detector = cv2.aruco.CharucoDetector(board)
    return board, aruco_dict, detector


def _detect_in_image(gray: np.ndarray, detector):
    """Return (charuco_corners, charuco_ids) or (None, None) if not enough."""
    ch_corners, ch_ids, _m_corners, _m_ids = detector.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < 6:
        return None, None
    return ch_corners, ch_ids


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images", required=True,
                   help="Directory of thermal images of the ChArUco board")
    p.add_argument("--squares-x", type=int, default=7,
                   help="Squares along X (default 7)")
    p.add_argument("--squares-y", type=int, default=9,
                   help="Squares along Y (default 9)")
    p.add_argument("--square-mm", type=float, default=30.0,
                   help="Square side length in mm (default 30)")
    p.add_argument("--marker-mm", type=float, default=22.0,
                   help="ArUco marker side length in mm (default 22)")
    p.add_argument("--rms-gate", type=float, default=1.0,
                   help="Reprojection RMS gate in pixels (default 1.0)")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute and report; do not write calibration.json")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    img_dir = Path(args.images)
    if not img_dir.is_dir():
        log.error("Not a directory: %s", img_dir)
        return 2

    board, aruco_dict, detector = _build_charuco_board(
        args.squares_x, args.squares_y, args.square_mm, args.marker_mm
    )

    image_files = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    )
    if not image_files:
        log.error("No images found in %s", img_dir)
        return 2

    # Pre-compute the 3D positions of every ChArUco corner on the board
    # in board frame (Z=0 plane). cv2.calibrateCamera takes one Nx3
    # array of object points per image; we slice this by detected ids.
    all_obj_pts = np.asarray(board.getChessboardCorners(), dtype=np.float32)

    log.info("Detecting ChArUco corners in %d image(s)…", len(image_files))
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None
    used = 0
    for f in image_files:
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            log.warning("Could not read %s", f.name)
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])  # (w, h)
        elif (img.shape[1], img.shape[0]) != img_size:
            log.warning("Image size mismatch in %s — skipping", f.name)
            continue
        corners, ids = _detect_in_image(img, detector)
        if corners is None:
            log.debug("  %s: no usable corners", f.name)
            continue
        ids_flat = ids.reshape(-1).astype(int)
        obj_points.append(all_obj_pts[ids_flat].astype(np.float32))
        img_points.append(corners.reshape(-1, 2).astype(np.float32))
        used += 1

    if used < 8:
        log.error("Only %d frame(s) yielded usable ChArUco corners; need ≥8", used)
        return 2

    log.info("Calibrating with %d frame(s) at %dx%d…", used, *img_size)

    # Rational model handles Boson barrel cleanly; standard 5-param
    # leaves ~1px residual at the FOV edges in our experience.
    # Note: cv2.aruco.calibrateCameraCharuco was removed in OpenCV
    # 4.7+; we now feed (obj_pts, img_pts) directly into the generic
    # cv2.calibrateCamera, which gives identical results (the legacy
    # function was a thin wrapper around it).
    flags = cv2.CALIB_RATIONAL_MODEL

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None, flags=flags
    )

    log.info("Calibration RMS reprojection error: %.3f px", rms)
    log.info("K =\n%s", np.array2string(K, precision=3, suppress_small=True))
    log.info("dist = %s",
             np.array2string(dist.reshape(-1), precision=4, suppress_small=True))

    if rms > args.rms_gate:
        log.warning("RMS %.3f > gate %.3f — calibration is questionable; "
                    "recapture recommended", rms, args.rms_gate)

    if args.dry_run:
        log.info("--dry-run: not writing calibration.json")
        return 0

    path = calibration_store.save_intrinsic(
        "thermal",
        K=K.tolist(),
        dist=dist.reshape(-1).tolist(),
        reproj_rms_px=float(rms),
    )
    log.info("Saved → %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
PROOF TOOL — Phase 2 calibration.

Reads the captured Y16 + AGC8 frames from `_y16_vs_agc8_proof.py`'s
.npz output, runs the production HeatDetector against both, sweeps
`threshold_k` on the Y16 stream, and reports the value that makes
Y16 detection rate match the AGC8 baseline. Outputs a ready-to-paste
YAML snippet.

This is the GATE that prevents the "heat blob flood" we saw on the
first Y16 attempt. If the calibrated threshold_k diverges wildly from
the legacy 20.0, the report says so and recommends NOT applying
Patch 2 until the operator inspects the Y16 frame statistics.

Usage:
    # First, capture frames (sensor must be free):
    python scripts/_y16_vs_agc8_proof.py --hfov 37.5
    # Then calibrate (purely offline):
    python scripts/_phase2_calibrate.py
    python scripts/_phase2_calibrate.py --npz <path>
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from common.config import load_config  # noqa: E402
from thermal.heat_detector import HeatDetector, HeatDetectorConfig  # noqa: E402

DEFAULT_NPZ_DIR = os.path.join("recordings", "thermal_compare")


def _newest_proof_npz() -> str:
    pattern = os.path.join(DEFAULT_NPZ_DIR, "y16_vs_agc8_*.npz")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not matches:
        raise FileNotFoundError(
            f"no y16_vs_agc8_*.npz found in {DEFAULT_NPZ_DIR}. "
            f"run scripts/_y16_vs_agc8_proof.py first."
        )
    return matches[-1]


def _build_detector(cfg_dict: dict, threshold_k: float) -> HeatDetector:
    """Build a HeatDetector from the live config, with threshold_k overridden."""
    return HeatDetector(HeatDetectorConfig(
        threshold_k=float(threshold_k),
        background_kernel=int(cfg_dict.get("background_kernel", 31)),
        min_blob_area_px=int(cfg_dict.get("min_blob_area_px", 500)),
        max_blob_area_px=int(cfg_dict.get("max_blob_area_px", 30000)),
        max_detections=int(cfg_dict.get("max_detections_per_frame", 6)),
        algorithm=str(cfg_dict.get("algorithm", "tophat")),
        tophat_kernel=int(cfg_dict.get("tophat_kernel", 31)),
    ))


def _detect_counts(detector: HeatDetector, frames_u16: np.ndarray) -> List[int]:
    """Return detection count per frame."""
    return [len(detector.detect(f)) for f in frames_u16]


def _agc8_to_u16(agc8_bgr: np.ndarray) -> np.ndarray:
    """Match thermal_manager's AGC8 fallback path: BGR -> gray -> uint16."""
    gray = cv2.cvtColor(agc8_bgr, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.uint16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npz", type=str, default=None,
                    help="Path to the .npz from _y16_vs_agc8_proof.py "
                         "(default: newest in recordings/thermal_compare/)")
    ap.add_argument("--k-min", type=float, default=20.0)
    ap.add_argument("--k-max", type=float, default=400.0)
    ap.add_argument("--k-step", type=float, default=5.0)
    ap.add_argument("--tolerance", type=float, default=1.0,
                    help="Max allowed mean-detection-count delta vs AGC8 baseline")
    args = ap.parse_args(argv)

    npz_path = args.npz or _newest_proof_npz()
    print(f"[calib] loading {npz_path}")
    data = np.load(npz_path)
    y16_frames = data["y16"]
    agc8_frames = data["agc8"]
    hfov = float(data["hfov_deg"])
    print(f"[calib] {y16_frames.shape[0]} Y16 frames {y16_frames.shape[1:]}, "
          f"hfov_deg={hfov}")
    print(f"[calib] {agc8_frames.shape[0]} AGC8 frames {agc8_frames.shape[1:]}")

    cfg_full = load_config()
    hd_cfg = cfg_full.get("heat_detector", {})
    legacy_k = float(hd_cfg.get("threshold_k", 20.0))
    print(f"[calib] live YAML threshold_k = {legacy_k}, "
          f"min_blob_area_px = {hd_cfg.get('min_blob_area_px')}")

    # ── 1) AGC8 baseline (live behavior today) ────────────────────
    det_legacy = _build_detector(hd_cfg, legacy_k)
    agc8_u16 = np.stack([_agc8_to_u16(f) for f in agc8_frames], axis=0)
    agc8_counts = _detect_counts(det_legacy, agc8_u16)
    agc8_mean = float(np.mean(agc8_counts))
    agc8_max = int(np.max(agc8_counts))
    print(f"[calib] AGC8 baseline:  mean detections/frame = {agc8_mean:.2f}  "
          f"max = {agc8_max}")

    # ── 2) Y16 at legacy threshold_k=20 (the prior failure mode) ──
    y16_legacy_counts = _detect_counts(det_legacy, y16_frames)
    y16_legacy_mean = float(np.mean(y16_legacy_counts))
    y16_legacy_max = int(np.max(y16_legacy_counts))
    print(f"[calib] Y16 @ k={legacy_k}: mean detections/frame = "
          f"{y16_legacy_mean:.2f}  max = {y16_legacy_max}  "
          f"(prior 'flood' state: {'YES' if y16_legacy_mean > agc8_mean + 1 else 'no'})")

    # ── 3) Sweep threshold_k on Y16 to find a match ───────────────
    print()
    print(f"[calib] sweeping threshold_k = {args.k_min} .. {args.k_max} step {args.k_step}")
    print(f"[calib] target: Y16 mean detections within ±{args.tolerance} "
          f"of AGC8 baseline {agc8_mean:.2f}")
    print()
    print(f"  {'threshold_k':>11}  {'Y16 mean':>9}  {'Y16 max':>7}  {'delta':>7}")

    chosen_k = None
    chosen_mean = None
    sweep_results: List[Tuple[float, float, int]] = []
    k = args.k_min
    while k <= args.k_max + 1e-6:
        det = _build_detector(hd_cfg, k)
        counts = _detect_counts(det, y16_frames)
        m = float(np.mean(counts))
        mx = int(np.max(counts))
        delta = m - agc8_mean
        sweep_results.append((k, m, mx))
        marker = ""
        if chosen_k is None and abs(delta) <= args.tolerance:
            chosen_k = k
            chosen_mean = m
            marker = "  <- FIRST MATCH"
        print(f"  {k:>11.1f}  {m:>9.2f}  {mx:>7d}  {delta:>+7.2f}{marker}")
        k += args.k_step

    print()
    if chosen_k is None:
        print(f"[calib] NO threshold_k in [{args.k_min}, {args.k_max}] produced "
              f"a Y16 detection rate within ±{args.tolerance} of AGC8 baseline "
              f"({agc8_mean:.2f}).", file=sys.stderr)
        print("[calib] inspect the sweep table above. If the Y16 detection rate "
              "stays high at large k, the scene has more real warm structure "
              "than AGC8 was rendering — consider raising min_blob_area_px or "
              "max_detections_per_frame.", file=sys.stderr)
        return 2

    # ── 4) Write the YAML recommendation ──────────────────────────
    print(f"[calib] *** RECOMMENDED threshold_k = {chosen_k:.1f} ***")
    print(f"[calib] (raises Y16 detection rate from {y16_legacy_mean:.2f} to "
          f"{chosen_mean:.2f}, vs AGC8 baseline {agc8_mean:.2f})")
    print()
    print("=" * 64)
    print("Apply to config/app_config.yaml::heat_detector:")
    print("=" * 64)
    print(f"  threshold_k: {chosen_k:.1f}             # Y16-calibrated "
          f"({chosen_mean:.2f} det/frame matches AGC8 baseline {agc8_mean:.2f})")
    print(f"  # min_blob_area_px: keep at {hd_cfg.get('min_blob_area_px', 500)}")
    print(f"  # other heat_detector knobs: unchanged")
    print("=" * 64)
    print()
    print("Then apply Patch 2 (Y16 enable in boson_capture.py — see docs/handoffs/2026-04-27_proposed_y16_roi_changes.md).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""CLI for the EO auto-calibrator.

Run:

    python -m eo.auto_calibrate \
        --duration 1200 \
        --target-mean 100 \
        --reference scripts/eo_snapshots/calibration/leopard_reference.bmp \
        --resume

What it does, end-to-end:
    1. Asserts the device is reachable AND that ExposureExt writes are
       actually moving the captured mean (this catches "CameraTool was
       left open" silently — the optimizer won't waste your night
       searching a frozen device).
    2. Loads ``state.json`` from a previous run if --resume is set, and
       seeds the differential-evolution population around the best
       previous params.
    3. Runs the optimizer up to ``duration_s`` or until plateau.
    4. Writes ``best.png`` snapshot continuously and final ``state.json``
       with the winning parameter vector.

Output files (under ``scripts/eo_snapshots/calibration/``):
    state.json          — current best params, history, weights, target
    snapshots/best.png  — best processed frame (updated on each
                          improvement)

A subsequent process (eo_processor) can read state.json and apply the
winning params at runtime — no rerun needed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from common.logging_setup import get_logger
from eo.auto_calibrate.metrics import CostWeights
from eo.auto_calibrate.optimizer import (
    FrameSource,
    STATE_PATH,
    SNAPSHOTS_DIR,
    load_state,
    run_optimizer,
)
from eo.auto_calibrate.pipeline import CalibParams

log = get_logger("eo.auto_calibrate")


# ───────────────────────────── self-test ──────────────────────────────

def _self_test_writes_take_effect(duration_budget_s: float = 30.0) -> bool:
    """Prove the SDK helper is actually moving the sensor.

    Procedure:
        1. Drive ExposureExt = 200 (very short — should produce a dark
           frame, mean << 80).
        2. Drive ExposureExt = 5000 (long — should be much brighter,
           mean >> drive #1).
        3. If the mean delta is < 10 gray levels we are NOT controlling
           the sensor (most common cause: CameraTool / VLC / Windows
           camera-properties dialog is open and holding the device).

    Returns True on pass. On fail, prints actionable instructions and
    returns False.
    """
    log.info("self-test: verifying ExposureExt writes change captured mean…")
    src = FrameSource(n_grab_per_eval=4)
    try:
        src.ensure_exposure(200)
        f1 = src.grab_median()
        m1 = float(f1.mean()) if f1 is not None else float("nan")
        src.ensure_exposure(5000)
        f2 = src.grab_median()
        m2 = float(f2.mean()) if f2 is not None else float("nan")
    finally:
        src.close()
    log.info("self-test: mean@ExposureExt=200 -> %.2f", m1)
    log.info("self-test: mean@ExposureExt=5000 -> %.2f", m2)
    delta = abs(m2 - m1)
    if delta < 5.0:
        log.error("self-test FAILED: mean delta %.2f is too small. "
                  "Something else is holding the camera.", delta)
        sys.stderr.write(
            "\n*** AUTO-CALIBRATOR SELF-TEST FAILED ***\n"
            "Setting ExposureExt 200 vs 5000 produced almost the same\n"
            "captured mean (delta=%.2f). That means our SDK writes are\n"
            "not reaching the sensor — most likely cause is that some\n"
            "other app has the camera open.\n\n"
            "Action:\n"
            "  1. Close Leopard CameraTool COMPLETELY (check Task\n"
            "     Manager for CameraTool.exe).\n"
            "  2. Close VLC, Windows Camera, OBS, Skype, Teams, or any\n"
            "     browser tab using the camera.\n"
            "  3. Close any 'LI-IMX568 Properties' DirectShow dialog.\n"
            "  4. Re-run.\n\n" % delta
        )
        return False
    log.info("self-test PASSED: mean delta %.2f → sensor under control.",
             delta)
    return True


# ─────────────────────────── reference loader ─────────────────────────

def _load_reference(path: Path) -> np.ndarray | None:
    if not path.exists():
        log.warning("reference path does not exist: %s", path)
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try as a 16-bit raw (Leopard's .raw — 2472x2064 mono16)
        try:
            data = np.fromfile(str(path), dtype=np.uint16)
            if data.size == 2472 * 2064:
                img16 = data.reshape(2064, 2472)
                # AGC-normalize for use as a comparison target
                p_lo, p_hi = np.percentile(img16, [0.5, 99.5])
                img = np.clip((img16 - p_lo) / max(p_hi - p_lo, 1) * 255.0,
                              0, 255).astype(np.uint8)
                log.info("reference loaded from raw16: shape=%s", img.shape)
                return img
        except Exception as e:
            log.warning("could not parse reference as raw16: %r", e)
        return None
    return img


# ─────────────────────────────── main ─────────────────────────────────

def main(argv=None) -> int:
    p = argparse.ArgumentParser("eo.auto_calibrate")
    p.add_argument("--duration", type=float, default=1200.0,
                   help="Wall-clock seconds before the optimizer stops "
                        "(also stops on plateau)")
    p.add_argument("--target-mean", type=float, default=100.0,
                   help="Desired output mean (0..255). 100 ≈ comfortable "
                        "midtone for a daylight indoor scene; the "
                        "Leopard CameraTool default produces ~16 "
                        "(very dim). Push higher than Leopard.")
    p.add_argument("--reference", type=Path, default=None,
                   help="Optional reference image (Leopard CameraTool "
                        "BMP or .raw16). Activates histogram-match and "
                        "pixel-match cost terms.")
    p.add_argument("--resume", action="store_true",
                   help="Seed the optimizer with best_params from the "
                        "previous state.json if present.")
    p.add_argument("--pop-size", type=int, default=12)
    p.add_argument("--plateau-iters", type=int, default=50,
                   help="Stop after this many iterations without "
                        "improvement of more than --plateau-eps.")
    p.add_argument("--plateau-eps", type=float, default=1e-3)
    p.add_argument("--skip-self-test", action="store_true",
                   help="Skip the 'writes actually move the sensor' "
                        "self-test. Use only if you are certain the "
                        "sensor is under SDK control.")
    p.add_argument("--w-target", type=float, default=1.0)
    p.add_argument("--w-saturation", type=float, default=5.0)
    p.add_argument("--w-blackclip", type=float, default=5.0)
    p.add_argument("--w-entropy", type=float, default=1.0)
    p.add_argument("--w-sharpness", type=float, default=1.5)
    p.add_argument("--w-snr", type=float, default=2.0)
    p.add_argument("--w-histmatch", type=float, default=0.0,
                   help="Auto-set to 2.0 if --reference is provided.")
    p.add_argument("--w-pixmatch", type=float, default=0.0,
                   help="Auto-set to 1.0 if --reference is provided "
                        "(weaker than histogram match — pixel match is "
                        "fragile to FOV / parallax differences).")
    args = p.parse_args(argv)

    # Sanity-check: the SDK helper actually controls the sensor.
    if not args.skip_self_test:
        if not _self_test_writes_take_effect():
            return 2

    # Compose weights — promote ref-match terms automatically when ref
    # is provided.
    weights = CostWeights(
        target_mean=args.w_target,
        saturation=args.w_saturation,
        black_clip=args.w_blackclip,
        histogram_entropy=args.w_entropy,
        sharpness=args.w_sharpness,
        snr=args.w_snr,
        histogram_match=args.w_histmatch,
        pixel_match=args.w_pixmatch,
    )
    ref_path = None
    if args.reference is not None and args.reference.exists():
        ref_path = args.reference
        if weights.histogram_match == 0.0:
            weights.histogram_match = 2.0
        if weights.pixel_match == 0.0:
            weights.pixel_match = 1.0
        log.info("using reference image: %s", ref_path)
        # Verify it loads cleanly as one of the supported formats.
        ref_img = _load_reference(ref_path)
        if ref_img is None:
            log.warning("reference at %s could not be read — "
                        "ref-match terms will be inactive.", ref_path)
            ref_path = None
            weights.histogram_match = 0.0
            weights.pixel_match = 0.0

    # Resume seeding.
    seed: CalibParams | None = None
    if args.resume:
        prior = load_state()
        if prior is not None and prior.best_params:
            try:
                seed = CalibParams(**prior.best_params)
                log.info("resuming from previous best (cost=%.4f, "
                         "iter=%d): %s", prior.best_cost, prior.iter,
                         prior.best_params)
            except Exception as e:
                log.warning("could not seed from prior state: %r", e)

    log.info("starting optimizer: duration=%ds target_mean=%.1f "
             "ref=%s pop_size=%d plateau_iters=%d",
             int(args.duration), args.target_mean,
             "yes" if ref_path else "no", args.pop_size, args.plateau_iters)

    t0 = time.time()
    final = run_optimizer(
        target_mean=args.target_mean,
        duration_s=args.duration,
        weights=weights,
        reference_path=ref_path,
        seed_params=seed,
        pop_size=args.pop_size,
        plateau_iters=args.plateau_iters,
        plateau_eps=args.plateau_eps,
    )
    elapsed = time.time() - t0
    log.info("───────── DONE ─────────")
    log.info("iterations: %d  elapsed: %.1fs  best_cost: %.4f",
             final.iter, elapsed, final.best_cost)
    log.info("best params: %s", final.best_params)
    log.info("breakdown:   %s", final.last_breakdown)
    log.info("state file:  %s", STATE_PATH)
    log.info("best image:  %s", SNAPSHOTS_DIR / "best.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

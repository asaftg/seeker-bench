"""Differential-evolution optimizer for the EO post-processing pipeline.

Two-tier search:

  Tier 1 — sensor + pipeline coupled. Each candidate vector includes
  ``exposure_ext``. When the candidate's ``exposure_ext`` differs from
  the current sensor state, we spawn the 32-bit Leopard SDK helper to
  apply it. Frame capture cost dominates iteration time.

  Tier 2 — pipeline-only. Once the optimizer converges on an
  ``exposure_ext``, we lock it and re-optimize only the post-processing
  knobs at much higher iteration speed (no helper spawn, no PyAV
  reopen, just process-the-cached-raw-frame).

Resumability:

  After every cost evaluation the optimizer writes
  ``scripts/eo_snapshots/calibration/state.json`` with:

      best_params, best_cost, last_breakdown, iter, history

  A new session can resume by reading that file and seeding the
  optimizer's initial population from ``best_params``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common.logging_setup import get_logger
from eo.imx568_capture import IMX568Capture
from eo.auto_calibrate.metrics import (
    CostBreakdown,
    CostWeights,
    composite_cost,
    precompute_ref_hist,
)
from eo.auto_calibrate.pipeline import (
    PARAM_BOUNDS,
    CalibParams,
    apply_pipeline,
)

log = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
PY32 = REPO / "tools" / "python311-x86" / "python.exe"
STATE_DIR = REPO / "scripts" / "eo_snapshots" / "calibration"
STATE_DIR.mkdir(parents=True, exist_ok=True)
STATE_PATH = STATE_DIR / "state.json"
SNAPSHOTS_DIR = STATE_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────── sensor control ──────────────────────────

def _apply_exposure_ext(value: int) -> dict:
    """Spawn the 32-bit helper to set ExposureExt + AE off. Blocking,
    ~0.7s per call. Don't call this inside the optimizer hot path
    unless ``value`` actually changed.
    """
    cmd = [str(PY32), str(HELPER),
           "--exposure-ext", str(int(value)),
           "--ae", "off",
           "--json"]
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return json.loads(cp.stdout) if cp.stdout else {}
    except Exception as e:
        return {"err": repr(e)}


# ───────────────────────── frame acquisition ─────────────────────────

class FrameSource:
    """Holds an open ``IMX568Capture`` so we don't pay reopen cost per
    candidate. Re-opens on demand when ``exposure_ext`` changes (the
    helper has to take exclusive access of the device, which forces us
    to release the PyAV handle first).
    """

    def __init__(self, n_grab_per_eval: int = 8) -> None:
        self.n_grab = n_grab_per_eval
        self._cap: Optional[IMX568Capture] = None
        self._current_exp_ext: Optional[int] = None

    def ensure_exposure(self, exp_ext: int) -> dict:
        if exp_ext == self._current_exp_ext and self._cap is not None:
            return {"unchanged": True}
        # Close PyAV first so the helper can claim the device.
        if self._cap is not None:
            try:
                self._cap.stop()
            except Exception:
                pass
            self._cap = None
        rep = _apply_exposure_ext(exp_ext)
        time.sleep(0.4)
        self._cap = IMX568Capture(device_index="auto")
        self._cap.start()
        self._current_exp_ext = exp_ext
        # warm up — first frame post-reopen is sometimes the previous
        # AE setting still being clocked out
        for _ in range(3):
            self._cap.grab()
        return rep

    def grab_median(self) -> Optional[np.ndarray]:
        """Capture N frames, return per-pixel median as a single uint8 mono.
        Median across frames removes per-frame noise without blurring
        spatial detail (better than mean — survives outlier frames).
        """
        if self._cap is None:
            return None
        frames = []
        for _ in range(self.n_grab):
            f = self._cap.grab()
            if f is None:
                continue
            y = f[..., 0] if f.ndim == 3 else f
            frames.append(y)
        if not frames:
            return None
        stack = np.stack(frames, axis=0)
        return np.median(stack, axis=0).astype(np.uint8)

    def close(self) -> None:
        if self._cap is not None:
            try:
                self._cap.stop()
            except Exception:
                pass
        self._cap = None


# ───────────────────────────── state I/O ─────────────────────────────

@dataclass
class OptimState:
    best_cost: float
    best_params: dict
    last_breakdown: dict
    iter: int
    history: list
    started_at: str
    last_update: str
    target_mean: float
    weights: dict


def save_state(state: OptimState) -> None:
    STATE_PATH.write_text(json.dumps(asdict(state), indent=2),
                          encoding="utf-8")


def load_state() -> Optional[OptimState]:
    if not STATE_PATH.exists():
        return None
    try:
        d = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        return OptimState(**d)
    except Exception as e:
        log.warning("could not load state: %r", e)
        return None


# ─────────────────────────── main loop ───────────────────────────────

def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def run_optimizer(
    target_mean: float = 100.0,
    duration_s: float = 1200.0,
    weights: Optional[CostWeights] = None,
    reference_path: Optional[Path] = None,
    seed_params: Optional[CalibParams] = None,
    pop_size: int = 12,
    plateau_iters: int = 50,
    plateau_eps: float = 1e-3,
) -> OptimState:
    """Run the optimizer up to ``duration_s`` or until plateau.

    Plateau: best_cost has not improved by more than ``plateau_eps``
    across ``plateau_iters`` consecutive evaluations. That's the
    "perfect" stopping rule.
    """
    weights = weights or CostWeights()

    ref_img = None
    ref_hist = None
    if reference_path is not None and reference_path.exists():
        ref = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
        if ref is not None:
            ref_img = ref
            ref_hist = precompute_ref_hist(ref)
            # Activate reference-match terms if they're at default 0.
            if weights.histogram_match == 0.0:
                weights.histogram_match = 2.0
            if weights.pixel_match == 0.0:
                weights.pixel_match = 2.0
            log.info("reference image loaded from %s — shape=%s mean=%.1f",
                     reference_path, ref.shape, ref.mean())

    src = FrameSource(n_grab_per_eval=8)
    started = _iso_now()
    history: list = []
    best_cost = float("inf")
    best_params: Optional[CalibParams] = seed_params
    best_breakdown: Optional[CostBreakdown] = None
    n_iter = 0
    last_improve_iter = 0
    t0 = time.time()
    last_exp_ext_for_pipeline_cache: Optional[int] = None
    cached_raw: Optional[np.ndarray] = None

    def evaluate(params: CalibParams) -> tuple[float, CostBreakdown, np.ndarray]:
        """Apply sensor knob if changed, capture, run pipeline, score."""
        nonlocal last_exp_ext_for_pipeline_cache, cached_raw
        # Sensor side
        if params.exposure_ext != last_exp_ext_for_pipeline_cache:
            src.ensure_exposure(int(params.exposure_ext))
            cached_raw = src.grab_median()
            last_exp_ext_for_pipeline_cache = int(params.exposure_ext)
        if cached_raw is None:
            return float("inf"), CostBreakdown(
                total=float("inf"), target_mean=0, saturation=0,
                black_clip=0, histogram_entropy=0, sharpness=0,
                snr=0, histogram_match=0, pixel_match=0), np.zeros((1, 1), dtype=np.uint8)
        processed = apply_pipeline(cached_raw, params)
        bd = composite_cost(processed, weights, target_mean,
                            ref_img=ref_img, ref_hist=ref_hist)
        return bd.total, bd, processed

    # Seed initial population. SciPy's DE samples uniformly from bounds;
    # we pre-bias by setting init='sobol' for better space-coverage.
    from scipy.optimize import differential_evolution

    def cost_fn(v: np.ndarray) -> float:
        nonlocal n_iter, best_cost, best_params, best_breakdown, last_improve_iter
        # Clamp + sanitize: agc_high_pct must be > agc_low_pct + 1
        params = CalibParams.from_vector(v)
        if params.agc_high_pct <= params.agc_low_pct + 1.0:
            params.agc_high_pct = params.agc_low_pct + 1.0
        try:
            cost, bd, processed = evaluate(params)
        except Exception as e:
            log.warning("evaluate failed: %r", e)
            return float("inf")
        n_iter += 1
        improved = cost < best_cost - plateau_eps
        if improved:
            best_cost = cost
            best_params = params
            best_breakdown = bd
            last_improve_iter = n_iter
            # save best snapshot
            cv2.imwrite(str(SNAPSHOTS_DIR / "best.png"), processed)
        history.append({
            "iter": n_iter,
            "cost": round(cost, 4),
            "params": params.to_dict(),
            "breakdown": asdict(bd),
            "improved": bool(improved),
            "elapsed_s": round(time.time() - t0, 1),
        })
        # Persist on every eval — resume safety
        save_state(OptimState(
            best_cost=best_cost,
            best_params=(best_params.to_dict() if best_params else {}),
            last_breakdown=(asdict(best_breakdown) if best_breakdown else {}),
            iter=n_iter,
            history=history[-200:],  # cap log size
            started_at=started,
            last_update=_iso_now(),
            target_mean=float(target_mean),
            weights=asdict(weights),
        ))
        if n_iter % 5 == 0 or improved:
            log.info(
                "iter=%4d cost=%.4f%s exp_ext=%4d  agc=[%.1f,%.1f] gamma=%.2f "
                "nlm=%.1f bilat=%d shp=%.2f clahe=%.1f  "
                "[mean=%.3f sat=%.3f blk=%.3f ent=%.3f shp=%.3f snr=%.3f]",
                n_iter, cost, " ★" if improved else "  ",
                params.exposure_ext, params.agc_low_pct, params.agc_high_pct,
                params.gamma, params.nlm_h, params.bilateral_d,
                params.sharpen_amount, params.clahe_clip,
                bd.target_mean, bd.saturation, bd.black_clip,
                bd.histogram_entropy, bd.sharpness, bd.snr,
            )
        # Stopping conditions checked via SciPy callback below.
        return cost

    def stop_callback(xk, convergence):  # noqa: ARG001
        if time.time() - t0 > duration_s:
            log.info("stopping: duration limit reached")
            return True
        if n_iter - last_improve_iter > plateau_iters:
            log.info("stopping: plateau (%d iters without improvement)",
                     n_iter - last_improve_iter)
            return True
        return False

    init = "sobol"
    if seed_params is not None:
        # Convert seed to a small population around it.
        seed_v = seed_params.to_vector()
        init_pop = np.tile(seed_v, (max(pop_size, 5), 1))
        rng = np.random.default_rng(0)
        for i in range(1, init_pop.shape[0]):
            init_pop[i] += rng.normal(scale=0.02, size=seed_v.size) * np.array(
                [b[1] - b[0] for b in PARAM_BOUNDS])
        init = init_pop

    try:
        differential_evolution(
            cost_fn,
            bounds=PARAM_BOUNDS,
            maxiter=10_000,
            popsize=pop_size,
            mutation=(0.4, 1.2),
            recombination=0.8,
            tol=1e-4,
            init=init,
            polish=False,
            updating="deferred",
            callback=stop_callback,
            workers=1,
        )
    except Exception as e:
        log.error("differential_evolution raised: %r", e)
    finally:
        src.close()

    return OptimState(
        best_cost=best_cost,
        best_params=(best_params.to_dict() if best_params else {}),
        last_breakdown=(asdict(best_breakdown) if best_breakdown else {}),
        iter=n_iter,
        history=history[-200:],
        started_at=started,
        last_update=_iso_now(),
        target_mean=float(target_mean),
        weights=asdict(weights),
    )

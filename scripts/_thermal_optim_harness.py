"""
Thermal optimization harness — autonomous parameter sweep.

Captures Y16 + AGC8 frames from the Boson, then sweeps the software
pipeline (AGC mode, post-AGC enhancement) over a config grid, scores
each via ``_thermal_metrics``, and writes a ranked leaderboard +
per-config previews to ``recordings/optim/<session>/``.

Tier A (software-only) runs against a single captured Y16 stack —
the camera config doesn't matter because we're stretching the raw
16-bit data ourselves. Tier B (camera SDK) re-captures for each
camera config; gated behind ``--sdk`` and only runs if BosonControl
can open a serial port.

Auto-resume: writes ``STATE.json`` after every config so a session
restart picks up where we left off. ``STOP`` sentinel file
short-circuits the loop cleanly.

Usage::

    python scripts/_thermal_optim_harness.py --pose pose1
    python scripts/_thermal_optim_harness.py --pose pose1 --resume
    python scripts/_thermal_optim_harness.py --pose pose1 --sdk
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from typing import Iterable, List, Optional

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# These imports are resolved relative to the seeker repo
from thermal.thermal_processor import (  # noqa: E402
    ThermalEnhanceParams,
    apply_agc, apply_agc_mode, apply_bilateral_denoise, apply_clahe_y16,
    apply_colormap, apply_dead_pixel_median, apply_gamma, apply_gates_agc,
    apply_roi_agc, apply_unsharp_mask, enhance_post_agc,
)
import importlib.util  # noqa: E402
_metrics_path = os.path.join(_REPO_ROOT, "scripts", "_thermal_metrics.py")
_spec = importlib.util.spec_from_file_location("_thermal_metrics", _metrics_path)
_tm = importlib.util.module_from_spec(_spec)
sys.modules["_thermal_metrics"] = _tm  # dataclass needs this BEFORE exec_module
_spec.loader.exec_module(_tm)  # type: ignore

OUT_BASE = os.path.join("recordings", "optim")


# ───────────────────────────────────────────────────────────────
# Config grid
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PipelineConfig:
    """One point in the software-pipeline parameter space."""
    name: str
    mode: str = "global"
    low_percentile: float = 2.0
    high_percentile: float = 98.0
    roi_top_frac: float = 0.4
    cold_count: int = 0
    hot_count: int = 65535
    clahe_y16_clip_limit: float = 2.0
    clahe_y16_tile_grid: int = 8
    # post-AGC chain
    dead_pixel_median: bool = False
    gamma: float = 1.0
    bilateral_enabled: bool = False
    bilateral_d: int = 3
    bilateral_sigma: float = 10.0
    unsharp_enabled: bool = False
    unsharp_amount: float = 0.0
    unsharp_radius: float = 1.0
    # display
    colormap: str = "WHITE_HOT"

    def to_params(self) -> ThermalEnhanceParams:
        return ThermalEnhanceParams(
            mode=self.mode,
            low_percentile=self.low_percentile,
            high_percentile=self.high_percentile,
            roi_top_frac=self.roi_top_frac,
            cold_count=self.cold_count,
            hot_count=self.hot_count,
            clahe_y16_clip_limit=self.clahe_y16_clip_limit,
            clahe_y16_tile_grid=self.clahe_y16_tile_grid,
            colormap=self.colormap,
            gamma=self.gamma,
            bilateral_enabled=self.bilateral_enabled,
            bilateral_d=self.bilateral_d,
            bilateral_sigma_color=self.bilateral_sigma,
            bilateral_sigma_space=self.bilateral_sigma,
            unsharp_enabled=self.unsharp_enabled,
            unsharp_amount=self.unsharp_amount,
            unsharp_radius=self.unsharp_radius,
            dead_pixel_median_enabled=self.dead_pixel_median,
        )


def tier1_baselines() -> List[PipelineConfig]:
    """Tier-1: the four core AGC modes at sensible defaults.

    Each is a candidate "winner" by itself. Subsequent tiers explore
    parameter variants of the best one.
    """
    return [
        PipelineConfig("agc_global_2_98",
                       mode="global", low_percentile=2.0, high_percentile=98.0),
        PipelineConfig("agc_roi_60",
                       mode="roi", roi_top_frac=0.4),
        PipelineConfig("agc_gates_p10_p95",
                       mode="gates", cold_count=19089, hot_count=22273),
        PipelineConfig("agc_clahe_y16_2_8",
                       mode="clahe_y16",
                       clahe_y16_clip_limit=2.0, clahe_y16_tile_grid=8),
    ]


def tier2_clahe_y16_sweep() -> List[PipelineConfig]:
    """Tier-2: CLAHE-Y16 parameter sweep (only if T1 winner is clahe_y16)."""
    out = []
    for clip in [1.0, 2.0, 3.0, 4.0, 6.0]:
        for tile in [4, 8, 16]:
            out.append(PipelineConfig(
                f"clahe_y16_clip{clip}_tile{tile}",
                mode="clahe_y16",
                clahe_y16_clip_limit=clip, clahe_y16_tile_grid=tile,
            ))
    return out


def tier3_post_enhance_sweep(base: PipelineConfig) -> List[PipelineConfig]:
    """Tier-3: post-AGC enhancement chain on top of the T2 winner."""
    out = []
    for gamma in [1.0, 0.85, 1.15]:
        for sharp in [0.0, 0.3, 0.6]:
            for denoise in [False, True]:
                cfg = dataclasses.replace(
                    base,
                    name=f"{base.name}_g{gamma}_us{sharp}_dn{int(denoise)}",
                    gamma=gamma,
                    unsharp_enabled=sharp > 0,
                    unsharp_amount=sharp,
                    bilateral_enabled=denoise,
                )
                out.append(cfg)
    return out


# ───────────────────────────────────────────────────────────────
# Capture (Y16 + AGC8)
# ───────────────────────────────────────────────────────────────

def _open_boson_y16(idx_hint: int = 1):
    for idx in [idx_hint, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        ok, test = cap.read()
        if ok and test is not None and test.ndim == 2 and test.shape == (512, 640):
            return cap, idx
        cap.release()
    return None, None


def _open_boson_agc8(idx_hint: int = 1):
    for idx in [idx_hint, 0, 1, 2, 3]:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', '1', '6', ' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 512)
        ok, test = cap.read()
        if ok and test is not None and test.ndim == 3 and test.shape == (512, 640, 3):
            return cap, idx
        cap.release()
    return None, None


def capture_stacks(n_frames: int = 60, warmup: int = 15
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Capture (y16_stack, agc8_stack) from the Boson. Each (N,H,W) / (N,H,W,3)."""
    cap, _ = _open_boson_agc8()
    if cap is None:
        raise RuntimeError("could not open Boson in AGC8 mode")
    for _ in range(warmup):
        cap.read()
    agc8 = []
    while len(agc8) < n_frames:
        ok, f = cap.read()
        if ok and f is not None and f.ndim == 3:
            agc8.append(f.copy())
    cap.release()

    time.sleep(0.5)
    cap, _ = _open_boson_y16()
    if cap is None:
        raise RuntimeError("could not open Boson in Y16 mode")
    for _ in range(warmup):
        cap.read()
    y16 = []
    while len(y16) < n_frames:
        ok, f = cap.read()
        if ok and f is not None and f.ndim == 2 and f.dtype == np.uint16:
            y16.append(f.copy())
    cap.release()

    return np.stack(y16, axis=0), np.stack(agc8, axis=0)


# ───────────────────────────────────────────────────────────────
# Sweep
# ───────────────────────────────────────────────────────────────

def render_pipeline(y16_stack: np.ndarray, cfg: PipelineConfig) -> np.ndarray:
    """Apply a PipelineConfig to a Y16 stack. Returns u8 grayscale stack."""
    p = cfg.to_params()
    out = np.zeros((y16_stack.shape[0], y16_stack.shape[1], y16_stack.shape[2]),
                   dtype=np.uint8)
    for i in range(y16_stack.shape[0]):
        f = y16_stack[i]
        if p.dead_pixel_median_enabled:
            f = apply_dead_pixel_median(f, 3)
        agc = apply_agc_mode(f, p)
        out[i] = enhance_post_agc(agc, p)
    return out


@dataclass
class SweepResult:
    config_name: str
    config: dict
    metrics: dict        # asdict(StackMetrics)
    composite: float
    error: Optional[str] = None


def sweep(y16_stack: np.ndarray, configs: Iterable[PipelineConfig],
          out_dir: str, on_progress=None) -> List[SweepResult]:
    """Run the sweep, save per-config preview + metrics, return ranked results."""
    os.makedirs(out_dir, exist_ok=True)
    results: List[SweepResult] = []
    configs_list = list(configs)
    for i, cfg in enumerate(configs_list):
        cfg_dir = os.path.join(out_dir, cfg.name)
        os.makedirs(cfg_dir, exist_ok=True)
        try:
            u8_stack = render_pipeline(y16_stack, cfg)
            mlist = [_tm.per_frame_metrics(u8_stack[k]) for k in range(u8_stack.shape[0])]
            sm = _tm.aggregate(mlist, u8_stack)
            # Save mid-frame preview (apply colormap)
            mid = u8_stack.shape[0] // 2
            preview = apply_colormap(u8_stack[mid], cfg.colormap)
            cv2.imwrite(os.path.join(cfg_dir, "preview.png"), preview)
            with open(os.path.join(cfg_dir, "metrics.json"), "w") as f:
                json.dump({
                    "config": asdict(cfg),
                    "metrics": _tm.metrics_to_dict(sm),
                }, f, indent=2)
            results.append(SweepResult(
                config_name=cfg.name,
                config=asdict(cfg),
                metrics=_tm.metrics_to_dict(sm),
                composite=sm.composite,
            ))
            if on_progress:
                on_progress(i + 1, len(configs_list), cfg.name, sm.composite)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            results.append(SweepResult(cfg.name, asdict(cfg), {}, -float("inf"), error=err))
            if on_progress:
                on_progress(i + 1, len(configs_list), cfg.name, -float("inf"))
    return sorted(results, key=lambda r: r.composite, reverse=True)


# ───────────────────────────────────────────────────────────────
# Reporting
# ───────────────────────────────────────────────────────────────

def write_leaderboard(results: List[SweepResult], out_path: str,
                      title: str = "Pipeline sweep leaderboard"):
    lines = [
        f"# {title}",
        "",
        f"_{len(results)} configs evaluated, sorted by composite score._",
        "",
        "| Rank | Composite | Config | Sharpness (Lap) | Edge density | Sat % | Notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(results, start=1):
        m = r.metrics
        lap = m.get("sharpness_lap", 0)
        ed = m.get("edge_density", 0)
        sat = m.get("saturation_pct", 0)
        note = f"`{r.error}`" if r.error else ""
        lines.append(
            f"| {rank} | {r.composite:.3f} | `{r.config_name}` | "
            f"{lap:.1f} | {ed:.4f} | {sat * 100:.2f} | {note} |"
        )
    with open(out_path, "w") as f:
        f.write("\n".join(lines))


# ───────────────────────────────────────────────────────────────
# State / sentinel
# ───────────────────────────────────────────────────────────────

@dataclass
class HarnessState:
    pose: str
    started_at: float
    last_heartbeat: float
    tier: str
    completed_configs: List[str] = field(default_factory=list)
    leaderboard_top: List[dict] = field(default_factory=list)


def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "STATE.json")


def _heartbeat_path(out_dir: str) -> str:
    return os.path.join(out_dir, "HEARTBEAT.txt")


def _stop_path(out_dir: str) -> str:
    return os.path.join(out_dir, "STOP")


def write_state(state: HarnessState, out_dir: str):
    with open(_state_path(out_dir), "w") as f:
        json.dump(asdict(state), f, indent=2)


def write_heartbeat(out_dir: str, msg: str):
    with open(_heartbeat_path(out_dir), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")


def stop_requested(out_dir: str) -> bool:
    return os.path.exists(_stop_path(out_dir))


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", required=True,
                    help="Tag for this pose (used in output dir)")
    ap.add_argument("--frames", type=int, default=30, help="Frames per capture")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--use-existing-npz", type=str, default=None,
                    help="Reuse a previously-captured .npz instead of capturing live")
    ap.add_argument("--resume", action="store_true",
                    help="Skip configs already in STATE.json")
    ap.add_argument("--no-tier3", action="store_true",
                    help="Skip the post-enhance sweep (tier 3)")
    args = ap.parse_args(argv)

    out_dir = os.path.join(OUT_BASE, args.pose)
    os.makedirs(out_dir, exist_ok=True)

    # Load or initialize state
    state_path = _state_path(out_dir)
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            state_dict = json.load(f)
        state = HarnessState(**state_dict)
        print(f"[harness] resuming from STATE.json — {len(state.completed_configs)} "
              f"configs already done")
    else:
        state = HarnessState(
            pose=args.pose,
            started_at=time.time(),
            last_heartbeat=time.time(),
            tier="capture",
        )
    write_state(state, out_dir)

    # Capture (or load existing)
    if args.use_existing_npz:
        write_heartbeat(out_dir, f"loading {args.use_existing_npz}")
        data = np.load(args.use_existing_npz)
        y16, agc8 = data["y16"], data["agc8"]
    else:
        write_heartbeat(out_dir, "capturing live frames")
        y16, agc8 = capture_stacks(n_frames=args.frames, warmup=args.warmup)
        np.savez_compressed(
            os.path.join(out_dir, "capture.npz"),
            y16=y16, agc8=agc8,
        )
    print(f"[harness] y16 stack: {y16.shape}, agc8 stack: {agc8.shape}")

    # Build full sweep
    sweep_configs = tier1_baselines() + tier2_clahe_y16_sweep()
    if args.resume:
        sweep_configs = [c for c in sweep_configs if c.name not in state.completed_configs]
    print(f"[harness] {len(sweep_configs)} configs to sweep")

    def on_prog(i, total, name, score):
        write_heartbeat(out_dir, f"sweep {i}/{total}: {name} -> {score:.3f}")
        state.completed_configs.append(name)
        state.last_heartbeat = time.time()
        write_state(state, out_dir)
        if stop_requested(out_dir):
            raise KeyboardInterrupt("STOP sentinel detected")

    state.tier = "tier12_sweep"
    write_state(state, out_dir)
    try:
        results = sweep(y16, sweep_configs, out_dir, on_progress=on_prog)
    except KeyboardInterrupt as e:
        print(f"[harness] stopped: {e}")
        return 130

    write_leaderboard(results, os.path.join(out_dir, "leaderboard_t1t2.md"),
                      f"Tier 1+2 sweep — pose {args.pose}")

    # Tier 3 — only if T2 winner is clahe_y16 family
    if not args.no_tier3 and results and "clahe_y16" in results[0].config_name:
        winner = next(c for c in sweep_configs if c.name == results[0].config_name)
        t3_configs = tier3_post_enhance_sweep(winner)
        state.tier = "tier3_post_enhance"
        write_state(state, out_dir)
        try:
            t3_results = sweep(y16, t3_configs, out_dir, on_progress=on_prog)
        except KeyboardInterrupt as e:
            print(f"[harness] stopped during T3: {e}")
            return 130
        write_leaderboard(t3_results, os.path.join(out_dir, "leaderboard_t3.md"),
                          f"Tier 3 post-enhance — pose {args.pose}")
        results = sorted(results + t3_results, key=lambda r: r.composite, reverse=True)

    write_leaderboard(results, os.path.join(out_dir, "leaderboard_overall.md"),
                      f"Overall — pose {args.pose}")
    state.tier = "complete"
    state.last_heartbeat = time.time()
    state.leaderboard_top = [
        {"name": r.config_name, "composite": r.composite}
        for r in results[:5]
    ]
    write_state(state, out_dir)
    write_heartbeat(out_dir, "complete")
    print(f"[harness] done. top 5:")
    for r in results[:5]:
        print(f"  {r.composite:7.3f}  {r.config_name}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

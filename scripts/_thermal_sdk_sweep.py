"""
SDK-extended thermal optimization sweep — Boson hardware controls.

Combines camera-side parameters (via flirpy) with software-side
parameters (existing harness) for a proper full-cartesian sweep.

Camera-side knobs swept:
    gain_mode: HIGH / LOW / DUAL  (FLR_BOSON_*_GAIN enum)
    averager:  0 (off) / 2 / 4   (frame averaging, noise reduction)

Software-side knobs swept:
    AGC mode: global / roi / clahe_y16
    For clahe_y16: tile_grid in [8, 12, 16, 20]

Per camera config:
    1. Apply via flirpy (set_gain_mode, set_averager)
    2. Trigger manual FFC for a clean baseline
    3. Wait for FFC to settle (~1.5s)
    4. Capture Y16 + AGC8 stack
    5. For each software config: render, score, save preview
    6. Move to next camera config

Total: 3 (gain) x 3 (averager) x 7 (sw modes) = 63 configs.
Per-config wall: ~5s capture + render. Total: ~5 minutes.

Output: recordings/optim/sdk_<tag>/<cam_cfg>/<sw_cfg>/preview.png +
metrics.json + leaderboard.

Usage:
    python scripts/_thermal_sdk_sweep.py --tag indoor_v1
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import struct
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from typing import List, Optional

import cv2
import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Pull metrics module
_metrics_path = os.path.join(_REPO_ROOT, "scripts", "_thermal_metrics.py")
_spec = importlib.util.spec_from_file_location("_thermal_metrics", _metrics_path)
_tm = importlib.util.module_from_spec(_spec)
sys.modules["_thermal_metrics"] = _tm
_spec.loader.exec_module(_tm)

from thermal.thermal_processor import (  # noqa: E402
    ThermalEnhanceParams, apply_agc_mode, apply_clahe_y16, apply_colormap,
    apply_dead_pixel_median, enhance_post_agc,
)

OUT_BASE = os.path.join("recordings", "optim")


# ───────────────────────────────────────────────────────────────
# Camera config (Boson SDK side)
# ───────────────────────────────────────────────────────────────

GAIN_MODE_HIGH = 0
GAIN_MODE_LOW = 1
GAIN_MODE_AUTO = 2
GAIN_MODE_DUAL = 3
GAIN_MODE_NAMES = {0: "HIGH", 1: "LOW", 2: "AUTO", 3: "DUAL"}


@dataclass(frozen=True)
class CameraConfig:
    name: str
    gain_mode: int = GAIN_MODE_HIGH
    averager: int = 0
    do_ffc: bool = True


def camera_grid() -> List[CameraConfig]:
    out = []
    for g in [GAIN_MODE_HIGH, GAIN_MODE_LOW, GAIN_MODE_AUTO]:
        for a in [0, 2, 4]:
            out.append(CameraConfig(
                name=f"gain{GAIN_MODE_NAMES[g]}_avg{a}",
                gain_mode=g, averager=a,
            ))
    return out


def apply_camera_config(boson, cfg: CameraConfig):
    """Apply a CameraConfig to the live camera. Returns time to wait
    after settling (sec)."""
    boson.set_gain_mode(cfg.gain_mode)
    try:
        boson.set_averager(cfg.averager)
    except Exception:
        pass  # not all firmware versions
    if cfg.do_ffc:
        try:
            boson.do_ffc()
        except Exception:
            pass
    return 1.5  # let things settle


# ───────────────────────────────────────────────────────────────
# Software config grid
# ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SwConfig:
    name: str
    mode: str = "global"
    low_pct: float = 2.0
    high_pct: float = 98.0
    roi_top_frac: float = 0.4
    cold_count: int = 0
    hot_count: int = 65535
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    colormap: str = "WHITE_HOT"
    dead_pixel: bool = False
    gamma: float = 1.0

    def to_params(self) -> ThermalEnhanceParams:
        return ThermalEnhanceParams(
            mode=self.mode,
            low_percentile=self.low_pct,
            high_percentile=self.high_pct,
            roi_top_frac=self.roi_top_frac,
            cold_count=self.cold_count,
            hot_count=self.hot_count,
            clahe_y16_clip_limit=self.clahe_clip,
            clahe_y16_tile_grid=self.clahe_tile,
            colormap=self.colormap,
            gamma=self.gamma,
            dead_pixel_median_enabled=self.dead_pixel,
        )


def sw_grid() -> List[SwConfig]:
    out = [
        SwConfig("global_2_98", mode="global"),
        SwConfig("roi_60", mode="roi", roi_top_frac=0.4),
    ]
    for tile in [8, 12, 16, 20]:
        out.append(SwConfig(f"clahe_y16_tile{tile}",
                            mode="clahe_y16", clahe_tile=tile))
    out.append(SwConfig("global_with_dead_pixel_median",
                        mode="global", dead_pixel=True))
    return out


# ───────────────────────────────────────────────────────────────
# Capture (using flirpy's already-open Boson wouldn't work for video;
# we need to open the UVC stream separately).
# ───────────────────────────────────────────────────────────────

def _open_y16(idx_hint: int = 1):
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


def _open_agc8(idx_hint: int = 1):
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


def capture_both(n_frames: int = 30, warmup: int = 8):
    """Capture (y16, agc8) stacks. Y16 + AGC8 sequentially because
    the Boson can only be in one output mode at a time over UVC."""
    cap, _ = _open_agc8()
    if cap is None:
        raise RuntimeError("could not open Boson AGC8 mode")
    for _ in range(warmup):
        cap.read()
    agc8 = []
    while len(agc8) < n_frames:
        ok, f = cap.read()
        if ok and f is not None and f.ndim == 3:
            agc8.append(f.copy())
    cap.release()
    time.sleep(0.5)

    cap, _ = _open_y16()
    if cap is None:
        raise RuntimeError("could not open Boson Y16 mode")
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
# Render & score
# ───────────────────────────────────────────────────────────────

def render_pipeline(y16_stack: np.ndarray, cfg: SwConfig) -> np.ndarray:
    p = cfg.to_params()
    out = np.zeros_like(y16_stack[..., 0:1].squeeze(-1) if y16_stack.ndim == 4
                        else np.empty(y16_stack.shape, dtype=np.uint8),
                        dtype=np.uint8)
    out = np.zeros((y16_stack.shape[0], y16_stack.shape[1], y16_stack.shape[2]),
                   dtype=np.uint8)
    for i in range(y16_stack.shape[0]):
        f = y16_stack[i]
        if p.dead_pixel_median_enabled:
            f = apply_dead_pixel_median(f, 3)
        agc = apply_agc_mode(f, p)
        out[i] = enhance_post_agc(agc, p)
    return out


def score_stack(u8_stack: np.ndarray, yolo_model=None) -> dict:
    yolo_results = [None] * u8_stack.shape[0]
    if yolo_model is not None:
        stride = max(1, u8_stack.shape[0] // 4)
        for k in range(0, u8_stack.shape[0], stride):
            bgr = cv2.cvtColor(u8_stack[k], cv2.COLOR_GRAY2BGR)
            try:
                ys = yolo_model.predict(bgr, conf=0.4, imgsz=640,
                                        verbose=False, device=0)
                if ys:
                    yolo_results[k] = ys[0]
            except Exception:
                pass
    mlist = [_tm.per_frame_metrics(u8_stack[k], yolo_results[k])
             for k in range(u8_stack.shape[0])]
    sm = _tm.aggregate(mlist, u8_stack)
    return _tm.metrics_to_dict(sm), sm.composite


def _try_yolo():
    p = os.path.join(_REPO_ROOT, "models", "seeker_thermal_hv.pt")
    if not os.path.exists(p):
        return None
    try:
        from ultralytics import YOLO
        return YOLO(p)
    except Exception:
        return None


# ───────────────────────────────────────────────────────────────
# State / sentinel
# ───────────────────────────────────────────────────────────────

def _state_path(out_dir: str) -> str:
    return os.path.join(out_dir, "STATE.json")


def _stop_path(out_dir: str) -> str:
    return os.path.join(out_dir, "STOP")


def _heartbeat_path(out_dir: str) -> str:
    return os.path.join(out_dir, "HEARTBEAT.txt")


def write_heartbeat(out_dir: str, msg: str):
    with open(_heartbeat_path(out_dir), "w") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}: {msg}\n")


# ───────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    cam_name: str
    sw_name: str
    composite: float
    metrics: dict
    error: Optional[str] = None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default=time.strftime("%Y%m%d_%H%M"),
                    help="Output dir tag under recordings/optim/sdk_<tag>")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--port", default="COM3", help="Boson serial port")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=0,
                    help="Cap total cam×sw configs (0=unlimited)")
    args = ap.parse_args(argv)

    out_dir = os.path.join(OUT_BASE, f"sdk_{args.tag}")
    os.makedirs(out_dir, exist_ok=True)

    # Load completed-set if resuming
    completed = set()
    state_path = _state_path(out_dir)
    if args.resume and os.path.exists(state_path):
        with open(state_path) as f:
            sd = json.load(f)
        completed = set(sd.get("completed", []))
        print(f"[sdk_sweep] resuming with {len(completed)} configs already done")

    yolo_model = _try_yolo()
    print(f"[sdk_sweep] YOLO model: {'loaded' if yolo_model else 'none'}")

    # Open Boson SDK control
    from flirpy.camera.boson import Boson
    print(f"[sdk_sweep] opening Boson on {args.port}...")
    boson = Boson(port=args.port)
    print(f"[sdk_sweep] camera SN={boson.get_camera_serial()}, "
          f"PN={boson.get_part_number()}, "
          f"FPA={boson.get_fpa_temperature():.1f}C, "
          f"gain_mode_now={boson.get_gain_mode()}")

    # Build grid
    cam_grid = camera_grid()
    sw_g = sw_grid()
    if args.limit:
        cam_grid = cam_grid[:max(1, args.limit // len(sw_g))]
    total = len(cam_grid) * len(sw_g)
    print(f"[sdk_sweep] grid: {len(cam_grid)} cam × {len(sw_g)} sw = {total} configs")

    results: List[SweepResult] = []
    counter = 0

    try:
        for cam in cam_grid:
            if os.path.exists(_stop_path(out_dir)):
                print("[sdk_sweep] STOP sentinel — exiting")
                break

            cam_dir = os.path.join(out_dir, cam.name)
            os.makedirs(cam_dir, exist_ok=True)

            # Apply camera config and capture once
            try:
                wait_s = apply_camera_config(boson, cam)
                time.sleep(wait_s)
            except Exception as e:
                print(f"[sdk_sweep] camera config {cam.name} failed: {e}")
                continue

            write_heartbeat(out_dir, f"capturing for cam={cam.name}")
            try:
                y16, agc8 = capture_both(n_frames=args.frames)
            except Exception as e:
                print(f"[sdk_sweep] capture failed for {cam.name}: {e}")
                continue
            np.savez_compressed(os.path.join(cam_dir, "capture.npz"),
                                 y16=y16, agc8=agc8)

            for sw in sw_g:
                key = f"{cam.name}__{sw.name}"
                if key in completed:
                    continue
                counter += 1
                cfg_dir = os.path.join(cam_dir, sw.name)
                os.makedirs(cfg_dir, exist_ok=True)
                try:
                    u8 = render_pipeline(y16, sw)
                    metrics, composite = score_stack(u8, yolo_model)
                    mid = u8.shape[0] // 2
                    preview = apply_colormap(u8[mid], sw.colormap)
                    cv2.imwrite(os.path.join(cfg_dir, "preview.png"), preview)
                    with open(os.path.join(cfg_dir, "metrics.json"), "w") as f:
                        json.dump({
                            "cam_config": asdict(cam),
                            "sw_config": asdict(sw),
                            "metrics": metrics,
                        }, f, indent=2)
                    results.append(SweepResult(cam.name, sw.name, composite, metrics))
                    completed.add(key)
                    write_heartbeat(out_dir,
                                     f"{counter}/{total}  {key}  -> {composite:.3f}")
                    print(f"  {counter:3d}/{total}  {key:48s} -> {composite:.3f}")
                    # Persist state after each config so we can resume
                    with open(_state_path(out_dir), "w") as f:
                        json.dump({"completed": sorted(completed)}, f)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    results.append(SweepResult(cam.name, sw.name, -float("inf"),
                                               {}, error=err))
                    print(f"  {counter:3d}/{total}  {key} -> ERROR: {err}")
    finally:
        try:
            boson.close()
        except Exception:
            pass

    # Leaderboard
    results_sorted = sorted(results, key=lambda r: r.composite, reverse=True)
    lines = [
        "# SDK + software sweep leaderboard",
        "",
        f"_{len(results_sorted)} configs evaluated, sorted by composite._",
        "",
        "| Rank | Composite | Camera | Software | Sharp(Lap) | Edges | YOLO_conf | Notes |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for rank, r in enumerate(results_sorted, start=1):
        m = r.metrics or {}
        note = f"`{r.error}`" if r.error else ""
        lines.append(
            f"| {rank} | {r.composite:.3f} | `{r.cam_name}` | `{r.sw_name}` | "
            f"{m.get('sharpness_lap', 0):.1f} | "
            f"{m.get('edge_density', 0):.4f} | "
            f"{m.get('yolo_confidence_sum', 0):.2f} | {note} |"
        )
    leaderboard_path = os.path.join(out_dir, "leaderboard.md")
    with open(leaderboard_path, "w") as f:
        f.write("\n".join(lines))

    print()
    print(f"[sdk_sweep] leaderboard: {leaderboard_path}")
    print(f"[sdk_sweep] top 5:")
    for r in results_sorted[:5]:
        print(f"  {r.composite:7.3f}  {r.cam_name:25s}  {r.sw_name}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)

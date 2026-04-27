"""Replay the fusion matcher against a captured JSONL recording.

Why this exists
---------------
The morning of 2026-04-27 a per-track gimbal-pose camera-frame
compensation was added to fusion_manager._update_tracks (to fix YOLO
id-swap during fast slews) and was reverted the same afternoon after
recordings/gimbal_not_tracking_static.jsonl exposed it merging two
distinct vehicles into one fused id. SESSION_SUMMARY.md explicitly
calls out the right next step: "Fusion in world frame (structural)
— track world_az = cam_az + gimbal_pan instead of camera-frame az.
Solves both the YOLO id-swap case AND the multi-vehicle merge case.
Needs a fusion-replay tool to validate before shipping."

This is that fusion-replay tool. It re-runs an offline fusion matcher
against the per-sensor observations + gimbal-state stream from a
recording and emits the same fused_track_born / fused_track_dropped
events the live fusion_manager would. Two variants:

    --variant camera-frame   Reproduces the live matcher (parity).
    --variant world-frame    Stores + matches tracks in world frame.

The camera-frame variant is a parity check: if its outputs don't match
the recorded events, the replay tool itself is wrong. Once parity is
established, the world-frame variant is judged against the regression
set:

    Human_and_vehicle_mistrack.jsonl   must keep the YOLO id-swap survival
    gimbal_not_tracking_static.jsonl   must NOT merge two distinct vehicles
    radar_opposite.jsonl               radar persistence unchanged or better

Usage:
    python scripts/replay_fusion.py --recording RECORDING.jsonl --variant camera-frame
    python scripts/replay_fusion.py --recording RECORDING.jsonl --variant world-frame
                                    --out cmp.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

# Repo-root import path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fusion.angular import angular_iou, bbox_to_angular   # type: ignore
# TargetClass values are strings (TargetClass.PERSON.value -> "person", etc.)


_FUSABLE_CLASSES = {"person", "vehicle", "drone", "radar_target"}
RADAR_TARGET = "radar_target"
TRACK_IOU = 0.15


# ──────────────────────────────────────────────────────────────────
# JSONL streaming + per-tick observation extraction
# ──────────────────────────────────────────────────────────────────
def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def thermal_obs_from_frame(msg: Dict[str, Any], thermal_az_bias: float,
                           thermal_el_bias: float) -> List[Dict[str, Any]]:
    if not msg.get("connected"):
        return []
    w = int(msg.get("width") or 0)
    h = int(msg.get("height") or 0)
    if w == 0 or h == 0:
        return []
    hfov = float(msg.get("hfov_deg", 75.0))
    vfov = float(msg.get("vfov_deg", 60.0))
    out = []
    for d in msg.get("detections") or []:
        cls = (d.get("classification") or {}).get("target_class")
        if cls not in _FUSABLE_CLASSES:
            continue
        bb = d.get("bbox") or {}
        az, el, aw, ah = bbox_to_angular(
            int(bb.get("x", 0)), int(bb.get("y", 0)),
            int(bb.get("w", 0)), int(bb.get("h", 0)),
            w, h, hfov, vfov,
        )
        az += thermal_az_bias
        el += thermal_el_bias
        out.append({
            "az": az, "el": el, "ang_w": aw, "ang_h": ah,
            "class": cls,
            "conf": float((d.get("classification") or {}).get("confidence", 0.0)),
            "primary": "thermal",
            "sensors": ["thermal"],
        })
    return out


def eo_obs_from_frame(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not msg.get("connected"):
        return []
    w = int(msg.get("width") or 0)
    h = int(msg.get("height") or 0)
    if w == 0 or h == 0:
        return []
    hfov = float(msg.get("hfov_deg", 11.05))
    vfov = float(msg.get("vfov_deg", 9.23))
    out = []
    for d in msg.get("detections") or []:
        cls = d.get("target_class")
        if cls not in _FUSABLE_CLASSES:
            continue
        bb = d.get("bbox") or {}
        az, el, aw, ah = bbox_to_angular(
            int(bb.get("x", 0)), int(bb.get("y", 0)),
            int(bb.get("w", 0)), int(bb.get("h", 0)),
            w, h, hfov, vfov,
        )
        out.append({
            "az": az, "el": el, "ang_w": aw, "ang_h": ah,
            "class": cls, "conf": float(d.get("confidence", 0.0)),
            "primary": "eo", "sensors": ["eo"],
        })
    return out


def radar_obs_from_frame(msg: Dict[str, Any], radar_az_bias: float,
                         radar_el_bias: float) -> List[Dict[str, Any]]:
    if not msg.get("connected"):
        return []
    out = []
    for t in msg.get("targets") or []:
        if t.get("coasting"):
            continue
        pos = t.get("pos") or [0.0, 0.0, 0.0]
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if y <= 0.1:
            continue
        slant = math.sqrt(x*x + y*y + z*z)
        if slant < 0.1:
            continue
        az = math.degrees(math.atan2(x, y)) + radar_az_bias
        el = math.degrees(math.atan2(z, math.sqrt(x*x + y*y))) + radar_el_bias
        size = t.get("size") or [0.5, 0.5, 0.5]
        ang_w = max(0.4, math.degrees(2.0 * math.atan2(float(size[0]), slant)))
        ang_h = max(0.4, math.degrees(2.0 * math.atan2(float(size[2]), slant)))
        out.append({
            "az": az, "el": el, "ang_w": ang_w, "ang_h": ang_h,
            "class": RADAR_TARGET, "conf": float(t.get("conf", 0.5)),
            "primary": "radar", "sensors": ["radar"],
        })
    return out


# ──────────────────────────────────────────────────────────────────
# Fusion matcher — two variants
# ──────────────────────────────────────────────────────────────────
@dataclass
class FusionState:
    """Tracks held in CAMERA frame (mirrors live FusionManager)."""
    tracks: List[Dict[str, Any]] = field(default_factory=list)
    next_id: int = 1
    max_misses: int = 30
    sensor_grace_ticks: int = 5

    def step(self, candidates: List[Dict[str, Any]],
             cur_pan: float, cur_tilt: float
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Run one fusion tick. Returns (births, deaths) — events that would
        have been emitted live."""
        n_existing = len(self.tracks)
        matched = [False] * n_existing
        births: List[Dict[str, Any]] = []
        for c in candidates:
            best_i, best_iou = -1, 0.0
            for i in range(n_existing):
                trk = self.tracks[i]
                if matched[i] or not _class_compat(trk["class"], c["class"]):
                    continue
                iou = angular_iou(
                    c["az"], c["el"], c["ang_w"], c["ang_h"],
                    trk["az"], trk["el"], trk["ang_w"], trk["ang_h"],
                )
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= TRACK_IOU:
                trk = self.tracks[best_i]
                a = 0.4
                trk["az"]    = a * trk["az"]    + (1 - a) * c["az"]
                trk["el"]    = a * trk["el"]    + (1 - a) * c["el"]
                trk["ang_w"] = a * trk["ang_w"] + (1 - a) * c["ang_w"]
                trk["ang_h"] = a * trk["ang_h"] + (1 - a) * c["ang_h"]
                if trk["class"] == RADAR_TARGET and c["class"] != RADAR_TARGET:
                    trk["class"] = c["class"]
                for s in list(trk["sensor_misses"].keys()):
                    trk["sensor_misses"][s] += 1
                for s in c["sensors"]:
                    trk["sensor_misses"][s] = 0
                trk["sensor_misses"] = {
                    s: m for s, m in trk["sensor_misses"].items()
                    if m <= self.sensor_grace_ticks
                }
                if not (c["primary"] == "radar" and trk["class"] != RADAR_TARGET):
                    trk["primary"] = c["primary"]
                trk["conf"] = max(trk["conf"] * 0.9, c["conf"])
                trk["hits"] += 1
                trk["misses"] = 0
                matched[best_i] = True
            else:
                new_track = {
                    "id": self.next_id,
                    "class": c["class"],
                    "az": c["az"], "el": c["el"],
                    "ang_w": c["ang_w"], "ang_h": c["ang_h"],
                    "sensor_misses": {s: 0 for s in c["sensors"]},
                    "primary": c["primary"],
                    "conf": c["conf"],
                    "hits": 1, "misses": 0,
                }
                self.tracks.append(new_track)
                births.append({
                    "id": self.next_id,
                    "class": c["class"],
                    "primary": c["primary"],
                    "az": float(c["az"]), "el": float(c["el"]),
                    "sensors": list(c["sensors"]),
                    "conf": float(c["conf"]),
                })
                self.next_id += 1

        # Age unmatched tracks; drop those past max_misses.
        kept: List[Dict[str, Any]] = []
        deaths: List[Dict[str, Any]] = []
        for i, trk in enumerate(self.tracks):
            if i < len(matched) and matched[i]:
                kept.append(trk)
            else:
                trk["misses"] += 1
                for s in list(trk["sensor_misses"].keys()):
                    trk["sensor_misses"][s] += 1
                trk["sensor_misses"] = {
                    s: m for s, m in trk["sensor_misses"].items()
                    if m <= self.sensor_grace_ticks
                }
                if trk["misses"] <= self.max_misses:
                    kept.append(trk)
                else:
                    deaths.append({
                        "id": int(trk["id"]),
                        "hits": int(trk["hits"]),
                        "misses": int(trk["misses"]),
                        "reason": "max_misses",
                    })
        self.tracks = kept
        return births, deaths


@dataclass
class WorldFrameFusionState(FusionState):
    """Tracks held in WORLD frame.

    The structural fix to YOLO id-swap and gimbal-slew dissociation. Each
    incoming candidate is converted to world frame at observation time:
        world_az = cam_az + cur_pan
        world_el = cam_el + cur_tilt
    Tracks are matched in world frame, so a target's world position
    survives any amount of gimbal motion between observations.

    All other matcher logic is identical to the camera-frame variant.
    """

    def step(self, candidates: List[Dict[str, Any]],
             cur_pan: float, cur_tilt: float
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # Convert candidates camera-frame az/el → world-frame az/el
        world_cands = []
        for c in candidates:
            wc = dict(c)
            wc["az"] = c["az"] + cur_pan
            wc["el"] = c["el"] + cur_tilt
            world_cands.append(wc)
        return super().step(world_cands, cur_pan, cur_tilt)


def _class_compat(a: str, b: str) -> bool:
    if a == b:
        return True
    return a == RADAR_TARGET or b == RADAR_TARGET


# ──────────────────────────────────────────────────────────────────
# Per-tick cross-sensor association + dedup + merge — mirrors
# fusion_manager._tick exactly.
# ──────────────────────────────────────────────────────────────────
XSENSOR_IOU = 0.15
DEDUP_IOU   = 0.35
MERGE_IOU   = 0.35


def cross_sensor_associate(thermal_obs: List[Dict[str, Any]],
                           eo_obs: List[Dict[str, Any]],
                           radar_obs: List[Dict[str, Any]],
                           radar_iou_gate: float
                           ) -> List[Dict[str, Any]]:
    """EO + thermal merge, then radar attach. Matches fusion_manager._tick."""
    candidates: List[Dict[str, Any]] = []
    used_t = [False] * len(thermal_obs)
    for e in eo_obs:
        best_i, best_iou = -1, 0.0
        for i, t in enumerate(thermal_obs):
            if used_t[i] or t["class"] != e["class"]:
                continue
            iou = angular_iou(
                e["az"], e["el"], e["ang_w"], e["ang_h"],
                t["az"], t["el"], t["ang_w"], t["ang_h"])
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= XSENSOR_IOU:
            t = thermal_obs[best_i]
            used_t[best_i] = True
            candidates.append({
                "sensors": ["eo", "thermal"], "primary": "eo",
                "class": e["class"],
                "az": e["az"], "el": e["el"],
                "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                "conf": max(e["conf"], t["conf"]),
            })
        else:
            candidates.append({
                "sensors": ["eo"], "primary": "eo",
                "class": e["class"],
                "az": e["az"], "el": e["el"],
                "ang_w": e["ang_w"], "ang_h": e["ang_h"],
                "conf": e["conf"],
            })
    for i, t in enumerate(thermal_obs):
        if used_t[i]:
            continue
        candidates.append({
            "sensors": ["thermal"], "primary": "thermal",
            "class": t["class"],
            "az": t["az"], "el": t["el"],
            "ang_w": t["ang_w"], "ang_h": t["ang_h"],
            "conf": t["conf"],
        })

    # Radar joins: attach to camera candidates via IoU; survivors
    # become standalone radar candidates.
    n_cam = len(candidates)
    used_c = [False] * n_cam
    for r in radar_obs:
        best_i, best_iou = -1, 0.0
        for i in range(n_cam):
            if used_c[i]:
                continue
            c = candidates[i]
            iou = angular_iou(
                r["az"], r["el"], r["ang_w"], r["ang_h"],
                c["az"], c["el"], c["ang_w"], c["ang_h"])
            if iou > best_iou:
                best_iou, best_i = iou, i
        if best_i >= 0 and best_iou >= radar_iou_gate:
            c = candidates[best_i]
            used_c[best_i] = True
            if "radar" not in c["sensors"]:
                c["sensors"].append("radar")
            c["conf"] = max(c["conf"], r["conf"])
        else:
            candidates.append({
                "sensors": ["radar"], "primary": "radar",
                "class": r["class"],
                "az": r["az"], "el": r["el"],
                "ang_w": r["ang_w"], "ang_h": r["ang_h"],
                "conf": r["conf"],
            })
    return candidates


def dedup_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(cands) <= 1:
        return cands
    order = sorted(
        range(len(cands)),
        key=lambda i: (len(cands[i]["sensors"]), cands[i]["conf"]),
        reverse=True)
    used = [False] * len(cands)
    out: List[Dict[str, Any]] = []
    for i in order:
        if used[i]:
            continue
        w = dict(cands[i])
        w["sensors"] = list(w["sensors"])
        used[i] = True
        for j in order:
            if used[j] or cands[j]["class"] != w["class"]:
                continue
            c = cands[j]
            iou = angular_iou(
                w["az"], w["el"], w["ang_w"], w["ang_h"],
                c["az"], c["el"], c["ang_w"], c["ang_h"])
            if iou >= DEDUP_IOU:
                used[j] = True
                for s in c["sensors"]:
                    if s not in w["sensors"]:
                        w["sensors"].append(s)
                if "eo" in w["sensors"]:
                    w["primary"] = "eo"
                w["conf"] = max(w["conf"], c["conf"])
        out.append(w)
    return out


def merge_overlapping_tracks(tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if len(tracks) <= 1:
        return tracks
    rt = RADAR_TARGET
    order = sorted(
        range(len(tracks)),
        key=lambda i: (tracks[i]["class"] != rt, tracks[i]["hits"], -tracks[i]["id"]),
        reverse=True)
    drop = [False] * len(tracks)
    for oi, i in enumerate(order):
        if drop[i]:
            continue
        a = tracks[i]
        for j in order[oi + 1:]:
            if drop[j]:
                continue
            b = tracks[j]
            if not _class_compat(a["class"], b["class"]):
                continue
            iou = angular_iou(
                a["az"], a["el"], a["ang_w"], a["ang_h"],
                b["az"], b["el"], b["ang_w"], b["ang_h"])
            if iou >= MERGE_IOU:
                for s, m in b["sensor_misses"].items():
                    cur = a["sensor_misses"].get(s)
                    a["sensor_misses"][s] = m if cur is None else min(cur, m)
                a["conf"] = max(a["conf"], b["conf"])
                if a["class"] == rt and b["class"] != rt:
                    a["class"] = b["class"]
                drop[j] = True
    return [t for k, t in enumerate(tracks) if not drop[k]]


# ──────────────────────────────────────────────────────────────────
# Driver
# ──────────────────────────────────────────────────────────────────
def replay(path: str, variant: str, out_csv: Optional[str]) -> int:
    print(f"recording: {path}")
    print(f"variant:   {variant}")

    # Pull config snapshot for fusion params + extrinsics.
    header = None
    fcfg = {}
    thermal_az = thermal_el = radar_az = radar_el = 0.0
    state: FusionState
    rate_hz = 15.0
    next_fusion_t = None
    fusion_dt = 1.0 / rate_hz

    # Latest-only buffers for each sensor (mirrors BUS.get_latest semantics)
    latest_thermal = None
    latest_eo = None
    latest_radar = None
    latest_pan = 0.0
    latest_tilt = 0.0

    rec_births: List[Tuple[float, Dict[str, Any]]] = []
    rec_deaths: List[Tuple[float, Dict[str, Any]]] = []
    sim_births: List[Tuple[float, Dict[str, Any]]] = []
    sim_deaths: List[Tuple[float, Dict[str, Any]]] = []
    rate_hz = 15.0
    state = None

    t0_ns = None

    for r in iter_records(path):
        ts = r.get("ts_ns")
        if ts is None:
            continue
        if t0_ns is None:
            t0_ns = int(ts)
        t = (int(ts) - t0_ns) / 1e9
        ch = r.get("channel", "")
        msg = r.get("msg") or {}

        if ch == "session/header":
            header = msg
            cfg = (msg.get("config_snapshot") or {})
            fcfg = cfg.get("fusion") or {}
            tcfg = (cfg.get("thermal") or {}).get("extrinsic") or {}
            rcfg = (cfg.get("radar") or {}).get("extrinsic") or {}
            thermal_az = float(tcfg.get("az_bias_deg", 0.0))
            thermal_el = float(tcfg.get("el_bias_deg", 0.0))
            radar_az = float(rcfg.get("az_bias_deg", 0.0))
            radar_el = float(rcfg.get("el_bias_deg", 0.0))
            rate_hz = float(fcfg.get("rate_hz", 15.0))
            cls = WorldFrameFusionState if variant == "world-frame" else FusionState
            state = cls(
                max_misses=int(fcfg.get("max_misses", 30)),
                sensor_grace_ticks=int(fcfg.get("sensor_grace_ticks", 5)),
            )
            fusion_dt = 1.0 / rate_hz
            print(f"  rate_hz={rate_hz}  max_misses={state.max_misses}  "
                  f"sensor_grace_ticks={state.sensor_grace_ticks}  "
                  f"thermal_bias=({thermal_az:.3f}, {thermal_el:.3f})  "
                  f"radar_bias=({radar_az:.3f}, {radar_el:.3f})")
            continue

        if state is None:
            # No header yet; assume defaults.
            cls = WorldFrameFusionState if variant == "world-frame" else FusionState
            state = cls(max_misses=30, sensor_grace_ticks=5)

        if ch == "events":
            tp = msg.get("type")
            pl = msg.get("payload") or {}
            if tp == "fused_track_born":
                rec_births.append((t, pl))
            elif tp == "fused_track_dropped":
                rec_deaths.append((t, pl))
            continue

        if ch == "thermal/frame":
            latest_thermal = msg
        elif ch == "eo/frame":
            latest_eo = msg
        elif ch == "radar/frame":
            latest_radar = msg
        elif ch == "gimbal/state":
            latest_pan = float(msg.get("pan_deg") or 0.0)
            latest_tilt = float(msg.get("tilt_deg") or 0.0)

        # Run a fusion tick at rate_hz.
        if next_fusion_t is None:
            next_fusion_t = t + fusion_dt
        radar_iou_gate = float(fcfg.get("radar_iou_gate", 0.05))
        while next_fusion_t is not None and t >= next_fusion_t:
            thermal_obs = thermal_obs_from_frame(latest_thermal or {},
                                                 thermal_az, thermal_el)
            eo_obs = eo_obs_from_frame(latest_eo or {})
            radar_obs_list = radar_obs_from_frame(latest_radar or {},
                                                  radar_az, radar_el)
            cands = cross_sensor_associate(thermal_obs, eo_obs,
                                            radar_obs_list, radar_iou_gate)
            cands = dedup_candidates(cands)
            births, deaths = state.step(cands, latest_pan, latest_tilt)
            state.tracks = merge_overlapping_tracks(state.tracks)
            for b in births:
                sim_births.append((next_fusion_t, b))
            for d in deaths:
                sim_deaths.append((next_fusion_t, d))
            next_fusion_t += fusion_dt

    # Print summary
    print(f"\nrecorded births: {len(rec_births)}, deaths: {len(rec_deaths)}")
    print(f"simulated births: {len(sim_births)}, deaths: {len(sim_deaths)}")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["t_rel", "kind", "source", "id", "class", "primary",
                        "az", "el", "sensors"])
            for t, b in rec_births:
                w.writerow([f"{t:.3f}", "born", "rec", b.get("id"),
                            b.get("class"), b.get("primary"),
                            b.get("az", 0), b.get("el", 0),
                            ",".join(b.get("sensors") or [])])
            for t, b in sim_births:
                w.writerow([f"{t:.3f}", "born", "sim", b.get("id"),
                            b.get("class"), b.get("primary"),
                            b.get("az", 0), b.get("el", 0),
                            ",".join(b.get("sensors") or [])])
            for t, d in rec_deaths:
                w.writerow([f"{t:.3f}", "dropped", "rec", d.get("id"),
                            "", "", "", "", ""])
            for t, d in sim_deaths:
                w.writerow([f"{t:.3f}", "dropped", "sim", d.get("id"),
                            "", "", "", "", ""])
        print(f"wrote {out_csv}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", "-r", required=True,
                    help="JSONL recording path")
    ap.add_argument("--variant", default="camera-frame",
                    choices=("camera-frame", "world-frame"),
                    help="Matcher variant (default camera-frame for parity)")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()
    if not os.path.exists(args.recording):
        print(f"recording not found: {args.recording}")
        return 2
    return replay(args.recording, args.variant, args.out)


if __name__ == "__main__":
    sys.exit(main())

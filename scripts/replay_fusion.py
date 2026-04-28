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
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                        List[int]]:
        """Run one fusion tick. Returns (births, deaths, matched_ids).

        matched_ids = the list of track ids that received an observation
        this tick (used by the lifespan analyzer to count observations
        per track).
        """
        n_existing = len(self.tracks)
        matched = [False] * n_existing
        matched_ids: List[int] = []
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
                matched_ids.append(int(trk["id"]))
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
        return births, deaths, matched_ids


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
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                        List[int]]:
        # Convert candidates camera-frame az/el → world-frame az/el
        world_cands = []
        for c in candidates:
            wc = dict(c)
            wc["az"] = c["az"] + cur_pan
            wc["el"] = c["el"] + cur_tilt
            world_cands.append(wc)
        return super().step(world_cands, cur_pan, cur_tilt)


@dataclass
class WorldFrameCaptureFusionState(FusionState):
    """Same as WorldFrameFusionState but uses per-candidate
    SENSOR-FRAME-CAPTURE-TIME pose instead of fusion-tick pose.

    This is the offline simulation of the 2a16649 timing-offset fix.
    Candidates carry `_pan_at` / `_tilt_at` set by the driver from
    a pose-history lookup against the source frame's timestamp.
    """

    def step(self, candidates: List[Dict[str, Any]],
             cur_pan: float, cur_tilt: float
             ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]],
                        List[int]]:
        world_cands = []
        for c in candidates:
            wc = dict(c)
            pan_at = float(c.get("_pan_at", cur_pan))
            tilt_at = float(c.get("_tilt_at", cur_tilt))
            wc["az"] = c["az"] + pan_at
            wc["el"] = c["el"] + tilt_at
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
    # Per-sensor capture-time bookkeeping for the world-frame-capture
    # variant: when a sensor frame arrives, record its rel-time `t`
    # alongside the gimbal pose at that moment, so the offline matcher
    # can convert each observation with the pose from when its source
    # frame was actually captured.
    latest_thermal_ts = 0.0
    latest_eo_ts = 0.0
    latest_radar_ts = 0.0
    pose_history: List[Tuple[float, float, float]] = []
    POSE_HISTORY_MAX = 60   # ~2 s at 30 Hz gimbal/state
    POSE_HISTORY_MAX_AGE_S = 2.0

    def _pose_at(ts: float) -> Tuple[float, float]:
        if not pose_history or ts <= 0.0:
            return (latest_pan, latest_tilt)
        latest_t = pose_history[-1][0]
        if latest_t - ts > POSE_HISTORY_MAX_AGE_S:
            return (latest_pan, latest_tilt)
        best = pose_history[0]
        best_dt = abs(best[0] - ts)
        for entry in pose_history:
            dt = abs(entry[0] - ts)
            if dt < best_dt:
                best_dt, best = dt, entry
        return (best[1], best[2])

    rec_births: List[Tuple[float, Dict[str, Any]]] = []
    rec_deaths: List[Tuple[float, Dict[str, Any]]] = []
    sim_births: List[Tuple[float, Dict[str, Any]]] = []
    sim_deaths: List[Tuple[float, Dict[str, Any]]] = []
    # Per-id lifespan accumulator: id -> dict(birth_t, death_t, n_obs,
    # birth_class, primary, born_az, born_el)
    sim_lives: Dict[int, Dict[str, Any]] = {}
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
            if variant == "world-frame-capture":
                cls = WorldFrameCaptureFusionState
            elif variant == "world-frame":
                cls = WorldFrameFusionState
            else:
                cls = FusionState
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
            if variant == "world-frame-capture":
                cls = WorldFrameCaptureFusionState
            elif variant == "world-frame":
                cls = WorldFrameFusionState
            else:
                cls = FusionState
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
            latest_thermal_ts = t
        elif ch == "eo/frame":
            latest_eo = msg
            latest_eo_ts = t
        elif ch == "radar/frame":
            latest_radar = msg
            latest_radar_ts = t
        elif ch == "gimbal/state":
            latest_pan = float(msg.get("pan_deg") or 0.0)
            latest_tilt = float(msg.get("tilt_deg") or 0.0)
            pose_history.append((t, latest_pan, latest_tilt))
            if len(pose_history) > POSE_HISTORY_MAX:
                pose_history.pop(0)

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
            # For the world-frame-capture variant: tag each candidate
            # with the gimbal pose at its source frame's capture time
            # (looked up from pose_history). The matcher uses these
            # rather than the fusion-tick pose snapshot.
            if variant == "world-frame-capture":
                for c in cands:
                    p = c.get("primary")
                    if p == "thermal":
                        ts = latest_thermal_ts
                    elif p == "radar":
                        ts = latest_radar_ts
                    else:
                        ts = latest_eo_ts
                    pan_at, tilt_at = _pose_at(ts)
                    c["_pan_at"] = pan_at
                    c["_tilt_at"] = tilt_at
            births, deaths, matched_ids = state.step(cands, latest_pan, latest_tilt)
            state.tracks = merge_overlapping_tracks(state.tracks)
            for b in births:
                sim_births.append((next_fusion_t, b))
                sim_lives[int(b["id"])] = {
                    "birth_t": float(next_fusion_t),
                    "death_t": None,
                    "n_obs": 1,
                    "class": b.get("class"),
                    "primary": b.get("primary"),
                    "born_az": b.get("az"),
                    "born_el": b.get("el"),
                }
            for d in deaths:
                sim_deaths.append((next_fusion_t, d))
                if int(d["id"]) in sim_lives:
                    sim_lives[int(d["id"])]["death_t"] = float(next_fusion_t)
            for tid in matched_ids:
                if tid in sim_lives:
                    sim_lives[tid]["n_obs"] += 1
            next_fusion_t += fusion_dt

    # Print summary
    print(f"\nrecorded births: {len(rec_births)}, deaths: {len(rec_deaths)}")
    print(f"simulated births: {len(sim_births)}, deaths: {len(sim_deaths)}")

    # ── Per-track lifespan summary ──
    # Close any tracks that never died (still alive at end of recording)
    if state and state.tracks:
        for trk in state.tracks:
            tid = int(trk["id"])
            if tid in sim_lives and sim_lives[tid]["death_t"] is None:
                sim_lives[tid]["death_t"] = float(next_fusion_t or 0.0)

    if sim_lives:
        # Compute lifespan for each track
        lifespans = []
        for tid, info in sim_lives.items():
            bt = info["birth_t"]
            dt = info["death_t"] if info["death_t"] is not None else bt
            life_s = dt - bt
            lifespans.append((tid, life_s, info["n_obs"], info["class"],
                              info["primary"]))
        lifespans.sort(key=lambda x: -x[1])  # longest first

        # Aggregate stats
        total = len(lifespans)
        life_values = [l[1] for l in lifespans]
        n_obs_values = [l[2] for l in lifespans]
        long_lived = [l for l in lifespans if l[1] >= 1.0]  # >= 1s alive
        orphan = [l for l in lifespans if l[2] <= 1]  # born and died alone

        def _stats(xs: List[float]) -> Dict[str, float]:
            if not xs:
                return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
            xs2 = sorted(xs)
            return {
                "min": xs2[0],
                "median": xs2[len(xs2) // 2],
                "mean": sum(xs2) / len(xs2),
                "max": xs2[-1],
            }

        ls = _stats(life_values)
        os = _stats(n_obs_values)
        print(f"\n== PER-TRACK LIFESPAN SUMMARY ==")
        print(f"  total tracks                  : {total}")
        print(f"  long-lived (>= 1.0 s alive)   : {len(long_lived)}  "
              f"({100*len(long_lived)/max(total,1):.1f}%)")
        print(f"  orphan (1 obs, died next tick): {len(orphan)}  "
              f"({100*len(orphan)/max(total,1):.1f}%)")
        print(f"  lifespan_s   min/median/mean/max: "
              f"{ls['min']:.2f} / {ls['median']:.2f} / "
              f"{ls['mean']:.2f} / {ls['max']:.2f}")
        print(f"  n_obs        min/median/mean/max: "
              f"{os['min']:.0f} / {os['median']:.0f} / "
              f"{os['mean']:.1f} / {os['max']:.0f}")
        print(f"\n  Top 8 longest-lived tracks:")
        print(f"  {'id':>4} {'lifespan_s':>11} {'n_obs':>6} {'class':>14} {'primary':>8}")
        for tid, life_s, n_obs, cls, primary in lifespans[:8]:
            print(f"  {tid:>4} {life_s:>11.2f} {n_obs:>6} "
                  f"{str(cls):>14} {str(primary):>8}")

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
                    choices=("camera-frame", "world-frame",
                             "world-frame-capture"),
                    help="Matcher variant (default camera-frame for parity)")
    ap.add_argument("--out", default=None, help="CSV output path")
    args = ap.parse_args()
    if not os.path.exists(args.recording):
        print(f"recording not found: {args.recording}")
        return 2
    return replay(args.recording, args.variant, args.out)


if __name__ == "__main__":
    sys.exit(main())

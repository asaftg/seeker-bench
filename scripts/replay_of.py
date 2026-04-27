"""
Optical-flow / heat-tracker replay against a captured JSONL session.

Decodes the thermal JPEG frames out of the recording, feeds them into
a fresh ``DetectionTracker``, and replays user actions (synthetic
target draws, clears) at their captured timestamps. Compares the
tracker's bbox trajectory against what was recorded.

This is the "did our proposed fix work?" tool for synthetic-target /
heat-tracking issues — no rig needed. Pair it with ``replay_algo.py``
(predictor) to validate algorithm changes against the same captured
inputs without having to re-run the bench.

Variants:
    --variant default                 # match recorded params exactly
    --variant max_dist_px=80,of_min_features=2,of_max_features=40,...
    --variant of_synth_shift_mult=2.5  # scale the synthetic OF cap
                                       # (custom replay-only knob —
                                       # see TrackerConfig docs for
                                       # what's tunable from YAML)

Output:
    Stdout/CSV table per thermal frame with bbox center, hits, misses,
    coasting, plus the recorded (live) values for side-by-side compare.
    A summary at the bottom flags how many frames the bbox CENTRE
    differs by > 5 px between live and replay — quick signal whether
    a variant materially changes tracker behaviour.
"""
from __future__ import annotations

import argparse
import base64
import csv
import dataclasses
import io
import json
import os
import sys
from typing import Any, Dict, Iterator, List, Optional, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import cv2

from common.frames import BBox, ThermalDetection
from thermal.detection_tracker import DetectionTracker, TrackerConfig
from scripts.replay_inspect import find_latest, iter_records


def _decode_jpeg(b64: Optional[str]) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        buf = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


def _det_from_recorded(d: Dict[str, Any]) -> ThermalDetection:
    """Rebuild a ThermalDetection from the recorder's JSON shape.

    Classification dict is dropped — the tracker doesn't care about it
    for matching, and we'd otherwise need to import the full enum.
    Synthetic flag is preserved.
    """
    bb = d.get("bbox") or {}
    return ThermalDetection(
        bbox=BBox(int(bb.get("x", 0)), int(bb.get("y", 0)),
                  int(bb.get("w", 0)), int(bb.get("h", 0))),
        area_px=int(d.get("area_px", 0)),
        contrast=float(d.get("contrast", 0.0)),
        synthetic=bool(d.get("synthetic", False)),
    )


def _make_tracker(cfg_snap: Dict[str, Any], variant: str) -> DetectionTracker:
    """Build a tracker from the recorded config snapshot, then apply
    KEY=VALUE overrides from the variant string."""
    # NB: the live thermal manager reads `cfg.heat_detector.tracker`,
    # NOT `cfg.thermal.tracker` — the YAML key sits under heat_detector.
    # Replay must mirror that or it ends up using TrackerConfig defaults
    # (max_dist_px=60) instead of the live values (max_dist_px=40).
    hd = (cfg_snap or {}).get("heat_detector", {}) or {}
    trk_cfg = (hd.get("tracker") or {}) if isinstance(hd, dict) else {}
    defaults = TrackerConfig()
    cfg = TrackerConfig(
        enabled=bool(trk_cfg.get("enabled", defaults.enabled)),
        max_dist_px=float(trk_cfg.get("max_dist_px", defaults.max_dist_px)),
        min_hits=int(trk_cfg.get("min_hits", defaults.min_hits)),
        max_misses=int(trk_cfg.get("max_misses", defaults.max_misses)),
        ema=float(trk_cfg.get("ema", defaults.ema)),
        of_enabled=bool(trk_cfg.get("of_enabled", defaults.of_enabled)),
        max_of_bridges=int(trk_cfg.get("max_of_bridges", defaults.max_of_bridges)),
        of_min_warmth_contrast=float(trk_cfg.get(
            "of_min_warmth_contrast", defaults.of_min_warmth_contrast)),
    )
    if variant and variant.lower() != "default":
        field_types = {f.name: f.type for f in dataclasses.fields(cfg)}
        for chunk in variant.split(","):
            if "=" not in chunk:
                continue
            k, v = chunk.split("=", 1)
            k, v = k.strip(), v.strip()
            if k not in field_types:
                sys.stderr.write(f"unknown variant field: {k}\n")
                continue
            cur = getattr(cfg, k)
            try:
                cast = type(cur)(v) if not isinstance(cur, bool) \
                    else (v.lower() in ("1", "true", "yes"))
            except Exception:
                cast = float(v)
            setattr(cfg, k, cast)
    return DetectionTracker(cfg), cfg


def replay_of(path: str, variant: str, out_csv: Optional[str]) -> int:
    cfg_snap: Dict[str, Any] = {}
    # Pre-pass: collect (ts_ns, channel, msg) for thermal frames and
    # synthetic-target events. Other channels ignored.
    thermal_frames: List[Tuple[int, Dict[str, Any]]] = []
    synth_events: List[Tuple[int, str, Dict[str, Any]]] = []
    recorded_heat_tracks: Dict[int, List[Tuple[int, Dict[str, Any]]]] = {}
    for rec in iter_records(path):
        ch = rec.get("channel"); ts = int(rec.get("ts_ns") or 0)
        msg = rec.get("msg") or {}
        if ch == "session/header":
            cfg_snap = msg.get("config_snapshot") or {}
        elif ch == "thermal/frame":
            thermal_frames.append((ts, msg))
            for ht in (msg.get("heat_tracks") or []):
                tid = ht.get("id")
                if tid is not None:
                    recorded_heat_tracks.setdefault(int(tid), []).append((ts, ht))
        elif ch == "events":
            t = msg.get("type", "")
            if t in ("synthetic_target_drawn", "synthetic_target_cleared"):
                synth_events.append((ts, t, msg.get("payload") or {}))

    if not thermal_frames:
        sys.stderr.write("no thermal/frame entries in recording\n")
        return 1
    if not synth_events:
        sys.stderr.write("no synthetic_target events in recording — "
                         "this replay only handles user-drawn targets\n")
        return 1

    tracker, applied_cfg = _make_tracker(cfg_snap, variant)
    sys.stderr.write(f"tracker config: {applied_cfg}\n")
    sys.stderr.write(f"thermal frames: {len(thermal_frames)}\n")
    sys.stderr.write(f"synthetic events: {len(synth_events)}\n")

    rows: List[Dict[str, Any]] = []
    se_idx = 0
    replay_synth_id: Optional[int] = None  # the id our REPLAY tracker assigned
    recorded_synth_id: Optional[int] = None  # the id the LIVE tracker assigned

    for ts, msg in thermal_frames:
        # Apply any synthetic events whose timestamp <= this frame's ts.
        while se_idx < len(synth_events) and synth_events[se_idx][0] <= ts:
            _, etype, payload = synth_events[se_idx]
            se_idx += 1
            if etype == "synthetic_target_drawn":
                bbox = payload.get("bbox") or [0, 0, 0, 0]
                # The live tracker's most recent display_gray was THIS
                # frame's image — use it for OF feature seeding so we
                # match live behaviour as closely as possible.
                img = _decode_jpeg(msg.get("jpeg_b64"))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
                replay_synth_id = tracker.seed_synthetic(
                    BBox(int(bbox[0]), int(bbox[1]),
                         int(bbox[2]), int(bbox[3])),
                    agc8=gray,
                )
                recorded_synth_id = payload.get("tid")
                sys.stderr.write(
                    f"replay seed_synthetic: replay_id={replay_synth_id} "
                    f"live_id={recorded_synth_id} bbox={bbox}\n")
            elif etype == "synthetic_target_cleared":
                tracker.clear_synthetic()
                replay_synth_id = None
                recorded_synth_id = None

        # Decode this frame and feed the tracker. We pass the recorded
        # detections list as the input — that's what the live manager
        # would have produced from the same image; replaying detection
        # is out of scope (and the tracker is the only thing under test).
        img = _decode_jpeg(msg.get("jpeg_b64"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None
        live_dets_recorded = msg.get("detections") or []
        live_dets = [_det_from_recorded(d) for d in live_dets_recorded
                     if not d.get("synthetic")]
        try:
            tracker.update(live_dets, gray)
        except Exception as e:
            sys.stderr.write(f"tracker.update failed: {e}\n")
            continue

        if replay_synth_id is None:
            continue

        # Look up our replay synthetic track and the recorded one
        replay_snap = None
        for s in tracker.snapshot():
            if s.id == replay_synth_id:
                replay_snap = s
                break
        rec_ht = None
        if recorded_synth_id is not None:
            for ht_ts, ht in recorded_heat_tracks.get(recorded_synth_id, []):
                if ht_ts == ts:
                    rec_ht = ht
                    break

        rcx = (replay_snap.bbox.x + replay_snap.bbox.w // 2) if replay_snap else None
        rcy = (replay_snap.bbox.y + replay_snap.bbox.h // 2) if replay_snap else None
        live_cx = live_cy = None
        if rec_ht:
            bb = rec_ht["bbox"]
            live_cx = bb["x"] + bb["w"] // 2
            live_cy = bb["y"] + bb["h"] // 2

        rows.append({
            "ts_ns": ts,
            "replay_cx": rcx, "replay_cy": rcy,
            "replay_misses": replay_snap.misses if replay_snap else None,
            "replay_coasting": replay_snap.coasting if replay_snap else None,
            "replay_hits": replay_snap.hits if replay_snap else None,
            "live_cx": live_cx, "live_cy": live_cy,
            "live_misses": rec_ht["misses"] if rec_ht else None,
            "live_coasting": rec_ht["coasting"] if rec_ht else None,
            "dx": (rcx - live_cx) if (rcx is not None and live_cx is not None) else None,
            "dy": (rcy - live_cy) if (rcy is not None and live_cy is not None) else None,
        })

    if not rows:
        sys.stderr.write("no rows produced (no synthetic track lifetime?)\n")
        return 1

    cols = list(rows[0].keys())
    fh = open(out_csv, "w", encoding="utf-8", newline="") if out_csv else sys.stdout
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    if out_csv:
        fh.close()

    # Summary
    n = len(rows)
    big_drift = sum(1 for r in rows
                    if r["dx"] is not None and (r["dx"] ** 2 + r["dy"] ** 2) > 25)
    max_drift = 0
    for r in rows:
        if r["dx"] is not None:
            d = (r["dx"] ** 2 + r["dy"] ** 2) ** 0.5
            if d > max_drift:
                max_drift = d
    sys.stderr.write(
        f"\nsummary: frames={n}  frames_with_>5px_centre_drift={big_drift}  "
        f"max_centre_drift_px={max_drift:.1f}\n")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="JSONL (default: --latest)")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--dir", default="recordings")
    p.add_argument("--variant", default="default",
                   help='K=V,K=V overrides for TrackerConfig fields '
                        '(e.g. "max_dist_px=80,max_misses=10")')
    p.add_argument("--out", default=None,
                   help="CSV path; default = stdout")
    args = p.parse_args()

    if args.latest or not args.file:
        path = find_latest(args.dir)
        if path is None:
            sys.stderr.write(f"no seeker_*.jsonl in {args.dir!r}\n")
            return 2
    else:
        path = args.file
    return replay_of(path, args.variant, args.out)


if __name__ == "__main__":
    sys.exit(main())

"""extract_drone_timeline.py

Build a 1-fps EO + thermal visual timeline from a seeker_bench JSONL recording so a human
can scrub through and see WHEN the drone is in scene.

Outputs (all under recordings/<stem>_visual/):
  eo/eo_t{seconds:05.2f}_fid{frame_id}.jpg          (~1 per recording-second)
  thermal/thermal_t{seconds:05.2f}_fid{frame_id}.jpg
  timeline.csv      one row per sampled second, EO + nearest-thermal joined
  radar_frames.csv  one row per radar/aa_frame event (NO subsampling) so we can map
                    t_rel_s -> aa frame_id -> raw .bin frame index

Usage:
    python tools/extract_drone_timeline.py "recordings/drone fly.jsonl"
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", type=Path, help="path to recording .jsonl")
    ap.add_argument("--fps", type=float, default=1.0, help="sample rate, frames/sec")
    return ap.parse_args()


def open_jsonl(path: Path):
    """Stream events; tolerate occasional truncated lines."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for ln_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                yield ln_no, json.loads(line)
            except json.JSONDecodeError:
                # tail-of-file truncation or rare bad line; skip silently
                continue


def write_jpeg(out_path: Path, jpeg_b64: str) -> int:
    raw = base64.b64decode(jpeg_b64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as fh:
        fh.write(raw)
    return len(raw)


def main() -> int:
    args = parse_args()
    jsonl_path: Path = args.jsonl.resolve()
    if not jsonl_path.exists():
        print(f"ERROR: file not found: {jsonl_path}", file=sys.stderr)
        return 2

    rec_dir = jsonl_path.parent
    stem = jsonl_path.stem  # "drone fly"
    safe_stem = stem.replace(" ", "_")
    out_root = rec_dir / f"{safe_stem}_visual"
    out_eo = out_root / "eo"
    out_th = out_root / "thermal"
    out_eo.mkdir(parents=True, exist_ok=True)
    out_th.mkdir(parents=True, exist_ok=True)

    sample_period_s = 1.0 / args.fps

    # Pass 1: stream the JSONL once. We hold per-second buckets in memory but only the
    # MOST RECENT frame per second (we keep the full latest event for EO and thermal).
    # We also remember the latest gimbal state at the time of each kept EO frame, and
    # the aa-frame and radar/frame counts since the previous kept EO frame.
    first_eo_ts: float | None = None  # wall seconds (from msg.timestamp)
    last_gimbal: dict | None = None
    pending_radar_n_targets = 0  # most recent radar/frame num_targets
    pending_aa_n_targets = 0     # most recent radar/aa_frame num_targets

    # bucket_idx -> dict with eo_event, thermal_event, gimbal_snapshot, radar_n, aa_n
    eo_buckets: dict[int, dict] = {}
    thermal_buckets: dict[int, dict] = {}

    aa_rows: list[dict] = []  # every aa_frame event

    n_events = 0
    n_eo = n_th = n_aa = n_radar = n_gimbal = 0

    for _, ev in open_jsonl(jsonl_path):
        n_events += 1
        ch = ev.get("channel")
        msg = ev.get("msg") or {}

        if ch == "gimbal/state":
            n_gimbal += 1
            last_gimbal = msg
            continue

        if ch == "radar/frame":
            n_radar += 1
            pending_radar_n_targets = int(msg.get("num_targets", 0) or 0)
            continue

        if ch == "radar/aa_frame":
            n_aa += 1
            pending_aa_n_targets = int(msg.get("num_targets", 0) or 0)
            aa_rows.append({
                "wall_ts": float(msg.get("timestamp", 0.0) or 0.0),
                "frame_id": int(msg.get("frame_id", -1)),
                "n_targets": int(msg.get("num_targets", 0) or 0),
                "n_points": int(msg.get("num_points", 0) or 0),
                "gimbal_pan": msg.get("gimbal_pan_at_capture"),
                "gimbal_tilt": msg.get("gimbal_tilt_at_capture"),
            })
            continue

        if ch == "eo/frame":
            n_eo += 1
            ts = msg.get("timestamp")
            if ts is None or "jpeg_b64" not in msg:
                continue
            ts = float(ts)
            if first_eo_ts is None:
                first_eo_ts = ts
            t_rel = ts - first_eo_ts
            bucket = int(t_rel // sample_period_s)
            # keep MOST RECENT frame in the bucket
            eo_buckets[bucket] = {
                "ts": ts,
                "frame_id": int(msg.get("frame_id", -1)),
                "jpeg_b64": msg["jpeg_b64"],
                "detections": msg.get("detections", []) or [],
                "gimbal_pan_at_capture": msg.get("gimbal_pan_at_capture"),
                "gimbal_tilt_at_capture": msg.get("gimbal_tilt_at_capture"),
                "gimbal_snapshot": last_gimbal,
                "radar_n": pending_radar_n_targets,
                "aa_n": pending_aa_n_targets,
            }
            continue

        if ch == "thermal/frame":
            n_th += 1
            ts = msg.get("timestamp")
            if ts is None or "jpeg_b64" not in msg:
                continue
            ts = float(ts)
            # use EO's first_eo_ts if known, else use thermal's own
            anchor = first_eo_ts if first_eo_ts is not None else ts
            t_rel = ts - anchor
            bucket = int(t_rel // sample_period_s)
            thermal_buckets[bucket] = {
                "ts": ts,
                "frame_id": int(msg.get("frame_id", -1)),
                "jpeg_b64": msg["jpeg_b64"],
                "detections": msg.get("detections", []) or [],
            }
            continue

        # ignore: fusion/tracks, events, session/header

    if first_eo_ts is None:
        print("ERROR: no eo/frame events found in recording", file=sys.stderr)
        return 3

    # --- write timeline CSV + dump JPEGs ---
    timeline_csv = out_root / "timeline.csv"
    n_rows = 0
    with open(timeline_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "t_rel_s", "wall_ts",
            "eo_frame_id", "eo_path",
            "thermal_frame_id", "thermal_path",
            "gimbal_pan", "gimbal_tilt",
            "n_eo_dets", "n_thermal_dets",
            "n_radar_targets", "n_aa_targets",
        ])
        for bucket in sorted(eo_buckets):
            eo = eo_buckets[bucket]
            t_rel = eo["ts"] - first_eo_ts
            eo_name = f"eo_t{t_rel:05.2f}_fid{eo['frame_id']}.jpg"
            eo_path = out_eo / eo_name
            write_jpeg(eo_path, eo["jpeg_b64"])

            th = thermal_buckets.get(bucket)
            th_fid = ""
            th_path_str = ""
            n_th_dets = 0
            if th is not None:
                th_t_rel = th["ts"] - first_eo_ts
                th_name = f"thermal_t{th_t_rel:05.2f}_fid{th['frame_id']}.jpg"
                th_path = out_th / th_name
                write_jpeg(th_path, th["jpeg_b64"])
                th_fid = th["frame_id"]
                th_path_str = str(th_path.relative_to(rec_dir))
                n_th_dets = len(th["detections"])

            # prefer the snapshot at-capture pose (sensor-side), fall back to ambient gimbal/state
            pan = eo["gimbal_pan_at_capture"]
            tilt = eo["gimbal_tilt_at_capture"]
            if pan is None and eo["gimbal_snapshot"]:
                pan = eo["gimbal_snapshot"].get("pan_deg")
            if tilt is None and eo["gimbal_snapshot"]:
                tilt = eo["gimbal_snapshot"].get("tilt_deg")

            w.writerow([
                f"{t_rel:.3f}", f"{eo['ts']:.3f}",
                eo["frame_id"], str(eo_path.relative_to(rec_dir)),
                th_fid, th_path_str,
                ("" if pan is None else f"{float(pan):.3f}"),
                ("" if tilt is None else f"{float(tilt):.3f}"),
                len(eo["detections"]), n_th_dets,
                eo["radar_n"], eo["aa_n"],
            ])
            n_rows += 1

    # --- write radar AA CSV (no subsampling) ---
    aa_csv = out_root / "radar_frames.csv"
    first_aa_fid = aa_rows[0]["frame_id"] if aa_rows else None
    with open(aa_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "t_rel_s", "wall_ts", "frame_id", "n_targets", "n_points",
            "gimbal_pan", "gimbal_tilt", "bin_idx_estimate",
        ])
        for r in aa_rows:
            t_rel = r["wall_ts"] - first_eo_ts
            bin_idx = r["frame_id"] - first_aa_fid if first_aa_fid is not None else ""
            pan = r["gimbal_pan"]
            tilt = r["gimbal_tilt"]
            w.writerow([
                f"{t_rel:.3f}", f"{r['wall_ts']:.3f}",
                r["frame_id"], r["n_targets"], r["n_points"],
                ("" if pan is None else f"{float(pan):.3f}"),
                ("" if tilt is None else f"{float(tilt):.3f}"),
                bin_idx,
            ])

    # --- print summary ---
    last_t = max(b["ts"] for b in eo_buckets.values()) - first_eo_ts
    print(f"Parsed {n_events} events:")
    print(f"  eo/frame:        {n_eo}")
    print(f"  thermal/frame:   {n_th}")
    print(f"  radar/frame:     {n_radar}")
    print(f"  radar/aa_frame:  {n_aa}")
    print(f"  gimbal/state:    {n_gimbal}")
    print(f"Recording duration (first->last EO): {last_t:.2f} s")
    print()
    print(f"Output dir:      {out_root}")
    print(f"  eo jpegs:      {len(eo_buckets)} in {out_eo}")
    print(f"  thermal jpegs: {len(thermal_buckets)} in {out_th}")
    print(f"  timeline csv:  {timeline_csv}  ({n_rows} rows)")
    print(f"  aa csv:        {aa_csv}  ({len(aa_rows)} rows)")
    print()

    # head of each csv
    def head(path: Path, n: int = 6):
        print(f"--- head {path.name} ({n} rows) ---")
        with open(path, "r", encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if i >= n + 1:
                    break
                print(line.rstrip())
        print()

    head(timeline_csv, 8)
    head(aa_csv, 8)

    if first_aa_fid is not None:
        print(f"first aa_frame.frame_id = {first_aa_fid}  -> bin_idx = frame_id - {first_aa_fid}")
        print(f"last  aa_frame.frame_id = {aa_rows[-1]['frame_id']} (n_aa={len(aa_rows)}, expected bin_frames ~ {len(aa_rows)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

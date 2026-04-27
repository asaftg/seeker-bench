"""
Inspect a Seeker-01 JSONL recording.

This is the FIRST tool to reach for when the user reports a bug they
captured. The expected loop is:

    user: "I recorded the tracking bug; gimbal slewed 40 deg right."
    agent: python scripts/replay_inspect.py --latest --summary
    agent: python scripts/replay_inspect.py --latest --track-id 12

No file paths or CLI flags from the user — ``--latest`` defaults to
the newest ``recordings/seeker_*.jsonl``. Designed to be the single
entry point an LLM agent uses to triage a session offline.

Modes:
    --summary               session duration, channel counts, event
                            counts by type, top tracks
    --track-id N            per-tick CSV of tracking pipeline state
                            (gimbal pose, observed az/el, predictor
                            internals) — the one to grep for "what
                            was the predictor doing at t=2.3s?"
    --grep <regex>          filter events by type
    --events-since <ts>     wall-clock time slice (s since epoch or
                            ISO 8601)
    --channels              list channels and message counts
    --head N                print first N JSONL lines verbatim
                            (useful sanity check on the file)

The single-file canonical "show me the last session" recipe documented
in recording/README.md is exactly:

    python scripts/replay_inspect.py --latest --summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, Iterator, List, Optional


DEFAULT_DIR = "recordings"


# ──────────────────────────────────────────────────────────────────
# JSONL streaming
# ──────────────────────────────────────────────────────────────────
def iter_records(path: str) -> Iterator[Dict[str, Any]]:
    with io.open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                # One bad line shouldn't kill an inspection — log and skip.
                sys.stderr.write(f"[inspect] {path}:{ln}: {e}\n")
                continue


def find_latest(directory: str = DEFAULT_DIR) -> Optional[str]:
    pattern = os.path.join(directory, "seeker_*.jsonl")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


# ──────────────────────────────────────────────────────────────────
# Modes
# ──────────────────────────────────────────────────────────────────
def cmd_summary(path: str) -> int:
    counts: Counter = Counter()
    event_types: Counter = Counter()
    track_ids: Counter = Counter()  # tracked_id from track_predictor_step
    fused_ids: Counter = Counter()  # fused track ids that ever appeared
    first_ts_ns: Optional[int] = None
    last_ts_ns: Optional[int] = None
    header: Optional[Dict[str, Any]] = None

    for rec in iter_records(path):
        ts = rec.get("ts_ns")
        ch = rec.get("channel", "?")
        msg = rec.get("msg") or {}
        counts[ch] += 1
        if ts is not None:
            if first_ts_ns is None:
                first_ts_ns = int(ts)
            last_ts_ns = int(ts)
        if ch == "session/header":
            header = msg
        elif ch == "events":
            t = msg.get("type", "?")
            event_types[t] += 1
            payload = msg.get("payload") or {}
            if t == "track_predictor_step":
                tid = payload.get("tracked_id")
                if tid is not None:
                    track_ids[int(tid)] += 1
            elif t in ("track_engaged", "fused_track_born"):
                fid = payload.get("target_id") or payload.get("id")
                if fid is not None:
                    fused_ids[int(fid)] += 1
        elif ch == "fusion/tracks":
            for tr in msg.get("tracks") or []:
                fid = tr.get("id")
                if fid is not None:
                    fused_ids[int(fid)] += 1

    print(f"file: {path}")
    if header:
        print(f"  version: {header.get('version')}")
        started = header.get("started_at")
        if started:
            try:
                print(f"  started: {datetime.fromtimestamp(float(started)).isoformat(sep=' ')}")
            except Exception:
                print(f"  started_at: {started}")
        print(f"  jpeg_quality: {header.get('jpeg_quality')}")
    if first_ts_ns is not None and last_ts_ns is not None:
        dur_s = (last_ts_ns - first_ts_ns) / 1e9
        print(f"  duration: {dur_s:.2f} s")
    print()
    print("channels (msg counts):")
    width = max((len(c) for c in counts), default=0)
    for ch, n in counts.most_common():
        print(f"  {ch.ljust(width)}  {n}")
    if event_types:
        print()
        print("events:")
        ewidth = max(len(t) for t in event_types)
        for t, n in event_types.most_common():
            print(f"  {t.ljust(ewidth)}  {n}")
    if fused_ids:
        print()
        print("fused track ids seen (top 10 by mentions):")
        for fid, n in fused_ids.most_common(10):
            print(f"  #{fid}  {n}")
    if track_ids:
        print()
        print("tracked ids in predictor stream (top 5):")
        for tid, n in track_ids.most_common(5):
            print(f"  #{tid}  ticks={n}")
    return 0


def cmd_channels(path: str) -> int:
    counts: Counter = Counter()
    for rec in iter_records(path):
        counts[rec.get("channel", "?")] += 1
    width = max((len(c) for c in counts), default=0)
    for ch, n in counts.most_common():
        print(f"{ch.ljust(width)}  {n}")
    return 0


def cmd_head(path: str, n: int) -> int:
    for i, rec in enumerate(iter_records(path)):
        if i >= n:
            break
        # Trim image payloads so the terminal doesn't drown in base64
        msg = rec.get("msg") or {}
        if isinstance(msg, dict) and "jpeg_b64" in msg and msg["jpeg_b64"]:
            msg = dict(msg)
            msg["jpeg_b64"] = f"<{len(msg['jpeg_b64'])} chars>"
            rec = dict(rec)
            rec["msg"] = msg
        print(json.dumps(rec, ensure_ascii=False))
    return 0


def cmd_grep(path: str, pattern: str) -> int:
    rx = re.compile(pattern)
    for rec in iter_records(path):
        if rec.get("channel") != "events":
            continue
        msg = rec.get("msg") or {}
        t = msg.get("type", "")
        if rx.search(t):
            ts = rec.get("ts_ns")
            ts_s = (int(ts) / 1e9) if ts else 0.0
            print(f"{ts_s:.6f}  {t}  {json.dumps(msg.get('payload') or {}, ensure_ascii=False)}")
    return 0


def cmd_events_since(path: str, since: str) -> int:
    # Accept ISO 8601 OR seconds-since-epoch.
    try:
        cutoff_s = float(since)
    except ValueError:
        try:
            cutoff_s = datetime.fromisoformat(since).timestamp()
        except Exception:
            sys.stderr.write(f"could not parse --events-since: {since!r}\n")
            return 2
    cutoff_ns = int(cutoff_s * 1e9)
    for rec in iter_records(path):
        if rec.get("channel") != "events":
            continue
        ts = rec.get("ts_ns") or 0
        if ts < cutoff_ns:
            continue
        msg = rec.get("msg") or {}
        ts_s = ts / 1e9
        print(f"{ts_s:.6f}  {msg.get('type','')}  {json.dumps(msg.get('payload') or {}, ensure_ascii=False)}")
    return 0


def cmd_track(path: str, track_id: int, out_csv: Optional[str] = None) -> int:
    """Per-tick CSV for one tracked target.

    Joins:
      * track_predictor_step events (filtered by tracked_id)
      * the gimbal/state line nearest in time (so the user can compare
        commanded sp vs. actual cur_pan/tilt)
      * fusion/tracks: az/el of the matched fused-track id

    Output is CSV by default to stdout (or --out PATH). Columns are
    chosen so a future agent can grep, sort, or feed it to pandas.
    """
    # Pass 1: collect predictor events for this id
    pred_rows: List[Dict[str, Any]] = []
    gimbal_samples: List[tuple[int, Dict[str, Any]]] = []
    fused_samples: List[tuple[int, Dict[str, Any]]] = []
    for rec in iter_records(path):
        ch = rec.get("channel")
        ts = int(rec.get("ts_ns") or 0)
        msg = rec.get("msg") or {}
        if ch == "events":
            if msg.get("type") == "track_predictor_step":
                pl = msg.get("payload") or {}
                if int(pl.get("tracked_id", -1)) == int(track_id):
                    pl = dict(pl)
                    pl["ts_ns"] = ts
                    pred_rows.append(pl)
        elif ch == "gimbal/state":
            gimbal_samples.append((ts, msg))
        elif ch == "fusion/tracks":
            fused_samples.append((ts, msg))

    if not pred_rows:
        sys.stderr.write(
            f"no track_predictor_step events for tracked_id={track_id} "
            f"in {path}\n")
        return 1

    # Helpers for nearest-in-time join
    g_ts = [t for t, _ in gimbal_samples]
    g_data = [m for _, m in gimbal_samples]
    f_ts = [t for t, _ in fused_samples]
    f_data = [m for _, m in fused_samples]

    def _nearest(ts_list: List[int], data: List[Any], target: int):
        if not ts_list:
            return None
        # Linear scan is fine; sessions are small.
        best_i = 0
        best_d = abs(ts_list[0] - target)
        for i in range(1, len(ts_list)):
            d = abs(ts_list[i] - target)
            if d < best_d:
                best_d = d
                best_i = i
        return data[best_i]

    # Pass 2: write CSV
    cols = [
        "t_s", "tracked_id",
        "cur_pan", "cur_tilt", "gimbal_dps", "settled", "fresh_fused",
        "obs_world_az", "obs_world_el",
        "world_az", "world_el", "world_az_dot", "world_el_dot",
        "obs_count", "age", "confidence", "lead",
        "shift_az", "shift_el",
        "sp_pan", "sp_tilt",
        # Joined columns
        "gimbal_actual_pan", "gimbal_actual_tilt", "gimbal_target_pan",
        "fused_az_deg", "fused_el_deg", "fused_hits",
    ]
    fh: io.IOBase
    if out_csv:
        fh = open(out_csv, "w", encoding="utf-8", newline="")
    else:
        fh = sys.stdout  # type: ignore[assignment]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    t0_ns = pred_rows[0]["ts_ns"]
    for r in pred_rows:
        ts = int(r["ts_ns"])
        gm = _nearest(g_ts, g_data, ts) or {}
        fm = _nearest(f_ts, f_data, ts) or {}
        # Pick the matching fused track from the fusion snapshot
        fused_az = fused_el = fused_hits = None
        for tr in fm.get("tracks") or []:
            if int(tr.get("id", -1)) == int(track_id):
                fused_az = tr.get("az_deg")
                fused_el = tr.get("el_deg")
                fused_hits = tr.get("hits")
                break
        row = dict(r)
        row["t_s"] = (ts - t0_ns) / 1e9
        row["gimbal_actual_pan"] = gm.get("pan_deg")
        row["gimbal_actual_tilt"] = gm.get("tilt_deg")
        row["gimbal_target_pan"] = gm.get("target_pan_deg")
        row["fused_az_deg"] = fused_az
        row["fused_el_deg"] = fused_el
        row["fused_hits"] = fused_hits
        w.writerow(row)
    if out_csv:
        fh.close()
        sys.stderr.write(f"wrote {out_csv}\n")
    return 0


# ──────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("file", nargs="?", help="JSONL file (default: --latest)")
    g.add_argument("--latest", action="store_true",
                   help="Open the newest recordings/seeker_*.jsonl")
    p.add_argument("--dir", default=DEFAULT_DIR,
                   help=f"Recordings directory (default: {DEFAULT_DIR})")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--summary", action="store_true",
                      help="Session summary (default mode)")
    mode.add_argument("--channels", action="store_true",
                      help="Per-channel message counts only")
    mode.add_argument("--head", type=int, metavar="N",
                      help="Print first N JSONL lines (jpeg payloads trimmed)")
    mode.add_argument("--grep", metavar="REGEX",
                      help="Filter events by type regex")
    mode.add_argument("--events-since", metavar="TS",
                      help="Print events at or after TS (ISO or epoch s)")
    mode.add_argument("--track-id", type=int, metavar="N",
                      help="Per-tick CSV for tracked id N")
    p.add_argument("--out", metavar="PATH",
                   help="(--track-id only) write CSV to PATH instead of stdout")
    args = p.parse_args()

    if args.latest or not args.file:
        path = find_latest(args.dir)
        if path is None:
            sys.stderr.write(
                f"no seeker_*.jsonl files in {args.dir!r}; "
                "record one with the REC button or --auto-record\n")
            return 2
    else:
        path = args.file
    if not os.path.exists(path):
        sys.stderr.write(f"file not found: {path}\n")
        return 2

    if args.channels:
        return cmd_channels(path)
    if args.head is not None:
        return cmd_head(path, int(args.head))
    if args.grep:
        return cmd_grep(path, args.grep)
    if args.events_since:
        return cmd_events_since(path, args.events_since)
    if args.track_id is not None:
        return cmd_track(path, int(args.track_id), out_csv=args.out)
    return cmd_summary(path)


if __name__ == "__main__":
    sys.exit(main())

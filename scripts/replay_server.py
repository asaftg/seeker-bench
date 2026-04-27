"""
Visual replay server.

Streams a JSONL recording back through the existing seeker GUI as if
it were live. The browser opens ``http://localhost:8081/`` and sees
the same dashboard layout, sensors, gimbal motion, and fused tracks
that were captured — at the original timestamps.

No Foxglove, no new viewer, just FastAPI + WebSocket re-emitting
captured frames into the existing static SPA.

URL params on /ws/sensors:
    speed=<float>   playback rate; 1.0 = real-time, 2.0 = double-speed.
                    Defaults to 1.0.
    from=<float>    seconds offset from session start to begin from.
                    Defaults to 0.

Limitations vs. live:
    - The fused-track projections onto thermal/EO panels (bbox_thermal,
      bbox_eo) are NOT rebuilt; the GUI just won't draw them. Raw
      sensor detections + radar boxes still render.
    - Live gimbal commands from the browser are ignored — this is a
      replay, not a sandbox. (Easy follow-up if the use case appears.)

Usage:
    python scripts/replay_server.py --latest
    python scripts/replay_server.py --latest --speed 2.0
    python scripts/replay_server.py recordings/seeker_2026-04-26_18-12.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# Add project root so the static dir resolves the same way as in
# gui/app.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from scripts.replay_inspect import find_latest, iter_records


def _static_dir() -> Path:
    return Path(_ROOT) / "gui" / "static"


# ──────────────────────────────────────────────────────────────
# Wire-envelope builder
# ──────────────────────────────────────────────────────────────
def _empty_thermal() -> Dict[str, Any]:
    return {"connected": False, "frame_id": 0, "timestamp": 0.0,
            "jpeg_b64": None, "width": 0, "height": 0,
            "hfov_deg": 75.0, "vfov_deg": 60.0,
            "zoom_preset": "full", "detections": [], "heat_tracks": []}


def _empty_eo() -> Dict[str, Any]:
    return {"connected": False, "initializing": False, "frame_id": 0,
            "timestamp": 0.0, "jpeg_b64": None, "width": 0, "height": 0,
            "hfov_deg": 11.05, "vfov_deg": 9.23,
            "source_device": None, "detections": []}


def _empty_radar() -> Dict[str, Any]:
    return {"connected": False, "frame_id": 0, "timestamp": 0.0,
            "profile": "", "max_range_m": 50.0, "fov_half_deg": 60.0,
            "num_points": 0, "num_targets": 0, "points": [],
            "targets": [], "detections": []}


def _empty_gimbal() -> Dict[str, Any]:
    return {"pan": 0.0, "tilt": 0.0, "mode": "manual",
            "connected": False, "tracked_target_id": None,
            "target_pan": 0.0, "target_tilt": 0.0, "error": None}


def _gimbal_wire(msg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pan": round(float(msg.get("pan_deg", 0.0)), 2),
        "tilt": round(float(msg.get("tilt_deg", 0.0)), 2),
        "mode": str(msg.get("mode", "manual")),
        "connected": bool(msg.get("connected", False)),
        "tracked_target_id": msg.get("tracked_target_id"),
        "target_pan": round(float(msg.get("target_pan_deg", 0.0)), 2),
        "target_tilt": round(float(msg.get("target_tilt_deg", 0.0)), 2),
        "error": msg.get("error"),
    }


def _eo_wire(msg: Dict[str, Any]) -> Dict[str, Any]:
    """Reshape recorded EO into wire shape (detections need classification dict)."""
    out = dict(msg)
    new_dets = []
    for d in out.get("detections") or []:
        new_dets.append({
            "bbox": d.get("bbox"),
            "track_id": d.get("track_id"),
            "classification": {
                "target_class": d.get("target_class", "unknown"),
                "confidence": d.get("confidence", 0.0),
                "classifier_used": "yolo_eo",
            },
        })
    out["detections"] = new_dets
    return out


def _radar_wire(msg: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(msg)
    # The wire form tags each target with class="radar_detection";
    # the GUI reads this for rendering colour.
    new_targets = []
    for t in out.get("targets") or []:
        t2 = dict(t)
        t2["class"] = "radar_detection"
        new_targets.append(t2)
    out["targets"] = new_targets
    out.setdefault("detections", [])
    return out


def _fused_wire(msg: Dict[str, Any]) -> list:
    """Return the list of fused tracks with the wire-required field names.

    The live wire shape includes bbox_thermal/bbox_eo projections we
    can't easily reconstruct here without the live frame buffers.
    GUI gracefully renders without them; the radar/fused panels still
    show angles + ids correctly.
    """
    out: list = []
    for tr in msg.get("tracks") or []:
        out.append({
            "id": tr["id"],
            "target_class": tr.get("target_class", "unknown"),
            "confidence": tr.get("confidence", 0.0),
            "sensors": tr.get("sensors", []),
            "primary": tr.get("primary", ""),
            "az_deg": tr.get("az_deg", 0.0),
            "el_deg": tr.get("el_deg", 0.0),
            "ang_w_deg": tr.get("ang_w_deg", 0.5),
            "ang_h_deg": tr.get("ang_h_deg", 0.5),
            "hits": tr.get("hits", 1),
            "bbox_thermal": None,
            "bbox_eo": None,
        })
    return out


# ──────────────────────────────────────────────────────────────
# Replay engine — single in-memory ordered timeline
# ──────────────────────────────────────────────────────────────
class _Timeline:
    """All records in (ts_ns, channel, msg) order, plus latest-by-channel."""

    def __init__(self, path: str) -> None:
        self.records: list[tuple[int, str, dict]] = []
        self.first_ts_ns: Optional[int] = None
        self.last_ts_ns: Optional[int] = None
        for rec in iter_records(path):
            ts = rec.get("ts_ns")
            ch = rec.get("channel", "")
            msg = rec.get("msg") or {}
            if ts is None:
                continue
            ts = int(ts)
            self.records.append((ts, ch, msg))
            if self.first_ts_ns is None:
                self.first_ts_ns = ts
            self.last_ts_ns = ts


async def _stream(ws: WebSocket, tl: _Timeline,
                  speed: float, from_offset_s: float) -> None:
    if not tl.records or tl.first_ts_ns is None:
        await ws.send_json({"error": "empty recording"})
        return

    base_ts_ns = tl.first_ts_ns + int(from_offset_s * 1e9)
    # Latest-by-channel snapshots feed the next outgoing envelope.
    latest = {
        "thermal/frame": _empty_thermal(),
        "eo/frame": _empty_eo(),
        "radar/frame": _empty_radar(),
        "gimbal/state": _empty_gimbal(),
        "fusion/tracks": {"tracks": []},
    }

    # Pacing: emit a wire envelope at the rate the underlying record
    # stream produces frames, but advance simulated wall-clock at
    # speed×real. We tick once per record; on every "frame" record
    # we batch up the envelope and send it. Events get folded as a
    # synthetic "event" message in the envelope.
    sim_t0 = asyncio.get_event_loop().time()
    rec_t0_ns = base_ts_ns
    pending_events: list[dict] = []
    for ts, ch, msg in tl.records:
        if ts < base_ts_ns:
            continue
        # Sleep until this record's wall-clock target.
        target_dt = (ts - rec_t0_ns) / 1e9 / max(0.01, speed)
        now_dt = asyncio.get_event_loop().time() - sim_t0
        sleep_s = target_dt - now_dt
        if sleep_s > 0:
            try:
                await asyncio.sleep(sleep_s)
            except asyncio.CancelledError:
                return

        if ch in latest:
            latest[ch] = msg

        if ch == "events":
            pending_events.append(
                {"ts_ns": ts, "type": msg.get("type"),
                 "payload": msg.get("payload")})
            # Don't emit a full envelope on event-only ticks; flush
            # events with the next sensor frame to keep the client
            # rate sane.
            continue

        # On every frame-shaped record, build + send the envelope.
        envelope = {
            "ts": ts / 1e9,
            "thermal": latest["thermal/frame"],
            "eo": _eo_wire(latest["eo/frame"]),
            "radar": _radar_wire(latest["radar/frame"]),
            "fused": _fused_wire(latest["fusion/tracks"]),
            "tracks": [],
            "top_targets": _fused_wire(latest["fusion/tracks"])[:5],
            "main_target_id": latest["gimbal/state"].get("tracked_target_id"),
            "gimbal": _gimbal_wire(latest["gimbal/state"]),
            "recording": False,
            "replay": True,
            "replay_t_s": (ts - tl.first_ts_ns) / 1e9,
            "events": pending_events,
        }
        pending_events = []
        try:
            await ws.send_text(json.dumps(envelope, default=str))
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return

    # Recording exhausted — send a sentinel and drop.
    try:
        await ws.send_json({"replay_done": True})
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
def build_app(jsonl_path: str) -> FastAPI:
    app = FastAPI(title="Seeker-01 Replay", version="0.1.0")
    static = _static_dir()
    if static.exists():
        app.mount("/static", StaticFiles(directory=str(static)), name="static")

    @app.get("/")
    def _root():
        return FileResponse(str(static / "index.html"),
                            headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def _health():
        return {"status": "ok", "replay": True, "file": jsonl_path}

    @app.websocket("/ws/sensors")
    async def _ws(ws: WebSocket) -> None:
        await ws.accept()
        speed = 1.0
        from_offset = 0.0
        try:
            qp = ws.query_params
            speed = float(qp.get("speed", 1.0))
            from_offset = float(qp.get("from", 0.0))
        except Exception:
            pass
        tl = _Timeline(jsonl_path)
        try:
            await _stream(ws, tl, speed=speed, from_offset_s=from_offset)
        except WebSocketDisconnect:
            return

    return app


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="JSONL file (default: --latest)")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--dir", default="recordings")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8081)
    p.add_argument("--speed", type=float, default=1.0,
                   help="(printed for convenience; the live knob is "
                        "the ?speed= query param on /ws/sensors)")
    args = p.parse_args()

    if args.latest or not args.file:
        path = find_latest(args.dir)
        if path is None:
            sys.stderr.write(f"no seeker_*.jsonl in {args.dir!r}\n")
            return 2
    else:
        path = args.file
    if not os.path.exists(path):
        sys.stderr.write(f"file not found: {path}\n")
        return 2

    app = build_app(path)
    print(f"Replay → http://{args.host}:{args.port}/  (file={path})")
    print(f"  use ?speed=2.0 on the WS to fast-forward; ?from=<s> to skip ahead")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

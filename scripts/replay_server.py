"""
Visual replay server.

Streams a JSONL recording back through the existing seeker GUI as if
it were live. The browser opens ``http://localhost:8081/`` and sees
the same dashboard layout, sensors, gimbal motion, and fused tracks
that were captured — at the original timestamps.

The GUI exposes a bottom control bar in replay mode: pause/play, a
seek slider, and a speed dropdown. Commands flow client→server over
the same /ws/sensors WebSocket as JSON `{cmd: ...}` messages.

URL params on /ws/sensors:
    speed=<float>   playback rate; 1.0 = real-time, 2.0 = double-speed.
                    Defaults to 1.0. Live speed changes are sent over
                    the WS as `{cmd:"speed", v:<float>}`.

Limitations vs. live:
    - The fused-track projections onto thermal/EO panels (bbox_thermal,
      bbox_eo) are NOT rebuilt; the GUI just won't draw them. Raw
      sensor detections + radar boxes still render.
    - Live gimbal commands from the browser are ignored — this is a
      replay, not a sandbox.

Usage:
    python scripts/replay_server.py --latest
    python scripts/replay_server.py --latest --speed 2.0
    python scripts/replay_server.py recordings/seeker_2026-04-26_18-12.jsonl
"""
from __future__ import annotations

import argparse
import asyncio
import bisect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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
    """All records in (ts_ns, channel, msg) order, plus precomputed
    seek lookup arrays."""

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
        # rel_ns[i] = offset of record i from session start, in ns.
        # Used by index_at() for O(log N) seek.
        first = self.first_ts_ns or 0
        self.rel_ns: list[int] = [t - first for t, _, _ in self.records]
        last = self.last_ts_ns or first
        self.duration_s: float = max(0.0, (last - first) / 1e9)

    def index_at(self, t_s: float) -> int:
        target_ns = int(max(0.0, t_s) * 1e9)
        return bisect.bisect_left(self.rel_ns, target_ns)


def _empty_latest() -> Dict[str, Any]:
    return {
        "thermal/frame": _empty_thermal(),
        "eo/frame": _empty_eo(),
        "radar/frame": _empty_radar(),
        "gimbal/state": _empty_gimbal(),
        "fusion/tracks": {"tracks": []},
    }


# Browser refresh-rate ceiling. Source records arrive at ~55 Hz at 1x,
# scaling linearly with speed (≈220 Hz at 4x) — far past what the JS
# thread can JSON-parse + base64-decode + render. Cap natural-playback
# emissions here so 2x/4x actually feel faster: each emitted envelope
# carries the latest state across all dropped sub-frames, trading
# per-record fidelity for smooth wall-clock pacing.
_EMIT_INTERVAL_S = 1.0 / 60.0


# ──────────────────────────────────────────────────────────────
# Player session — one per WebSocket connection. Owns playback
# state machine: pause/play/seek/speed driven by inbound WS commands.
# ──────────────────────────────────────────────────────────────
class _PlayerSession:
    def __init__(self, tl: _Timeline, speed: float = 1.0) -> None:
        self.tl = tl
        self.speed: float = max(0.01, float(speed))
        self.current_index: int = 0
        self.playhead_t_s: float = 0.0
        self.is_paused: bool = False
        self.latest: Dict[str, Any] = _empty_latest()
        self.pending_events: list = []
        # wake fires on any command arrival, breaking out of the
        # interruptible asyncio.wait_for(wake) sleep so the loop can
        # pick the command up before emitting the next envelope.
        self.wake = asyncio.Event()
        self.cmd_queue: asyncio.Queue = asyncio.Queue()
        # Wall-clock anchor: at this loop.time(), the playhead is 0 s.
        # Re-anchored by play/seek/speed.
        self.wall_origin: float = 0.0
        self._done_sent = False
        # Last envelope emit timestamp (loop.time()) — used to rate-cap
        # the run loop at _EMIT_INTERVAL_S so 4x doesn't drown the
        # browser. Updated inside _send_envelope so command-driven
        # snapshots also push back the next natural emit.
        self._last_emit_t: float = 0.0

    # ── inbound commands ─────────────────────────────────────
    def submit(self, cmd: dict) -> None:
        try:
            self.cmd_queue.put_nowait(cmd)
        except asyncio.QueueFull:
            return
        self.wake.set()

    async def _apply_cmd(self, cmd: dict, ws: WebSocket) -> None:
        loop = asyncio.get_event_loop()
        name = cmd.get("cmd")
        if name == "pause":
            self.is_paused = True
        elif name == "play":
            self.is_paused = False
            self.wall_origin = loop.time() - self.playhead_t_s / self.speed
        elif name == "seek":
            try:
                t = float(cmd.get("t_s", 0.0))
            except (TypeError, ValueError):
                return
            self._do_seek(t)
        elif name == "speed":
            try:
                v = float(cmd.get("v", 1.0))
            except (TypeError, ValueError):
                return
            v = max(0.01, v)
            self.wall_origin = loop.time() - self.playhead_t_s / v
            self.speed = v
        else:
            return  # unknown command — don't broadcast

        # Broadcast new state. Critical for pause/speed-while-paused
        # because the run loop parks without emitting in those states,
        # so the GUI would otherwise never see the flag flip.
        try:
            await self._send_envelope(ws, seeked=(name == "seek"))
        except (WebSocketDisconnect, RuntimeError):
            pass

    def _do_seek(self, t_s: float) -> None:
        loop = asyncio.get_event_loop()
        duration = self.tl.duration_s
        # Clamp to [0, duration); stay just before end so we don't
        # immediately re-trigger replay_done.
        t = max(0.0, min(t_s, max(0.0, duration - 1e-6)))
        target_idx = self.tl.index_at(t)
        target_idx = min(target_idx, len(self.tl.records))
        # Rebuild `latest` by replaying records [0, target_idx) — the
        # only correct way to compute state-at-time-t after a backward
        # seek, since the running `latest` accumulates forward only.
        self.latest = _empty_latest()
        for i in range(target_idx):
            _ts, ch, msg = self.tl.records[i]
            if ch in self.latest:
                self.latest[ch] = msg
        self.current_index = target_idx
        self.playhead_t_s = t
        # Past events are not replayed — clear the buffer rather than
        # flush stale entries the GUI's append-only event log would
        # treat as fresh.
        self.pending_events = []
        self.wall_origin = loop.time() - t / self.speed
        self._done_sent = False

    # ── outbound envelope ────────────────────────────────────
    async def _send_envelope(self, ws: WebSocket, *, seeked: bool = False) -> None:
        if self.current_index < len(self.tl.records):
            ts = self.tl.records[self.current_index][0]
        elif self.tl.last_ts_ns is not None:
            ts = self.tl.last_ts_ns
        else:
            ts = 0
        envelope = {
            "ts": ts / 1e9,
            "thermal": self.latest["thermal/frame"],
            "eo": _eo_wire(self.latest["eo/frame"]),
            "radar": _radar_wire(self.latest["radar/frame"]),
            "fused": _fused_wire(self.latest["fusion/tracks"]),
            "tracks": [],
            "top_targets": _fused_wire(self.latest["fusion/tracks"])[:5],
            "main_target_id": self.latest["gimbal/state"].get("tracked_target_id"),
            "gimbal": _gimbal_wire(self.latest["gimbal/state"]),
            "recording": False,
            "replay": True,
            # replay_t_s kept as alias of playhead_t_s for the existing
            # _setReplayBadge clock in main.js.
            "replay_t_s": self.playhead_t_s,
            "playhead_t_s": self.playhead_t_s,
            "duration_s": self.tl.duration_s,
            "is_paused": self.is_paused,
            "speed": self.speed,
            "events": self.pending_events,
        }
        if seeked:
            envelope["seeked"] = True
        self.pending_events = []
        await ws.send_text(json.dumps(envelope, default=str))
        # Track last emit so the run loop's rate cap knows when to skip.
        # Cmd-driven snapshots count too — they push back the next
        # natural emit by one interval, avoiding back-to-back floods.
        self._last_emit_t = asyncio.get_event_loop().time()

    # ── main loop ────────────────────────────────────────────
    async def run(self, ws: WebSocket) -> None:
        if not self.tl.records or self.tl.first_ts_ns is None:
            try:
                await ws.send_json({"error": "empty recording"})
            except Exception:
                pass
            return
        loop = asyncio.get_event_loop()
        self.wall_origin = loop.time()  # playhead 0 ↔ now

        while True:
            # Drain any pending commands (pause/play/seek/speed).
            while not self.cmd_queue.empty():
                try:
                    cmd = self.cmd_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                await self._apply_cmd(cmd, ws)

            # End-of-timeline park: emit replay_done once, then wait
            # for a command (typically a backward seek).
            if self.current_index >= len(self.tl.records):
                if not self._done_sent:
                    self.is_paused = True
                    try:
                        await ws.send_json({"replay_done": True})
                    except (WebSocketDisconnect, RuntimeError):
                        return
                    self._done_sent = True
                self.wake.clear()
                try:
                    await self.wake.wait()
                except asyncio.CancelledError:
                    return
                continue

            # Pause park.
            if self.is_paused:
                self.wake.clear()
                try:
                    await self.wake.wait()
                except asyncio.CancelledError:
                    return
                continue

            ts, ch, msg = self.tl.records[self.current_index]
            # Refresh playhead at the top of every tick — keeps a
            # subsequent pause re-anchor from reading a stale value.
            self.playhead_t_s = (ts - self.tl.first_ts_ns) / 1e9

            # Sleep until target wall time, interruptible via wake.
            target_dt = self.playhead_t_s / self.speed
            now_dt = loop.time() - self.wall_origin
            sleep_s = target_dt - now_dt
            if sleep_s > 0:
                self.wake.clear()
                try:
                    await asyncio.wait_for(self.wake.wait(), timeout=sleep_s)
                    # Woken by a command — re-loop without advancing
                    # so the command applies before the next emit.
                    continue
                except asyncio.TimeoutError:
                    pass
                except asyncio.CancelledError:
                    return

            if ch in self.latest:
                self.latest[ch] = msg

            if ch == "events":
                self.pending_events.append({
                    "ts_ns": ts,
                    "type": msg.get("type"),
                    "payload": msg.get("payload"),
                })
                # Don't emit on event-only ticks; flush with the next
                # frame to keep client rate sane.
                self.current_index += 1
                continue

            # Rate cap: drop natural emits that would arrive faster than
            # 60 Hz. The dropped record's data is already folded into
            # `latest` above, so the next non-dropped emit carries it.
            # This is what makes 2x/4x actually feel faster instead of
            # "stuck" — the browser stops choking on the JPEG flood.
            if (loop.time() - self._last_emit_t) < _EMIT_INTERVAL_S:
                self.current_index += 1
                continue

            try:
                await self._send_envelope(ws)
            except (WebSocketDisconnect, RuntimeError):
                return
            except Exception:
                return
            self.current_index += 1


# ──────────────────────────────────────────────────────────────
# Auto-shutdown — when every browser tab closes, exit the server so
# the user doesn't have to manually kill the cmd window between runs.
# ──────────────────────────────────────────────────────────────
_active_conns: int = 0
_shutdown_task: Optional[asyncio.Task] = None
_AUTO_SHUTDOWN_GRACE_S = 2.0


async def _auto_shutdown_after_grace() -> None:
    try:
        await asyncio.sleep(_AUTO_SHUTDOWN_GRACE_S)
    except asyncio.CancelledError:
        return
    if _active_conns == 0:
        # Hard exit beats a graceful uvicorn shutdown here — the server
        # has no persistent state to flush and the user just wants the
        # cmd window to close.
        print("[replay] no clients — shutting down")
        os._exit(0)


def _on_connect() -> None:
    global _active_conns, _shutdown_task
    _active_conns += 1
    if _shutdown_task and not _shutdown_task.done():
        _shutdown_task.cancel()
        _shutdown_task = None


def _on_disconnect() -> None:
    global _active_conns, _shutdown_task
    _active_conns = max(0, _active_conns - 1)
    if _active_conns == 0:
        if _shutdown_task and not _shutdown_task.done():
            _shutdown_task.cancel()
        _shutdown_task = asyncio.create_task(_auto_shutdown_after_grace())


async def _inbound_loop(ws: WebSocket, session: _PlayerSession) -> None:
    while True:
        try:
            txt = await ws.receive_text()
        except (WebSocketDisconnect, RuntimeError):
            return
        try:
            cmd = json.loads(txt)
        except Exception:
            continue
        if isinstance(cmd, dict) and cmd.get("cmd"):
            session.submit(cmd)


# ──────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────
def build_app(jsonl_path: str) -> FastAPI:
    app = FastAPI(title="Seeker-01 Replay", version="0.2.0")
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
        _on_connect()
        speed = 1.0
        try:
            speed = float(ws.query_params.get("speed", 1.0))
        except Exception:
            pass
        tl = _Timeline(jsonl_path)
        session = _PlayerSession(tl, speed=speed)

        run_task = asyncio.create_task(session.run(ws))
        inbound_task = asyncio.create_task(_inbound_loop(ws, session))
        try:
            _done, pending = await asyncio.wait(
                {run_task, inbound_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        except WebSocketDisconnect:
            pass
        finally:
            _on_disconnect()

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
    print(f"Replay -> http://{args.host}:{args.port}/  (file={path})")
    print(f"  pause/seek controls in the bottom bar; ?speed=N still works")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())

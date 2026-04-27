"""Headless harness for the Stage A/B optical-correction A/B test.

Drives the running bench via the GUI's WebSocket — same path as a
human operator clicking the GUI. Programmatically:

    1. Optionally slews the gimbal to a "good scene" anchor pose
    2. Starts a JSONL recording
    3. Draws N synthetic BBs in sequence (gimbal auto-locks each, the
       optical-residual integrator runs against it for SETTLE_S, then
       we clear the lock and move on to the next BB)
    4. Stops + renames the recording

The point: produce two recordings — one with `optical_correction_enabled:
false` (baseline), one with `true` (Stage B engaged) — that I can compare
offline by reading the optical_residual events in each.

Usage (with main.py already running on localhost:8080):
    python scripts/ab_optical_correction.py --label ab_stageB_off
    # ... edit config to flip optical_correction_enabled, restart main.py ...
    python scripts/ab_optical_correction.py --label ab_stageB_on

Defaults choose 4 BB positions across the 640x512 thermal frame plus an
initial slew toward the operator-reported "good scene" near pan=-20,
tilt=4. Override via --bbs / --pan / --tilt.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
from typing import List, Optional, Tuple

import websockets


def set_zoom(host: str, port: int, preset: str) -> bool:
    """Hit the /api/config/thermal REST endpoint to change zoom preset."""
    url = f"http://{host}:{port}/api/config/thermal"
    body = json.dumps({"zoom_preset": preset}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        ok = data.get("zoom_preset") == preset
        print(f"  zoom -> {preset}  (echo: {data})")
        return ok
    except Exception as e:
        print(f"  set_zoom({preset}) failed: {e}")
        return False


DEFAULT_BBS: List[Tuple[int, int, int, int]] = [
    # (x, y, w, h) — 640x512 thermal frame.  Choose 4 positions: top-left,
    # top-right, bottom-left, near-center.  Boxes ~80x80 -> good feature
    # density inside.
    (110, 110, 80, 80),   # top-left quadrant
    (450, 100, 80, 80),   # top-right
    ( 80, 320, 80, 80),   # bottom-left
    (260, 220, 100, 80),  # near-center
]


async def _send(ws, msg: dict) -> None:
    await ws.send(json.dumps(msg))


async def _wait_for_recording_state(ws, want_on: bool, timeout_s: float = 5.0
                                    ) -> bool:
    """Drain incoming WS messages (mostly frame payloads) until we see the
    recording state we asked for. Returns True on success, False on
    timeout. Frames carry `recording: true|false` in the payload.
    """
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
        except asyncio.TimeoutError:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("recording") == want_on:
            return True
    return False


async def run(uri: str, pan_deg: float, tilt_deg: float,
              settle_pre_s: float, settle_per_bb_s: float,
              clear_pause_s: float, bbs: List[Tuple[int, int, int, int]],
              label: str, dry_run: bool,
              zooms: Optional[List[str]] = None,
              host: str = "localhost", http_port: int = 8080) -> int:
    print(f"connecting to {uri}")
    # ping_interval set high; we drain frames in a background task so the
    # websockets library's pings get acknowledged.
    async with websockets.connect(uri, max_size=None,
                                  ping_interval=30,
                                  ping_timeout=20) as ws:
        # Background drainer: reads every incoming frame so the WS
        # connection's pings get acked. Tracks the latest "recording"
        # bool from the payloads.
        state = {"recording": None, "n_drained": 0, "stop": False}

        async def _drain() -> None:
            while not state["stop"]:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    state["stop"] = True
                    return
                try:
                    p = json.loads(raw)
                except Exception:
                    continue
                if isinstance(p, dict):
                    if "recording" in p:
                        state["recording"] = bool(p["recording"])
                    state["n_drained"] += 1

        async def _wait_recording(want: bool, timeout_s: float = 5.0) -> bool:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                if state["recording"] == want:
                    return True
                await asyncio.sleep(0.1)
            return False

        drain_task = asyncio.create_task(_drain())
        # Wait for at least one inbound frame to confirm WS is live.
        t0 = time.monotonic()
        while state["n_drained"] == 0 and time.monotonic() - t0 < 3.0:
            await asyncio.sleep(0.1)
        if state["n_drained"] == 0:
            print("WS open but no frames received in 3s; aborting")
            state["stop"] = True
            await asyncio.gather(drain_task, return_exceptions=True)
            return 2
        print(f"WS connected (drained {state['n_drained']} initial frames)")

        try:
            # Initial gimbal aim: operator-reported good scene.
            print(f"slewing to pan={pan_deg:+.2f} tilt={tilt_deg:+.2f}")
            await _send(ws, {"command": "gimbal_absolute",
                             "pan_deg": float(pan_deg),
                             "tilt_deg": float(tilt_deg)})
            await asyncio.sleep(settle_pre_s)

            if dry_run:
                print("[dry-run] would now start recording, draw BBs, stop")
                return 0

            # Start recording.
            print(f"recording -> ON")
            await _send(ws, {"command": "record", "on": True})
            ok = await _wait_recording(True, timeout_s=5.0)
            if not ok:
                print("warning: didn't observe recording=True (continuing)")
            await asyncio.sleep(0.5)

            # If zooms is None, use whatever is currently set; else
            # iterate through each zoom and draw all BBs at each.
            zoom_loop = zooms if zooms else [None]
            bb_count = 0
            for z in zoom_loop:
                if z is not None:
                    print(f"\n=== Switching to zoom={z} ===")
                    set_zoom(host, http_port, z)
                    await asyncio.sleep(1.5)  # let zoom settle
                for i, (x, y, w, h) in enumerate(bbs, start=1):
                    bb_count += 1
                    label_z = f"zoom={z}" if z else ""
                    print(f"\nBB#{bb_count}{f' [{label_z}]' if z else ''}: "
                          f"bbox=({x},{y},{w},{h})")
                    await _send(ws, {"type": "synthetic_target",
                                     "bbox": [int(x), int(y), int(w), int(h)]})
                    print(f"  settling for {settle_per_bb_s:.1f}s ...")
                    await asyncio.sleep(settle_per_bb_s)
                    print(f"  clearing")
                    await _send(ws, {"command": "track_heat", "heat_id": None})
                    await asyncio.sleep(clear_pause_s)

            # Stop recording with rename.
            print(f"\nrecording -> OFF (rename={label!r})")
            await _send(ws, {"command": "record", "on": False,
                             "rename_to": label})
            ok = await _wait_recording(False, timeout_s=5.0)
            if not ok:
                print("warning: didn't observe recording=False")
            await asyncio.sleep(1.0)
        finally:
            state["stop"] = True
            await asyncio.gather(drain_task, return_exceptions=True)
    print("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://localhost:8080/ws/sensors",
                    help="WS URI of the running main.py GUI")
    ap.add_argument("--label", required=True,
                    help="Recording filename (.jsonl appended)")
    ap.add_argument("--pan", type=float, default=-20.0,
                    help="Initial gimbal pan degrees (default -20)")
    ap.add_argument("--tilt", type=float, default=4.0,
                    help="Initial gimbal tilt degrees (default 4)")
    ap.add_argument("--settle-pre-s", type=float, default=2.5,
                    help="Wait for gimbal to settle to initial pose")
    ap.add_argument("--settle-per-bb-s", type=float, default=8.0,
                    help="Hold each BB this long for residual to converge")
    ap.add_argument("--clear-pause-s", type=float, default=2.0,
                    help="Wait between clearing one BB and drawing next")
    ap.add_argument("--bbs", default=None,
                    help="Override BB list, JSON: '[[x,y,w,h], ...]'")
    ap.add_argument("--zooms", default=None,
                    help="Comma-separated zoom presets to iterate through "
                         "(e.g. 'full,mid,narrow'). Each draws all BBs.")
    ap.add_argument("--http-port", type=int, default=8080,
                    help="HTTP port for /api/config/thermal (default 8080)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Connect, slew, but don't record/draw")
    args = ap.parse_args()
    if args.bbs:
        bbs = [tuple(map(int, b)) for b in json.loads(args.bbs)]
    else:
        bbs = DEFAULT_BBS
    zooms = args.zooms.split(",") if args.zooms else None
    print(f"will draw {len(bbs)} BBs at: {bbs}")
    if zooms:
        print(f"iterating zooms: {zooms} (total {len(bbs)*len(zooms)} BBs)")
    rc = asyncio.run(run(args.uri, args.pan, args.tilt,
                         args.settle_pre_s, args.settle_per_bb_s,
                         args.clear_pause_s, bbs, args.label, args.dry_run,
                         zooms=zooms, http_port=args.http_port))
    return rc


if __name__ == "__main__":
    sys.exit(main())

"""Headless A/B harness for the TRACK-button path on fused-managed tracks.

Drives the live bench via WebSocket the same way a human would:
    1. Slew gimbal to a static-target scene
    2. Wait for the fused tracker to populate the targets list
    3. Pick a target (highest-confidence non-radar by default)
    4. Send the TRACK command (`command: track, track_id: N`)
    5. Hold for SETTLE_S so the closed loop converges
    6. Release TRACK, save the recording

The point: produce two recordings — one with `fused_track_closed_loop:
true` (the default we just shipped), one with `false` (the legacy
predictor-driven setpoint) — that we can compare offline by reading
the optical_residual events to see which gives smaller centering
residual on the same scene.

Usage:
    python scripts/ab_track_button.py --label phase1_track_closed_loop
    # ... edit config to flip the flag, restart main.py ...
    python scripts/ab_track_button.py --label phase1_track_predictor
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any, Dict, List, Optional

import websockets


async def _send(ws, msg: dict) -> None:
    await ws.send(json.dumps(msg))


async def run(uri: str, pan_deg: float, tilt_deg: float,
              settle_pre_s: float, settle_track_s: float,
              label: str, target_class: Optional[str],
              min_conf: float, dry_run: bool) -> int:
    print(f"connecting to {uri}")
    async with websockets.connect(uri, max_size=None,
                                  ping_interval=30,
                                  ping_timeout=20) as ws:
        # Background drainer.
        state: Dict[str, Any] = {"recording": None, "tracks": [],
                                  "n_drained": 0, "stop": False,
                                  "tracked_target_id": None}

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
                if not isinstance(p, dict):
                    continue
                if "recording" in p:
                    state["recording"] = bool(p["recording"])
                if "fused" in p or "tracks" in p:
                    state["tracks"] = p.get("fused") or p.get("tracks") or []
                state["tracked_target_id"] = p.get("tracked_target_id")
                state["n_drained"] += 1

        async def _wait(predicate, timeout_s: float) -> bool:
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout_s:
                if predicate():
                    return True
                await asyncio.sleep(0.1)
            return False

        drain_task = asyncio.create_task(_drain())
        if not await _wait(lambda: state["n_drained"] > 0, 3.0):
            print("WS open but no frames received in 3s; aborting")
            state["stop"] = True
            await asyncio.gather(drain_task, return_exceptions=True)
            return 2
        print(f"WS connected ({state['n_drained']} initial frames)")

        try:
            # 1. Slew to scene.
            print(f"slewing to pan={pan_deg:+.2f} tilt={tilt_deg:+.2f}")
            await _send(ws, {"command": "gimbal_absolute",
                              "pan_deg": float(pan_deg),
                              "tilt_deg": float(tilt_deg)})
            await asyncio.sleep(settle_pre_s)

            # 2. Wait for at least one fused track of the desired class.
            def has_target() -> bool:
                for t in state["tracks"]:
                    if t.get("confidence", 0.0) < min_conf:
                        continue
                    if target_class:
                        cls = t.get("target_class") or t.get("class")
                        if cls != target_class:
                            continue
                    return True
                return False

            print(f"waiting up to 6 s for a target "
                  f"(class={target_class!r}, conf>={min_conf:.2f})...")
            if not await _wait(has_target, 6.0):
                print("no target appeared. Either no detection or wrong "
                      "class/conf threshold. Tracks currently visible:")
                for t in state["tracks"]:
                    print(f"  id={t.get('id')} class={t.get('target_class') or t.get('class')} "
                          f"conf={t.get('confidence', t.get('conf', 0)):.2f} "
                          f"sensors={t.get('sensors')} az={t.get('az_deg', 0):+.2f}")
                state["stop"] = True
                await asyncio.gather(drain_task, return_exceptions=True)
                return 2

            # Pick the highest-confidence target matching the filter.
            best = None
            for t in state["tracks"]:
                if t.get("confidence", 0.0) < min_conf:
                    continue
                if target_class:
                    cls = t.get("target_class") or t.get("class")
                    if cls != target_class:
                        continue
                if best is None or t.get("confidence", 0) > best.get("confidence", 0):
                    best = t
            if best is None:
                print("FAILED to pick a target after wait")
                state["stop"] = True
                await asyncio.gather(drain_task, return_exceptions=True)
                return 2
            tid = int(best.get("id"))
            cls = best.get("target_class") or best.get("class")
            print(f"picked tid={tid} class={cls} conf={best.get('confidence', 0):.2f} "
                  f"az={best.get('az_deg', 0):+.2f} el={best.get('el_deg', 0):+.2f} "
                  f"sensors={best.get('sensors')}")

            if dry_run:
                print("[dry-run] not engaging TRACK or recording")
                return 0

            # 3. Start recording.
            await _send(ws, {"command": "record", "on": True})
            await _wait(lambda: state["recording"] is True, 5.0)
            await asyncio.sleep(0.5)

            # 4. Engage TRACK.
            print(f"engaging TRACK on tid={tid}")
            await _send(ws, {"command": "track", "track_id": tid})
            print(f"  holding for {settle_track_s:.1f} s ...")
            await asyncio.sleep(settle_track_s)

            # 5. Release TRACK.
            print(f"releasing TRACK")
            await _send(ws, {"command": "track", "track_id": None})
            await asyncio.sleep(1.0)

            # 6. Stop recording with rename.
            print(f"recording -> OFF (rename={label!r})")
            await _send(ws, {"command": "record", "on": False,
                              "rename_to": label})
            await _wait(lambda: state["recording"] is False, 5.0)
            await asyncio.sleep(1.0)
        finally:
            state["stop"] = True
            await asyncio.gather(drain_task, return_exceptions=True)
    print("done")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--uri", default="ws://localhost:8080/ws/sensors")
    ap.add_argument("--label", required=True)
    ap.add_argument("--pan", type=float, default=-20.0)
    ap.add_argument("--tilt", type=float, default=4.0)
    ap.add_argument("--settle-pre-s", type=float, default=3.0)
    ap.add_argument("--settle-track-s", type=float, default=12.0)
    ap.add_argument("--target-class", default="vehicle",
                    help="Filter targets by class (default 'vehicle'); "
                         "empty/none = any class")
    ap.add_argument("--min-conf", type=float, default=0.3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cls = args.target_class if args.target_class not in ("", "none") else None
    rc = asyncio.run(run(args.uri, args.pan, args.tilt,
                         args.settle_pre_s, args.settle_track_s,
                         args.label, cls, args.min_conf, args.dry_run))
    return rc


if __name__ == "__main__":
    sys.exit(main())

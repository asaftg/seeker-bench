"""Watch the WS feed and report each track-state transition.

For each fused track ID, record (sensors_set, target_class). Whenever
either changes — new ID born, sensors gain/lose a sensor, class
promoted — print a one-line event with timestamp. Used to verify
Phase 2 radar fusion behaviour live.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Dict, Tuple

import websockets


def _summarize(t: dict) -> Tuple[str, str]:
    sensors = tuple(sorted(t.get("sensors") or []))
    cls = t.get("target_class") or "?"
    return sensors, cls


async def main(duration_s: float = 300.0) -> None:
    state: Dict[int, Tuple[Tuple[str, ...], str]] = {}
    t_start = time.time()
    n_frames = 0
    n_radar_targets_seen = 0
    radar_only_ids_seen = set()
    promotions_seen = []  # (id, from_cls, to_cls)
    class_locked_after_radar_only = []  # (id, locked_class)

    print(f"[watch] connecting … duration={duration_s:.0f}s")
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors") as ws:
        while time.time() - t_start < duration_s:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=2.0)
            except asyncio.TimeoutError:
                continue
            n_frames += 1
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            ts_rel = time.time() - t_start
            radar = msg.get("radar") or {}
            n_radar_targets_seen += len(radar.get("targets") or [])

            fused = msg.get("fused") or []
            seen_now = set()
            for t in fused:
                tid = int(t.get("id"))
                seen_now.add(tid)
                key = _summarize(t)
                prev = state.get(tid)
                if prev != key:
                    if prev is None:
                        print(f"[{ts_rel:6.1f}s] BORN     #{tid:3d} sensors={key[0]} class={key[1]}")
                        if key[0] == ("radar",):
                            radar_only_ids_seen.add(tid)
                    else:
                        # Class promotion?
                        if prev[1] != key[1]:
                            print(f"[{ts_rel:6.1f}s] CLASS    #{tid:3d} {prev[1]} -> {key[1]}  sensors={key[0]}")
                            promotions_seen.append((tid, prev[1], key[1]))
                        # Sensor set change?
                        if prev[0] != key[0]:
                            added = set(key[0]) - set(prev[0])
                            dropped = set(prev[0]) - set(key[0])
                            label = []
                            if added:
                                label.append(f"+{','.join(sorted(added))}")
                            if dropped:
                                label.append(f"-{','.join(sorted(dropped))}")
                            print(f"[{ts_rel:6.1f}s] SENSORS  #{tid:3d} {' '.join(label):<14} now={key[0]} class={key[1]}")
                            # Class locked? (track went radar-only AFTER having a real class)
                            if key[0] == ("radar",) and key[1] not in ("radar_target", "?"):
                                class_locked_after_radar_only.append((tid, key[1]))
                                # ASCII only — Windows cp1252 console can't encode box-drawing chars.
                                print(f"          -> class LOCKED at '{key[1]}' while only radar sees it [OK]")
                    state[tid] = key
            # Track death
            for tid in list(state.keys()):
                if tid not in seen_now:
                    prev = state.pop(tid)
                    print(f"[{ts_rel:6.1f}s] DROPPED  #{tid:3d} last sensors={prev[0]} class={prev[1]}")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"frames: {n_frames}")
    print(f"radar_targets_seen total: {n_radar_targets_seen}")
    print(f"radar-only fused tracks born: {len(radar_only_ids_seen)} ids={sorted(radar_only_ids_seen)}")
    print(f"class promotions: {len(promotions_seen)}")
    for p in promotions_seen[:10]:
        print(f"  #{p[0]}: {p[1]} -> {p[2]}")
    print(f"class-locked-after-radar-only: {len(class_locked_after_radar_only)}")
    for c in class_locked_after_radar_only[:10]:
        print(f"  #{c[0]}: locked at '{c[1]}'")


if __name__ == "__main__":
    import sys
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 300.0
    asyncio.run(main(dur))

"""Minimal WS probe — connect, grab one frame, print the structural keys
(stripping the bulky jpeg_b64 fields). Used for quick smoke-tests after
fusion changes."""
from __future__ import annotations

import asyncio
import json

import websockets


async def main() -> None:
    async with websockets.connect("ws://127.0.0.1:8080/ws/sensors") as ws:
        for _ in range(3):  # let a few frames flow so radar/fusion warms up
            raw = await ws.recv()
        msg = json.loads(raw)

    # Strip heavy frame payloads.
    for sec in ("thermal", "eo"):
        s = msg.get(sec)
        if isinstance(s, dict):
            s.pop("jpeg_b64", None)

    print("== top keys ==")
    print(sorted(msg.keys()))
    print("\n== extrinsic ==")
    print(msg.get("extrinsic"))
    print("\n== radar ==")
    r = msg.get("radar") or {}
    print({k: v for k, v in r.items() if k != "targets"})
    print(f"  targets: n={len(r.get('targets') or [])}")
    print("\n== fused ==")
    fused = msg.get("fused") or []
    print(f"n={len(fused)}")
    for f in fused[:5]:
        keep = {k: f.get(k) for k in
                ("id", "target_class", "sensors", "primary",
                 "confidence", "az_deg", "el_deg")}
        print(f"  {keep}")
    print("\n== top_targets ==")
    tt = msg.get("top_targets") or []
    print(f"n={len(tt)}")
    for t in tt[:5]:
        keep = {k: t.get(k) for k in
                ("id", "target_class", "sensors", "confidence")}
        print(f"  {keep}")


if __name__ == "__main__":
    asyncio.run(main())

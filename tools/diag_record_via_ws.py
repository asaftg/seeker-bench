"""Trigger a 10-second record cycle on the running seeker app via WebSocket.

Sends the same `{"command":"record","on":true}` the GUI button sends, waits
N seconds, then sends `{"command":"record","on":false}`. This is the only
path that creates the paired `_radar.bin` from DCA1000 raw ADC.

Usage:
    python tools/diag_record_via_ws.py            # 10 sec record
    python tools/diag_record_via_ws.py --seconds 5
    python tools/diag_record_via_ws.py --rename channelCfg_15_15
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("websockets package not available — install: pip install websockets", file=sys.stderr)
    sys.exit(1)


async def run(seconds: float, rename: str | None) -> int:
    url = "ws://127.0.0.1:8080/ws/sensors"
    print(f"connecting to {url}...")
    try:
        ws = await asyncio.wait_for(websockets.connect(url), timeout=10)
    except Exception as e:
        print(f"FATAL: ws connect failed: {e}", file=sys.stderr)
        return 1

    print("→ record ON")
    await ws.send(json.dumps({"command": "record", "on": True}))

    print(f"recording for {seconds:.1f} seconds...")
    await asyncio.sleep(seconds)

    print("→ record OFF" + (f" (rename to '{rename}')" if rename else ""))
    cmd = {"command": "record", "on": False}
    if rename:
        cmd["rename_to"] = rename
    await ws.send(json.dumps(cmd))

    # Drain a few messages to ensure server processed the off command
    try:
        for _ in range(5):
            await asyncio.wait_for(ws.recv(), timeout=1.0)
    except Exception:
        pass

    await ws.close()
    print("done.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--rename", default=None,
                    help="If set, rename the JSONL to <name>.jsonl after stop")
    args = ap.parse_args()
    return asyncio.run(run(args.seconds, args.rename))


if __name__ == "__main__":
    sys.exit(main())

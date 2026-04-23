"""Verify radar_to_wire() emits valid JSON with points — Phase F2 smoke test.

Runs RadarManager briefly, grabs the latest RadarFrame off the bus, and
checks that sensor_bridge.radar_to_wire serializes cleanly (no NaN, points
present). This is what the GUI will receive over WS.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import load_config  # noqa: E402
from common.frame_bus import BUS  # noqa: E402
from common.frames import Topic  # noqa: E402
from common.logging_setup import configure  # noqa: E402
from gui.sensor_bridge import radar_to_wire  # noqa: E402
from radar.radar_manager import RadarManager  # noqa: E402


def main() -> int:
    configure(level="WARNING")
    cfg = load_config()
    rcfg = cfg["radar"]

    rm = RadarManager(
        cli_port=rcfg["cli_port"], data_port=rcfg["data_port"],
        cfg_path=rcfg["cfg_path"],
        cli_baud=int(rcfg.get("cli_baud", 115200)),
        data_baud=int(rcfg.get("data_baud", 3_125_000)),
        snr_min_db=float(rcfg.get("snr_min_db", 12.0)),
        max_range_m=float(rcfg.get("max_range_m", 50.0)),
        profile_name=str(rcfg.get("profile_name", "awr2944p_ddm")),
    )
    rm.start()

    try:
        rf = None
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            latest = BUS.get_latest(Topic.RADAR)
            if latest is not None and latest.connected and latest.num_points > 0:
                rf = latest
                break
            time.sleep(0.1)
        if rf is None:
            print("FAIL: no connected RadarFrame with points within 6s")
            return 1

        wire = radar_to_wire(rf)
        # Round-trip via strict JSON (no NaN/Inf allowed).
        s = json.dumps(wire, allow_nan=False)
        print(f"OK  frame_id={wire['frame_id']}  points={len(wire['points'])}  "
              f"targets={len(wire['targets'])}  payload_bytes={len(s)}")
        if wire["points"]:
            p = wire["points"][0]
            print(f"    sample point: x={p['x']}m y={p['y']}m z={p['z']}m "
                  f"v={p['v']}m/s snr={p['snr']}dB")
        return 0
    finally:
        rm.stop()


if __name__ == "__main__":
    sys.exit(main())

"""Run RadarManager for N seconds and print what it publishes on the bus.

Sanity-check end-to-end: connect → parse TLVs → publish RadarFrame.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import load_config  # noqa: E402
from common.frame_bus import BUS  # noqa: E402
from common.frames import Topic  # noqa: E402
from common.logging_setup import configure  # noqa: E402
from radar.radar_manager import RadarManager  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=8.0)
    args = ap.parse_args()

    configure(level="INFO")
    cfg = load_config()
    rcfg = cfg["radar"]

    rm = RadarManager(
        cli_port=rcfg["cli_port"],
        data_port=rcfg["data_port"],
        cfg_path=rcfg["cfg_path"],
        cli_baud=int(rcfg.get("cli_baud", 115200)),
        data_baud=int(rcfg.get("data_baud", 3_125_000)),
        snr_min_db=float(rcfg.get("snr_min_db", 12.0)),
        max_range_m=float(rcfg.get("max_range_m", 50.0)),
        profile_name=str(rcfg.get("profile_name", "awr2944p_ddm")),
    )
    rm.start()
    deadline = time.monotonic() + args.seconds
    last_seen_fid = -1
    connected_seen = False
    n_frames = 0
    n_pts_max = 0
    n_tgt_max = 0
    try:
        while time.monotonic() < deadline:
            rf = BUS.get_latest(Topic.RADAR)
            if rf is not None and rf.frame_id != last_seen_fid:
                last_seen_fid = rf.frame_id
                if rf.connected:
                    connected_seen = True
                    n_frames += 1
                    n_pts_max = max(n_pts_max, rf.num_points)
                    n_tgt_max = max(n_tgt_max, rf.num_targets)
                    print(f"  fid={rf.frame_id:>4}  connected=True  "
                          f"pts={rf.num_points:>3}  targets={rf.num_targets}")
                else:
                    print(f"  fid={rf.frame_id:>4}  connected=False (sentinel)")
            time.sleep(0.05)
    finally:
        rm.stop()

    print("---")
    print(f"connected_seen={connected_seen}  frames={n_frames}  "
          f"max_pts={n_pts_max}  max_targets={n_tgt_max}")
    return 0 if connected_seen and n_frames > 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Push a TI mmw_demo .cfg profile to the radar CLI UART manually.

Useful for first-light sanity without needing to spin up the whole
Seeker app. Used in Phase C of Ticket 5a to verify that a stock .cfg
is accepted by the firmware and causes the chip to stream TLVs.

Example:
    python scripts/radar_send_cfg.py \\
        --port COM6 \\
        --cfg "C:/ti/mmwave_mcuplus_sdk_04_07_02_01/ti/demo/awr2x44P/mmw_ddm/profiles/awr2944P/profile_3d_3Azim_1ElevTx_DDMA_awr2944P_highRange.cfg"
"""
from __future__ import annotations

import argparse
import sys

import serial

# Make sure ``radar.cfg_sender`` resolves when running this script
# directly from scripts/ — the repo uses flat-package imports.
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radar.cfg_sender import send_cfg  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="CLI UART port (e.g. COM6)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cfg", required=True, help="Path to a mmw_demo .cfg file")
    args = ap.parse_args()

    def _print(cmd: str, resp: str) -> None:
        resp_preview = " ".join(resp.split())[:80]
        print(f"  {cmd}\n      -> {resp_preview}")

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as e:
        print(f"[FATAL] cannot open {args.port}: {e}")
        return 2

    with ser:
        # Always send sensorStop first — if the chip is already
        # streaming from a prior run, CLI commands go un-parsed.
        ser.write(b"sensorStop\n")
        ser.flush()
        print("Sent sensorStop (preflight)")
        import time as _t; _t.sleep(0.1)
        ser.reset_input_buffer()
        print(f"Pushing {args.cfg} ...")
        send_cfg(ser, args.cfg, on_line=_print)
        print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

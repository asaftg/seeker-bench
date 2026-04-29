"""Bypass RadarManager entirely — push cfg once, read raw bytes for N seconds.

Tests whether the chip emits TLV continuously when *nothing* on the host
is reconnecting / kicking it. If this script shows continuous TLV bytes
arriving over the full duration, then the bug is in RadarManager's
reconnect loop. If it shows the same burst-then-die pattern, the chip
itself is broken and a reflash is the only fix.

Usage:
    python scripts/radar_raw_test.py [--duration 30]

Reads counters every second and prints bytes-per-second on COM10.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import serial

# Make repo importable when run from the seeker_bench/ root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radar.cfg_sender import send_cfg

CLI_PORT = "COM11"
DATA_PORT = "COM10"
CLI_BAUD = 115200
DATA_BAUD = 3125000
CFG_PATH = "radar/cfg/awr2944P_unified.cfg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds to read after pushing cfg")
    args = ap.parse_args()

    print(f"[1/3] pushing {CFG_PATH} on {CLI_PORT}@{CLI_BAUD}…")
    with serial.Serial(CLI_PORT, CLI_BAUD, timeout=0.5) as cli:
        responses = send_cfg(cli, CFG_PATH)
    last_resp = (responses[-1] if responses else "").strip().splitlines()[-1] \
        if responses else ""
    print(f"      cfg-push final response: {last_resp!r}")
    if "Error" in last_resp and "Invalid" in last_resp:
        print("      (chip wasn't in INIT — sending sensorStart 0 as recovery)")
        with serial.Serial(CLI_PORT, CLI_BAUD, timeout=0.5) as cli:
            cli.write(b"sensorStart 0\n")
            cli.flush()
            time.sleep(0.5)
            print("      sensorStart 0 ACK:",
                  cli.read(cli.in_waiting or 0).decode("ascii", errors="replace").strip())

    print(f"[2/3] opening data port {DATA_PORT}@{DATA_BAUD}, reading "
          f"{args.duration:.0f} s…")
    bytes_total = 0
    bytes_per_sec_history = []
    last_byte_t = time.monotonic()

    with serial.Serial(DATA_PORT, DATA_BAUD, timeout=0.2) as data:
        deadline = time.monotonic() + args.duration
        last_print = time.monotonic()
        bytes_this_sec = 0
        while time.monotonic() < deadline:
            chunk = data.read(8192)
            now = time.monotonic()
            if chunk:
                bytes_total += len(chunk)
                bytes_this_sec += len(chunk)
                last_byte_t = now
            if now - last_print >= 1.0:
                age_s = now - last_byte_t
                marker = "" if bytes_this_sec > 0 else f"  (silent for {age_s:.1f}s)"
                print(f"  t+{int(now - (deadline - args.duration)):3d}s  "
                      f"{bytes_this_sec:>10d} B/s  total={bytes_total:>10d}{marker}")
                bytes_per_sec_history.append(bytes_this_sec)
                bytes_this_sec = 0
                last_print = now

    print(f"[3/3] done. total={bytes_total:,} B over {args.duration:.0f}s")
    if not bytes_per_sec_history:
        print("      (no measurements collected)")
        return
    streaming_secs = sum(1 for x in bytes_per_sec_history if x > 1000)
    silent_secs = sum(1 for x in bytes_per_sec_history if x <= 1000)
    print(f"      streaming seconds: {streaming_secs}/{len(bytes_per_sec_history)}")
    print(f"      silent seconds:    {silent_secs}/{len(bytes_per_sec_history)}")
    if silent_secs > streaming_secs:
        print("\n      VERDICT: chip stops streaming during the run.")
        print("      Bug is chip-side (firmware / flash). Reflash mmw_demoDDM via UniFlash.")
    elif streaming_secs == len(bytes_per_sec_history):
        print("\n      VERDICT: chip streams continuously when not interrupted.")
        print("      Bug was in RadarManager's reconnect loop.")
    else:
        print("\n      VERDICT: intermittent. Chip emits bursts; firmware likely flaky.")


if __name__ == "__main__":
    main()

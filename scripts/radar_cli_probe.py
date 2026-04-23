"""Interactive CLI probe: send a single command and print the full response.

Used to debug firmware quirks without going through the full cfg push.
"""
from __future__ import annotations

import argparse
import sys
import time

import serial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--cmd", required=True, help="CLI command to send")
    ap.add_argument("--wait-s", type=float, default=1.5)
    args = ap.parse_args()

    with serial.Serial(args.port, args.baud, timeout=0.2) as ser:
        ser.reset_input_buffer()
        ser.write((args.cmd + "\n").encode("ascii"))
        ser.flush()
        deadline = time.monotonic() + args.wait_s
        buf = bytearray()
        while time.monotonic() < deadline:
            n = ser.in_waiting
            if n:
                buf.extend(ser.read(n))
            else:
                time.sleep(0.02)
    sys.stdout.buffer.write(buf.decode("ascii", errors="replace").encode("utf-8"))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())

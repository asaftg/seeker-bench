"""Scan a data port for TLV magic word across a set of baud rates."""
from __future__ import annotations

import argparse
import sys
import time

import serial

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--listen-s", type=float, default=2.0)
    ap.add_argument("--bauds", nargs="*", type=int,
                    default=[892857, 921600, 1250000, 2000000, 3125000, 3000000])
    args = ap.parse_args()

    for baud in args.bauds:
        try:
            s = serial.Serial(args.port, baud, timeout=0.05)
        except serial.SerialException as e:
            print(f"{baud:>10}: open failed: {e}")
            continue
        deadline = time.monotonic() + args.listen_s
        buf = bytearray()
        with s:
            while time.monotonic() < deadline:
                n = s.in_waiting
                if n:
                    buf.extend(s.read(n))
                else:
                    time.sleep(0.01)
        idx = bytes(buf).find(MAGIC)
        tag = f"MAGIC@{idx}" if idx >= 0 else "no-magic"
        print(f"{baud:>10}: {len(buf):>6}B  {tag}  head={bytes(buf[:16]).hex()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

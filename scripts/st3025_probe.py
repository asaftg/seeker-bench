"""Waveshare ST-series bus-servo probe.

Pings each requested ID and prints raw position, voltage, temperature,
and the status/error byte. Useful for:
    - confirming both servos are alive on the bus
    - finding zero_raw for the calibration block (rotate the bracket
      to system 0° by hand and read raw position)
    - monitoring temperature during stress runs

Read-only. Safe to run alongside main.py only if main.py is NOT also
using the same port — half-duplex bus collisions otherwise.

Usage:
    python scripts/st3025_probe.py                 # probe IDs 1 and 2 once
    python scripts/st3025_probe.py --ids 1,2,3
    python scripts/st3025_probe.py --watch         # continuous, 5 Hz
    python scripts/st3025_probe.py --port COM7
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gimbal.bus_servo_calibration import UNITS_PER_DEG
from gimbal.bus_servo_driver import BusServoDriver


def _parse_ids(s: str) -> list[int]:
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _print_row(servo_id: int, drv: BusServoDriver) -> None:
    raw = drv.read_position(servo_id)
    status = drv.read_status(servo_id)
    if raw is None:
        deg_str = "  --  "
    else:
        deg = raw / UNITS_PER_DEG
        deg_str = f"{deg:6.2f}"
    if status is None:
        v_str = "--"; t_str = "--"; e_str = "--"
    else:
        v_str = f"{status['voltage_v']:4.1f}"
        t_str = f"{status['temperature_c']:3d}"
        e_str = f"0x{status['status']:02X}"
    raw_str = "----" if raw is None else f"{raw:4d}"
    print(f"  id={servo_id:3d}  raw={raw_str}  deg={deg_str}  "
          f"V={v_str}  T={t_str}°C  err={e_str}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="1,2",
                    help="Comma-separated servo IDs to probe (default 1,2).")
    ap.add_argument("--watch", action="store_true",
                    help="Loop at --rate Hz until Ctrl-C.")
    ap.add_argument("--rate", type=float, default=5.0,
                    help="Watch-mode poll rate (Hz, default 5).")
    args = ap.parse_args(argv)

    ids = _parse_ids(args.ids)
    if not ids:
        print("ERROR: no IDs to probe (parse --ids failed).")
        return 2

    drv = BusServoDriver(port=args.port, baud=args.baud)
    if not drv.open():
        print("ERROR: could not open the bus-servo adapter — is it plugged in "
              "and the DIP switch on position A?")
        return 1

    try:
        if not args.watch:
            print(f"Probing IDs {ids} on {drv.port}:")
            for sid in ids:
                _print_row(sid, drv)
            return 0

        period = 1.0 / max(0.5, args.rate)
        try:
            while True:
                ts = time.strftime("%H:%M:%S")
                print(f"[{ts}]")
                for sid in ids:
                    _print_row(sid, drv)
                time.sleep(period)
        except KeyboardInterrupt:
            print()
            return 0
    finally:
        drv.close()


if __name__ == "__main__":
    sys.exit(main())

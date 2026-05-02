"""One-shot helper to reassign a Waveshare ST-series servo's bus ID.

Both ST3025 servos ship with factory ID = 1, and the bench gimbal needs
ID 1 (tilt) and ID 2 (pan) on the same daisy-chain. This script reads
the current ID, writes the requested new ID, and pings the new ID to
confirm.

Workflow:
    1. Power ONLY the servo you want to re-flash. If both are on the
       bus and both have ID 1, every command gets two replies and the
       protocol can't disambiguate.
    2. Connect the Waveshare Bus Servo Adapter (A) to the host. Make
       sure the DIP switch is on position A (UART–SERVO).
    3. Run:
           python scripts/st3025_set_id.py --new-id 2
       (omit --port to auto-detect; pass --port COMn to pin it)
    4. Power down. Wire the second servo. It still has ID 1. Power on
       and verify with scripts/st3025_probe.py.

Defaults: assumes the servo currently has ID = 1 (factory). If you're
re-flashing a servo that's already been reassigned, pass --old-id N.
"""
from __future__ import annotations

import argparse
import sys

# allow running from repo root: `python scripts/st3025_set_id.py`
import os
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from gimbal.bus_servo_driver import BusServoDriver


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", default=None,
                    help="Serial device (e.g. COM7). Auto-detect if omitted.")
    ap.add_argument("--baud", type=int, default=1_000_000,
                    help="Baud rate (default 1_000_000).")
    ap.add_argument("--old-id", type=int, default=1,
                    help="Current servo ID (default 1, factory).")
    ap.add_argument("--new-id", type=int, required=True,
                    help="Target servo ID (1..253).")
    args = ap.parse_args(argv)

    if not (1 <= args.new_id <= 253):
        print(f"ERROR: --new-id must be in [1, 253] (got {args.new_id})")
        return 2
    if args.new_id == args.old_id:
        print(f"ERROR: --new-id must differ from --old-id (both {args.old_id})")
        return 2

    drv = BusServoDriver(port=args.port, baud=args.baud)
    if not drv.open():
        print("ERROR: could not open the bus-servo adapter — is it plugged in "
              "and the DIP switch on position A?")
        return 1

    try:
        if not drv.ping(args.old_id):
            print(f"ERROR: no servo answered to ID {args.old_id}. Is the "
                  "servo powered? Is it the only one on the bus?")
            return 1
        print(f"Found servo at ID {args.old_id}; writing new ID {args.new_id}…")
        if not drv.set_id(args.old_id, args.new_id):
            print("ERROR: write failed mid-sequence. Servo may be in an "
                  "intermediate state — power-cycle and retry.")
            return 1
        if not drv.ping(args.new_id):
            print(f"ERROR: servo did not answer at new ID {args.new_id}. "
                  "Power-cycle and probe.")
            return 1
        print(f"OK: servo now responds at ID {args.new_id}.")
        return 0
    finally:
        drv.close()


if __name__ == "__main__":
    sys.exit(main())

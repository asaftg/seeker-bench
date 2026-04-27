"""Pololu Maestro diagnostic — what does the Maestro itself think is
happening on the servo channels?

Polls (via the Pololu compact protocol over the Command Port):
    - getErrors  (0xA1)        the Maestro error register
    - getPosition (0x90 ch)    current commanded pulse width per channel
    - getMovingState (0x93)    1 if any servo is internally ramping

Notes on what is and is NOT available:
    - Servo *actual* position: NOT readable. The Maestro only knows what
      pulse it's commanding, not where the servo physically is. Hobby
      servos don't expose position feedback.
    - Servo voltage: NOT readable on the Micro Maestro 6. (The 12/18/24
      variants have unused analog inputs that *could* be wired to a
      voltage divider on the servo rail, but the 6 has none.)
    - Servo current: NOT readable, period. There's no current sensor on
      any Maestro variant.
    - Channel settings (min/max pulse, speed, accel, period, mode):
      stored in EEPROM, only readable through the Pololu Maestro Control
      Center GUI or USC SDK — not over the compact protocol.

What the script DOES tell you:
    - Whether the Maestro is throwing serial / protocol / script errors
      that the gimbal_manager has been silently ignoring
    - Whether the position the Maestro is commanding matches what we
      think it is — discrepancy = pulse-clamp on min/max
    - Whether the Maestro's internal speed/accel ramp is active (which
      it shouldn't be — gimbal_manager does its own slew limiting)

Usage:
    python scripts/maestro_diag.py             # poll for 10 s
    python scripts/maestro_diag.py --duration 30 --interval 0.1
    python scripts/maestro_diag.py --port COM4

This is read-only — it does not write servo targets. Safe to run
alongside main.py if main.py is using the Maestro on the SAME port,
because the Maestro's Command Port is a single-channel USB CDC and
two writers will fight. So: use either main.py OR this script, not
both at once.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from typing import List, Optional, Tuple

try:
    import serial
    from serial.tools import list_ports
except Exception as e:
    sys.stderr.write(f"pyserial missing: {e}\n")
    sys.exit(2)


POLOLU_VID = 0x1FFB


# ── Error bits (Pololu compact protocol getErrors) ────────────────
ERROR_BIT_LABELS = {
    0: "Serial Signal Error",
    1: "Serial Overrun Error",
    2: "Serial RX Buffer Full",
    3: "Serial CRC Error",
    4: "Serial Protocol Error",
    5: "Serial Timeout",
    6: "Script Stack Error",
    7: "Script Call Stack Error",
    8: "Script Program Counter Error",
}


def find_command_port() -> Optional[str]:
    primary, secondary = [], []
    for p in list_ports.comports():
        if getattr(p, "vid", None) != POLOLU_VID:
            continue
        desc = (p.description or "") + " " + (p.product or "")
        if "Command Port" in desc:
            primary.append((p.device, desc.strip()))
        else:
            secondary.append((p.device, desc.strip()))
    hits = primary + secondary
    return hits[0][0] if hits else None


def get_position_us(ser: serial.Serial, ch: int) -> Optional[float]:
    """Get current commanded pulse width (microseconds) on channel ch."""
    try:
        ser.reset_input_buffer()
        ser.write(bytes([0x90, int(ch) & 0x7F]))
        raw = ser.read(2)
        if len(raw) != 2:
            return None
        qus = struct.unpack("<H", raw)[0]   # quarter-microseconds
        return qus / 4.0
    except Exception:
        return None


def get_errors(ser: serial.Serial) -> Optional[int]:
    """Read and clear the Maestro error register (16-bit)."""
    try:
        ser.reset_input_buffer()
        ser.write(bytes([0xA1]))
        raw = ser.read(2)
        if len(raw) != 2:
            return None
        return struct.unpack("<H", raw)[0]
    except Exception:
        return None


def get_moving_state(ser: serial.Serial) -> Optional[int]:
    """Returns 1 if any servo is internally ramping, 0 otherwise."""
    try:
        ser.reset_input_buffer()
        ser.write(bytes([0x93]))
        raw = ser.read(1)
        if len(raw) != 1:
            return None
        return raw[0]
    except Exception:
        return None


def explain_errors(bits: int) -> str:
    if bits == 0:
        return "(none)"
    names = []
    for b, label in ERROR_BIT_LABELS.items():
        if bits & (1 << b):
            names.append(label)
    return ", ".join(names) if names else f"unknown bits 0x{bits:04x}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None,
                    help="Maestro COM port (auto-detect if omitted)")
    ap.add_argument("--channels", type=int, nargs="+", default=[0, 1],
                    help="Channels to poll (default 0,1 = pan+tilt)")
    ap.add_argument("--duration", type=float, default=10.0,
                    help="Total seconds to poll (default 10)")
    ap.add_argument("--interval", type=float, default=0.2,
                    help="Poll interval in seconds (default 0.2 = 5 Hz)")
    args = ap.parse_args()

    port = args.port or find_command_port()
    if port is None:
        print("No Pololu Maestro Command Port found.")
        print("Plug it in and re-run, or pass --port COMx.")
        return 2
    print(f"Maestro: opening {port}")
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.2,
                            write_timeout=0.5)
    except Exception as e:
        print(f"open failed: {e}")
        return 2

    # First read of errors — this clears any pre-existing latched bits
    initial_errors = get_errors(ser)
    print(f"Initial error register (cleared on read): 0x{initial_errors or 0:04x}"
          f" — {explain_errors(initial_errors or 0)}")

    print(f"\nPolling {args.channels} every {args.interval}s "
          f"for {args.duration}s. Watching for:")
    print("  - error register bits (anything non-zero is bad)")
    print("  - pulse-position changes (should match what gimbal_manager commands)")
    print("  - moving-state == 1 (means Maestro is internally rate-limiting "
          "us — we don't want this)")
    print()
    print(f"{'t_s':>6s}  {'errors':>8s}  {'moving':>6s}  "
          + "  ".join(f"ch{c}_us".rjust(8) for c in args.channels))

    t_start = time.monotonic()
    last_pos = {c: None for c in args.channels}
    moving_seen = False
    error_seen_after_start = False

    while time.monotonic() - t_start < args.duration:
        t_rel = time.monotonic() - t_start
        errs = get_errors(ser) or 0
        if errs and t_rel > 0.5:
            error_seen_after_start = True
        moving = get_moving_state(ser) or 0
        if moving:
            moving_seen = True
        positions: List[Optional[float]] = []
        for c in args.channels:
            positions.append(get_position_us(ser, c))
        # Print only on change OR every ~2s
        changed = any(
            p is not None and last_pos[c] is not None
            and abs(p - last_pos[c]) > 0.5
            for c, p in zip(args.channels, positions))
        if changed or int(t_rel * 5) % 10 == 0:
            print(f"{t_rel:6.2f}  0x{errs:04x}  {moving:6d}  "
                  + "  ".join(f"{p:8.1f}" if p is not None else "    None"
                              for p in positions))
        for c, p in zip(args.channels, positions):
            if p is not None:
                last_pos[c] = p

        time.sleep(max(0.05, args.interval))

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if error_seen_after_start:
        print("- Errors observed AFTER startup. Common causes:")
        print("    Serial Signal/Overrun → bad USB cable or driver")
        print("    Serial Timeout       → main.py crashed mid-session")
        print("    Protocol Error       → another program writing to the "
              "Maestro at the same time")
        print("  Recommend opening Pololu Maestro Control Center → "
              "'Errors' tab.")
    else:
        print("- No new errors after startup. Maestro is healthy.")
    if moving_seen:
        print("- moving_state == 1 was observed. The Maestro is internally "
              "rate-limiting servo motion via its 'Speed' or 'Acceleration' "
              "channel settings.")
        print("  Open Pololu Maestro Control Center → 'Channel Settings' → "
              "set Speed = 0 and Acceleration = 0 for both channels.")
        print("  (Otherwise the gimbal_manager's slew calls will be smoothed "
              "*twice*, once by us and once by the Maestro.)")
    else:
        print("- moving_state == 0 throughout. The Maestro is not internally "
              "rate-limiting (good).")
    print()
    print("What this script CANNOT see:")
    print("  - Servo voltage / current (no Maestro support)")
    print("  - Actual servo physical position (no feedback in hobby servos)")
    print("  - PWM period / channel min/max — open Pololu Maestro Control "
          "Center → 'Channel Settings' to inspect and edit those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

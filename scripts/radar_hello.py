"""First-light UART sanity check for the AWR2944P.

Usage:
    # CLI port (115200) — expects a "version" / build-date banner.
    python scripts/radar_hello.py --port COM6 --baud 115200

    # Data port (921600) — hex-dump the first bytes; expects magic word.
    python scripts/radar_hello.py --port COM7 --baud 921600 --hex

Used in Phase C of Ticket 5a to figure out which of the quad-FTDI
COMs is the CLI UART vs the data UART on this particular board. On
the AR-DevPack-EVM-012 the lowest COM is conventionally the CLI
(UART-A / SBL-UART) and the next sequential one is the data port
(UART-B), but this script avoids assumptions by just printing what
comes back.
"""
from __future__ import annotations

import argparse
import sys
import time

import serial


def _hex_dump(ser: serial.Serial, n_bytes: int = 256, timeout_s: float = 2.0) -> None:
    """Dump up to n_bytes of bytes as hex — useful for the data UART.

    The mmw_demo magic word (02 01 04 03 06 05 08 07) should appear
    in the first few hundred bytes of any active data port.
    """
    print(f"Hex-dumping up to {n_bytes} bytes (timeout {timeout_s}s)...")
    buf = bytearray()
    deadline = time.monotonic() + timeout_s
    while len(buf) < n_bytes and time.monotonic() < deadline:
        w = ser.in_waiting
        if w:
            buf.extend(ser.read(min(w, n_bytes - len(buf))))
        else:
            time.sleep(0.02)
    if not buf:
        print("  (no bytes received)")
        return
    print(f"  received {len(buf)} bytes")
    # 16 bytes per row.
    for i in range(0, len(buf), 16):
        row = buf[i : i + 16]
        hex_part = " ".join(f"{b:02x}" for b in row)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in row)
        print(f"  {i:04x}: {hex_part:<48}  {ascii_part}")
    magic = b"\x02\x01\x04\x03\x06\x05\x08\x07"
    found = buf.find(magic)
    print(f"  magic word {'FOUND' if found >= 0 else 'NOT FOUND'} (offset={found})")


def _cli_banner(ser: serial.Serial) -> None:
    """Send 'version' to the CLI UART and print whatever comes back."""
    ser.reset_input_buffer()
    ser.write(b"version\n")
    ser.flush()
    time.sleep(0.5)
    data = ser.read(ser.in_waiting)
    if not data:
        print("  (no banner — wrong port, wrong baud, or firmware not running)")
        return
    print(data.decode("ascii", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True, help="Serial port (e.g. COM6)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate (default 115200)")
    ap.add_argument("--hex", action="store_true",
                    help="Hex-dump incoming bytes instead of sending 'version'. "
                         "Use for the data UART at 921600.")
    args = ap.parse_args()

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.5)
    except serial.SerialException as e:
        print(f"[FATAL] cannot open {args.port} @ {args.baud}: {e}")
        return 2

    with ser:
        if args.hex:
            _hex_dump(ser)
        else:
            _cli_banner(ser)
            # Courtesy sensorStop so we don't leave the chip streaming
            # if the CLI port was pointed at an already-running demo.
            try:
                ser.write(b"sensorStop\n")
                ser.flush()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    sys.exit(main())

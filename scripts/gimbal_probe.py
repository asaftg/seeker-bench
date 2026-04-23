"""Gimbal diagnostic probe.

Bypasses the whole app pipeline. Just opens every Pololu COM port in
turn, sends a few raw "set target" packets to channel 0 (pan) and
channel 1 (tilt), and reports which port(s) actually talk to the
Maestro.

Usage:
    python -m scripts.gimbal_probe
    python -m scripts.gimbal_probe --port COM5
    python -m scripts.gimbal_probe --channel 0 --us 1500

Close Maestro Control Center before running — it holds the port
exclusively and writes from here will fail with PermissionError 13.
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Tuple


POLOLU_VID = 0x1FFB


def list_all_serial() -> List[Tuple[str, str, int | None, int | None]]:
    try:
        from serial.tools import list_ports
    except Exception as e:
        print(f"[!] pyserial not installed: {e}")
        sys.exit(2)
    out = []
    for p in list_ports.comports():
        out.append((
            p.device,
            (p.description or "") + " | " + (p.product or ""),
            getattr(p, "vid", None),
            getattr(p, "pid", None),
        ))
    return out


def pololu_ports() -> List[str]:
    return [row[0] for row in list_all_serial() if row[2] == POLOLU_VID]


def probe_port(port: str, channel: int, us: float) -> bool:
    try:
        import serial
    except Exception as e:
        print(f"[!] pyserial missing: {e}")
        return False
    try:
        ser = serial.Serial(port, baudrate=115200, timeout=0.1, write_timeout=0.5)
    except Exception as e:
        print(f"[{port}] open FAILED: {e}")
        return False

    ok = True
    try:
        target_qus = int(round(us * 4.0))
        lo = target_qus & 0x7F
        hi = (target_qus >> 7) & 0x7F
        packet = bytes([0x84, channel & 0x7F, lo, hi])
        try:
            ser.write(packet)
            ser.flush()
            print(f"[{port}] ch{channel} <- {us:.0f} us  WRITE OK")
        except Exception as e:
            print(f"[{port}] ch{channel} <- {us:.0f} us  WRITE FAILED: {e}")
            ok = False
    finally:
        try:
            ser.close()
        except Exception:
            pass
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="Specific COM port to test (e.g. COM5). Default: every Pololu port.")
    ap.add_argument("--channel", type=int, default=None, help="Only test this channel (0=pan, 1=tilt). Default: both.")
    ap.add_argument("--us", type=float, default=1500.0, help="Target pulse width in microseconds. Default 1500 (center).")
    ap.add_argument("--sweep", action="store_true", help="Sweep 1200→1800 us on given channel(s) to produce visible motion.")
    args = ap.parse_args()

    print("── All serial ports visible to pyserial ──")
    for dev, desc, vid, pid in list_all_serial():
        flag = " <-- Pololu" if vid == POLOLU_VID else ""
        vid_s = f"{vid:04X}" if vid is not None else "----"
        pid_s = f"{pid:04X}" if pid is not None else "----"
        print(f"  {dev:<8} VID={vid_s} PID={pid_s}  {desc}{flag}")
    print()

    if args.port:
        ports = [args.port]
    else:
        ports = pololu_ports()
        if not ports:
            print("[!] No Pololu-VID ports found. Is the Maestro plugged in?")
            return 1

    channels = [args.channel] if args.channel is not None else [0, 1]

    print(f"── Probing ports {ports} on channels {channels} ──")
    print("Tip: CLOSE Maestro Control Center before this test. If it's")
    print("     still open, writes will appear to succeed but the servo")
    print("     won't move (or will return PermissionError 13).")
    print()

    for port in ports:
        if args.sweep:
            for us in (1200, 1400, 1500, 1600, 1800, 1500):
                for ch in channels:
                    probe_port(port, ch, us)
                time.sleep(0.6)
        else:
            for ch in channels:
                probe_port(port, ch, args.us)

    print()
    print("Done. If you saw physical servo motion for one of the ports,")
    print("that's your Command Port — set it explicitly in app_config.yaml:")
    print("    gimbal:")
    print("      port: COM5      # ← whichever one worked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

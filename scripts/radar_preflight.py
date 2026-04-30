"""Pre-flight radar health check. Run this BEFORE starting Seeker
or pressing REC for a drone test.

Verifies in 10 seconds whether the chip will produce continuous
LVDS data. If yes, ready to fly. If no, kicks the chip and re-tests;
if it still fails, prints exactly what to do next.

Stop Seeker first (this script holds COM10/COM11 + UDP port 4098).

Usage:
    python scripts/radar_preflight.py
"""
from __future__ import annotations

import socket
import struct
import sys
import time
from pathlib import Path

import serial

# Make repo importable when run from the seeker_bench root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radar.cfg_sender import send_cfg

CLI_PORT = "COM11"
DATA_PORT = "COM10"
CLI_BAUD = 115200
HOST_IP = "192.168.33.30"
DATA_UDP_PORT = 4098
CFG_PATH = "radar/cfg/awr2944P_unified.cfg"

OK = "\033[92m✓"
FAIL = "\033[91m✗"
NC = "\033[0m"


def _kick_chip() -> bool:
    """sensorStop + sensorStart 0 over CLI. Returns True if chip
    ack'd both 'Done'. The push of the cfg ending in `sensorStart`
    is what the chip's mmw_demoDDM build apparently can't sustain;
    we use the warm-restart path that scripts/radar_raw_test.py
    proved keeps streaming for 30+ s."""
    try:
        with serial.Serial(CLI_PORT, CLI_BAUD, timeout=0.5) as cli:
            # Push cfg first so the chip has the unified profile loaded.
            try:
                responses = send_cfg(cli, CFG_PATH)
                tail = responses[-1] if responses else ""
            except Exception as e:
                print(f"  {FAIL}{NC} cfg push failed: {e}")
                return False
            cli.reset_input_buffer()
            for cmd, wait in [("sensorStop", 1.0), ("sensorStart 0", 2.0)]:
                cli.write((cmd + "\n").encode("ascii"))
                cli.flush()
                deadline = time.monotonic() + wait
                buf = bytearray()
                while time.monotonic() < deadline:
                    if cli.in_waiting:
                        buf.extend(cli.read(cli.in_waiting))
                        if b"Done" in buf or b"Error" in buf:
                            break
                    else:
                        time.sleep(0.05)
                if b"Done" not in buf:
                    print(f"  {FAIL}{NC} {cmd!r} did not ack: "
                          f"{buf.decode('ascii', errors='replace').strip()!r}")
                    return False
        return True
    except serial.SerialException as e:
        print(f"  {FAIL}{NC} CLI port busy ({e}). Stop Seeker first.")
        return False


def _test_lvds(duration_s: float = 8.0) -> tuple[bool, int]:
    """Listen on the DCA UDP data port for `duration_s` seconds.
    Returns (passed, total_bytes_received). 'Passed' = continuous
    streaming for the whole duration, no >1 s gap."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    except OSError:
        pass
    try:
        sock.bind((HOST_IP, DATA_UDP_PORT))
    except OSError as e:
        print(f"  {FAIL}{NC} could not bind {HOST_IP}:{DATA_UDP_PORT} ({e}). "
              "Stop Seeker first.")
        sock.close()
        return False, 0
    sock.settimeout(0.5)

    bytes_total = 0
    last_byte_t = time.monotonic()
    longest_gap = 0.0
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(16384)
            bytes_total += len(data)
            now = time.monotonic()
            longest_gap = max(longest_gap, now - last_byte_t)
            last_byte_t = now
        except socket.timeout:
            continue
    sock.close()
    final_gap = time.monotonic() - last_byte_t
    passed = (bytes_total > 100_000   # at least ~100 KB in `duration_s`
              and longest_gap < 1.0
              and final_gap < 1.0)
    print(f"     received {bytes_total:>12,} B  longest gap {longest_gap:.2f}s  "
          f"final gap {final_gap:.2f}s")
    return passed, bytes_total


def main() -> int:
    print("Radar pre-flight check")
    print("======================")
    print()
    print("[1/2] Kicking chip (cfg push + sensorStart 0)...")
    if not _kick_chip():
        print()
        print(f"  {FAIL}{NC} Chip did not respond. Try:")
        print("       1. Confirm COM11 is the AWR XDS110 port")
        print("       2. Power-cycle the AWR (pull 12V, plug back)")
        print("       3. Re-run this script")
        return 2
    print(f"  {OK}{NC} chip kicked")
    print()
    print("[2/2] Listening for LVDS for 8 seconds...")
    passed, _ = _test_lvds(duration_s=8.0)
    print()
    if passed:
        print(f"  {OK} READY TO FLY{NC}")
        print("  LVDS is streaming continuously. Start Seeker, switch to A/A,")
        print("  press REC, and run the drone test.")
        return 0
    print(f"  {FAIL} NOT READY{NC}")
    print("  LVDS halted or never streamed cleanly.")
    print("  Likely causes:")
    print("    - chip's mmw_demoDDM hit FFT clipping (RX gain too high")
    print("      for the indoor scene). Move the radar away from a wall")
    print("      or strong reflector and re-run.")
    print("    - chip flash needs reload via UniFlash.")
    print("    - 12V brownout (try a fresh barrel adapter).")
    return 1


if __name__ == "__main__":
    sys.exit(main())

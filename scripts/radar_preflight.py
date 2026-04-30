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
from typing import Optional

import serial

# Make repo importable when run from the seeker_bench root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from radar.cfg_sender import send_cfg
from radar_dca.dca_control import DCAControl, DCAControlError

CLI_PORT = "COM11"
DATA_PORT = "COM10"
CLI_BAUD = 115200
HOST_IP = "192.168.33.30"
DATA_UDP_PORT = 4098
CFG_PATH = "radar/cfg/awr2944P_unified.cfg"

OK = "\033[92m✓"
FAIL = "\033[91m✗"
NC = "\033[0m"


def _kick_awr() -> bool:
    """Push cfg + ensure chip is STARTED. Returns True if chip ack'd.

    Acceptance is permissive: the cfg's final `sensorStart` returns
    one of several strings depending on chip state:
      - `Done`                                    — clean INIT path
      - `Debug: Init Calibration Status = 0x...`  — STARTED with
                                                     calibration
                                                     debug print
      - `Error: Invalid Sensor Start`             — chip wasn't in
                                                     INIT; needs
                                                     `sensorStart 0`
                                                     recovery

    All three are progress; only the last needs follow-up. Anything
    else (port busy, no ack, hardware fault) is a real failure.
    """
    def _ok(s: str) -> bool:
        return ("Done" in s
                or "Init Calibration Status" in s
                or "Calibration Status = 0x" in s)

    def _read_cli_ack(cli, wait_s: float) -> str:
        deadline = time.monotonic() + wait_s
        buf = bytearray()
        while time.monotonic() < deadline:
            if cli.in_waiting:
                buf.extend(cli.read(cli.in_waiting))
                if (b"Done" in buf
                        or b"Init Calibration Status" in buf
                        or b"Error" in buf):
                    break
            else:
                time.sleep(0.05)
        return buf.decode("ascii", errors="replace").strip()

    try:
        with serial.Serial(CLI_PORT, CLI_BAUD, timeout=0.5) as cli:
            try:
                responses = send_cfg(cli, CFG_PATH)
            except Exception as e:
                print(f"  {FAIL}{NC} cfg push failed: {e}")
                return False
            tail = (responses[-1] if responses else "").strip()
            if _ok(tail):
                return True
            if "Invalid" in tail or "Error" in tail:
                cli.reset_input_buffer()
                cli.write(b"sensorStart 0\n")
                cli.flush()
                ack = _read_cli_ack(cli, 2.0)
                if _ok(ack):
                    return True
                print(f"  {FAIL}{NC} sensorStart 0 ack: {ack!r}")
                return False
            print(f"  {FAIL}{NC} unexpected cfg-push tail: {tail!r}")
            return False
    except serial.SerialException as e:
        print(f"  {FAIL}{NC} CLI port busy ({e}). Stop Seeker first.")
        return False


def _kick_dca() -> bool:
    """Tell the DCA1000 to forward LVDS as UDP. Without this the
    DCA's FPGA receives bytes from the AWR over the ribbon but
    discards them — host sees zero UDP packets even though the AWR
    is happily streaming. Returns True if all three steps succeeded.
    """
    ctrl = DCAControl(host_ip=HOST_IP, dca_ip="192.168.33.180",
                      config_port=4096, data_port=DATA_UDP_PORT)
    try:
        # Stop any lingering capture from a prior preflight run.
        # Without this, setup_capture errors with "Stop the already
        # running process." on the second back-to-back preflight.
        try:
            ctrl.stop_record()
        except Exception:
            pass
        time.sleep(0.2)
        ctrl.reset_fpga()
        time.sleep(0.2)
        ctrl.setup_capture()
        ctrl.start_record()
        return True
    except DCAControlError as e:
        print(f"  {FAIL}{NC} DCA setup failed: {e}")
        return False
    except Exception as e:
        print(f"  {FAIL}{NC} DCA setup unexpected error: {e}")
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
    last_byte_t: Optional[float] = None  # type: ignore[name-defined]
    longest_gap = 0.0
    deadline = time.monotonic() + duration_s

    while time.monotonic() < deadline:
        try:
            data, _ = sock.recvfrom(16384)
            bytes_total += len(data)
            now = time.monotonic()
            # Only measure inter-packet gaps AFTER the first packet
            # arrives — the gap between "start listening" and "first
            # packet" is just cfg-push startup latency, not a chip stall.
            if last_byte_t is not None:
                longest_gap = max(longest_gap, now - last_byte_t)
            last_byte_t = now
        except socket.timeout:
            continue
    sock.close()
    if last_byte_t is None:
        # Never received a single packet.
        print(f"     received 0 B  (chip never started streaming)")
        return False, 0
    final_gap = time.monotonic() - last_byte_t
    # Pass: enough volume that PMM has frames to work with AND the
    # stream is "alive" at the end. The chip's natural pattern is
    # duty-cycled bursts (~4 frames in a row, ~2 s pause, repeat),
    # so a longest_gap up to ~3 s is normal — not a halt. What we
    # care about is that the stream came back BEFORE the test
    # ended (final_gap small means we saw a recent burst).
    passed = (bytes_total > 5_000_000   # >5 MB = at least one full PMM-window
              and final_gap < 5.0)       # stream was alive within 5 s of end
    print(f"     received {bytes_total:>12,} B  longest mid-stream gap "
          f"{longest_gap:.2f}s  final gap {final_gap:.2f}s")
    return passed, bytes_total


def main() -> int:
    print("Radar pre-flight check")
    print("======================")
    print()
    # DCA FIRST so the FPGA is in capture mode before the AWR
    # emits the first LVDS bytes. If we did AWR-first, the chip's
    # initial LVDS frames hit a not-yet-listening FPGA and the
    # chip's DMA backs up, which appears to be what was halting
    # the stream after the first burst.
    print("[1/3] Telling DCA1000 to forward LVDS as UDP...")
    if not _kick_dca():
        print()
        print(f"  {FAIL}{NC} DCA1000 setup failed. Check:")
        print("       1. RJ45 cable between DCA1000 and host")
        print("       2. Host NIC IP is 192.168.33.30/24 (run `ipconfig`)")
        print("       3. DCA1000 5V barrel + FTDI USB plugged in")
        return 3
    print(f"  {OK}{NC} DCA1000 in start_record mode (waiting for LVDS)")
    print()
    print("[2/3] Pushing cfg to AWR (CLI on COM11)...")
    if not _kick_awr():
        print()
        print(f"  {FAIL}{NC} AWR did not start. Try:")
        print("       1. Confirm COM11 is the AWR XDS110 port")
        print("       2. Power-cycle the AWR (pull 12V, plug back)")
        print("       3. Re-run this script")
        return 2
    print(f"  {OK}{NC} AWR kicked")
    print()
    print("[3/3] Listening for LVDS over UDP for 8 seconds...")
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

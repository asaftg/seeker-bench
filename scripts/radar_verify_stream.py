"""End-to-end diagnostic: push cfg, verify sensorStart ack, then confirm
TLV magic word on the data port.

Prints one line per stage so we can see exactly where the pipeline breaks:

    [1/4] CLI open + sensorStop preflight
    [2/4] cfg push (with per-line Done/Error verdict)
    [3/4] sensorStart ack verification  <-- the usual failure point
    [4/4] data port — magic-word search at configured baud

Run after a fresh power cycle for cleanest results.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import serial

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from radar.cfg_sender import _load_cfg_lines, _read_until_ack  # noqa: E402

MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"


def _stage(n: int, msg: str) -> None:
    print(f"[{n}/4] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli-port", required=True)
    ap.add_argument("--data-port", required=True)
    ap.add_argument("--cli-baud", type=int, default=115200)
    ap.add_argument("--data-baud", type=int, default=3_125_000)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--settle-s", type=float, default=2.0,
                    help="Delay after sensorStart before opening data port")
    ap.add_argument("--listen-s", type=float, default=3.0)
    args = ap.parse_args()

    # --- stage 1 --------------------------------------------------------
    _stage(1, f"opening {args.cli_port} @ {args.cli_baud}")
    try:
        cli = serial.Serial(args.cli_port, args.cli_baud, timeout=0.2)
    except serial.SerialException as e:
        print(f"  [FATAL] cannot open CLI: {e}")
        return 2
    cli.write(b"sensorStop\n")
    cli.flush()
    time.sleep(0.3)
    cli.reset_input_buffer()
    print("      sensorStop sent, buffer drained")

    # --- stage 2 --------------------------------------------------------
    lines = _load_cfg_lines(args.cfg)
    _stage(2, f"pushing {len(lines)} cfg lines from {os.path.basename(args.cfg)}")
    sensor_start_idx = -1
    for i, line in enumerate(lines):
        if line.strip().lower() == "sensorstart":
            sensor_start_idx = i
            continue
        cli.reset_input_buffer()
        cli.write((line + "\n").encode("ascii"))
        cli.flush()
        resp = _read_until_ack(cli, 2.0)
        ok = "Done" in resp
        err = "Error" in resp or "not recognized" in resp
        tag = "OK " if ok and not err else ("ERR" if err else "???")
        first_tok = line.split()[0] if line.split() else "(blank)"
        print(f"      [{tag}] {first_tok:<32} {resp.strip()[-60:]!r}")
        time.sleep(0.05)

    # --- stage 3: sensorStart ack verification -------------------------
    _stage(3, "sending sensorStart and waiting for ack")
    if sensor_start_idx < 0:
        print("      [WARN] cfg did not contain 'sensorStart' — appending it")
    cli.reset_input_buffer()
    cli.write(b"sensorStart\n")
    cli.flush()
    resp = _read_until_ack(cli, 3.0)
    print(f"      response={resp.strip()!r}")
    if "Done" in resp:
        print("      [OK] sensorStart acknowledged")
    elif "Error" in resp:
        print("      [FATAL] sensorStart rejected — chip will not stream")
        cli.close()
        return 3
    else:
        print("      [WARN] no Done within 3s — chip may or may not be streaming")
    cli.close()

    # --- stage 4: data port --------------------------------------------
    time.sleep(args.settle_s)
    _stage(4, f"opening {args.data_port} @ {args.data_baud} and listening {args.listen_s}s")
    try:
        data = serial.Serial(args.data_port, args.data_baud, timeout=0.1)
    except serial.SerialException as e:
        print(f"      [FATAL] cannot open data port: {e}")
        return 4

    deadline = time.monotonic() + args.listen_s
    buf = bytearray()
    with data:
        while time.monotonic() < deadline:
            n = data.in_waiting
            if n:
                buf.extend(data.read(n))
            else:
                time.sleep(0.01)

    print(f"      got {len(buf)} bytes")
    if not buf:
        print("      [FATAL] no bytes — chip is not streaming")
        return 5
    idx = bytes(buf).find(MAGIC)
    if idx < 0:
        print("      [FATAL] bytes received but no magic word — wrong baud or corrupted stream")
        print(f"      first 32 bytes: {bytes(buf[:32]).hex()}")
        return 6
    print(f"      [OK] magic word at offset {idx} — stream is live")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Send a TI mmw_demo ``.cfg`` profile to the radar CLI UART.

The stock ``mmw_demoDDM`` firmware will not stream anything until the
host pushes a full profile ending in ``sensorStart``. The TI convention
is one command per line, terminated by ``\\n``, with a short pause
between commands so the firmware has time to ack ``Done``.

This helper is shared by ``radar.radar_manager.RadarManager`` (on
connect) and ``scripts/radar_send_cfg.py`` (for manual first-light
sanity), so the two always agree on framing / pacing / ack handling.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional

import serial

from common.logging_setup import get_logger

log = get_logger(__name__)


def _load_cfg_lines(cfg_path: str | Path) -> List[str]:
    """Read a mmw_demo .cfg file, stripping comments + blank lines."""
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = f.readlines()
    out: List[str] = []
    for ln in raw:
        s = ln.strip()
        if not s or s.startswith("%") or s.startswith("#"):
            continue
        out.append(s)
    return out


def send_cfg(
    ser: serial.Serial,
    cfg_path: str | Path,
    inter_line_delay_s: float = 0.05,
    ack_timeout_s: float = 2.0,
    on_line: Optional[Callable[[str, str], None]] = None,
) -> List[str]:
    """Push a .cfg line-by-line over an already-open CLI serial port.

    Returns the list of line-by-line responses from the firmware (for
    debugging / docs). ``on_line(cmd, resp)`` is called for each line
    if the caller wants live feedback (the scripts/radar_send_cfg.py
    CLI uses it to print to stdout).

    This function does NOT open or close ``ser`` — caller owns the
    port lifecycle. That's deliberate: ``RadarManager`` reuses the CLI
    port for ``sensorStop`` on shutdown, and the sanity-check script
    holds it open across multiple commands.
    """
    lines = _load_cfg_lines(cfg_path)
    responses: List[str] = []

    # Clear any stale bytes sitting in the OS buffer — the firmware's
    # previous run often leaves a tail of "Done" / version banner
    # that would otherwise confuse the first ack read.
    try:
        ser.reset_input_buffer()
    except Exception:  # pragma: no cover — non-POSIX / non-Win serial backends
        pass

    for line in lines:
        # Drain any stale bytes from the previous line's delayed ack BEFORE
        # sending the next command. Without this, a slow ack on line N
        # (e.g. antennaCalibParams) sits in the buffer when we write line
        # N+1, and the next _read_until_ack returns immediately on the
        # stale "Done" — shifting every subsequent ack by one and causing
        # sensorStart's real response to be missed entirely.
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write((line + "\n").encode("ascii"))
        ser.flush()

        # Read the response until we either see "Done" / "Error" /
        # "mmwDemo:" (the prompt between commands) or the timeout hits.
        resp = _read_until_ack(ser, ack_timeout_s)
        responses.append(resp)
        if on_line is not None:
            try:
                on_line(line, resp)
            except Exception:
                log.exception("cfg_sender on_line callback failed")

        if "Error" in resp or "error" in resp:
            log.warning("radar cfg line rejected by firmware: %r -> %r", line, resp)

        time.sleep(inter_line_delay_s)

    log.info("Pushed %d cfg lines from %s", len(lines), cfg_path)
    return responses


def _read_until_ack(ser: serial.Serial, timeout_s: float) -> str:
    """Accumulate CLI UART bytes until we see an ack or time out."""
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n_waiting = ser.in_waiting
        if n_waiting:
            buf.extend(ser.read(n_waiting))
            text = buf.decode("ascii", errors="replace")
            # TI CLI shows "Done" on success, "Error" on rejection,
            # and drops "mmwDemo:/>" as the next prompt. Any of the
            # three means the firmware has finished processing the
            # previous line.
            if ("Done" in text
                    or "Error" in text
                    or "mmwDemo:/>" in text
                    or "mmw_pro:/>" in text):
                return text
        else:
            # Short idle sleep to avoid hot-looping. 2 ms is well
            # under the firmware's ack latency on 115200.
            time.sleep(0.002)
    return buf.decode("ascii", errors="replace")

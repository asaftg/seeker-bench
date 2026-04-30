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

    # mmw_demo on AWR2944P (SDK 4.7.2.1) self-resets on commands that
    # change RF/DFE state (`dfeDataOutputMode`, sometimes `profileCfg`),
    # to apply the new state. The reset wipes the in-RAM mode back to
    # default. So a single linear cfg push fails — lines after the
    # reset see chip-in-default-state, not the partially-configured
    # chip we expected.
    #
    # Workaround: when we detect a chip self-reset mid-push, restart
    # the cfg from line 1. Eventually we converge: once the chip is
    # already in mode 1 (carried over in NVM/RAM from the previous
    # attempt), `dfeDataOutputMode 1` is a no-op (no reset), the rest
    # of the cfg pushes cleanly through `sensorStart`. Cap retries so
    # a genuinely-broken chip fails loudly instead of looping.
    max_restarts = 5
    restart_count = 0

    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]

        # Drain stale bytes from the previous line's response BEFORE
        # sending the next command. reset_input_buffer alone is NOT
        # enough — the chip's trailing "mmwDemo:/>" prompt arrives
        # ~5-30 ms AFTER the "Done" line, often AFTER our reset call.
        # Wait for the UART to go quiet for 50 ms (no new bytes), then
        # reset_input_buffer to clear it.
        idle_deadline = time.monotonic() + 0.25
        while time.monotonic() < idle_deadline:
            n = ser.in_waiting
            if n:
                ser.read(n)
                idle_deadline = time.monotonic() + 0.05
            else:
                time.sleep(0.01)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write((line + "\n").encode("ascii"))
        ser.flush()

        # Read the response until we see "Done" / "Error" or time out.
        resp = _read_until_ack(ser, ack_timeout_s)
        responses.append(resp)
        if on_line is not None:
            try:
                on_line(line, resp)
            except Exception:
                log.exception("cfg_sender on_line callback failed")

        if "Error" in resp or "error" in resp:
            log.warning("radar cfg line rejected by firmware: %r -> %r", line, resp)

        # Detect a chip self-reset and decide whether to restart the
        # cfg from line 1. The chip self-resets when commands like
        # `dfeDataOutputMode` change DFE state — the reset wipes RAM
        # back to default, so subsequent lines fail. Restart from line 1
        # converges: on the second pass `dfeDataOutputMode 1` is a no-op
        # (chip already in mode 1) and the rest pushes cleanly.
        rebooted = ("Bootloader" in resp or "Starting QSPI" in resp)
        if not rebooted:
            # Peek a moment for a late reboot (Done arrives, then reset
            # starts a few ms later).
            peek_end = time.monotonic() + 0.3
            peek = bytearray()
            while time.monotonic() < peek_end:
                n = ser.in_waiting
                if n:
                    peek.extend(ser.read(n))
                else:
                    time.sleep(0.02)
            if b"Starting QSPI" in peek or b"Bootloader" in peek:
                rebooted = True

        if rebooted:
            _wait_for_app_ready(ser, timeout_s=8.0)
            if restart_count < max_restarts:
                restart_count += 1
                log.info(
                    "cfg_sender: chip self-reset on line %d (%s); "
                    "restarting cfg from line 1 (attempt %d/%d)",
                    line_idx + 1, line.split()[0], restart_count, max_restarts)
                line_idx = 0
                continue
            else:
                log.error(
                    "cfg_sender: chip kept self-resetting after %d attempts; "
                    "giving up. Last failing line: %s",
                    max_restarts, line)
                break

        line_idx += 1
        time.sleep(inter_line_delay_s)

    log.info("Pushed %d cfg lines from %s", len(lines), cfg_path)
    return responses


def _wait_for_app_ready(ser: serial.Serial, timeout_s: float = 8.0) -> bool:
    """Block until we see the application banner ('AWR2X44P MMW Demo')
    OR the prompt 'mmwDemo:/>' from the post-boot app, indicating the
    chip is back from a self-reset and ready for the next command.

    Returns True if banner / prompt seen within ``timeout_s``, else False.
    Reads opportunistically, doesn't echo to the caller.
    """
    log.info("cfg_sender: chip self-rebooted, waiting for app ready (up to %.1fs)",
             timeout_s)
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n = ser.in_waiting
        if n:
            buf.extend(ser.read(n))
            text = buf.decode("ascii", errors="replace")
            if "MMW Demo" in text or "mmwDemo:/>" in text:
                # Drain a bit more so the next reset_input_buffer
                # actually clears the post-boot tail.
                end = time.monotonic() + 0.3
                while time.monotonic() < end:
                    if ser.in_waiting:
                        buf.extend(ser.read(ser.in_waiting))
                    else:
                        time.sleep(0.02)
                log.info("cfg_sender: chip app ready")
                return True
        else:
            time.sleep(0.02)
    log.warning("cfg_sender: timed out waiting for app banner after self-reset")
    return False


def _read_until_ack(ser: serial.Serial, timeout_s: float) -> str:
    """Accumulate CLI UART bytes until we see an ack or time out.

    Only "Done" or "Error" count as a real ack — NOT the bare
    "mmwDemo:/>" prompt. Returning on a stale prompt that arrived
    after a previous reset_input_buffer() was the cfg-push race
    we hit on 2026-04-30: the prompt sits in the buffer when we
    write the next line, _read_until_ack returns immediately on
    it, and the next line's true ack lands too late and gets
    eaten by the next reset_input_buffer.

    Drain a small window after Done/Error so the trailing prompt
    arrives in this read rather than leaking into the next.
    """
    deadline = time.monotonic() + timeout_s
    buf = bytearray()
    while time.monotonic() < deadline:
        n_waiting = ser.in_waiting
        if n_waiting:
            buf.extend(ser.read(n_waiting))
            text = buf.decode("ascii", errors="replace")
            if "Done" in text or "Error" in text:
                # Drain the trailing prompt so it doesn't leak into
                # the next line's read.
                end = time.monotonic() + 0.08
                while time.monotonic() < end:
                    m = ser.in_waiting
                    if m:
                        buf.extend(ser.read(m))
                    else:
                        time.sleep(0.005)
                return buf.decode("ascii", errors="replace")
        else:
            # Short idle sleep to avoid hot-looping. 2 ms is well
            # under the firmware's ack latency on 115200.
            time.sleep(0.002)
    return buf.decode("ascii", errors="replace")

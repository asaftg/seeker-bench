"""Pololu Micro Maestro 6-channel — thin USB/serial driver.

The Maestro enumerates as two USB CDC virtual COM ports on Windows:

    • "Command Port"   — what we write servo commands to
    • "TTL Port"       — mirrors the TTL UART header on the board

Both share the same USB VID (0x1ffb). We auto-detect the command
port by VID+product string ("Command Port" suffix) so the user
never has to pick COM numbers by hand. If the device is absent,
``MaestroDriver.open()`` returns False and the caller can degrade
gracefully.

Protocol: we use the Pololu "compact protocol" — each command is
a small binary sequence. For Set Target (the only thing we need
for pan+tilt):

    0x84, channel, target_lo (7-bit), target_hi (7-bit)

where target is in *quarter-microseconds* (so 1500 µs = 6000). A
hobby servo takes pulses in [500, 2500] µs, i.e. target in
[2000, 10000]. The Maestro also accepts 0 as "release/float" —
we use that on shutdown so servos don't keep holding torque.

No dependency on pololu's USB SDK — pyserial is enough.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from common.logging_setup import get_logger

log = get_logger(__name__)


POLOLU_VID = 0x1FFB  # Pololu Corporation


class MaestroDriver:
    """Minimal Maestro command interface over pyserial."""

    def __init__(self, port: Optional[str] = None) -> None:
        self._explicit_port = port
        self._port: Optional[str] = None
        self._ser = None  # serial.Serial | None

    # ── discovery ─────────────────────────────────────────────

    @staticmethod
    def list_maestro_ports() -> List[Tuple[str, str]]:
        """Return [(port, description), ...] for every Pololu Maestro
        command port we can find on the system.

        Order: "Command Port" first (preferred), then anything else
        with the Pololu VID as a fallback.
        """
        try:
            from serial.tools import list_ports
        except Exception as e:
            log.warning("pyserial not installed (%s) — gimbal disabled", e)
            return []
        primary: List[Tuple[str, str]] = []
        secondary: List[Tuple[str, str]] = []
        for p in list_ports.comports():
            vid = getattr(p, "vid", None)
            if vid != POLOLU_VID:
                continue
            desc = (p.description or "") + " " + (p.product or "")
            if "Command Port" in desc:
                primary.append((p.device, desc.strip()))
            else:
                secondary.append((p.device, desc.strip()))
        return primary + secondary

    # ── lifecycle ─────────────────────────────────────────────

    def open(self) -> bool:
        """Try to open the Maestro. Returns True on success.

        Never raises on absent hardware — logs and returns False so
        callers can run in "disconnected" mode.
        """
        try:
            import serial
        except Exception as e:
            log.warning("pyserial missing (%s) — gimbal unavailable", e)
            return False

        port = self._explicit_port
        if port is None:
            hits = self.list_maestro_ports()
            if not hits:
                log.warning("No Pololu Maestro found on any COM port")
                return False
            port, desc = hits[0]
            log.info("Maestro auto-detected on %s (%s)", port, desc)

        try:
            # Baud rate is ignored on native USB CDC, but pyserial
            # still wants a number. Short timeout so any accidental
            # read doesn't block the app.
            self._ser = serial.Serial(port, baudrate=115200, timeout=0.1, write_timeout=0.5)
            self._port = port
            log.info("Maestro opened on %s", port)
            return True
        except Exception as e:
            log.warning("Failed to open Maestro on %s: %s", port, e)
            self._ser = None
            self._port = None
            return False

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
        self._ser = None
        self._port = None

    @property
    def is_open(self) -> bool:
        return self._ser is not None and getattr(self._ser, "is_open", False)

    @property
    def port(self) -> Optional[str]:
        return self._port

    # ── commands ──────────────────────────────────────────────

    def set_target_us(self, channel: int, microseconds: float) -> bool:
        """Command channel `channel` to a pulse width in microseconds.

        Pass ``microseconds == 0`` to release (servo goes floppy).
        Returns False if the port isn't open or the write fails.
        """
        if not self.is_open:
            return False
        target_qus = 0 if microseconds <= 0 else int(round(microseconds * 4.0))
        lo = target_qus & 0x7F
        hi = (target_qus >> 7) & 0x7F
        packet = bytes([0x84, int(channel) & 0x7F, lo, hi])
        try:
            self._ser.write(packet)
            return True
        except Exception as e:
            log.warning("Maestro write failed on ch %d: %s", channel, e)
            return False

    def release_all(self, channels: List[int]) -> None:
        """Float every listed channel so servos stop holding torque."""
        for ch in channels:
            self.set_target_us(ch, 0)

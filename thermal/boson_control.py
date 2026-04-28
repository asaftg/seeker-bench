"""
FLIR Boson serial control — pyserial wrapper for the camera's
hardware AGC, DDE, gain, FFC, etc.

The Boson exposes its full command set over a CDC-class USB serial
port (separate from the UVC video stream we already use in
``boson_capture.py``). Commands use FLIR's "FFC" (Frame Flow Control)
packet framing — variable-length packets with CRC-16/CCITT and
byte-stuffing escape codes.

This module ONLY handles serial control. Video capture stays in
``boson_capture.py``. The two are decoupled — you can run either
without the other.

Usage::

    with BosonControl(port="auto") as bc:
        sw = bc.get_software_rev()      # confidence-test the link
        bc.set_dde_state(True)          # turn the camera's hardware
                                         # Digital Detail Enhancement on
        bc.set_dde_gain(6)               # pump it up
        bc.run_ffc()                     # manual flat-field correction

Function IDs mirror the names in FLIR's Boson IDD (Interface
Description Document). Some IDs are flagged ``# IDD-VERIFIED-PENDING``
— they're built from public references / library examples and need
to be confirmed against the actual IDD before being trusted in
production. The ``ping()`` and ``get_camera_sn()`` commands are the
safest no-op probes; we use those to validate the link before issuing
any state-changing command.

Protocol summary (FLIR FFC over UART, 921600 8N1)::

    [0x8E][header bytes][CRC16 BE][0xAE]

      header (in CRC scope) = channel(1) | seq(1) | function_id(4)
                              [| status(4) — response only]
                              | payload(variable)

    Byte stuffing in CRC scope: 0x8E -> 0x9E 0x81,
                                 0xAE -> 0x9E 0xA1,
                                 0x9E -> 0x9E 0x91.

    CRC16: poly 0x1021, init 0x1D0F (CRC-CCITT-XMODEM variant per
    Boson IDD). Computed on raw header bytes BEFORE byte-stuffing,
    transmitted in big-endian after byte-stuffing.

    Function ID is 32-bit big-endian. High bit (0x80000000) set on
    response packets. Status follows function ID on responses (0 = OK,
    nonzero = error code per IDD).
"""
from __future__ import annotations

import struct
import threading
import time
from contextlib import contextmanager
from typing import Optional

import serial  # pyserial — already a project dep (gimbal, radar)

from common.logging_setup import get_logger

log = get_logger(__name__)


# ───────────────────────────────────────────────────────────────
# Wire constants
# ───────────────────────────────────────────────────────────────

START_BYTE = 0x8E
END_BYTE = 0xAE
ESCAPE_BYTE = 0x9E
ESCAPE_MASK = 0x10  # XOR mask used by Boson's stuffing scheme

DEFAULT_BAUD = 921600
DEFAULT_TIMEOUT_S = 0.5


# ───────────────────────────────────────────────────────────────
# Function IDs — names mirror the Boson IDD
# ───────────────────────────────────────────────────────────────
#
# All values are best-effort transcriptions from public Boson IDD
# excerpts and reference implementations (e.g. flirpy). EVERY ID
# should be verified against the current Boson IDD PDF before being
# relied on in production. Tagged ``# IDD-VERIFIED-PENDING`` until a
# live ping confirms the function ID + payload format.

class FFC:
    GET_CAMERA_PN = 0x00050003           # IDD-VERIFIED-PENDING
    GET_CAMERA_SN = 0x00050004           # IDD-VERIFIED-PENDING
    GET_SOFTWARE_REV = 0x00050006        # IDD-VERIFIED-PENDING

    # FFC (flat-field correction) — manual NUC trigger
    RUN_FFC = 0x00050201                 # IDD-VERIFIED-PENDING
    GET_FFC_MODE = 0x00050202            # IDD-VERIFIED-PENDING
    SET_FFC_MODE = 0x00050203            # IDD-VERIFIED-PENDING

    # GAO = Gain & Adjustment Operations (camera-side AGC stack)
    GET_GAO_MODE = 0x00050800            # IDD-VERIFIED-PENDING
    SET_GAO_MODE = 0x00050801            # IDD-VERIFIED-PENDING
    GET_DDE_STATE = 0x00050810           # IDD-VERIFIED-PENDING
    SET_DDE_STATE = 0x00050811           # IDD-VERIFIED-PENDING
    GET_DDE_GAIN = 0x00050812            # IDD-VERIFIED-PENDING
    SET_DDE_GAIN = 0x00050813            # IDD-VERIFIED-PENDING
    GET_SSO_STATE = 0x00050820           # IDD-VERIFIED-PENDING
    SET_SSO_STATE = 0x00050821           # IDD-VERIFIED-PENDING
    GET_PLATEAU_VALUE = 0x00050830       # IDD-VERIFIED-PENDING
    SET_PLATEAU_VALUE = 0x00050831       # IDD-VERIFIED-PENDING
    GET_BRIGHTNESS_BIAS = 0x00050840     # IDD-VERIFIED-PENDING
    SET_BRIGHTNESS_BIAS = 0x00050841     # IDD-VERIFIED-PENDING

    GET_GAIN_MODE = 0x00050B00           # IDD-VERIFIED-PENDING
    SET_GAIN_MODE = 0x00050B01           # IDD-VERIFIED-PENDING


# Enum constants used in payloads. Verify before relying on them.
class GAOMode:
    LINEAR = 0
    PLATEAU_HISTOGRAM = 1
    INFORMATION_BASED = 2     # IGE
    INFORMATION_BASED_EQ = 3
    MANUAL = 4
    NOT_DEFINED = 5
    AUTO_BRIGHT = 6
    LINEAR_AGC = 7


class GainMode:
    HIGH = 0
    LOW = 1
    AUTO = 2
    DUAL = 3                  # if firmware supports HDR


class FFCMode:
    MANUAL = 0
    AUTO = 1
    EXTERNAL = 2


# ───────────────────────────────────────────────────────────────
# CRC-16 (CCITT, init 0x1D0F)
# ───────────────────────────────────────────────────────────────

_CRC16_TABLE: Optional[bytes] = None


def _crc16_table() -> bytes:
    global _CRC16_TABLE
    if _CRC16_TABLE is not None:
        return _CRC16_TABLE
    poly = 0x1021
    out = bytearray(512)
    for i in range(256):
        crc = i << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
        out[2 * i] = (crc >> 8) & 0xFF
        out[2 * i + 1] = crc & 0xFF
    _CRC16_TABLE = bytes(out)
    return _CRC16_TABLE


def crc16_ccitt_xmodem(data: bytes, init: int = 0x1D0F) -> int:
    """CRC-16/CCITT with init 0x1D0F (Boson IDD variant)."""
    tbl = _crc16_table()
    crc = init & 0xFFFF
    for b in data:
        idx = ((crc >> 8) ^ b) & 0xFF
        crc = ((crc << 8) ^ ((tbl[2 * idx] << 8) | tbl[2 * idx + 1])) & 0xFFFF
    return crc


# ───────────────────────────────────────────────────────────────
# Byte stuffing (escape any 0x8E / 0xAE / 0x9E inside the body)
# ───────────────────────────────────────────────────────────────

def stuff(body: bytes) -> bytes:
    out = bytearray()
    for b in body:
        if b in (START_BYTE, END_BYTE, ESCAPE_BYTE):
            out.append(ESCAPE_BYTE)
            out.append(b ^ ESCAPE_MASK)
        else:
            out.append(b)
    return bytes(out)


def unstuff(stuffed: bytes) -> bytes:
    out = bytearray()
    i = 0
    while i < len(stuffed):
        b = stuffed[i]
        if b == ESCAPE_BYTE:
            i += 1
            if i >= len(stuffed):
                raise ValueError("truncated escape sequence")
            out.append(stuffed[i] ^ ESCAPE_MASK)
        else:
            out.append(b)
        i += 1
    return bytes(out)


# ───────────────────────────────────────────────────────────────
# Packet framing
# ───────────────────────────────────────────────────────────────

class FFCError(Exception):
    """Raised for malformed packets / non-zero status codes."""


def build_command(function_id: int, payload: bytes = b"",
                  channel: int = 0, seq: int = 0) -> bytes:
    """Build a transmit-ready FFC command packet (with start/end bytes,
    byte stuffing, and CRC).
    """
    header = struct.pack(
        ">BBI", channel & 0xFF, seq & 0xFF, function_id & 0xFFFFFFFF
    ) + payload
    crc = crc16_ccitt_xmodem(header)
    body = header + struct.pack(">H", crc)
    return bytes([START_BYTE]) + stuff(body) + bytes([END_BYTE])


def parse_response(packet: bytes) -> tuple[int, int, bytes]:
    """Parse a received packet (with start/end bytes still attached).

    Returns ``(function_id, status, payload)``. Raises FFCError on
    framing / CRC / length errors.
    """
    if len(packet) < 4 or packet[0] != START_BYTE or packet[-1] != END_BYTE:
        raise FFCError(f"bad framing (len={len(packet)})")
    body = unstuff(packet[1:-1])
    if len(body) < 12:  # 1+1+4+4+2 minimum (channel, seq, fn, status, crc)
        raise FFCError(f"body too short: {len(body)}")
    received_crc = struct.unpack(">H", body[-2:])[0]
    expected_crc = crc16_ccitt_xmodem(body[:-2])
    if received_crc != expected_crc:
        raise FFCError(
            f"CRC mismatch: got {received_crc:#06x}, expected {expected_crc:#06x}"
        )
    channel = body[0]
    seq = body[1]
    function_id = struct.unpack(">I", body[2:6])[0]
    status = struct.unpack(">I", body[6:10])[0]
    payload = body[10:-2]
    # Boson convention: response has high bit set in function_id
    function_id_response_bit = function_id & 0x80000000
    function_id_clean = function_id & 0x7FFFFFFF
    log.debug(
        "FFC response: ch=%d seq=%d fn=0x%08x (resp_bit=%s) status=%d payload_len=%d",
        channel, seq, function_id_clean,
        bool(function_id_response_bit), status, len(payload),
    )
    return function_id_clean, status, payload


# ───────────────────────────────────────────────────────────────
# Serial port detection
# ───────────────────────────────────────────────────────────────

def find_boson_port(exclude: Optional[list[str]] = None) -> Optional[str]:
    """Probe COM ports to find the Boson's CDC serial port.

    Heuristic: enumerate ports, exclude known ones (gimbal COM4, radar
    COM10/COM11), try opening each remaining port and ping at 921600.
    Return the first that responds with valid FFC framing.

    Returns None if no Boson found. Caller can then either set
    ``port=`` explicitly or skip control entirely.
    """
    from serial.tools import list_ports
    excluded = set(p.upper() for p in (exclude or ["COM4", "COM10", "COM11"]))
    candidates = [
        p.device for p in list_ports.comports()
        if p.device.upper() not in excluded
    ]
    log.info("Boson port probe: candidates=%s", candidates)
    for dev in candidates:
        try:
            with serial.Serial(dev, DEFAULT_BAUD, timeout=DEFAULT_TIMEOUT_S) as s:
                # Try the safest read-only command: get_software_rev
                pkt = build_command(FFC.GET_SOFTWARE_REV)
                s.write(pkt)
                s.flush()
                resp = _read_packet(s, timeout_s=0.5)
                if resp is None:
                    continue
                fn, status, payload = parse_response(resp)
                if fn == FFC.GET_SOFTWARE_REV and status == 0:
                    log.info("Boson found on %s (%d-byte sw_rev payload)",
                             dev, len(payload))
                    return dev
        except (serial.SerialException, FFCError, OSError) as e:
            log.debug("port %s: not Boson (%s)", dev, e)
            continue
    return None


def _read_packet(s: serial.Serial, timeout_s: float = 0.5) -> Optional[bytes]:
    """Read one FFC packet from `s`. Synchronizes on START_BYTE,
    accumulates until END_BYTE (with escape awareness)."""
    deadline = time.time() + timeout_s
    buf = bytearray()
    in_packet = False
    while time.time() < deadline:
        b = s.read(1)
        if not b:
            continue
        byte = b[0]
        if not in_packet:
            if byte == START_BYTE:
                buf.clear()
                buf.append(byte)
                in_packet = True
            continue
        # In-packet
        buf.append(byte)
        # END_BYTE that wasn't escaped (escapes appear as 0x9E 0xA1 in stream)
        if byte == END_BYTE:
            return bytes(buf)
        # Cap on packet size — Boson packets fit easily in a few hundred bytes
        if len(buf) > 1024:
            log.warning("FFC read: packet exceeds 1024 bytes — resyncing")
            in_packet = False
            buf.clear()
    return None


# ───────────────────────────────────────────────────────────────
# High-level wrapper
# ───────────────────────────────────────────────────────────────

class BosonControl:
    """High-level access to Boson camera controls.

    Thread-safe (one command in flight at a time). Each command
    builds a packet, sends it, and reads the response with a per-call
    timeout. Failures raise :class:`FFCError`.

    Use as a context manager to ensure the serial port closes::

        with BosonControl() as bc:
            bc.set_dde_state(True)
            bc.set_dde_gain(6)
    """

    def __init__(
        self,
        port: str = "auto",
        baud: int = DEFAULT_BAUD,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        exclude_ports: Optional[list[str]] = None,
    ) -> None:
        self.requested_port = port
        self.baud = baud
        self.timeout_s = timeout_s
        self._exclude = list(exclude_ports or ["COM4", "COM10", "COM11"])
        self._port: Optional[str] = None
        self._ser: Optional[serial.Serial] = None
        self._seq = 0
        self._lock = threading.Lock()

    # ─────────── lifecycle ───────────

    def open(self) -> None:
        if self._ser is not None:
            return
        if self.requested_port == "auto":
            dev = find_boson_port(exclude=self._exclude)
            if dev is None:
                raise FFCError("Boson serial port not found via auto-probe")
            self._port = dev
        else:
            self._port = self.requested_port
        self._ser = serial.Serial(self._port, self.baud, timeout=self.timeout_s)
        log.info("BosonControl opened %s @ %d baud", self._port, self.baud)

    def close(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def __enter__(self) -> "BosonControl":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def port(self) -> Optional[str]:
        return self._port

    # ─────────── core transport ───────────

    def _command(self, function_id: int, payload: bytes = b"",
                 timeout_s: Optional[float] = None) -> bytes:
        """Send command, read response, validate status. Returns payload."""
        if self._ser is None:
            raise FFCError("BosonControl not open — call .open()")
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        with self._lock:
            self._seq = (self._seq + 1) & 0xFF
            pkt = build_command(function_id, payload, seq=self._seq)
            self._ser.reset_input_buffer()
            self._ser.write(pkt)
            self._ser.flush()
            resp = _read_packet(self._ser, timeout_s=timeout)
        if resp is None:
            raise FFCError(
                f"timeout waiting for response to fn=0x{function_id:08x}"
            )
        fn, status, body = parse_response(resp)
        if fn != function_id:
            raise FFCError(
                f"function_id mismatch: sent 0x{function_id:08x}, got 0x{fn:08x}"
            )
        if status != 0:
            raise FFCError(
                f"command 0x{function_id:08x} returned status=0x{status:08x}"
            )
        return body

    # ─────────── identity / link probe ───────────

    def get_software_rev(self) -> bytes:
        """Return raw software-revision payload (firmware version
        bytes; format per IDD). Used as a link-validate command."""
        return self._command(FFC.GET_SOFTWARE_REV)

    def get_camera_sn(self) -> bytes:
        """Return raw camera serial-number payload."""
        return self._command(FFC.GET_CAMERA_SN)

    def get_camera_pn(self) -> bytes:
        """Return raw camera part-number payload."""
        return self._command(FFC.GET_CAMERA_PN)

    # ─────────── FFC (flat-field correction) ───────────

    def run_ffc(self) -> None:
        """Trigger an immediate manual flat-field correction.
        Camera will internally close the shutter, capture a flat
        reference, and update its NUC table. Takes ~1 second on the
        Boson 640."""
        self._command(FFC.RUN_FFC, timeout_s=2.0)

    def get_ffc_mode(self) -> int:
        body = self._command(FFC.GET_FFC_MODE)
        if len(body) < 4:
            raise FFCError(f"unexpected FFC mode payload: {body!r}")
        return struct.unpack(">I", body[:4])[0]

    def set_ffc_mode(self, mode: int) -> None:
        self._command(FFC.SET_FFC_MODE, struct.pack(">I", mode & 0xFFFFFFFF))

    # ─────────── GAO (camera-side AGC) ───────────

    def get_gao_mode(self) -> int:
        body = self._command(FFC.GET_GAO_MODE)
        return struct.unpack(">I", body[:4])[0]

    def set_gao_mode(self, mode: int) -> None:
        self._command(FFC.SET_GAO_MODE, struct.pack(">I", mode & 0xFFFFFFFF))

    def get_dde_state(self) -> bool:
        body = self._command(FFC.GET_DDE_STATE)
        return bool(struct.unpack(">I", body[:4])[0])

    def set_dde_state(self, on: bool) -> None:
        self._command(FFC.SET_DDE_STATE,
                      struct.pack(">I", 1 if on else 0))

    def get_dde_gain(self) -> int:
        body = self._command(FFC.GET_DDE_GAIN)
        return struct.unpack(">i", body[:4])[0]  # signed per IDD

    def set_dde_gain(self, gain: int) -> None:
        """DDE gain. Boson IDD documents a range of about -20..+20 or
        0..9 depending on firmware. Verify against your IDD."""
        self._command(FFC.SET_DDE_GAIN, struct.pack(">i", int(gain)))

    def get_sso_state(self) -> bool:
        body = self._command(FFC.GET_SSO_STATE)
        return bool(struct.unpack(">I", body[:4])[0])

    def set_sso_state(self, on: bool) -> None:
        self._command(FFC.SET_SSO_STATE,
                      struct.pack(">I", 1 if on else 0))

    def get_plateau_value(self) -> int:
        body = self._command(FFC.GET_PLATEAU_VALUE)
        return struct.unpack(">I", body[:4])[0]

    def set_plateau_value(self, value: int) -> None:
        self._command(FFC.SET_PLATEAU_VALUE,
                      struct.pack(">I", int(value) & 0xFFFFFFFF))

    def get_brightness_bias(self) -> int:
        body = self._command(FFC.GET_BRIGHTNESS_BIAS)
        return struct.unpack(">i", body[:4])[0]

    def set_brightness_bias(self, bias: int) -> None:
        self._command(FFC.SET_BRIGHTNESS_BIAS, struct.pack(">i", int(bias)))

    def get_gain_mode(self) -> int:
        body = self._command(FFC.GET_GAIN_MODE)
        return struct.unpack(">I", body[:4])[0]

    def set_gain_mode(self, mode: int) -> None:
        self._command(FFC.SET_GAIN_MODE, struct.pack(">I", mode & 0xFFFFFFFF))


# ───────────────────────────────────────────────────────────────
# CLI: quick probe / one-off commands
# ───────────────────────────────────────────────────────────────

def _format_payload_hex(payload: bytes) -> str:
    return " ".join(f"{b:02x}" for b in payload)


@contextmanager
def _open(port: str = "auto"):
    bc = BosonControl(port=port)
    bc.open()
    try:
        yield bc
    finally:
        bc.close()


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Boson serial probe + one-off commands."
    )
    ap.add_argument("--port", default="auto",
                    help="COM port. 'auto' probes for Boson by sending "
                         "GET_SOFTWARE_REV at 921600 and watching for a "
                         "valid FFC response.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("probe", help="Open + read identity, no state changes")
    sub.add_parser("ffc", help="Trigger manual FFC")
    sub.add_parser("status", help="Read current GAO/DDE/SSO state")
    p_dde = sub.add_parser("dde", help="Set DDE on/off and gain")
    p_dde.add_argument("--on", action="store_true")
    p_dde.add_argument("--off", dest="on", action="store_false")
    p_dde.add_argument("--gain", type=int, default=None)
    p_gao = sub.add_parser("gao", help="Set camera-side AGC mode")
    p_gao.add_argument("mode", type=int,
                       help="GAOMode enum: 0=Linear 1=Plateau 2=IGE "
                            "3=IGE-EQ 4=Manual 6=AutoBright 7=LinearAGC")
    args = ap.parse_args(argv)

    with _open(args.port) as bc:
        if args.cmd == "probe":
            sw = bc.get_software_rev()
            try:
                sn = bc.get_camera_sn()
                pn = bc.get_camera_pn()
            except FFCError as e:
                sn = pn = b""
                log.warning("optional identity command failed: %s", e)
            print(f"port:    {bc.port}")
            print(f"sw_rev:  {_format_payload_hex(sw)}")
            print(f"sn:      {_format_payload_hex(sn)}")
            print(f"pn:      {_format_payload_hex(pn)}")
        elif args.cmd == "ffc":
            print("triggering manual FFC...")
            bc.run_ffc()
            print("FFC complete")
        elif args.cmd == "status":
            print(f"GAO mode: {bc.get_gao_mode()}")
            print(f"DDE state: {bc.get_dde_state()}")
            print(f"DDE gain: {bc.get_dde_gain()}")
            print(f"SSO state: {bc.get_sso_state()}")
            print(f"Plateau: {bc.get_plateau_value()}")
            print(f"Brightness bias: {bc.get_brightness_bias()}")
            print(f"Gain mode: {bc.get_gain_mode()}")
        elif args.cmd == "dde":
            bc.set_dde_state(args.on)
            print(f"DDE state -> {args.on}")
            if args.gain is not None:
                bc.set_dde_gain(args.gain)
                print(f"DDE gain  -> {args.gain}")
        elif args.cmd == "gao":
            bc.set_gao_mode(args.mode)
            print(f"GAO mode  -> {args.mode}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

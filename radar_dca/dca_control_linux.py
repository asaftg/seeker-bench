"""Linux-native DCA1000EVM controller — Python UDP, no TI .exe.

Drop-in replacement for ``radar_dca.dca_control.DCAControl`` on Linux,
where ``DCA1000EVM_CLI_Control.exe`` (the TI Windows binary the
original implementation shells out to) doesn't exist.

Implements the same six high-level methods (fpga_version,
query_sys_status, reset_fpga, setup_capture, start_record, stop_record)
by talking the documented DCA1000 control protocol directly over
UDP/4096. Protocol confirmed working on Jetson 2026-05-08 against
DCA1000EVM FPGA v4.130:

    SYSTEM_CONNECT     (0x0009) -> reply OK
    READ_FPGA_VERSION  (0x000e) -> reply 0x0482 = v4.130
    RESET_FPGA         (0x000a) -> reply OK
    CONFIG_FPGA_GEN    (0x0003) -> sets lvds/format/timer
    CONFIG_PACKET_DATA (0x000b) -> sets per-packet delay
    RECORD_START       (0x0005) -> begin streaming UDP/4098
    RECORD_STOP        (0x0006) -> stop streaming

Wire format per TI SPRUIJ4A:
    HEADER(2=0xa55a) CMD(2) DATA_SIZE(2) [DATA] FOOTER(2=0xeeaa)
    All multi-byte fields little-endian.

Reply format:
    HEADER(2) CMD(2) STATUS_OR_DATA(2) FOOTER(2)
    STATUS_OR_DATA: 0 = success on commands without return data; for
    READ_FPGA_VERSION it carries the version code as
    (minor | major<<8) packed (e.g. 0x82 0x04 = minor=130, major=4).

Mirrors the cf.json defaults the Windows path uses, so the wire-side
behaviour is byte-identical to the Windows code path:
    lvdsMode=1, dataTransferMode=1, dataCaptureMode=2,
    dataFormatMode=3, packetDelay_us=25, packetSize=1470.
"""
from __future__ import annotations

import logging
import socket
import struct
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# Protocol constants
_HEADER = 0xA55A
_FOOTER = 0xEEAA

_CMD_RESET_FPGA          = 0x000A
_CMD_RESET_AR_DEV        = 0x000C
_CMD_CONFIG_FPGA_GEN     = 0x0003
_CMD_CONFIG_EEPROM       = 0x0004
_CMD_RECORD_START        = 0x0005
_CMD_RECORD_STOP         = 0x0006
_CMD_PLAYBACK_START      = 0x0007
_CMD_PLAYBACK_STOP       = 0x0008
_CMD_SYSTEM_CONNECT      = 0x0009
_CMD_READ_FPGA_VERSION   = 0x000E
_CMD_CONFIG_PACKET_DATA  = 0x000B
_CMD_QUERY_SYS_STATUS    = 0x000F


class DCAControlError(RuntimeError):
    """Raised when a DCA control packet times out or returns non-zero status."""


@dataclass
class FpgaVersion:
    raw: str
    version: str          # "4.130" etc.
    flavor: str           # "ASIC"/"RF" — Linux backend distinguishes via the MSB of the version byte


@dataclass
class SystemStatus:
    raw: str
    connected: bool


def _build(cmd: int, data: bytes = b"") -> bytes:
    """HEADER + CMD + DATA_SIZE + DATA + FOOTER, all little-endian."""
    return (
        struct.pack("<HHH", _HEADER, cmd, len(data))
        + data
        + struct.pack("<H", _FOOTER)
    )


def _parse_reply(reply: bytes, expected_cmd: int) -> int:
    """Verify header/cmd/footer; return the 16-bit status/data field."""
    if len(reply) < 8:
        raise DCAControlError(f"reply too short ({len(reply)}B): {reply.hex()}")
    hdr, cmd, status = struct.unpack("<HHH", reply[:6])
    (footer,) = struct.unpack("<H", reply[-2:])
    if hdr != _HEADER or footer != _FOOTER:
        raise DCAControlError(
            f"bad framing: hdr=0x{hdr:04x} footer=0x{footer:04x} (expected 0xa55a / 0xeeaa)"
        )
    if cmd != expected_cmd:
        raise DCAControlError(
            f"reply cmd mismatch: got 0x{cmd:04x}, expected 0x{expected_cmd:04x}"
        )
    return status


class DCAControl:
    """Linux-native DCA1000EVM controller. Same public API as the
    Windows-side ``radar_dca.dca_control.DCAControl`` so callers don't
    need to know which platform they're on.
    """

    # CONFIG_FPGA_GEN payload is 6 bytes per TI DCA1000EVM CLI Software
    # Developer Guide v1.01 Table 6:
    #   byte0  dataLoggingMode  (1=Raw, 2=Multi)
    #   byte1  lvdsMode         (1=4-lane, 2=2-lane)
    #   byte2  dataTransferMode (1=LVDS capture, 2=playback)
    #   byte3  dataCaptureMode  (1=SD, 2=Ethernet)
    #   byte4  dataFormatMode   (1=12bit, 2=14bit, 3=16bit)
    #   byte5  timer            (seconds; 0=run forever, 30=default)
    # CRITICAL: an earlier revision of this file omitted byte0 entirely.
    # The FPGA only range-validates at start time, not parse time, so it
    # replied status=0 OK but the bytes shifted left and the device sat
    # in playback+invalid-capture-mode silently. ZERO UDP would arrive.
    _DEFAULT_DATA_LOGGING_MODE  = 1   # 1=Raw mode
    _DEFAULT_LVDS_MODE          = 2   # 2=2-lane (correct for AWR2944P)
    _DEFAULT_DATA_XFER_MODE     = 1   # 1=LVDS capture
    _DEFAULT_DATA_CAPTURE_MODE  = 2   # 2=ethernetStream
    _DEFAULT_DATA_FORMAT_MODE   = 3   # 3=16-bit real
    _DEFAULT_TIMER_S            = 30  # capture-timeout in seconds (0=disabled)
    _DEFAULT_PACKET_SIZE        = 1470
    _DEFAULT_PACKET_DELAY_US    = 25  # 25us — matches Windows cf.json

    def __init__(
        self,
        *,
        cli_path: Optional[str] = None,   # accepted for API compat, ignored on Linux
        host_ip: str = "192.168.33.30",
        dca_ip: str = "192.168.33.180",
        config_port: int = 4096,
        data_port: int = 4098,
        timeout_s: float = 1.5,
        lvds_mode: Optional[int] = None,
        packet_delay_us: Optional[int] = None,
    ) -> None:
        self.cli_path = cli_path  # unused; here so Windows callers don't error
        self.host_ip = host_ip
        self.dca_ip = dca_ip
        self.config_port = int(config_port)
        self.data_port = int(data_port)
        self.timeout_s = float(timeout_s)
        self._lvds_mode = lvds_mode if lvds_mode is not None else self._DEFAULT_LVDS_MODE
        self._packet_delay_us = (
            packet_delay_us if packet_delay_us is not None else self._DEFAULT_PACKET_DELAY_US
        )

    # ---------------------- low-level UDP RPC ----------------------
    def _send(self, cmd: int, data: bytes = b"", *, fire_and_forget: bool = False) -> int:
        """Send one command, return reply status. Binds an ephemeral
        UDP socket on (host_ip, config_port) for each call so the FPGA
        returns the reply to the same port we sent from."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((self.host_ip, self.config_port))
        except OSError as e:
            sock.close()
            raise DCAControlError(
                f"bind {self.host_ip}:{self.config_port} failed ({e}). "
                f"Another process may be holding the DCA control port."
            ) from e
        sock.settimeout(self.timeout_s)
        try:
            pkt = _build(cmd, data)
            log.debug("DCA tx cmd=0x%04x size=%d hex=%s", cmd, len(data), pkt.hex())
            sock.sendto(pkt, (self.dca_ip, self.config_port))
            if fire_and_forget:
                return 0
            try:
                reply, addr = sock.recvfrom(1024)
            except socket.timeout as e:
                raise DCAControlError(
                    f"DCA cmd 0x{cmd:04x} timed out after {self.timeout_s}s"
                ) from e
            log.debug("DCA rx from %s hex=%s", addr, reply.hex())
            return _parse_reply(reply, cmd)
        finally:
            sock.close()

    # ---------------------- high-level commands ----------------------
    def fpga_version(self) -> FpgaVersion:
        """Read FPGA bitstream version. Status field encodes
        (minor | major<<8); the high bit of the low byte distinguishes
        ASIC vs RF flavor."""
        status = self._send(_CMD_READ_FPGA_VERSION)
        major = (status >> 8) & 0xFF
        minor = status & 0x7F
        flavor = "RF" if (status & 0x0080) else "ASIC"
        ver = f"{major}.{minor}"
        return FpgaVersion(raw=f"FPGA Version: {ver} [{flavor}]", version=ver, flavor=flavor)

    def query_sys_status(self) -> SystemStatus:
        """Ask FPGA whether the AWR is actively pushing LVDS samples.
        Status 0 = connected, nonzero = not connected."""
        try:
            status = self._send(_CMD_QUERY_SYS_STATUS)
        except DCAControlError as e:
            return SystemStatus(raw=str(e), connected=False)
        connected = status == 0
        raw = "System is connected" if connected else f"System status: 0x{status:04x}"
        return SystemStatus(raw=raw, connected=connected)

    # ---------------------- capture lifecycle ----------------------
    def reset_fpga(self) -> None:
        """Soft-reset DCA FPGA. Non-fatal on failure."""
        try:
            self._send(_CMD_RESET_FPGA)
        except DCAControlError as e:
            log.warning("reset_fpga: %s (continuing -- non-fatal)", e)

    def setup_capture(self) -> None:
        """Two commands in TI order:
        (1) SYSTEM_CONNECT — handshake (the Windows DLL does this implicitly)
        (2) CONFIG_FPGA_GEN — lvds/format/timer
        (3) CONFIG_PACKET_DATA — packetSize + packetDelay_us.
        After this, ``start_record()`` begins UDP streaming."""
        try:
            self._send(_CMD_SYSTEM_CONNECT)
        except DCAControlError as e:
            raise DCAControlError(f"system_connect failed: {e}") from e

        cfg = struct.pack(
            "<BBBBBB",
            self._DEFAULT_DATA_LOGGING_MODE,    # byte0 — was missing pre-2026-05-08 fix
            self._lvds_mode,                    # byte1 (2-lane for AWR2944P)
            self._DEFAULT_DATA_XFER_MODE,       # byte2 (LVDS capture, NOT playback)
            self._DEFAULT_DATA_CAPTURE_MODE,    # byte3 (Ethernet stream)
            self._DEFAULT_DATA_FORMAT_MODE,     # byte4 (16-bit)
            self._DEFAULT_TIMER_S,              # byte5
        )
        try:
            status = self._send(_CMD_CONFIG_FPGA_GEN, cfg)
        except DCAControlError as e:
            raise DCAControlError(f"CONFIG_FPGA_GEN failed: {e}") from e
        if status != 0:
            raise DCAControlError(f"CONFIG_FPGA_GEN returned status 0x{status:04x}")

        pkt = struct.pack("<HI", self._DEFAULT_PACKET_SIZE, self._packet_delay_us)
        try:
            status = self._send(_CMD_CONFIG_PACKET_DATA, pkt)
        except DCAControlError as e:
            raise DCAControlError(f"CONFIG_PACKET_DATA failed: {e}") from e
        if status != 0:
            raise DCAControlError(f"CONFIG_PACKET_DATA returned status 0x{status:04x}")

        log.info(
            "DCA setup_capture OK (lvdsMode=%d, dataFormatMode=%d, packetDelay=%dus)",
            self._lvds_mode, self._DEFAULT_DATA_FORMAT_MODE, self._packet_delay_us,
        )

    def start_record(self) -> None:
        """Begin forwarding LVDS samples to UDP/4098. Non-fatal on
        failure — a missing UDP-4098 stream downstream is the right
        diagnostic anyway."""
        try:
            status = self._send(_CMD_RECORD_START)
            if status != 0:
                log.warning("RECORD_START returned status 0x%04x", status)
        except DCAControlError as e:
            log.warning("RECORD_START: %s", e)

    def stop_record(self) -> None:
        """Stop UDP streaming. Safe even if start_record was never issued."""
        try:
            self._send(_CMD_RECORD_STOP)
        except DCAControlError:
            pass


# Sanity smoke test if run directly.
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    c = DCAControl()
    print(c.fpga_version())
    c.reset_fpga()
    c.setup_capture()
    print(c.query_sys_status())
    c.start_record()
    time.sleep(0.5)
    c.stop_record()

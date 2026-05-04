"""Waveshare ST-series serial-bus servo driver (STS protocol over pyserial).

Wire model: every servo on the bus shares one half-duplex UART line. The
Waveshare Bus Servo Adapter (A) is a USB-C ↔ TTL bridge that handles
direction switching, so plain pyserial works at 1 Mbps (the ST-series
default). Each servo has a unique 1-byte ID; we daisy-chain them and
address by ID. Two servos are needed for pan/tilt — one must be
re-flashed from the factory ID 1 to ID 2 before first use; see
scripts/st3025_set_id.py.

Protocol summary (STS, used by Waveshare ST3025 / Feetech STS series):

    request:  0xFF 0xFF [ID] [LEN] [INSTR] [PARAMS...] [CHECKSUM]
    reply  :  0xFF 0xFF [ID] [LEN] [ERROR] [PARAMS...] [CHECKSUM]

LEN = number of params + 2 (covers INSTR/ERROR + checksum). CHECKSUM =
~(ID + LEN + INSTR + Σparams) & 0xFF.

INSTR codes used here:
    0x01 PING
    0x02 READ   params: [start_addr, num_bytes_to_read]
    0x03 WRITE  params: [start_addr, byte0, byte1, ...]

Word ordering: STS is little-endian (low byte first). SCS-protocol
variants of the same hardware family use big-endian — if a future
firmware ships with that, swap the helpers below.

Register map for ST3025 (subset we need):

    0x05  ID                             (1 B)
    0x09  Min Position Limit             (2 B)
    0x0B  Max Position Limit             (2 B)
    0x37  EEPROM Lock                    (1 B)   write 0 to unlock
    0x2A  Goal Position                  (2 B)
    0x38  Present Position               (2 B)
    0x3E  Present Voltage (×0.1 V)       (1 B)
    0x3F  Present Temperature (°C)       (1 B)
    0x41  Status / Error                 (1 B)

Driver contract (mirrors MaestroDriver shape so GimbalManager can swap
without restructuring):

    open() -> bool                        try to open the port
    close() -> None
    is_open: bool                         property
    port: str | None                      currently-open device
    ping(servo_id) -> bool                round-trip a 0x01 packet
    set_target_units(servo_id, raw)       write 2-byte goal position
    read_position(servo_id) -> int|None   read 2-byte present position
    get_last_written_units(servo_id)      cache of last successful write
    read_status(servo_id) -> dict|None    voltage / temp / error byte
    set_id(old_id, new_id) -> bool        admin helper for set_id script
    release_all(ids) -> None              float servos (best-effort)

Never raises on absent hardware — open() returns False so callers can
run in disconnected mode.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

from common.logging_setup import get_logger

log = get_logger(__name__)


# ── protocol constants ──────────────────────────────────────────

HEADER = b"\xff\xff"

INSTR_PING = 0x01
INSTR_READ = 0x02
INSTR_WRITE = 0x03

# register addresses (ST3025 / STS)
REG_ID = 0x05
REG_MIN_POS = 0x09
REG_MAX_POS = 0x0B
REG_EEPROM_LOCK = 0x37
REG_KP = 0x15        # Position P gain
REG_KD = 0x16        # Position D gain
REG_KI = 0x17        # Position I gain
REG_ACCELERATION = 0x29
REG_GOAL_POS = 0x2A
REG_PRESENT_POS = 0x38
REG_PRESENT_VOLTAGE = 0x3E
REG_PRESENT_TEMP = 0x3F
REG_PRESENT_STATUS = 0x41

# Adapter (A) shows up on Windows as a USB-CDC port; the WCH CH343 chip
# inside is the most common variant.
WAVESHARE_VID_PID_HINTS: Tuple[Tuple[int, int], ...] = (
    (0x1A86, 0x55D3),  # WCH CH343 (Adapter A typical VID/PID)
    (0x1A86, 0x7523),  # CH340 fallback (older Waveshare variants)
)

DEFAULT_BAUD = 1_000_000
READ_TIMEOUT_S = 0.05      # half-duplex round-trip is ~0.5 ms; 50 ms is generous
WRITE_TIMEOUT_S = 0.5


def _checksum(buf: bytes) -> int:
    return (~sum(buf)) & 0xFF


def _build_packet(servo_id: int, instr: int, params: bytes) -> bytes:
    length = len(params) + 2
    body = bytes([servo_id & 0xFF, length & 0xFF, instr & 0xFF]) + params
    return HEADER + body + bytes([_checksum(body)])


def _u16le(v: int) -> bytes:
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def _read_u16le(buf: bytes, offset: int) -> int:
    return buf[offset] | (buf[offset + 1] << 8)


class BusServoDriver:
    """Half-duplex STS bus driver for one or many servos."""

    def __init__(self, port: Optional[str] = None,
                 baud: int = DEFAULT_BAUD,
                 min_packet_gap_s: float = 0.0015) -> None:
        self._explicit_port = port
        self._port: Optional[str] = None
        self._ser = None  # serial.Serial | None
        self._baud = int(baud)
        self._last_units: Dict[int, int] = {}
        # Minimum spacing between successive bus operations. The half-
        # duplex transceiver eats the second of two back-to-back packets
        # if they hit the wire too fast (~80 µs apart at 1 Mbps).
        # 1.5 ms is reliable on the Waveshare adapter and barely
        # noticeable inside the manager's 16 ms tick.
        self._min_packet_gap_s = float(min_packet_gap_s)
        self._last_bus_t = 0.0
        # Warn at most once per process when no adapter is found. Callers
        # may invoke open() in a hot reconnect loop; without this, the
        # warning floods the global logger queue and stalls publisher
        # threads.
        self._missing_warned = False

    # ── discovery ───────────────────────────────────────────────

    @staticmethod
    def list_candidate_ports() -> List[Tuple[str, str]]:
        """Return [(port, description), ...] for every CDC port that
        looks like a Waveshare bus-servo adapter (CH343/CH340 by VID/PID).
        Description includes any vendor product string for the operator
        to pick the right one if multiple matches exist.
        """
        try:
            from serial.tools import list_ports
        except Exception as e:
            log.warning("pyserial not installed (%s) — gimbal disabled", e)
            return []
        hits: List[Tuple[str, str]] = []
        for p in list_ports.comports():
            vid = getattr(p, "vid", None)
            pid = getattr(p, "pid", None)
            if vid is None or pid is None:
                continue
            if (vid, pid) in WAVESHARE_VID_PID_HINTS:
                desc = (p.description or "") + " " + (p.product or "")
                hits.append((p.device, desc.strip()))
        return hits

    # ── lifecycle ───────────────────────────────────────────────

    def open(self) -> bool:
        """Open the port. Returns True on success, False otherwise.

        Never raises if the device is missing — logs and returns False.
        """
        try:
            import serial
        except Exception as e:
            log.warning("pyserial missing (%s) — gimbal unavailable", e)
            return False

        port = self._explicit_port
        if port is None:
            hits = self.list_candidate_ports()
            if not hits:
                if not self._missing_warned:
                    log.warning(
                        "No Waveshare bus-servo adapter found on any COM "
                        "port — gimbal disconnected. Subsequent reconnect "
                        "attempts will be silent."
                    )
                    self._missing_warned = True
                else:
                    log.debug("No Waveshare bus-servo adapter (retry)")
                return False
            port, desc = hits[0]
            log.info("Bus-servo adapter auto-detected on %s (%s)", port, desc)
            # Adapter has reappeared since the last warning — clear the
            # one-shot so a future disconnect logs again.
            self._missing_warned = False

        try:
            # rtscts/dsrdtr off + DTR/RTS deasserted at open: some adapter
            # variants tie RTS to the half-duplex transceiver's DIR pin,
            # so leaving it asserted (pyserial's default) pins the bus in
            # TX mode and the servo's reply never reaches us. Toggle them
            # off explicitly to be safe.
            self._ser = serial.Serial()
            self._ser.port = port
            self._ser.baudrate = self._baud
            self._ser.timeout = READ_TIMEOUT_S
            self._ser.write_timeout = WRITE_TIMEOUT_S
            self._ser.rtscts = False
            self._ser.dsrdtr = False
            self._ser.dtr = False
            self._ser.rts = False
            self._ser.open()
            self._port = port
            log.info("BusServoDriver opened on %s @ %d baud", port, self._baud)
            return True
        except Exception as e:
            log.warning("Failed to open bus-servo adapter on %s: %s", port, e)
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

    # ── low-level transport ─────────────────────────────────────

    def _write_packet(self, packet: bytes) -> bool:
        if not self.is_open:
            return False
        try:
            # Enforce minimum inter-packet gap. Spaced naturally in full
            # app (other threads steal CPU), but in gimbal-only mode the
            # loop fires writes too fast and the bus eats them.
            import time as _t
            now = _t.perf_counter()
            wait = self._min_packet_gap_s - (now - self._last_bus_t)
            if wait > 0:
                _t.sleep(wait)
            # Flush any stale reply bytes from a prior failed exchange so
            # the next read aligns to a fresh response header.
            try:
                self._ser.reset_input_buffer()
            except Exception:
                pass
            self._ser.write(packet)
            self._last_bus_t = _t.perf_counter()
            return True
        except Exception as e:
            log.warning("BusServoDriver write failed: %s", e)
            return False

    def _read_status(self, expected_id: int, expected_params: int) -> Optional[bytes]:
        """Read one status reply. Returns the params bytes (without
        header/ID/LEN/ERROR/checksum) or None on protocol error.

        We pull the fixed-size frame in one read; the timeout caps the
        wait so a missing servo doesn't hang us.
        """
        if not self.is_open:
            return None
        # Frame layout: 0xFF 0xFF ID LEN ERROR [params...] CHK
        # total = 2 + 1 + 1 + 1 + expected_params + 1
        n = 6 + expected_params
        try:
            buf = self._ser.read(n)
        except Exception as e:
            log.warning("BusServoDriver read failed: %s", e)
            return None
        finally:
            # Mark bus activity time so the next write respects the gap.
            import time as _t
            self._last_bus_t = _t.perf_counter()
        if len(buf) < n:
            return None
        if buf[0:2] != HEADER:
            return None
        servo_id = buf[2]
        length = buf[3]
        error = buf[4]
        if servo_id != expected_id:
            return None
        if length != expected_params + 2:
            return None
        body = bytes(buf[2:-1])
        if _checksum(body) != buf[-1]:
            return None
        if error != 0:
            log.debug("Servo %d returned error byte 0x%02X", servo_id, error)
        return bytes(buf[5:5 + expected_params])

    # ── high-level commands ─────────────────────────────────────

    def ping(self, servo_id: int) -> bool:
        if not self._write_packet(_build_packet(servo_id, INSTR_PING, b"")):
            return False
        return self._read_status(servo_id, expected_params=0) is not None

    def set_target_units(self, servo_id: int, units: int) -> bool:
        """Write the 2-byte Goal Position register on `servo_id`.

        ST3025 has no acknowledge mode by default — we don't read back
        a status frame to keep the loop cheap. Caller can verify with
        read_position() if it needs closed-loop confirmation.
        """
        if not self.is_open:
            return False
        u = max(0, min(UNITS_PER_REV_GUARD, int(units)))
        params = bytes([REG_GOAL_POS]) + _u16le(u)
        ok = self._write_packet(_build_packet(servo_id, INSTR_WRITE, params))
        if ok:
            self._last_units[int(servo_id)] = u
        return ok

    def read_position(self, servo_id: int) -> Optional[int]:
        """Read the 2-byte Present Position register. Returns raw units
        in [0, 4095] on success, None on transport / protocol error.
        """
        params = bytes([REG_PRESENT_POS, 2])
        if not self._write_packet(_build_packet(servo_id, INSTR_READ, params)):
            return None
        reply = self._read_status(servo_id, expected_params=2)
        if reply is None:
            return None
        return _read_u16le(reply, 0)

    def get_last_written_units(self, servo_id: int) -> Optional[int]:
        return self._last_units.get(int(servo_id))

    def set_position_i_gain(self, servo_id: int, ki: int) -> bool:
        """Write the Position-Integral gain register (0x17, 1 byte).

        I=0 (factory default) leaves a steady-state error against any
        constant load — gravity on a tilted-up gimbal sits the axis a
        couple degrees below the commanded angle. A small I drives that
        residual to zero. Too large hunts. 1..4 is the safe range.
        """
        if not self.is_open:
            return False
        ki = max(0, min(255, int(ki)))
        params = bytes([REG_KI, ki])
        return self._write_packet(_build_packet(servo_id, INSTR_WRITE, params))

    def set_acceleration(self, servo_id: int, accel: int) -> bool:
        """Write the Acceleration register (0x29, 1 byte) for `servo_id`.

        Higher = snappier (closer to step input). Lower = smoother ramp.
        0 disables internal accel limiting (default — produces noticeable
        jerk on per-tick goal-position writes at high update rates).
        Sensible smoothing values: 30..80.
        """
        if not self.is_open:
            return False
        a = max(0, min(255, int(accel)))
        params = bytes([REG_ACCELERATION, a])
        return self._write_packet(_build_packet(servo_id, INSTR_WRITE, params))

    def read_status(self, servo_id: int) -> Optional[dict]:
        """Read voltage, temperature, and status byte in one go.

        Registers 0x3E..0x41 are contiguous (4 bytes: voltage, temp,
        reserved, status). One READ covers them all.
        """
        params = bytes([REG_PRESENT_VOLTAGE, 4])
        if not self._write_packet(_build_packet(servo_id, INSTR_READ, params)):
            return None
        reply = self._read_status(servo_id, expected_params=4)
        if reply is None:
            return None
        return {
            "voltage_v": reply[0] / 10.0,
            "temperature_c": reply[1],
            "status": reply[3],
        }

    def set_id(self, old_id: int, new_id: int) -> bool:
        """Reassign a servo's bus ID. Workflow expects the bus to have
        only the target servo powered to avoid an ID-conflict reply storm.

        Sequence: unlock EEPROM (0x37=0) → write new ID (0x05) → lock
        EEPROM (0x37=1). Each EEPROM write needs a few ms to commit
        before the next packet hits or the change won't persist past
        a power cycle. Returns True if all writes go out cleanly; the
        caller should ping the NEW id afterwards to confirm.
        """
        import time as _t
        unlock = bytes([REG_EEPROM_LOCK, 0])
        if not self._write_packet(_build_packet(old_id, INSTR_WRITE, unlock)):
            return False
        _t.sleep(0.02)
        write_id = bytes([REG_ID, new_id & 0xFF])
        if not self._write_packet(_build_packet(old_id, INSTR_WRITE, write_id)):
            return False
        _t.sleep(0.02)
        # After the ID change, the servo answers to new_id. Re-lock via
        # the new id.
        lock = bytes([REG_EEPROM_LOCK, 1])
        ok = self._write_packet(_build_packet(new_id, INSTR_WRITE, lock))
        _t.sleep(0.02)
        return ok

    def set_position_limits(self, servo_id: int, raw_min: int, raw_max: int) -> bool:
        """Write the EEPROM Min/Max Position Limits (firmware-side hard
        clamp). Software clamps in BusServoCalibration are the primary
        defence; this sets a backstop the servo enforces itself.
        """
        unlock = bytes([REG_EEPROM_LOCK, 0])
        if not self._write_packet(_build_packet(servo_id, INSTR_WRITE, unlock)):
            return False
        # Min and Max Position Limits live at 0x09 / 0x0B (2 B each).
        # Write both in one go starting at 0x09 (4 bytes total).
        block = bytes([REG_MIN_POS]) + _u16le(raw_min) + _u16le(raw_max)
        if not self._write_packet(_build_packet(servo_id, INSTR_WRITE, block)):
            return False
        lock = bytes([REG_EEPROM_LOCK, 1])
        return self._write_packet(_build_packet(servo_id, INSTR_WRITE, lock))

    def release_all(self, ids: Iterable[int]) -> None:
        """Best-effort float — the ST3025 holds torque continuously and
        has no PWM-zero release equivalent. The closest real action is
        to disable Torque (register 0x28). We do that here so the user
        can backdrive the gimbal by hand on shutdown without fighting
        the motor.

        Sends a small inter-packet delay between IDs because back-to-
        back writes on the half-duplex bus occasionally get the second
        packet eaten — symptom: one axis stays locked on shutdown.
        """
        import time as _t
        REG_TORQUE_ENABLE = 0x28
        for sid in ids:
            params = bytes([REG_TORQUE_ENABLE, 0])
            self._write_packet(_build_packet(sid, INSTR_WRITE, params))
            _t.sleep(0.02)


# Sentinel for the set_target_units bound. Pulled into a module-level
# constant so the calibration's UNITS_PER_REV-1 isn't a magic number
# duplicated here. Kept local to this module because the driver is the
# one place that talks raw to the wire.
UNITS_PER_REV_GUARD = 4095

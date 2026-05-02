"""Unit tests for gimbal.bus_servo_driver — packet round-trips against
an in-memory fake serial port. Verifies the STS protocol byte layout
and high-level command behavior without hardware.
"""
from __future__ import annotations

import pytest

from gimbal import bus_servo_driver as bsd


# ── fake serial ──────────────────────────────────────────────────

class FakeSerial:
    """Minimal serial.Serial stand-in.

    write() captures every outgoing byte sequence in `writes`. read(n)
    pops the next pre-queued reply (or returns b'' on timeout).
    """
    def __init__(self):
        self.is_open = True
        self.writes: list[bytes] = []
        self._reply_queue: list[bytes] = []

    def queue_reply(self, b: bytes) -> None:
        self._reply_queue.append(b)

    def write(self, b: bytes) -> int:
        self.writes.append(bytes(b))
        return len(b)

    def read(self, n: int) -> bytes:
        if not self._reply_queue:
            return b""
        return self._reply_queue.pop(0)

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        self.is_open = False


def _make_driver(fake: FakeSerial) -> bsd.BusServoDriver:
    drv = bsd.BusServoDriver(port="FAKE")
    drv._ser = fake
    drv._port = "FAKE"
    return drv


def _status_frame(servo_id: int, params: bytes, error: int = 0) -> bytes:
    """Build a well-formed STS status reply for the fake to return."""
    length = len(params) + 2
    body = bytes([servo_id, length, error]) + params
    chk = (~sum(body)) & 0xFF
    return bsd.HEADER + body + bytes([chk])


# ── checksum / packet construction ───────────────────────────────

def test_checksum_matches_spec():
    body = bytes([0x01, 0x02, 0x01])
    assert bsd._checksum(body) == 0xFB


def test_build_ping_packet_layout():
    pkt = bsd._build_packet(servo_id=1, instr=bsd.INSTR_PING, params=b"")
    assert pkt == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_build_write_goal_position_packet():
    # WRITE 0x2A with raw=1024 (0x0400) → params [0x2A, 0x00, 0x04]
    pkt = bsd._build_packet(servo_id=1, instr=bsd.INSTR_WRITE,
                            params=bytes([bsd.REG_GOAL_POS]) + bsd._u16le(1024))
    # FF FF 01 05 03 2A 00 04 [chk]
    body = bytes([0x01, 0x05, 0x03, 0x2A, 0x00, 0x04])
    chk = (~sum(body)) & 0xFF
    assert pkt == bytes([0xFF, 0xFF]) + body + bytes([chk])


# ── ping ─────────────────────────────────────────────────────────

def test_ping_success_with_valid_reply():
    fake = FakeSerial()
    fake.queue_reply(_status_frame(servo_id=1, params=b""))
    drv = _make_driver(fake)
    assert drv.ping(1) is True
    assert fake.writes[0] == bytes([0xFF, 0xFF, 0x01, 0x02, 0x01, 0xFB])


def test_ping_fails_on_no_reply():
    fake = FakeSerial()
    drv = _make_driver(fake)
    assert drv.ping(1) is False


def test_ping_fails_on_wrong_servo_id_in_reply():
    fake = FakeSerial()
    fake.queue_reply(_status_frame(servo_id=99, params=b""))
    drv = _make_driver(fake)
    assert drv.ping(1) is False


def test_ping_fails_on_corrupted_checksum():
    fake = FakeSerial()
    bad = bytearray(_status_frame(servo_id=1, params=b""))
    bad[-1] ^= 0xFF
    fake.queue_reply(bytes(bad))
    drv = _make_driver(fake)
    assert drv.ping(1) is False


# ── set_target_units ─────────────────────────────────────────────

def test_set_target_units_writes_goal_position_register():
    fake = FakeSerial()
    drv = _make_driver(fake)
    assert drv.set_target_units(servo_id=2, units=2048) is True
    body = bytes([0x02, 0x05, 0x03, bsd.REG_GOAL_POS, 0x00, 0x08])
    chk = (~sum(body)) & 0xFF
    assert fake.writes[0] == bytes([0xFF, 0xFF]) + body + bytes([chk])


def test_set_target_units_caches_last_written():
    fake = FakeSerial()
    drv = _make_driver(fake)
    drv.set_target_units(1, 1024)
    assert drv.get_last_written_units(1) == 1024
    drv.set_target_units(1, 2000)
    assert drv.get_last_written_units(1) == 2000


def test_set_target_units_clamps_above_max_raw():
    fake = FakeSerial()
    drv = _make_driver(fake)
    drv.set_target_units(1, 9999)
    assert drv.get_last_written_units(1) == bsd.UNITS_PER_REV_GUARD


def test_set_target_units_returns_false_when_closed():
    drv = bsd.BusServoDriver(port="FAKE")
    assert drv.set_target_units(1, 1024) is False


# ── read_position ────────────────────────────────────────────────

def test_read_position_decodes_little_endian_u16():
    fake = FakeSerial()
    # raw = 0x07D0 = 2000 → bytes 0xD0 0x07 (little-endian)
    fake.queue_reply(_status_frame(servo_id=1, params=bytes([0xD0, 0x07])))
    drv = _make_driver(fake)
    assert drv.read_position(1) == 2000


def test_read_position_request_packet_bytes():
    fake = FakeSerial()
    fake.queue_reply(_status_frame(servo_id=1, params=bytes([0x00, 0x04])))
    drv = _make_driver(fake)
    drv.read_position(1)
    body = bytes([0x01, 0x04, 0x02, bsd.REG_PRESENT_POS, 0x02])
    chk = (~sum(body)) & 0xFF
    assert fake.writes[0] == bytes([0xFF, 0xFF]) + body + bytes([chk])


def test_read_position_returns_none_on_no_reply():
    fake = FakeSerial()
    drv = _make_driver(fake)
    assert drv.read_position(1) is None


# ── read_status ──────────────────────────────────────────────────

def test_read_status_decodes_voltage_temp_and_status_byte():
    fake = FakeSerial()
    # voltage=0x78 (12.0 V), temp=0x2D (45 °C), reserved=0x00, status=0x00
    fake.queue_reply(_status_frame(
        servo_id=2, params=bytes([0x78, 0x2D, 0x00, 0x00])))
    drv = _make_driver(fake)
    s = drv.read_status(2)
    assert s == {"voltage_v": 12.0, "temperature_c": 45, "status": 0}


# ── set_id ───────────────────────────────────────────────────────

def test_set_id_emits_unlock_writeid_lock_sequence():
    fake = FakeSerial()
    drv = _make_driver(fake)
    drv.set_id(old_id=1, new_id=2)
    assert len(fake.writes) == 3
    # 1) unlock EEPROM on old id
    assert fake.writes[0][2] == 1                         # ID byte
    assert fake.writes[0][5] == bsd.REG_EEPROM_LOCK       # register
    assert fake.writes[0][6] == 0                         # value = unlock
    # 2) write new ID on old id
    assert fake.writes[1][2] == 1
    assert fake.writes[1][5] == bsd.REG_ID
    assert fake.writes[1][6] == 2
    # 3) re-lock EEPROM addressed to NEW id
    assert fake.writes[2][2] == 2
    assert fake.writes[2][5] == bsd.REG_EEPROM_LOCK
    assert fake.writes[2][6] == 1


# ── release_all ──────────────────────────────────────────────────

def test_release_all_writes_torque_disable_for_each_id():
    fake = FakeSerial()
    drv = _make_driver(fake)
    drv.release_all([1, 2])
    assert len(fake.writes) == 2
    REG_TORQUE_ENABLE = 0x28
    for w, sid in zip(fake.writes, [1, 2]):
        assert w[2] == sid
        assert w[5] == REG_TORQUE_ENABLE
        assert w[6] == 0

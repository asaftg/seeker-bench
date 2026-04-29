"""DCA1000 raw-ADC data port listener.

UDP socket bound to the host's data-port endpoint (default
``192.168.33.30:4098``). The DCA forwards LVDS samples from the AWR
as a stream of UDP packets:

::

    ┌──────────────────────────┬──────────────────┬─────────────────┐
    │ sequence_number (4 LE)   │ byte_count (6 LE)│ payload (≤1456) │
    └──────────────────────────┴──────────────────┴─────────────────┘
    └─────────── header (10 bytes) ─────────────┘

Per SPRUIJ4A §4 (Data Format). The 6-byte byte_count is a cumulative
counter of payload bytes since record-start; the sequence number is
1-based and increments by 1 per packet. We use both for drop
detection — gaps in the sequence number indicate UDP loss (jumbo
frames + firewall whitelist usually fix this).

The listener runs on a daemon thread, tallies counters, and exposes
``stats()`` that returns a snapshot dict. The DCAManager polls
``stats()`` once per heartbeat and publishes the numbers downstream.

This module does NOT parse ADC samples — that's M4 (range-Doppler
FFT). For now we just count bytes so the GUI can show a meaningful
"DCA data plane: 1.4 MB/s" indicator.
"""
from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from common.logging_setup import get_logger

log = get_logger(__name__)

# Per SPRUIJ4A: header is 4-byte seq + 6-byte cumulative byte count = 10 bytes
_HEADER_LEN = 10
# UDP MTU on a jumbo-frames link can be up to ~9 KB; default DCA packet
# is ~1466 bytes (1456 payload + 10 header). Use a generous receive
# buffer so we don't truncate.
_RECV_BUF = 16384


@dataclass
class DataPortStats:
    """Snapshot of listener counters at a moment in time."""
    listening: bool = False
    bound_addr: str = ""
    packets_total: int = 0
    bytes_total: int = 0
    seq_drops_total: int = 0
    last_seq: int = 0
    last_byte_count: int = 0
    # Rates over the last refresh window (computed by stats())
    packets_per_s: float = 0.0
    bytes_per_s: float = 0.0
    last_packet_age_s: float = float("inf")  # seconds since last UDP packet


class DataPortListener:
    """Background thread that counts UDP packets on the DCA data port.

    Lifecycle:

    - ``start()``: bind socket, spawn daemon thread.
    - ``stop()``: close socket, join thread.
    - ``stats()``: snapshot counters + compute rolling rates.

    The listener is safe to run even when no data is flowing — it
    blocks in ``recvfrom()`` with a short timeout and just loops.
    """

    def __init__(
        self,
        *,
        host_ip: str = "192.168.33.30",
        data_port: int = 4098,
        recv_timeout_s: float = 0.25,
    ) -> None:
        self.host_ip = host_ip
        self.data_port = int(data_port)
        self.recv_timeout_s = float(recv_timeout_s)
        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        # Counters (mutated only from the listener thread; read under _lock)
        self._packets_total = 0
        self._bytes_total = 0
        self._seq_drops_total = 0
        self._last_seq = 0
        self._last_byte_count = 0
        self._last_packet_t = 0.0
        # For rate computation: snapshot of the previous stats() call
        self._prev_t = 0.0
        self._prev_packets = 0
        self._prev_bytes = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Larger kernel receive buffer — DCA can burst at 100s of Mb/s.
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
        except OSError:
            pass  # not all platforms allow 8MB; best-effort
        sock.bind((self.host_ip, self.data_port))
        sock.settimeout(self.recv_timeout_s)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DCADataPort", daemon=True,
        )
        self._thread.start()
        self._prev_t = time.time()
        log.info("DataPortListener bound on %s:%d", self.host_ip, self.data_port)

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("DataPortListener stopped (packets=%d bytes=%d drops=%d)",
                 self._packets_total, self._bytes_total, self._seq_drops_total)

    # ─────────────────────── receive loop ──────────────────────────────────
    def _loop(self) -> None:
        sock = self._sock
        assert sock is not None
        while not self._stop.is_set():
            try:
                data, _ = sock.recvfrom(_RECV_BUF)
            except socket.timeout:
                continue
            except OSError:
                # Socket was closed from stop(); exit cleanly.
                break
            if len(data) < _HEADER_LEN:
                # Malformed packet — count nothing, keep going.
                continue
            seq = struct.unpack("<I", data[:4])[0]
            # 6-byte cumulative byte counter, LE — pad to 8 for unpack.
            byte_count = struct.unpack("<Q", data[4:10] + b"\x00\x00")[0]
            payload_len = len(data) - _HEADER_LEN
            now = time.time()
            with self._lock:
                # Sequence-gap detection. The first packet sets the
                # baseline; afterwards every gap >1 is a drop.
                if self._last_seq > 0:
                    expected = self._last_seq + 1
                    if seq != expected:
                        self._seq_drops_total += max(0, seq - expected)
                self._last_seq = seq
                self._last_byte_count = byte_count
                self._packets_total += 1
                self._bytes_total += payload_len
                self._last_packet_t = now

    # ─────────────────────── stats ─────────────────────────────────────────
    def stats(self) -> DataPortStats:
        """Snapshot counters + compute rolling rates since the last
        call. Designed to be called once per GUI heartbeat (~1 Hz)."""
        now = time.time()
        with self._lock:
            packets = self._packets_total
            bytes_ = self._bytes_total
            drops = self._seq_drops_total
            last_seq = self._last_seq
            last_bc = self._last_byte_count
            last_pkt_t = self._last_packet_t
            prev_t = self._prev_t
            prev_packets = self._prev_packets
            prev_bytes = self._prev_bytes
            self._prev_t = now
            self._prev_packets = packets
            self._prev_bytes = bytes_
        dt = max(1e-6, now - prev_t)
        listening = self._thread is not None and self._thread.is_alive()
        bound = f"{self.host_ip}:{self.data_port}" if self._sock is not None else ""
        return DataPortStats(
            listening=listening,
            bound_addr=bound,
            packets_total=packets,
            bytes_total=bytes_,
            seq_drops_total=drops,
            last_seq=last_seq,
            last_byte_count=last_bc,
            packets_per_s=(packets - prev_packets) / dt,
            bytes_per_s=(bytes_ - prev_bytes) / dt,
            last_packet_age_s=(now - last_pkt_t) if last_pkt_t > 0 else float("inf"),
        )

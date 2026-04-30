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

import os
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, TextIO

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
        queue_max_packets: int = 16384,
        record_bin_path: Optional[str] = None,
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
        # Payload queue. The listener appends each packet's payload
        # (header stripped) to this deque; consumers (DCAPipeline) call
        # drain_payloads() once per tick to pull all queued bytes.
        # maxlen bounds memory if the consumer stalls — deque silently
        # discards oldest. At ~22 Kpps × 1456 B/packet, 16k packets ≈
        # 23 MB ≈ 750 ms of buffering, plenty of slack for a slow tick.
        # The seq_drops_total counter still reports drops on the wire,
        # so a stalled consumer is visible separately as "queue rolled
        # while seq numbers were continuous".
        self._payload_queue: Deque[bytes] = deque(maxlen=queue_max_packets)
        self._payloads_dropped_queue_full = 0
        # Optional .bin recording — write raw payloads to a file as
        # we receive them. The format is the standard DCA1000 raw bin
        # (no headers, just concatenated chirp bytes) so it's
        # re-readable by parse_bin_full / parse_bin_streaming and
        # mmWave Studio. Set via constructor or recording_start().
        self._record_path: Optional[str] = record_bin_path
        self._record_fp = None  # type: ignore[var-annotated]
        # Phase-3 ``dca_index.csv`` writer — one row per UDP packet so
        # post-flight tooling can seek into the .bin without re-parsing
        # 60 MB/s from the start. Columns (per
        # docs/PHASE_3_RECORDER_FIELDS.md §"dca_index.csv"):
        #   ts_host_ns, byte_offset, payload_len, seq_num, chunk_offset
        # ``ts_host_ns`` is monotonic_ns at packet receive — the listener
        # already takes wall-clock time.time() for stats, but monotonic
        # is what the spec asks for and what's safe across NTP slews.
        self._index_fp: Optional[TextIO] = None
        self._index_path: Optional[str] = None
        # Running byte offset INTO ``_record_fp`` — the CSV's
        # ``byte_offset`` column. Reset to 0 when recording starts.
        self._index_offset_bytes: int = 0
        # Buffer rows in memory; flush every N rows OR every T seconds,
        # whichever comes first. At ~30 kpps, per-row flushes would
        # tank the listener. With N=100 rows / T=0.1 s we flush at
        # roughly the same cadence either way and bound row-loss on
        # crash to ~3 ms of capture.
        self._index_buf: List[str] = []
        self._INDEX_FLUSH_ROWS = 100
        self._INDEX_FLUSH_INTERVAL_S = 0.1
        self._index_last_flush_mono: float = 0.0

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
        # Open the recording file if requested at construction time.
        if self._record_path is not None and self._record_fp is None:
            try:
                self._record_fp = open(self._record_path, "wb")
                log.info("DataPortListener recording raw payloads to %s",
                         self._record_path)
                self._open_index_for(self._record_path)
            except OSError as e:
                log.warning("Could not open record path %s: %s",
                            self._record_path, e)
                self._record_fp = None
        self._thread = threading.Thread(
            target=self._loop, name="DCADataPort", daemon=True,
        )
        self._thread.start()
        self._prev_t = time.time()
        log.info("DataPortListener bound on %s:%d", self.host_ip, self.data_port)

    # ─────────────────────── recording control ─────────────────────────────
    def recording_start(self, path: str) -> None:
        """Begin appending raw payload bytes to ``path``. Safe to call
        while the listener thread is running — the open() happens
        under the lock so the next packet either lands in the new file
        or is silently buffered until the file opens.

        Also opens the sibling ``<stem>_radar.csv`` index for
        per-packet seek records (Phase 3, see
        docs/PHASE_3_RECORDER_FIELDS.md)."""
        with self._lock:
            if self._record_fp is not None:
                try:
                    self._record_fp.close()
                except OSError:
                    pass
                self._record_fp = None
            self._close_index_locked()
            try:
                self._record_fp = open(path, "wb")
                self._record_path = path
                log.info("DataPortListener: recording started → %s", path)
                self._open_index_for(path)
            except OSError as e:
                log.warning("recording_start: open(%s) failed: %s", path, e)

    def recording_stop(self) -> Optional[str]:
        """Close the recording file. Returns the path that was being
        written, or None if no recording was active."""
        with self._lock:
            if self._record_fp is None:
                return None
            try:
                self._record_fp.close()
            except OSError:
                pass
            self._record_fp = None
            path = self._record_path
            self._close_index_locked()
            log.info("DataPortListener: recording stopped (was %s)", path)
            return path

    # ─────────────────────── dca_index.csv plumbing ────────────────────────
    def _open_index_for(self, bin_path: str) -> None:
        """Open ``<stem>_radar.csv`` next to the .bin and write the
        header row. Caller holds ``_lock``. Best-effort: a failure to
        open the index does not affect .bin recording — we just log
        and skip. ``<stem>_radar.csv`` mirrors the .bin's
        ``<stem>_radar.bin`` naming so paired-file globbing works."""
        try:
            stem, _ext = os.path.splitext(bin_path)
            index_path = stem + ".csv"
            # Line-buffered text mode so an external tool tailing the
            # CSV sees rows promptly. Buffering is layered on top in
            # the in-memory list — the file handle's buffering is just
            # the OS write cadence.
            self._index_fp = open(index_path, "w", encoding="utf-8",
                                  newline="")
            self._index_path = index_path
            self._index_offset_bytes = 0
            self._index_buf = []
            self._index_last_flush_mono = time.monotonic()
            self._index_fp.write(
                "ts_host_ns,byte_offset,payload_len,seq_num,chunk_offset\n"
            )
            self._index_fp.flush()
            log.info("DataPortListener: index started → %s", index_path)
        except OSError as e:
            log.warning("could not open index file for %s: %s", bin_path, e)
            self._index_fp = None
            self._index_path = None

    def _close_index_locked(self) -> None:
        """Flush any buffered rows and close the index file. Caller
        holds ``_lock``."""
        if self._index_fp is None:
            return
        try:
            if self._index_buf:
                self._index_fp.write("".join(self._index_buf))
                self._index_buf = []
            self._index_fp.flush()
            self._index_fp.close()
        except OSError as e:
            log.warning("index close failed: %s", e)
        self._index_fp = None
        self._index_path = None

    def _index_record(self, ts_host_ns: int, byte_offset: int,
                      payload_len: int, seq: int, chunk_offset: int) -> None:
        """Buffer one CSV row and flush if the threshold is hit. Caller
        holds ``_lock``. Hot path — the row is built with str() and
        f-strings rather than the csv module to avoid the extra
        attribute lookups (the format is fixed and rectangular)."""
        if self._index_fp is None:
            return
        # Row format: 5 ints, comma-separated, single newline.
        self._index_buf.append(
            f"{ts_host_ns},{byte_offset},{payload_len},{seq},{chunk_offset}\n"
        )
        # Flush when either the row count or the time-since-last flush
        # threshold is exceeded. Using monotonic() (seconds, float)
        # rather than ns to keep the comparison cheap — we don't need
        # ns precision for a 100 ms window.
        now = time.monotonic()
        if (len(self._index_buf) >= self._INDEX_FLUSH_ROWS
                or (now - self._index_last_flush_mono)
                >= self._INDEX_FLUSH_INTERVAL_S):
            try:
                self._index_fp.write("".join(self._index_buf))
                self._index_buf = []
                self._index_last_flush_mono = now
            except OSError as e:
                log.warning("index write failed: %s; closing index", e)
                try:
                    self._index_fp.close()
                except OSError:
                    pass
                self._index_fp = None
                self._index_path = None

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
        # Flush any open recording file so the .bin is readable.
        with self._lock:
            if self._record_fp is not None:
                try:
                    self._record_fp.close()
                except OSError:
                    pass
                self._record_fp = None
            self._close_index_locked()
        log.info(
            "DataPortListener stopped (packets=%d bytes=%d drops_seq=%d drops_queue=%d)",
            self._packets_total, self._bytes_total,
            self._seq_drops_total, self._payloads_dropped_queue_full,
        )

    # ─────────────────────── consumer interface ────────────────────────────
    def drain_payloads(self) -> List[bytes]:
        """Pop and return every payload queued since the previous call.

        Returns a Python list of bytes in arrival order. Order matches
        the wire (sequence-number ordered) — assuming no UDP reorder
        on a direct host↔DCA cable, which is the supported topology.

        Designed for the DCAPipeline consumer thread to call once per
        tick. Empty list = consumer is keeping up. List grows back to
        the queue's maxlen (16 K packets ≈ 23 MB) → consumer is
        falling behind, payloads are about to be dropped (which would
        bump _payloads_dropped_queue_full)."""
        with self._lock:
            if not self._payload_queue:
                return []
            out = list(self._payload_queue)
            self._payload_queue.clear()
            return out

    def queue_depth(self) -> int:
        """Number of payloads currently queued (for diagnostics)."""
        with self._lock:
            return len(self._payload_queue)

    def queue_drops(self) -> int:
        """Cumulative count of payloads dropped because the queue was
        full at append time (consumer too slow). Distinct from
        ``seq_drops_total`` which counts wire-level UDP loss."""
        with self._lock:
            return self._payloads_dropped_queue_full

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
            payload = data[_HEADER_LEN:]
            payload_len = len(payload)
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
                # Queue the payload for the consumer (DCAPipeline).
                # If the deque is full the OLDEST payload is silently
                # dropped — counted separately so we can distinguish
                # "consumer too slow" from "wire-level UDP drops".
                if len(self._payload_queue) == self._payload_queue.maxlen:
                    self._payloads_dropped_queue_full += 1
                self._payload_queue.append(payload)
                # Optional .bin record. Write OUTSIDE the lock-critical
                # work would be nicer for latency, but writes are tiny
                # (~1.4 KB) and OS-buffered, so it's fine here.
                if self._record_fp is not None:
                    # Capture the byte_offset BEFORE writing so the
                    # index row points at the start of this packet's
                    # payload in the .bin (matches the spec's column
                    # semantics: "offset where this packet's payload
                    # begins").
                    bin_offset = self._index_offset_bytes
                    try:
                        self._record_fp.write(payload)
                        self._index_offset_bytes += payload_len
                    except OSError as e:
                        log.warning("record write failed: %s; closing file", e)
                        try:
                            self._record_fp.close()
                        except OSError:
                            pass
                        self._record_fp = None
                    else:
                        # .bin write succeeded → emit one CSV row.
                        # ``ts_host_ns`` per spec is monotonic_ns; the
                        # rest of the listener uses time.time() for
                        # wall-clock stats, but seek/scrub semantics
                        # need a monotonic reference.
                        self._index_record(
                            ts_host_ns=time.monotonic_ns(),
                            byte_offset=bin_offset,
                            payload_len=payload_len,
                            seq=seq,
                            chunk_offset=byte_count,
                        )

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

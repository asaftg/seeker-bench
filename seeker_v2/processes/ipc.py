"""Shared-memory IPC primitives for cross-process frame passing.

Architecture: each producer (EO/thermal/radar capture process) owns
N shared-memory slots in a ring. The producer writes to slot i,
publishes (i, frame_id, mtime) to a queue. Consumer reads from queue,
processes from slot i, optionally returns slot to producer (or just
overwrites at i+1 mod N — single-producer-single-consumer).

For LATEST-FRAME semantics (which is what real-time GUI wants), N=2
suffices: producer writes slot 0, signals, then writes slot 1 next
frame, etc. Consumer always reads the most-recently-signaled slot.
Old frames are silently overwritten — same drop policy as the v1
single-slot Condition variable.
"""
from __future__ import annotations
import os
import time
import struct
from dataclasses import dataclass
from typing import Optional
from multiprocessing import shared_memory, Lock, Value

@dataclass
class FrameDescriptor:
    """Tells the consumer which shm slot has a fresh frame."""
    slot: int
    frame_id: int
    mtime: float  # monotonic timestamp at producer publish
    width: int
    height: int
    channels: int
    dtype: str  # numpy dtype name
    extra: bytes = b""  # optional sidecar bytes (e.g., JPEG)


class FrameRing:
    """Latest-frame ring buffer in shared memory.

    Producer writes np.ndarray frames into rotating slots; publishes
    a descriptor via shared atomics. Consumer reads the latest slot.
    No explicit lock — atomic seq counter does the synchronization.
    """

    def __init__(self, name: str, n_slots: int,
                 frame_bytes: int, create: bool = False):
        self._name = name
        self._n = int(n_slots)
        self._frame_bytes = int(frame_bytes)
        self._total = self._n * self._frame_bytes
        if create:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=self._total
            )
        else:
            self._shm = shared_memory.SharedMemory(name=name)
        # Latest published slot index (atomic via Value)
        self._latest = Value("q", -1, lock=False)
        self._latest_seq = Value("q", 0, lock=False)

    def writer_view(self, slot_idx: int):
        """Return memoryview into the slot for the producer to write."""
        offset = slot_idx * self._frame_bytes
        return self._shm.buf[offset:offset + self._frame_bytes]

    def reader_view(self, slot_idx: int):
        """Same; intended for consumer."""
        return self.writer_view(slot_idx)

    def publish(self, slot_idx: int):
        """Atomically advance latest pointer + seq."""
        self._latest.value = int(slot_idx)
        self._latest_seq.value = int(self._latest_seq.value) + 1

    def latest_slot(self) -> tuple[int, int]:
        """Return (slot_idx, seq) of the most recent publish.
        Slot may be -1 if no frame yet."""
        return int(self._latest.value), int(self._latest_seq.value)

    def close(self):
        self._shm.close()
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass

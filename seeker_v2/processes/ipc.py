"""Shared-memory IPC primitives for cross-process frame passing.

Architecture
============
Each producer (capture process per sensor) owns N shared-memory slots
in a ring. The producer:
  1. picks the next slot (round-robin)
  2. writes the frame data into the slot's mmap
  3. writes a descriptor (frame_id, slot_idx, timestamp, shape, dtype) to
     a single-slot atomic publish point + Queue (for back-pressure-aware
     consumers that want to be notified on every frame)

Two consumption styles are supported:
  - LATEST: read the most-recently-published descriptor (drops old frames).
            This is what the GUI/WS sender wants - no point sending stale.
  - FIFO:   pop from a multiprocessing.Queue. This is what the Inference
            process wants - it must process every frame in order so
            tracking IDs stay consistent.

Synchronization
===============
Single-producer-multiple-consumer per ring. Atomic publish uses
multiprocessing.Value with no lock (we never read partial 8-byte values
on aarch64 - atomic by default for naturally-aligned 64-bit). Slot
buffers are ring-overwritten - old data is silently lost if a consumer
is too slow to keep up (matching v1 single-slot Condition semantics).

Lifecycle
=========
The ring is created by the *parent* (orchestrator) and named via
/dev/shm/<name>. Children attach by name. On shutdown, the parent
closes + unlinks. Children only call close().

All shared-memory objects survive a parent crash unless explicitly
unlinked, which leaks /dev/shm files. The orchestrator's signal handler
must always unlink. We register an atexit hook as a fallback.
"""
from __future__ import annotations

import atexit
import logging
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from multiprocessing import shared_memory, Queue, Value
from queue import Empty, Full
from typing import Iterator, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# Track named shm objects we created so we always unlink on exit.
_OWNED_SHM: list[str] = []


def _unlink_owned():
    for name in list(_OWNED_SHM):
        try:
            shm = shared_memory.SharedMemory(name=name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("atexit unlink of %s failed: %r", name, e)


atexit.register(_unlink_owned)


@dataclass
class FrameDescriptor:
    """What the consumer reads off the queue / atomic publish point."""

    frame_id: int
    slot_idx: int
    mtime: float  # producer's time.monotonic() at publish
    width: int
    height: int
    channels: int  # 1 (mono) or 3 (BGR) or 2 (raw YUYV bytes-per-pixel)
    dtype: str  # numpy dtype name; "uint8" / "uint16"
    extra_size: int = 0  # bytes of optional sidecar (e.g. JPEG)
    # Sensor-specific arbitrary metadata. Kept tiny - serialized
    # inline with the descriptor in the queue.
    meta: dict = field(default_factory=dict)

    # struct packing for the atomic publish point.
    # We only put the essentials in the atomic slot (it's a single-writer
    # single-reader ABA-safe 64-byte block). Full descriptor with meta
    # goes through the Queue.
    _ATOMIC_FMT = "<qqdiiii"  # frame_id, slot, mtime, w, h, ch, dt_idx
    _ATOMIC_SIZE = struct.calcsize(_ATOMIC_FMT)
    _DTYPE_TO_IDX = {"uint8": 0, "uint16": 1, "<u2": 1, "uint32": 2,
                     "float32": 3, "int16": 4, "int32": 5}
    _IDX_TO_DTYPE = {0: "uint8", 1: "uint16", 2: "uint32",
                     3: "float32", 4: "int16", 5: "int32"}

    def to_atomic_bytes(self) -> bytes:
        return struct.pack(
            self._ATOMIC_FMT,
            self.frame_id, self.slot_idx, self.mtime,
            self.width, self.height, self.channels,
            self._DTYPE_TO_IDX.get(self.dtype, 0),
        )

    @classmethod
    def from_atomic_bytes(cls, b: bytes) -> "FrameDescriptor":
        fid, slot, mt, w, h, ch, dt_idx = struct.unpack(cls._ATOMIC_FMT, b)
        return cls(
            frame_id=fid, slot_idx=slot, mtime=mt,
            width=w, height=h, channels=ch,
            dtype=cls._IDX_TO_DTYPE.get(dt_idx, "uint8"),
        )


class FrameRing:
    """Latest-frame ring buffer in shared memory.

    Producer side:
        ring = FrameRing.create("eo_bgr", n_slots=4, frame_bytes=W*H*3)
        for ...:
            slot = ring.next_slot()
            view = ring.writer_view(slot)
            # write frame bytes into view
            ring.publish(FrameDescriptor(...))
        ring.close_and_unlink()

    Consumer side:
        ring = FrameRing.attach("eo_bgr", n_slots=4, frame_bytes=W*H*3)
        last_seq = 0
        while running:
            desc, seq = ring.latest()
            if desc is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            view = ring.reader_view(desc.slot_idx)
            frame = np.frombuffer(view, dtype=desc.dtype).reshape(...)
            # process frame; do NOT keep the view past the next publish
        ring.close()  # do NOT unlink - producer owns lifecycle
    """

    def __init__(
        self,
        name: str,
        n_slots: int,
        frame_bytes: int,
        *,
        owns: bool,
    ) -> None:
        self._name = name
        self._n = int(n_slots)
        self._frame_bytes = int(frame_bytes)
        self._owns = owns
        self._slot_idx = -1  # producer-side rolling slot

        total_size = self._n * self._frame_bytes + FrameDescriptor._ATOMIC_SIZE
        if owns:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=total_size
            )
            _OWNED_SHM.append(name)
        else:
            self._shm = shared_memory.SharedMemory(name=name)
            if self._shm.size < total_size:
                raise ValueError(
                    f"FrameRing {name}: shm size {self._shm.size} < required {total_size}"
                )
        # Atomic publish slot is the LAST FrameDescriptor._ATOMIC_SIZE bytes
        self._atomic_offset = self._n * self._frame_bytes

        # Atomic seq counter (multiprocessing.Value, no lock - naturally
        # aligned int64 reads/writes are atomic on aarch64+x86_64).
        self._seq = Value("q", 0, lock=False)

    @classmethod
    def create(cls, name: str, n_slots: int, frame_bytes: int) -> "FrameRing":
        return cls(name, n_slots, frame_bytes, owns=True)

    @classmethod
    def attach(cls, name: str, n_slots: int, frame_bytes: int) -> "FrameRing":
        return cls(name, n_slots, frame_bytes, owns=False)

    @property
    def n_slots(self) -> int:
        return self._n

    @property
    def frame_bytes(self) -> int:
        return self._frame_bytes

    def next_slot(self) -> int:
        """Producer: pick the next ring slot (round-robin)."""
        self._slot_idx = (self._slot_idx + 1) % self._n
        return self._slot_idx

    def writer_view(self, slot_idx: int) -> memoryview:
        """Producer: get a memoryview into the slot's frame bytes."""
        offset = slot_idx * self._frame_bytes
        return self._shm.buf[offset : offset + self._frame_bytes]

    def reader_view(self, slot_idx: int) -> memoryview:
        """Consumer: same - read frame bytes from a slot.

        Caller MUST consume the bytes before the producer publishes the
        next frame to the same slot. With n_slots >= 2 and a consumer
        faster than 1/N producer rate, this is safe.
        """
        offset = slot_idx * self._frame_bytes
        return self._shm.buf[offset : offset + self._frame_bytes]

    def publish(self, desc: FrameDescriptor) -> None:
        """Producer: atomically publish a descriptor as the latest frame."""
        if desc.slot_idx < 0 or desc.slot_idx >= self._n:
            raise ValueError(f"slot {desc.slot_idx} out of range")
        b = desc.to_atomic_bytes()
        # Write into the atomic slot. The struct fits in a single cache
        # line (32 bytes), so on aarch64 with 64-byte cache lines the
        # write is naturally atomic from a torn-read perspective.
        self._shm.buf[
            self._atomic_offset : self._atomic_offset + len(b)
        ] = b
        # Bump seq AFTER the descriptor write. Consumers read seq first,
        # then descriptor - and re-check seq after - to detect a torn
        # read across the publish boundary.
        self._seq.value = int(self._seq.value) + 1

    def latest(self) -> Tuple[Optional[FrameDescriptor], int]:
        """Consumer: return (latest_descriptor, seq).

        Returns (None, 0) before the first publish.
        """
        seq_a = int(self._seq.value)
        if seq_a == 0:
            return None, 0
        b = bytes(
            self._shm.buf[
                self._atomic_offset
                : self._atomic_offset + FrameDescriptor._ATOMIC_SIZE
            ]
        )
        seq_b = int(self._seq.value)
        if seq_a != seq_b:
            # Torn read across publish - caller can retry next tick.
            return None, seq_a
        try:
            desc = FrameDescriptor.from_atomic_bytes(b)
        except struct.error:
            return None, seq_a
        return desc, seq_b

    def close(self) -> None:
        """Detach from shared memory. Does NOT unlink (producer owns)."""
        try:
            self._shm.close()
        except Exception:
            pass

    def close_and_unlink(self) -> None:
        """Producer-only: close + unlink the shared memory block."""
        try:
            self._shm.close()
        except Exception:
            pass
        if self._owns:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass
            try:
                _OWNED_SHM.remove(self._name)
            except ValueError:
                pass


class DescriptorQueue:
    """multiprocessing.Queue wrapped to publish + receive FrameDescriptors.

    Use this when a consumer must process every frame (Inference: tracking
    IDs need consistent stride). For consumers that only care about the
    latest, use FrameRing.latest() directly.
    """

    def __init__(self, maxsize: int = 8):
        # Bounded so a slow consumer can't blow memory.
        self._q: Queue = Queue(maxsize=maxsize)

    def put(self, desc: FrameDescriptor, drop_old: bool = True) -> bool:
        """Put descriptor on the queue. Returns True if accepted.

        If queue is full and drop_old=True, discard the oldest queued
        descriptor and put the new one. This matches real-time video
        semantics (latest frame wins).
        """
        try:
            self._q.put_nowait(desc)
            return True
        except Full:
            if not drop_old:
                return False
            try:
                _ = self._q.get_nowait()  # drop oldest
            except Empty:
                pass
            try:
                self._q.put_nowait(desc)
                return True
            except Full:
                return False

    def get(self, timeout: Optional[float] = 0.5) -> Optional[FrameDescriptor]:
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None

    def qsize(self) -> int:
        try:
            return self._q.qsize()
        except NotImplementedError:
            # macOS doesn't support qsize - return 0 as a fallback.
            return 0

    def close(self) -> None:
        self._q.close()
        self._q.join_thread()


@contextmanager
def published_frame(ring: FrameRing, frame_id: int, *,
                    width: int, height: int, channels: int,
                    dtype: str, meta: Optional[dict] = None,
                    extra_size: int = 0) -> Iterator[memoryview]:
    """Context manager for publishing a frame.

    Usage:
        with published_frame(ring, frame_id, width=W, height=H,
                             channels=3, dtype="uint8") as buf:
            buf[:] = bgr_bytes  # write into shm slot

    The slot is auto-chosen and the descriptor auto-published on
    successful exit.
    """
    slot = ring.next_slot()
    view = ring.writer_view(slot)
    yield view
    desc = FrameDescriptor(
        frame_id=frame_id,
        slot_idx=slot,
        mtime=time.monotonic(),
        width=width,
        height=height,
        channels=channels,
        dtype=dtype,
        extra_size=extra_size,
        meta=meta or {},
    )
    ring.publish(desc)


# Smoke test (run with `python -m seeker_v2.processes.ipc`)
def _smoke_self_test() -> int:
    """Spawn a producer + consumer subprocess, push N frames,
    verify consumer received frames."""
    import multiprocessing as mp

    NAME = "_seeker_v2_ipc_smoke"
    H, W = 64, 64
    FRAME_BYTES = H * W
    N_SLOTS = 4
    N_FRAMES = 100

    def producer(name: str, ready_evt):
        ring = FrameRing.create(name, n_slots=N_SLOTS, frame_bytes=FRAME_BYTES)
        ready_evt.set()
        time.sleep(0.05)  # let consumers attach
        for i in range(N_FRAMES):
            with published_frame(
                ring, frame_id=i,
                width=W, height=H, channels=1, dtype="uint8",
            ) as view:
                arr = np.frombuffer(view, dtype=np.uint8).reshape(H, W)
                arr[:] = i % 256
            time.sleep(0.005)
        time.sleep(0.5)  # let consumers drain
        ring.close_and_unlink()

    def consumer_latest(name: str, result_q):
        time.sleep(0.1)  # give producer time to create
        ring = FrameRing.attach(name, n_slots=N_SLOTS, frame_bytes=FRAME_BYTES)
        seen = set()
        last_seq = 0
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(seen) < N_FRAMES:
            desc, seq = ring.latest()
            if desc is None or seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq
            seen.add(desc.frame_id)
        ring.close()
        result_q.put(("latest", sorted(seen)))

    mp_ctx = mp.get_context("spawn")
    ready = mp_ctx.Event()
    result_q = mp_ctx.Queue()
    p_prod = mp_ctx.Process(target=producer, args=(NAME, ready))
    p_prod.start()
    ready.wait(timeout=5)
    p_cons = mp_ctx.Process(target=consumer_latest, args=(NAME, result_q))
    p_cons.start()
    p_prod.join(timeout=10)
    p_cons.join(timeout=10)
    if p_prod.exitcode != 0 or p_cons.exitcode != 0:
        print(f"FAIL: producer exit={p_prod.exitcode} consumer={p_cons.exitcode}")
        return 1
    label, seen = result_q.get_nowait()
    n_seen = len(seen)
    print(f"smoke: consumer saw {n_seen}/{N_FRAMES} frames "
          f"(latest-wins semantics expected to drop some)")
    if n_seen < 5:
        print("FAIL: consumer saw too few frames")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(_smoke_self_test())

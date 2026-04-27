"""
JSONL session recorder.

One daemon thread per FrameBus topic blocks on ``wait_new``, encodes
the latest frame to a JSON-safe dict, and appends a single line:

    {"ts_ns": <int>, "channel": "<name>", "msg": {...}}

Discrete events go through ``common.events.subscribe`` instead of the
bus — see ``common/events.py`` for why (FrameBus is latest-only).

Lifecycle:
    rec = JSONLRecorder(BUS, output_dir="recordings", jpeg_quality=92)
    rec.start(config_snapshot=cfg_dict)        # opens a fresh file
    ...
    rec.stop()                                 # flushes and closes

A recorder instance is reusable: ``start`` after ``stop`` opens a new
file with a new timestamp in the name. Single file per session — no
rotation, the operator chunks by stop/start.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from common.events import subscribe as events_subscribe
from common.frame_bus import FrameBus
from common.frames import Topic

from . import encoders


log = logging.getLogger(__name__)


# Channel name (in JSONL) -> (topic name on bus, encoder function).
# Adding a new channel is just one entry here. The encoder runs on the
# recorder thread, so it must be cheap-or-decoupled (JPEG is cheap-ish
# at q=92 — measured ~3-5 ms per 640×512 frame on the bench rig).
def _channels_table(jpeg_quality: int) -> Dict[str, tuple[str, Callable]]:
    return {
        "thermal/frame":  (Topic.THERMAL, lambda f: encoders.encode_thermal(f, jpeg_quality)),
        "eo/frame":       (Topic.EO,      lambda f: encoders.encode_eo(f, jpeg_quality)),
        "radar/frame":    (Topic.RADAR,   encoders.encode_radar),
        "gimbal/state":   (Topic.GIMBAL,  encoders.encode_gimbal),
        "fusion/tracks":  (Topic.FUSED,   encoders.encode_fused),
    }


class JSONLRecorder:
    def __init__(
        self,
        bus: FrameBus,
        output_dir: str = "recordings",
        jpeg_quality: int = 92,
        channel_enable: Optional[Dict[str, bool]] = None,
    ) -> None:
        self._bus = bus
        self._output_dir = output_dir
        self._jpeg_quality = int(jpeg_quality)
        self._channel_enable = channel_enable or {}

        # Active-session state — None when not recording
        self._fh = None  # type: Optional[Any]
        self._fh_lock = threading.Lock()
        self._path: Optional[str] = None
        self._started_ns: Optional[int] = None
        self._threads: list[threading.Thread] = []
        self._stop_evt = threading.Event()
        self._unsubscribe_events: Optional[Callable[[], None]] = None

        # Counters for the stop-summary
        self._counts: Dict[str, int] = {}
        self._counts_lock = threading.Lock()

    # ──────────────────────────────────────────────────────────
    @property
    def is_recording(self) -> bool:
        return self._fh is not None

    @property
    def current_path(self) -> Optional[str]:
        return self._path

    # ──────────────────────────────────────────────────────────
    def start(self, config_snapshot: Optional[Dict[str, Any]] = None,
              path: Optional[str] = None) -> str:
        """Open a new recording file and spawn topic threads.

        Returns the absolute path of the file being written.
        Idempotent: a second call while already recording is a no-op
        (returns the current path).
        """
        if self.is_recording:
            log.warning("[recorder] start() called while already recording: %s", self._path)
            return self._path or ""

        os.makedirs(self._output_dir, exist_ok=True)
        if path is None:
            stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = os.path.join(self._output_dir, f"seeker_{stamp}.jsonl")
        self._path = os.path.abspath(path)
        self._fh = open(self._path, "w", encoding="utf-8", buffering=1)  # line-buffered
        self._started_ns = time.time_ns()
        self._stop_evt.clear()
        self._counts = {}

        # Header line — config snapshot + format version
        header = {
            "version": 1,
            "started_at": time.time(),
            "started_at_ns": self._started_ns,
            "jpeg_quality": self._jpeg_quality,
            "config_snapshot": config_snapshot or {},
        }
        self._write_line("session/header", header, ts_ns=self._started_ns)

        # Subscribe to events FIRST so anything emitted during thread
        # spinup is captured.
        self._unsubscribe_events = events_subscribe(self._on_event)

        # Spawn one thread per enabled channel.
        table = _channels_table(self._jpeg_quality)
        self._threads = []
        for ch_name, (topic, encoder) in table.items():
            if self._channel_enable.get(ch_name, True) is False:
                log.info("[recorder] channel %s disabled by config", ch_name)
                continue
            t = threading.Thread(
                target=self._channel_loop,
                name=f"rec[{ch_name}]",
                args=(ch_name, topic, encoder),
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        log.info("[recorder] STARTED → %s", self._path)
        return self._path

    # ──────────────────────────────────────────────────────────
    def stop(self) -> Optional[str]:
        """Stop recording, join threads, close the file.

        Returns the closed file's absolute path (or None if not
        recording). Threads exit via ``_stop_evt``; ``wait_new`` is
        bounded with a 0.5s timeout so each thread cycles within a
        bus-tick or half a second of stop().
        """
        if not self.is_recording:
            return None
        path = self._path
        log.info("[recorder] stopping; channel counts=%s",
                 dict(self._counts))
        self._stop_evt.set()
        # Unsubscribe events first so post-stop emits don't try to
        # write to a closed file.
        if self._unsubscribe_events is not None:
            try:
                self._unsubscribe_events()
            except Exception:
                pass
            self._unsubscribe_events = None
        # Join with a generous deadline; threads should exit quickly.
        for t in self._threads:
            t.join(timeout=2.0)
        self._threads = []
        with self._fh_lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass
            self._fh = None
        self._path = None
        self._started_ns = None
        return path

    # ──────────────────────────────────────────────────────────
    def _channel_loop(
        self, channel: str, topic: str, encoder: Callable[[Any], Optional[Dict]]
    ) -> None:
        """Per-topic loop: wait for a new frame, encode, write."""
        last_id = None
        while not self._stop_evt.is_set():
            got = self._bus.wait_new(topic, timeout=0.5)
            if not got:
                continue
            frame = self._bus.get_latest(topic)
            if frame is None:
                continue
            # Skip duplicate frames if the publisher republished the
            # same object (cheap identity check; not sufficient for
            # equal-by-value, but our publishers always allocate fresh
            # dataclasses).
            try:
                fid = (id(frame), getattr(frame, "frame_id", None),
                       getattr(frame, "timestamp", None))
            except Exception:
                fid = None
            if fid is not None and fid == last_id:
                continue
            last_id = fid

            try:
                msg = encoder(frame)
            except Exception:
                log.exception("[recorder] encode error on channel=%s", channel)
                continue
            if msg is None:
                continue
            self._write_line(channel, msg)

    # ──────────────────────────────────────────────────────────
    def _on_event(self, evt: Dict[str, Any]) -> None:
        """Subscriber callback for the events stream.

        ``evt`` shape: ``{"ts_ns": int, "type": str, "payload": dict}``.
        We unpack into the standard ``{ts_ns, channel, msg}`` line so
        the JSONL is uniformly schemed.
        """
        msg = {"type": evt.get("type", "unknown"),
               "payload": evt.get("payload", {})}
        ts = int(evt.get("ts_ns") or time.time_ns())
        self._write_line("events", msg, ts_ns=ts)

    # ──────────────────────────────────────────────────────────
    def _write_line(self, channel: str, msg: Dict[str, Any],
                    ts_ns: Optional[int] = None) -> None:
        """Append one JSON line to the open file. Thread-safe."""
        if self._fh is None:
            return
        if ts_ns is None:
            ts_ns = time.time_ns()
        line = json.dumps(
            {"ts_ns": int(ts_ns), "channel": channel, "msg": msg},
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._fh_lock:
            if self._fh is None:
                return
            try:
                self._fh.write(line)
                self._fh.write("\n")
            except Exception:
                log.exception("[recorder] write failed (channel=%s)", channel)
                return
        with self._counts_lock:
            self._counts[channel] = self._counts.get(channel, 0) + 1

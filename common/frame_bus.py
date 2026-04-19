"""
In-memory latest-frame bus.

This is intentionally the simplest thing that works:

    - One slot per topic holding the most recent frame.
    - Publish is lock-free-ish (GIL-protected dict assign).
    - Subscribers either poll `get_latest(topic)` or wait on a
      per-topic Event that is set whenever a new frame arrives.

No queueing, no history, no backpressure. If the consumer is
slow it just sees the newest frame and skips older ones, which
is exactly what the GUI wants.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional


class FrameBus:
    def __init__(self) -> None:
        self._latest: Dict[str, Any] = {}
        self._events: Dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def _event_for(self, topic: str) -> threading.Event:
        with self._lock:
            ev = self._events.get(topic)
            if ev is None:
                ev = threading.Event()
                self._events[topic] = ev
            return ev

    def publish(self, topic: str, frame: Any) -> None:
        """Publish the newest frame on `topic`. Overwrites previous."""
        self._latest[topic] = frame
        self._event_for(topic).set()

    def get_latest(self, topic: str) -> Optional[Any]:
        """Return the most recent frame, or None if nothing published yet."""
        return self._latest.get(topic)

    def wait_new(self, topic: str, timeout: Optional[float] = None) -> bool:
        """Block until a new frame arrives on `topic`.

        Returns True if a new frame is available, False on timeout.
        Resets the event so subsequent calls block again.
        """
        ev = self._event_for(topic)
        got = ev.wait(timeout=timeout)
        if got:
            ev.clear()
        return got

    def clear(self, topic: str) -> None:
        self._latest.pop(topic, None)
        self._events.pop(topic, None)


# Module-level singleton. Modules import `BUS` rather than
# constructing their own, so the GUI backend and sensor threads
# share the same in-process bus.
BUS = FrameBus()

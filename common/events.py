"""
Discrete-event publisher.

Every "thing that happened" — user pressed TRACK, AE converged,
fused track born, predictor stepped — funnels through ``emit()``.
Two delivery paths:

1. **Direct subscribers** (synchronous, FIFO): the JSONL recorder
   registers via ``subscribe()`` and is invoked inline. This avoids
   the FrameBus "latest only" semantics — two events emitted in the
   same millisecond would otherwise overwrite each other on the bus.

2. **FrameBus topic** (``Topic.EVENTS``): for any other consumer
   that wants the live stream. Publish is best-effort; if two events
   land within one bus tick the older one is lost. Subscribers that
   need every event must use ``subscribe()`` instead.

Format: ``{"type": str, "payload": dict, "ts_ns": int}``. Free-form,
no schema validation. Type strings live in `recording/README.md`.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List

from common.frame_bus import BUS
from common.frames import Topic


_subs_lock = threading.Lock()
_subscribers: List[Callable[[Dict[str, Any]], None]] = []


def subscribe(callback: Callable[[Dict[str, Any]], None]) -> Callable[[], None]:
    """Register a synchronous callback. Returns an unsubscribe function.

    Callback receives the full event dict
    ``{"ts_ns": int, "type": str, "payload": dict}``.

    Callbacks run on the emitter's thread — keep them fast and
    exception-safe. The recorder's callback just appends one JSONL
    line, so this is fine.
    """
    with _subs_lock:
        _subscribers.append(callback)

    def _unsubscribe() -> None:
        with _subs_lock:
            try:
                _subscribers.remove(callback)
            except ValueError:
                pass

    return _unsubscribe


def emit(type_: str, payload: Dict[str, Any] | None = None) -> None:
    """Publish one event.

    ``type_`` is a free-form string; see `recording/README.md` for the
    canonical vocabulary. ``payload`` is any JSON-able dict (or None);
    callers pass primitives that survive ``json.dumps`` without coercion.
    """
    evt = {
        "ts_ns": time.time_ns(),
        "type": str(type_),
        "payload": dict(payload) if payload else {},
    }
    # Snapshot subscribers under the lock, invoke without it
    with _subs_lock:
        subs = list(_subscribers)
    for cb in subs:
        try:
            cb(evt)
        except Exception:
            # Swallow — events are diagnostic, never break the caller
            pass
    # Best-effort bus publish for live consumers
    try:
        BUS.publish(Topic.EVENTS, evt)
    except Exception:
        pass

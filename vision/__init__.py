"""Vision utilities — frame-rate per-target trackers, residual measurement, etc.

Lives between sensor capture and detection: takes a frame + bbox(es)
and emits updated bbox(es) at frame rate, decoupling tracking from
the (slow, non-deterministic) detector.
"""

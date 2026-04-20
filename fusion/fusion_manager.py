"""Cross-sensor fusion thread.

Pulls the latest ThermalFrame + EOFrame from the bus at a fixed rate,
converts each sensor's raw detections into angular space, associates
them across sensors, and publishes a list of FusedTrack objects on
Topic.FUSED.

Association (keep it simple, iterate later):
    - Each sensor contributes a list of "observations" in angular
      coordinates (az, el, ang_w, ang_h, class, confidence).
    - EO is primary: for every EO observation we look for the best
      thermal observation of the same class whose angular center is
      within ``match_gate_deg`` (defaults to max of target angular
      size and 0.7°). If found, the pair fuses into a 2-sensor track.
    - Unmatched EO observations become 1-sensor EO tracks.
    - Unmatched thermal observations become 1-sensor thermal tracks.

Persistence:
    - Tracks live across fusion ticks. Matching existing tracks to
      this tick's observations is another angular-distance +
      same-class check.
    - hits count confirmations, misses count fusion ticks without
      a match; a track is dropped after ``max_misses`` misses.

This module reads from the bus but owns no hardware — it's safe to
stop and restart at will.
"""
from __future__ import annotations

import math
import threading
import time
from typing import Optional, Tuple

from common.config import load_config
from common.frame_bus import BUS
from common.frames import (
    EOFrame,
    FusedTrack,
    TargetClass,
    ThermalFrame,
    Topic,
)
from common.logging_setup import get_logger
from fusion.angular import bbox_to_angular

log = get_logger(__name__)


# ─── Only these classes get fused. Raw "heat" / UNKNOWN detections
# ─── don't have a reliable class to associate on.
_FUSABLE_CLASSES = {
    TargetClass.PERSON,
    TargetClass.VEHICLE,
    TargetClass.DRONE,
}


class FusionManager:
    """Background thread that emits FusedTrack lists on Topic.FUSED."""

    def __init__(
        self,
        rate_hz: float = 15.0,
        match_gate_deg: float = 0.7,
        track_match_gate_deg: float = 1.2,
        min_hits: int = 1,
        max_misses: int = 10,
    ) -> None:
        cfg = load_config()
        fcfg = cfg.get("fusion", {}) or {}
        self.rate_hz = float(fcfg.get("rate_hz", rate_hz))
        self.match_gate_deg = float(fcfg.get("match_gate_deg", match_gate_deg))
        self.track_match_gate_deg = float(
            fcfg.get("track_match_gate_deg", track_match_gate_deg)
        )
        self.min_hits = int(fcfg.get("min_hits", min_hits))
        self.max_misses = int(fcfg.get("max_misses", max_misses))

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tracks: list[dict] = []
        self._next_id = 1

    # ───────────────────────── lifecycle ─────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="FusionManager", daemon=True
        )
        self._thread.start()
        log.info(
            "FusionManager started (rate=%.1fHz, match_gate=%.2f°, "
            "track_gate=%.2f°, min_hits=%d, max_misses=%d)",
            self.rate_hz, self.match_gate_deg,
            self.track_match_gate_deg, self.min_hits, self.max_misses,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ───────────────────────── main loop ─────────────────────────
    def _loop(self) -> None:
        period = 1.0 / max(1.0, self.rate_hz)
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                log.exception("Fusion tick failed: %s", e)
            elapsed = time.time() - t0
            self._stop.wait(max(0.0, period - elapsed))
        log.info("FusionManager stopped")

    def _tick(self) -> None:
        tf: Optional[ThermalFrame] = BUS.get_latest(Topic.THERMAL)
        ef: Optional[EOFrame] = BUS.get_latest(Topic.EO)

        thermal_obs = self._observations_from_thermal(tf)
        eo_obs = self._observations_from_eo(ef)

        # ── Cross-sensor association (EO primary) ──
        # Produce a list of "candidates" per tick. Each candidate is a
        # single real-world target with a class, best angular pose, and
        # a set of contributing sensors. These then match into the
        # persistent tracker.
        candidates: list[dict] = []
        used_t = [False] * len(thermal_obs)
        for e in eo_obs:
            best_i, best_d = -1, math.inf
            for i, t in enumerate(thermal_obs):
                if used_t[i] or t["class"] != e["class"]:
                    continue
                d = math.hypot(e["az"] - t["az"], e["el"] - t["el"])
                if d < best_d:
                    best_d, best_i = d, i
            gate = max(self.match_gate_deg,
                       max(e["ang_w"], e["ang_h"]) * 0.8)
            if best_i >= 0 and best_d < gate:
                t = thermal_obs[best_i]
                used_t[best_i] = True
                candidates.append({
                    "sensors": ["eo", "thermal"],
                    "primary": "eo",
                    "class":   e["class"],
                    "az":      e["az"],   "el":    e["el"],
                    "ang_w":   e["ang_w"],"ang_h": e["ang_h"],
                    "conf":    max(e["conf"], t["conf"]),
                })
            else:
                candidates.append({
                    "sensors": ["eo"],
                    "primary": "eo",
                    "class":   e["class"],
                    "az":      e["az"],   "el":    e["el"],
                    "ang_w":   e["ang_w"],"ang_h": e["ang_h"],
                    "conf":    e["conf"],
                })
        for i, t in enumerate(thermal_obs):
            if used_t[i]:
                continue
            candidates.append({
                "sensors": ["thermal"],
                "primary": "thermal",
                "class":   t["class"],
                "az":      t["az"],   "el":    t["el"],
                "ang_w":   t["ang_w"],"ang_h": t["ang_h"],
                "conf":    t["conf"],
            })

        self._update_tracks(candidates)
        self._publish()

    # ───────────────────────── observation builders ──────────────
    def _observations_from_thermal(self, tf: Optional[ThermalFrame]) -> list[dict]:
        if tf is None or not tf.connected or tf.agc8 is None:
            return []
        h, w = tf.agc8.shape[:2]
        out = []
        for d in tf.detections:
            if d.classification is None:
                continue
            cls = d.classification.target_class
            if cls not in _FUSABLE_CLASSES:
                continue
            az, el, aw, ah = bbox_to_angular(
                d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h,
                w, h, tf.hfov_deg, tf.vfov_deg,
            )
            out.append({
                "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                "class": cls.value,
                "conf": float(d.classification.confidence),
            })
        return out

    def _observations_from_eo(self, ef: Optional[EOFrame]) -> list[dict]:
        if ef is None or not ef.connected or ef.bgr is None:
            return []
        h, w = ef.bgr.shape[:2]
        out = []
        for d in ef.detections:
            cls = d.target_class
            if cls not in _FUSABLE_CLASSES:
                continue
            az, el, aw, ah = bbox_to_angular(
                d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h,
                w, h, ef.hfov_deg, ef.vfov_deg,
            )
            out.append({
                "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                "class": cls.value,
                "conf": float(d.confidence),
            })
        return out

    # ───────────────────────── persistence tracker ───────────────
    def _update_tracks(self, candidates: list[dict]) -> None:
        matched = [False] * len(self._tracks)
        for c in candidates:
            best_i, best_d = -1, math.inf
            for i, trk in enumerate(self._tracks):
                if matched[i] or trk["class"] != c["class"]:
                    continue
                d = math.hypot(c["az"] - trk["az"], c["el"] - trk["el"])
                if d < best_d:
                    best_d, best_i = d, i
            gate = max(self.track_match_gate_deg,
                       max(c["ang_w"], c["ang_h"]) * 1.2)
            if best_i >= 0 and best_d < gate:
                trk = self._tracks[best_i]
                a = 0.4  # EMA
                trk["az"]    = a * trk["az"]    + (1 - a) * c["az"]
                trk["el"]    = a * trk["el"]    + (1 - a) * c["el"]
                trk["ang_w"] = a * trk["ang_w"] + (1 - a) * c["ang_w"]
                trk["ang_h"] = a * trk["ang_h"] + (1 - a) * c["ang_h"]
                # Sensor set accumulates — a track that has ever been
                # fused stays "2-sensor" if the next tick only saw EO.
                # Resetting per tick would make the green box flicker.
                trk["sensors_now"] = list(c["sensors"])
                for s in c["sensors"]:
                    if s not in trk["sensors_ever"]:
                        trk["sensors_ever"].append(s)
                trk["primary"] = c["primary"]
                trk["conf"] = max(trk["conf"] * 0.9, c["conf"])
                trk["hits"] += 1
                trk["misses"] = 0
                matched[best_i] = True
            else:
                self._tracks.append({
                    "id": self._next_id,
                    "class": c["class"],
                    "az": c["az"], "el": c["el"],
                    "ang_w": c["ang_w"], "ang_h": c["ang_h"],
                    "sensors_now": list(c["sensors"]),
                    "sensors_ever": list(c["sensors"]),
                    "primary": c["primary"],
                    "conf": c["conf"],
                    "hits": 1, "misses": 0,
                })
                self._next_id += 1

        kept = []
        for i, trk in enumerate(self._tracks):
            if i < len(matched) and matched[i]:
                kept.append(trk)
            else:
                trk["misses"] += 1
                # A track that isn't seen this tick is still alive, but
                # its "current sensor set" drops to empty so the GUI
                # knows to dim it.
                trk["sensors_now"] = []
                if trk["misses"] <= self.max_misses:
                    kept.append(trk)
        self._tracks = kept

    # ───────────────────────── publish ───────────────────────────
    def _publish(self) -> None:
        out: list[FusedTrack] = []
        for trk in self._tracks:
            if trk["hits"] < self.min_hits:
                continue
            try:
                tc = TargetClass(trk["class"])
            except ValueError:
                tc = TargetClass.UNKNOWN
            out.append(FusedTrack(
                id=int(trk["id"]),
                target_class=tc,
                confidence=float(trk["conf"]),
                sensors=list(trk["sensors_ever"]),
                primary=str(trk["primary"]),
                az_deg=float(trk["az"]),
                el_deg=float(trk["el"]),
                ang_w_deg=float(trk["ang_w"]),
                ang_h_deg=float(trk["ang_h"]),
                hits=int(trk["hits"]),
                misses=int(trk["misses"]),
            ))
        BUS.publish(Topic.FUSED, out)

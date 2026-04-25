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

import threading
import time
from typing import Optional, Tuple

from common.config import load_config
from common.frame_bus import BUS
from common.frames import (
    EOFrame,
    FusedTrack,
    RadarFrame,
    TargetClass,
    ThermalFrame,
    Topic,
)
from common.logging_setup import get_logger
from fusion.angular import angular_iou, bbox_to_angular

log = get_logger(__name__)


# ─── Only these classes get fused. Raw "heat" / UNKNOWN detections
# ─── don't have a reliable class to associate on.
# RADAR_TARGET is a class-agnostic sentinel: radar contributes position
# but no classifier output, so it's allowed in but treated as a wildcard
# during cross-sensor association (see _tick).
_FUSABLE_CLASSES = {
    TargetClass.PERSON,
    TargetClass.VEHICLE,
    TargetClass.DRONE,
    TargetClass.RADAR_TARGET,
}

# Real, classifier-derived classes. Used to decide whether a hit can
# upgrade a track's class — radar's RADAR_TARGET is NOT real, so it
# never overwrites an existing real class (per Phase 2: once
# EO/thermal say "vehicle", radar can't downgrade that).
_REAL_CLASSES = {
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
        track_match_gate_deg: float = 2.0,
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
        # Phase 2 — radar association uses a more permissive IoU gate
        # because radar bboxes are derived from 3D cluster size at slant
        # range and can barely brush an EO/thermal pixel-detector box on
        # the same target. Tunable via YAML; promote to DEV-tab slider
        # once we know the right operating range.
        self.radar_iou_gate = float(fcfg.get("radar_iou_gate", 0.05))

        # Software extrinsic for THERMAL → EO alignment. Applied to thermal
        # observations' az/el only; EO stays as ground truth. Tuned live
        # from GUI sliders via set_extrinsic().
        tcfg = cfg.get("thermal", {}) or {}
        tex = (tcfg.get("extrinsic") or {})
        self.thermal_az_bias_deg = float(tex.get("az_bias_deg", 0.0))
        self.thermal_el_bias_deg = float(tex.get("el_bias_deg", 0.0))
        # Phase 2 — software extrinsic for RADAR → EO alignment. Mirrors
        # thermal pattern. RadarManager also holds its own copy of these
        # biases (used in the projection-overlay path); the GUI's
        # extrinsic_tune handler keeps them in sync by routing radar
        # fields to BOTH managers. EO remains the ground-truth reference.
        rcfg = cfg.get("radar", {}) or {}
        rex = (rcfg.get("extrinsic") or {})
        self.radar_az_bias_deg = float(rex.get("az_bias_deg", 0.0))
        self.radar_el_bias_deg = float(rex.get("el_bias_deg", 0.0))
        self._ext_lock = threading.Lock()

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

    # ─────────────────────── live extrinsic tuning ───────────────────────
    def set_extrinsic(
        self,
        *,
        thermal_az_bias_deg: Optional[float] = None,
        thermal_el_bias_deg: Optional[float] = None,
        radar_az_bias_deg: Optional[float] = None,
        radar_el_bias_deg: Optional[float] = None,
    ) -> None:
        """Hot-update thermal/radar az/el bias used to align them with EO.

        EO is the ground-truth reference, so only thermal and radar get
        biased. Applied in ``_observations_from_{thermal,radar}`` on the
        next tick. Radar bias is kept in sync with RadarManager's own
        copy by the GUI's extrinsic_tune handler.
        """
        with self._ext_lock:
            if thermal_az_bias_deg is not None:
                self.thermal_az_bias_deg = float(thermal_az_bias_deg)
            if thermal_el_bias_deg is not None:
                self.thermal_el_bias_deg = float(thermal_el_bias_deg)
            if radar_az_bias_deg is not None:
                self.radar_az_bias_deg = float(radar_az_bias_deg)
            if radar_el_bias_deg is not None:
                self.radar_el_bias_deg = float(radar_el_bias_deg)

    def get_extrinsic(self) -> dict:
        with self._ext_lock:
            return {
                "thermal_az_bias_deg": self.thermal_az_bias_deg,
                "thermal_el_bias_deg": self.thermal_el_bias_deg,
                "radar_az_bias_deg":   self.radar_az_bias_deg,
                "radar_el_bias_deg":   self.radar_el_bias_deg,
            }

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
        rf: Optional[RadarFrame] = BUS.get_latest(Topic.RADAR)

        thermal_obs = self._observations_from_thermal(tf)
        eo_obs = self._observations_from_eo(ef)
        radar_obs = self._observations_from_radar(rf)

        # ── Cross-sensor association (EO primary) ──
        # Produce a list of "candidates" per tick. Each candidate is a
        # single real-world target with a class, best angular pose, and
        # a set of contributing sensors. These then match into the
        # persistent tracker.
        candidates: list[dict] = []
        used_t = [False] * len(thermal_obs)
        # Cross-sensor association now uses angular IoU: two observations
        # of the same class with bbox overlap >= XSENSOR_IOU are the same
        # real target. IoU is robust to the "big foreground target eats
        # small background target" bug that plagued center-distance
        # gates, because a small bbox inside a big bbox has IoU ≈ 0.
        XSENSOR_IOU = 0.15  # loose — FOV estimates & parallax can shift centers
        for e in eo_obs:
            best_i, best_iou = -1, 0.0
            for i, t in enumerate(thermal_obs):
                if used_t[i] or t["class"] != e["class"]:
                    continue
                iou = angular_iou(
                    e["az"], e["el"], e["ang_w"], e["ang_h"],
                    t["az"], t["el"], t["ang_w"], t["ang_h"],
                )
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= XSENSOR_IOU:
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

        # ── Pass 3: radar joins ──
        # Each radar observation tries to attach to the best existing
        # EO/thermal candidate by angular IoU (class wildcard — radar
        # has no classifier). Surviving radar obs become standalone
        # candidates with class=RADAR_TARGET so the operator still
        # sees the target in the top-5 list. Uses self.radar_iou_gate
        # (more permissive than XSENSOR_IOU because radar bboxes are
        # cluster-extent-derived and tend to be coarser).
        used_c = [False] * len(candidates)
        for r in radar_obs:
            best_i, best_iou = -1, 0.0
            for i, c in enumerate(candidates):
                if used_c[i]:
                    continue
                iou = angular_iou(
                    r["az"], r["el"], r["ang_w"], r["ang_h"],
                    c["az"], c["el"], c["ang_w"], c["ang_h"],
                )
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= self.radar_iou_gate:
                c = candidates[best_i]
                used_c[best_i] = True
                if "radar" not in c["sensors"]:
                    c["sensors"].append("radar")
                # Don't move the angular pose — EO/thermal pixels are
                # finer than radar's cluster centroid. Just take the
                # max confidence so the row score (sensors+conf) ranks
                # correctly.
                c["conf"] = max(c["conf"], r["conf"])
            else:
                # Standalone radar candidate. Class is the sentinel —
                # the persistence tracker will keep it as RADAR_TARGET
                # until an EO/thermal observation joins later and
                # promotes the class (see _update_tracks).
                candidates.append({
                    "sensors": ["radar"],
                    "primary": "radar",
                    "class":   r["class"],
                    "az":      r["az"],   "el":    r["el"],
                    "ang_w":   r["ang_w"],"ang_h": r["ang_h"],
                    "conf":    r["conf"],
                })

        # Collapse near-duplicate candidates within this tick before
        # they hit the persistence tracker. Without this, two YOLO
        # boxes on the same car (one per sensor, or two from EO) each
        # become a candidate, and greedy track matching means only one
        # claims the existing track — the other spawns a duplicate.
        candidates = self._dedup_candidates(candidates)

        self._update_tracks(candidates)
        # Final safety net: merge any fusion tracks that now overlap in
        # angular space with another track of the same class. The elder
        # (more hits, then lower id) keeps its ID; the younger is dropped.
        self._merge_overlapping_tracks()
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
            # Software extrinsic: bias thermal az/el to align with EO
            # (ground truth). Read under the lock so a mid-tick GUI
            # slider update doesn't tear the two reads.
            with self._ext_lock:
                az += self.thermal_az_bias_deg
                el += self.thermal_el_bias_deg
            out.append({
                "az": az, "el": el, "ang_w": aw, "ang_h": ah,
                "class": cls.value,
                "conf": float(d.classification.confidence),
            })
        return out

    def _observations_from_radar(self, rf: Optional[RadarFrame]) -> list[dict]:
        """Convert RadarTargets into angular observations (az/el/extent).

        Radar observations carry class=RADAR_TARGET (sentinel) — radar
        has no classifier, so association uses IoU only and class is a
        wildcard at match time. Coasting targets are skipped to keep
        cross-sensor fusion conservative; the projection-overlay path
        (gui/sensor_bridge.py) still draws them so the operator can
        tell a dead-reckoned radar box from a fresh one.
        """
        import math
        if rf is None or not rf.connected:
            return []
        out = []
        # Pull the bias once under the lock — match _observations_from_thermal
        # so a mid-tick GUI slider update doesn't tear the two reads.
        with self._ext_lock:
            az_bias = self.radar_az_bias_deg
            el_bias = self.radar_el_bias_deg
        for t in rf.targets:
            # Skip coasting (Kalman-only) targets — fusion shouldn't
            # drag a cross-sensor lock around on dead-reckoned positions.
            if getattr(t, "coasting", False):
                continue
            # Behind/below sensor — skip (also avoids atan2 weirdness).
            if t.pos_y_m <= 0.1:
                continue
            slant = math.sqrt(t.pos_x_m * t.pos_x_m
                              + t.pos_y_m * t.pos_y_m
                              + t.pos_z_m * t.pos_z_m)
            if slant < 0.1:
                continue
            # Cartesian → angular (radar convention: x=right, y=forward, z=up).
            az = math.degrees(math.atan2(t.pos_x_m, t.pos_y_m)) + az_bias
            el = math.degrees(math.atan2(
                t.pos_z_m,
                math.sqrt(t.pos_x_m * t.pos_x_m + t.pos_y_m * t.pos_y_m)
            )) + el_bias
            # Bbox angular extent from physical half-size at slant range.
            # Floor at 0.4° so a tiny cluster still gates against EO/thermal.
            ang_w = max(0.4, math.degrees(2.0 * math.atan2(t.size_x_m, slant)))
            ang_h = max(0.4, math.degrees(2.0 * math.atan2(t.size_z_m, slant)))
            out.append({
                "az": az, "el": el, "ang_w": ang_w, "ang_h": ang_h,
                "class": TargetClass.RADAR_TARGET.value,
                "conf": float(t.confidence),
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
    @staticmethod
    def _class_compatible(a: str, b: str) -> bool:
        """Two tracks/candidates can match if classes are equal OR
        either side is the radar sentinel (radar has no classifier so
        it's a wildcard). Pure equality otherwise — we don't want
        person↔vehicle association."""
        if a == b:
            return True
        rt = TargetClass.RADAR_TARGET.value
        return a == rt or b == rt

    def _update_tracks(self, candidates: list[dict]) -> None:
        # Snapshot the count BEFORE iterating — unmatched candidates
        # append new tracks below, and `matched` only covers pre-existing.
        n_existing = len(self._tracks)
        matched = [False] * n_existing
        # IoU-based matching tolerates EMA drift: even if the track's
        # smoothed bbox drifts, a new observation that clearly overlaps
        # the track still matches, so we don't spawn a duplicate ID.
        TRACK_IOU = 0.15
        rt = TargetClass.RADAR_TARGET.value
        for c in candidates:
            best_i, best_iou = -1, 0.0
            for i in range(n_existing):
                trk = self._tracks[i]
                if matched[i] or not self._class_compatible(trk["class"], c["class"]):
                    continue
                iou = angular_iou(
                    c["az"], c["el"], c["ang_w"], c["ang_h"],
                    trk["az"], trk["el"], trk["ang_w"], trk["ang_h"],
                )
                if iou > best_iou:
                    best_iou, best_i = iou, i
            if best_i >= 0 and best_iou >= TRACK_IOU:
                trk = self._tracks[best_i]
                a = 0.4  # EMA
                trk["az"]    = a * trk["az"]    + (1 - a) * c["az"]
                trk["el"]    = a * trk["el"]    + (1 - a) * c["el"]
                trk["ang_w"] = a * trk["ang_w"] + (1 - a) * c["ang_w"]
                trk["ang_h"] = a * trk["ang_h"] + (1 - a) * c["ang_h"]
                # Class promotion: a radar-born track stays RADAR_TARGET
                # until an EO/thermal observation joins, at which point
                # we lock in the real class. Once locked, never overwrite
                # (per Phase 2 design — EO/thermal classification wins).
                if trk["class"] == rt and c["class"] != rt:
                    trk["class"] = c["class"]
                # Sensor set accumulates — a track that has ever been
                # fused stays "2-sensor" if the next tick only saw EO.
                # Resetting per tick would make the green box flicker.
                trk["sensors_now"] = list(c["sensors"])
                for s in c["sensors"]:
                    if s not in trk["sensors_ever"]:
                        trk["sensors_ever"].append(s)
                # Don't let a radar-only update steal `primary` from
                # a real-class track — cameras own the primary sensor
                # for any track that's been seen by EO/thermal.
                if not (c["primary"] == "radar" and trk["class"] != rt):
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

    # ───────────────────────── dedup helpers ─────────────────────
    def _dedup_candidates(self, cands: list[dict]) -> list[dict]:
        """Merge same-class candidates whose angular centers are within
        gate of each other. Keeps the richer one (more sensors, then
        higher conf). Cameras are boresight-aligned <5cm apart so two
        candidates on one real target should land nearly on top of each
        other in (az, el)."""
        if len(cands) <= 1:
            return cands
        # Sort so the "winners" are visited first: more sensors first,
        # then higher confidence. Losers merge into winners.
        order = sorted(
            range(len(cands)),
            key=lambda i: (len(cands[i]["sensors"]), cands[i]["conf"]),
            reverse=True,
        )
        used = [False] * len(cands)
        out: list[dict] = []
        for i in order:
            if used[i]:
                continue
            w = dict(cands[i])
            w["sensors"] = list(w["sensors"])
            used[i] = True
            # Gate: generous — anything roughly co-angular & same class
            # is the same real-world target given our geometry.
            # Within-tick dedup: strict IoU threshold. Must really
            # overlap — don't collapse neighbors that merely classify
            # the same.
            DEDUP_IOU = 0.35
            for j in order:
                if used[j] or cands[j]["class"] != w["class"]:
                    continue
                c = cands[j]
                iou = angular_iou(
                    w["az"], w["el"], w["ang_w"], w["ang_h"],
                    c["az"], c["el"], c["ang_w"], c["ang_h"],
                )
                if iou >= DEDUP_IOU:
                    used[j] = True
                    for s in c["sensors"]:
                        if s not in w["sensors"]:
                            w["sensors"].append(s)
                    # If merging brings in a thermal hit, promote to
                    # 2-sensor. Primary stays EO if EO was present.
                    if "eo" in w["sensors"]:
                        w["primary"] = "eo"
                    w["conf"] = max(w["conf"], c["conf"])
            out.append(w)
        return out

    def _merge_overlapping_tracks(self) -> None:
        """Collapse fusion tracks that now live on top of each other.

        Scans all track pairs of the same class; if angular centers are
        within gate, keeps the elder (more hits, then lower id) and
        drops the other. Runs once per tick — cheap, O(N²) where N is
        typically <10.
        """
        if len(self._tracks) <= 1:
            return
        # Elder first: more hits, then lower id.
        # Prefer real-class tracks over RADAR_TARGET when picking the
        # survivor — if a vehicle track and a radar-only track collapse,
        # the vehicle track's class must win.
        rt = TargetClass.RADAR_TARGET.value
        order = sorted(
            range(len(self._tracks)),
            key=lambda i: (
                self._tracks[i]["class"] != rt,        # real class first
                self._tracks[i]["hits"],
                -self._tracks[i]["id"],
            ),
            reverse=True,
        )
        drop = [False] * len(self._tracks)
        for oi, i in enumerate(order):
            if drop[i]:
                continue
            a = self._tracks[i]
            for j in order[oi + 1:]:
                if drop[j]:
                    continue
                b = self._tracks[j]
                if not self._class_compatible(a["class"], b["class"]):
                    continue
                # IoU-based merge: two live tracks collapse only when
                # their angular bboxes substantially overlap. A small
                # distant same-class track sitting inside a big track's
                # bbox has IoU ≈ 0, so it's safe.
                MERGE_IOU = 0.35
                iou = angular_iou(
                    a["az"], a["el"], a["ang_w"], a["ang_h"],
                    b["az"], b["el"], b["ang_w"], b["ang_h"],
                )
                if iou >= MERGE_IOU:
                    # Fold b into a: union the sensor sets, bump hits.
                    for s in b["sensors_ever"]:
                        if s not in a["sensors_ever"]:
                            a["sensors_ever"].append(s)
                    a["conf"] = max(a["conf"], b["conf"])
                    # If `a` is RADAR_TARGET and `b` carries a real
                    # class, promote — class always upgrades, never
                    # downgrades.
                    if a["class"] == rt and b["class"] != rt:
                        a["class"] = b["class"]
                    drop[j] = True
        if any(drop):
            self._tracks = [t for k, t in enumerate(self._tracks) if not drop[k]]

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

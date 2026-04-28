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
        # Per-sensor grace window: how many fusion ticks a sensor can
        # miss a track before it's removed from the published `sensors`
        # list. Larger = more "sticky" (fewer green→red flickers from
        # 1-frame dropouts), smaller = the displayed sensor set tracks
        # ground truth more tightly. Default 5 ticks = ~333ms at 15Hz.
        self.sensor_grace_ticks = int(fcfg.get("sensor_grace_ticks", 5))

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

        # Phase 2 — world-frame fusion. When enabled (default), candidate
        # az/el is converted to world frame at ingest:
        #     world_az = cam_az + cur_pan_at_obs
        #     world_el = cam_el + cur_tilt_at_obs
        # Tracks STORE world az/el. Matching is in world frame, so a
        # target's identity survives gimbal motion (camera-frame matching
        # silently transfers a track to whichever vehicle currently sits
        # at the same camera-frame az — the bug behind both the YOLO
        # id-swap mistrack case AND the apparent "oscillation" on a
        # static target visible in 'single track oscilating after being
        # in the center.jsonl').
        # Output FusedTrack carries CAMERA-frame az/el (= world - cur_pan
        # at publish time) so the gimbal control path and GUI overlay
        # don't need any changes — they keep reading az_deg/el_deg as
        # offsets from current boresight.
        # Set false to revert to legacy camera-frame matching for A/B.
        self._world_frame: bool = bool(
            fcfg.get("world_frame_fusion", True))
        # Cached gimbal pose snapshot — captured at the start of each
        # _tick so all candidates use a consistent pose for world-frame
        # conversion. Default 0,0 until first tick (effectively
        # camera-frame for the first publish).
        self._cur_gimbal_pose: tuple[float, float] = (0.0, 0.0)

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
        # Snapshot the current gimbal pose so we can compensate
        # tracker-track az/el during matching. Without this, fusion
        # matches new observations to existing tracks in CAMERA frame
        # — and a fast gimbal slew between observations shifts the
        # same world target's camera-az enough to miss the IoU gate
        # → new track id every time YOLO+ByteTrack briefly drops the
        # target, which is what killed every TRACK lock in the
        # 2026-04-27 Human_and_vehicle_mistrack recording.
        from common.frames import GimbalState as _GS
        gs = BUS.get_latest(Topic.GIMBAL)
        cur_pan = float(gs.pan_deg) if isinstance(gs, _GS) else 0.0
        cur_tilt = float(gs.tilt_deg) if isinstance(gs, _GS) else 0.0
        self._cur_gimbal_pose = (cur_pan, cur_tilt)

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
        # Snapshot the camera-candidate count BEFORE the outer loop
        # because the unmatched-radar `else` branch appends to
        # `candidates`, and the next outer iteration would otherwise
        # walk past the end of `used_c`. Radar-vs-radar matching is
        # already handled at radar-track level; only camera candidates
        # are association targets here.
        n_cam_cands = len(candidates)
        used_c = [False] * n_cam_cands
        for r in radar_obs:
            best_i, best_iou = -1, 0.0
            for i in range(n_cam_cands):
                if used_c[i]:
                    continue
                c = candidates[i]
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

        # ── Phase 2: convert to WORLD frame for persistence-tracker
        # matching. Camera-frame matching silently transfers a track
        # to whichever vehicle currently sits at the same camera-frame
        # az during a gimbal slew — the bug behind the YOLO id-swap
        # in `Human_and_vehicle_mistrack.jsonl` AND the apparent
        # oscillation on a static target in
        # 'single track oscilating after being in the center.jsonl'.
        # World-frame matching preserves the target's identity across
        # arbitrary gimbal motion, since a static target's world az/el
        # is constant.
        if self._world_frame:
            for c in candidates:
                c["az"] = c["az"] + cur_pan
                c["el"] = c["el"] + cur_tilt

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
        # 2026-04-27 morning fix added (a) a gimbal-pose-compensated
        # track-az shift `trk_az_now = trk["az"] - (cur_pan - last_pan)`
        # before IoU and (b) a 3° centroid-distance fallback gate.
        # Both REVERTED the same afternoon: `gimbal_not_tracking_static.jsonl`
        # showed fused track #20 merging two distinct nearby vehicles
        # — bbox angular size bouncing between (3.27, 1.76) and
        # (5.05, 4.34) as the IoU "match" alternated between them.
        # The original IoU-only matcher with no gimbal compensation
        # is correct for this scene density (multiple vehicles within
        # one bbox-width). Re-introducing either fix needs a multi-
        # vehicle-scene replay test, not just a YOLO-id-swap one.
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
                # Per-sensor decay: bump everyone's miss count first,
                # then reset to 0 for sensors actually seen this tick.
                # Sensors that exceed the grace window get pruned, so
                # `sensors` reflects WHO IS CURRENTLY SEEING IT (with
                # a small grace) rather than who has ever seen it. The
                # green "2+ sensor" pill therefore decays back to single-
                # sensor automatically when radar leaves the scene.
                for s in list(trk["sensor_misses"].keys()):
                    trk["sensor_misses"][s] += 1
                for s in c["sensors"]:
                    trk["sensor_misses"][s] = 0
                trk["sensor_misses"] = {
                    s: m for s, m in trk["sensor_misses"].items()
                    if m <= self.sensor_grace_ticks
                }
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
                    "sensor_misses": {s: 0 for s in c["sensors"]},
                    "primary": c["primary"],
                    "conf": c["conf"],
                    "hits": 1, "misses": 0,
                })
                try:
                    from common.events import emit as _emit
                    _emit("fused_track_born", {
                        "id": int(self._next_id),
                        "class": c["class"],
                        "primary": c["primary"],
                        "az": float(c["az"]),
                        "el": float(c["el"]),
                        "sensors": list(c["sensors"]),
                        "conf": float(c["conf"]),
                    })
                except Exception:
                    pass
                self._next_id += 1

        kept = []
        for i, trk in enumerate(self._tracks):
            if i < len(matched) and matched[i]:
                kept.append(trk)
            else:
                trk["misses"] += 1
                # No sensor saw this track this tick — bump every
                # contributing sensor's miss counter and prune any that
                # crossed the grace threshold. This is what makes the
                # "RADAR" tag drop off ~333ms after radar stops seeing
                # it, even while EO/thermal still hold the track alive.
                for s in list(trk["sensor_misses"].keys()):
                    trk["sensor_misses"][s] += 1
                trk["sensor_misses"] = {
                    s: m for s, m in trk["sensor_misses"].items()
                    if m <= self.sensor_grace_ticks
                }
                if trk["misses"] <= self.max_misses:
                    kept.append(trk)
                else:
                    try:
                        from common.events import emit as _emit
                        _emit("fused_track_dropped", {
                            "id": int(trk["id"]),
                            "hits": int(trk["hits"]),
                            "misses": int(trk["misses"]),
                            "reason": "max_misses",
                        })
                    except Exception:
                        pass
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
                    # Fold b into a: union the active-sensor dict
                    # taking min misses for any shared sensor (so a
                    # freshly-seen sensor on either track wins).
                    for s, m in b["sensor_misses"].items():
                        cur = a["sensor_misses"].get(s)
                        a["sensor_misses"][s] = m if cur is None else min(cur, m)
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
        # When matching in world frame (Phase 2), tracks store world
        # az/el. Output FusedTrack carries CAMERA-frame az/el so the
        # gimbal control path and GUI overlay (which subtract cur_pan
        # from track az implicitly via "image position from boresight"
        # math) don't need any changes.
        cur_pan, cur_tilt = self._cur_gimbal_pose if self._world_frame else (0.0, 0.0)
        out: list[FusedTrack] = []
        for trk in self._tracks:
            if trk["hits"] < self.min_hits:
                continue
            # Currently-active sensors only (per Phase 2: a sensor that
            # stops contributing for > sensor_grace_ticks is pruned, so
            # the "2+ sensor" outline decays back to single-sensor when
            # e.g. radar leaves the scene).
            active_sensors = sorted(trk["sensor_misses"].keys())
            # Skip publishing tracks whose active-sensor set is empty.
            # The track stays in self._tracks (Kalman-style coasting up
            # to max_misses) so a re-acquire keeps the same id, but
            # without active sensors it has nothing to project from
            # confidently — drawing a stale dashed bbox on every panel
            # was misleading.
            if not active_sensors:
                continue
            try:
                tc = TargetClass(trk["class"])
            except ValueError:
                tc = TargetClass.UNKNOWN
            # Convert track-stored world az/el back to camera frame for
            # the published FusedTrack. When _world_frame is False the
            # track az/el is already camera frame and we subtract zero.
            pub_az = float(trk["az"]) - cur_pan
            pub_el = float(trk["el"]) - cur_tilt
            out.append(FusedTrack(
                id=int(trk["id"]),
                target_class=tc,
                confidence=float(trk["conf"]),
                sensors=active_sensors,
                primary=str(trk["primary"]),
                az_deg=pub_az,
                el_deg=pub_el,
                ang_w_deg=float(trk["ang_w"]),
                ang_h_deg=float(trk["ang_h"]),
                hits=int(trk["hits"]),
                misses=int(trk["misses"]),
            ))
        BUS.publish(Topic.FUSED, out)

"""
Temporal tracker for thermal detections.

Pipeline (per frame):

    1. Kalman predict — every existing track extrapolates (x, y) forward
       one tick from its own velocity. Matching is done against the
       PREDICTED centroid, not the last observation. This is what makes
       fast gimbal slews survivable: even if the blob jumps 80 px in
       one frame, the Kalman predicts ~80 px because last tick's
       velocity already said so, and the match still succeeds.

    2. Detection <-> track matching by centroid distance, greedy best-first.
       Distance is predicted-centroid <-> detection-centroid, so the
       "budget" only has to cover *acceleration* error, not raw motion.

    3. For matched pairs: Kalman update with the detection centroid as
       measurement. Bbox is EMA-smoothed.

    4. For UNmatched confirmed tracks: try to bridge with sparse Lucas-
       Kanade optical flow. We saved good feature points inside the
       bbox last frame; find them in the new frame, compute the
       centroid shift, and feed that as a "virtual measurement" to
       Kalman with inflated noise. This keeps the lock alive when the
       heat detector drops a frame (blob momentarily merges into warm
       background, etc.).

       The OF bridge is GATED on several sanity checks to avoid
       "ghost track" drift (see `_of_estimate_shift` and the bridge
       block in `update`):

         - `max_of_bridges`: a track can only be kept alive by OF for
           at most N consecutive frames. After that it must see a real
           detection or it is dropped. Prevents permanent ghosts.
         - Per-bridge displacement cap at `max_dist_px * 0.5`: LK
           occasionally returns "consistent" large shifts on pure
           texture; reject anything that looks like teleportation.
         - Post-shift warmth check: the new ROI must still contain a
           patch measurably hotter than its surround. If the blob has
           evaporated, the OF "lock" is onto background texture and we
           let the track die naturally.
         - Inlier spread check (unchanged): if the feature points
           disagree with each other, we don't trust the median.

    5. Still-unmatched tracks just predict (pure coast) and accumulate
       misses. Past `max_misses` they die.

    6. Unmatched detections spawn new tracks with zero initial velocity.

Why not IoU matching? Thermal blobs morph unpredictably — a blob can
halve in area between frames. Centroid distance on a Kalman prediction
is far more stable for small warm targets.

Classification stickiness: when a track is matched to a new detection,
the new detection inherits the previous track's classification if the
new one is None (classifier runs every Nth frame, we don't want the
label to blink on the in-between frames).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

from common.frames import BBox, ThermalDetection
from common.logging_setup import get_logger

log = get_logger(__name__)


# --------------------------------- Kalman -----------------------------
#
# 4-state constant-velocity model per track:
#
#     state  x = [px, py, vx, vy]^T
#     F = [[1 0 dt 0],
#          [0 1 0 dt],
#          [0 0 1  0],
#          [0 0 0  1]]
#     H = [[1 0 0 0],
#          [0 1 0 0]]
#
# dt = 1 tick (we run at a fixed ~20 Hz, no need to carry wall time).
#
# Noise tuning rationale (defaults in TrackerConfig):
#
#   q_pos = 4  (px^2 per tick)
#     Sigma_pos ~ 2 px/tick of untracked "position wander" the model
#     doesn't account for — roughly the frame-to-frame centroid jitter
#     we see on a stationary warm blob from detector re-centering.
#
#   q_vel = 9  ((px/tick)^2 per tick)
#     Sigma_vel ~ 3 px/tick^2 acceleration budget. Typical gimbal slews
#     are ~20 px/tick of velocity; the filter needs a few ticks to catch
#     up. q_vel=9 lets vx/vy change by ~3 px/tick each step without the
#     filter fighting it; any larger and we start tracking detector
#     noise as acceleration, any smaller and we lag fast maneuvers.
#
#   r_meas = 4  (px^2)
#     Sigma_meas ~ 2 px on a real detection centroid (blob edge ambiguity
#     on low-contrast warm targets). Matches observed jitter. Q/R ~ 1:1
#     on position means the filter weights prediction and measurement
#     roughly equally, which is the behavior we want: prediction
#     dominates during one-tick gaps, measurement dominates on steady
#     hits.
#
# Override via TrackerConfig when the scene changes (e.g. on-gimbal
# motion → try q_vel=16; extremely stable tripod → q_vel=4).


class _Kalman2D:
    """Minimal constant-velocity Kalman filter. Pure NumPy — no cv2
    dependency so we can keep the tracker hot-path tight and avoid the
    cv2.KalmanFilter weirdness with matrix dtypes."""

    __slots__ = ("x", "P", "F", "H", "Q", "R")

    def __init__(self, px: float, py: float,
                 q_pos: float, q_vel: float, r_meas: float) -> None:
        # State and covariance
        self.x = np.array([px, py, 0.0, 0.0], dtype=np.float64)
        # Initial uncertainty: position tight (we have a detection),
        # velocity wide (we have no idea yet).
        self.P = np.diag([4.0, 4.0, 200.0, 200.0]).astype(np.float64)

        self.F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        self.Q = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float64)
        self.R = np.diag([r_meas, r_meas]).astype(np.float64)

    def predict(self) -> Tuple[float, float]:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, mx: float, my: float, r_scale: float = 1.0) -> None:
        """Apply a measurement. ``r_scale`` inflates R for low-confidence
        observations (e.g. the optical-flow bridge)."""
        z = np.array([mx, my], dtype=np.float64)
        R = self.R * float(r_scale)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    @property
    def pos(self) -> Tuple[float, float]:
        return float(self.x[0]), float(self.x[1])

    @property
    def vel(self) -> Tuple[float, float]:
        return float(self.x[2]), float(self.x[3])


# --------------------------------- data types -------------------------

def _centroid(b: BBox) -> Tuple[float, float]:
    return (b.x + b.w * 0.5, b.y + b.h * 0.5)


@dataclass
class _Track:
    det: ThermalDetection
    kf: _Kalman2D
    hits: int = 1
    misses: int = 0
    age: int = 1
    id: int = 0
    # Count of consecutive frames the track has been kept alive ONLY by
    # the optical-flow bridge (reset to 0 on any real-detection match).
    # Capped by TrackerConfig.max_of_bridges.
    of_bridge_streak: int = 0
    # Whether the track has ever been emitted as CONFIRMED (first time
    # hits reached min_hits). Used to log the CONFIRMED transition once.
    announced_confirmed: bool = False
    # Optical-flow bridge state: last frame's feature points (Nx1x2 float32)
    # sampled inside the bbox, used by cv2.calcOpticalFlowPyrLK to locate
    # the track when the detector drops a frame. Refreshed on every
    # real-detection match.
    of_pts: Optional[np.ndarray] = field(default=None, repr=False)
    # Whether the most recent tick was a coast (no real-detection
    # match AND no OF bridge match). For synthetic tracks the GUI
    # uses this instead of (misses > 0), since misses-on-synthetic
    # is a one-way counter that never resets and would otherwise
    # paint the target as "coasting" forever after a single OF miss
    # during a fast slew.
    coasted_last_tick: bool = False
    # Track is born from a user "Draw Target" bbox rather than a heat
    # detection. Propagates purely via optical flow — no warmth check,
    # no streak / misses accounting (never needs a real detection), no
    # classifier pass. Lives until user clears or OF loses lock entirely.
    synthetic: bool = False


@dataclass
class HeatTrackSnapshot:
    """Debug view of a single heat-blob track. Exposed via
    ``DetectionTracker.snapshot()`` and pushed to the GUI in developer
    mode so the user can watch the temporal tracker's internal state
    (IDs, hits, misses, whether a track is coasting)."""
    id: int
    bbox: BBox
    hits: int
    misses: int
    age: int
    confirmed: bool       # hits >= min_hits (would be drawn as a box)
    coasting: bool        # misses > 0 (not matched this tick)
    synthetic: bool = False  # user-drawn "Draw Target" seed


@dataclass
class TrackerConfig:
    enabled: bool = True
    # Maximum centroid distance from the KALMAN-PREDICTED position to
    # an incoming detection for them to be matched. Because we match
    # against the prediction, this only has to cover acceleration /
    # jerk — not raw motion — so a relatively tight value is fine.
    max_dist_px: float = 60.0
    min_hits: int = 5            # frames before a track is emitted
    # Frames a confirmed track survives without a detection. At 20 Hz,
    # 6 ticks ~ 300 ms of Kalman coast — enough to bridge a single
    # detector hiccup without producing "ghost" boxes that visibly drift
    # across the screen on longer dropouts.
    max_misses: int = 6
    ema: float = 0.4             # bbox smoothing: 0 = raw, 1 = frozen

    # Kalman noise knobs (in px^2 for pos / (px/tick)^2 for vel / px^2 for meas).
    # See module docstring ("Noise tuning rationale") for why these values.
    kf_q_pos: float = 4.0
    kf_q_vel: float = 9.0
    kf_r_meas: float = 4.0

    # ---- Optical-flow bridge ------------------------------------------
    # Re-enabled with guards (see module docstring). The bridge is
    # tightly gated to avoid the "purple ghost" drift seen in early
    # testing:
    #   - max_of_bridges limits how long a track can survive on OF
    #     alone before it must see a real detection
    #   - of_max_shift_px (computed as max_dist_px * 0.5) caps per-bridge
    #     displacement, rejecting LK teleportation on texture
    #   - post-shift warmth check rejects bridges where the ROI no
    #     longer contains a measurably warm patch
    #   - inlier-spread check rejects scattered feature point clouds
    of_enabled: bool = True
    of_r_scale: float = 25.0     # how much to inflate R for OF measurements
    of_max_features: int = 20    # max corners to track per bbox
    of_min_features: int = 3     # min surviving corners to trust OF
    of_win_size: int = 21        # LK window (larger = handles more motion)
    of_max_level: int = 3        # LK pyramid levels
    # Maximum allowed displacement between the median feature shift and
    # any individual feature, in px, before we consider the OF result
    # unreliable (scene change, occlusion) and discard it.
    of_max_inlier_spread: float = 15.0
    # Max consecutive frames a confirmed track may be kept alive SOLELY
    # by the OF bridge with no real detection. After this many bridges
    # in a row the bridge is refused; Kalman coast then max_misses will
    # drop the track normally. 3 @ 20 Hz = 150 ms of blind coast, which
    # is about as much as we want to trust LK on a warm target.
    max_of_bridges: int = 3
    # Minimum mean-intensity contrast (ROI - surround) for the bridged
    # ROI to be considered "still warm". In uint8 display space. 5 is a
    # loose floor; pure-texture ROIs come out within +/-1 of their
    # surround and get rejected, real warm blobs easily clear this.
    of_min_warmth_contrast: float = 5.0


# --------------------------------- tracker ----------------------------

class DetectionTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self._tracks: List[_Track] = []
        self._next_id: int = 1
        # Previous frame grayscale for optical flow. Kept as a reference
        # (the manager passes a fresh ndarray each tick, and we just
        # swap it out). None on the first frame.
        self._prev_gray: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._tracks.clear()
        self._prev_gray = None
        # IDs stay globally unique across resets.

    def snapshot(self) -> List[HeatTrackSnapshot]:
        if not self.cfg.enabled:
            return []
        out: List[HeatTrackSnapshot] = []
        for trk in self._tracks:
            # Synthetic tracks have no detector to "match" against, so
            # `misses` only ever increments and never resets — making the
            # legacy `coasting=(misses > 0)` flag sticky from the second
            # tick onward. For the GUI it's cosmetic, but it lies about
            # whether OF is actually keeping up. Use a short rolling
            # window: synthetic tracks are coasting only if the LAST
            # tick was an actual coast (no OF match). For real tracks
            # the legacy semantics are preserved — they get reset by
            # _merge_detection on a real heat-blob match.
            if trk.synthetic:
                coasting = bool(trk.coasted_last_tick)
            else:
                coasting = (trk.misses > 0)
            out.append(HeatTrackSnapshot(
                id=trk.id,
                bbox=trk.det.bbox,
                hits=trk.hits,
                misses=trk.misses,
                age=trk.age,
                confirmed=trk.synthetic or (trk.hits >= self.cfg.min_hits),
                coasting=coasting,
                synthetic=trk.synthetic,
            ))
        return out

    # ------------------------------------------------------------------
    # Synthetic "Draw Target" tracks
    # ------------------------------------------------------------------
    def seed_synthetic(self,
                       bbox: BBox,
                       agc8: Optional[np.ndarray] = None,
                       ) -> Optional[int]:
        """Seed a user-drawn bbox as a synthetic track.

        The track is born CONFIRMED (hits = min_hits), carries
        ``synthetic=True`` on both the _Track and its ThermalDetection,
        and will propagate via optical flow only. Any previously-seeded
        synthetic track is replaced — one user target at a time.

        Returns the new track's internal ID, or None if the bbox is
        degenerate.
        """
        if bbox.w < 2 or bbox.h < 2:
            return None
        # Only one synthetic target at a time — replace any existing one.
        self._tracks = [t for t in self._tracks if not t.synthetic]

        cx = bbox.x + bbox.w * 0.5
        cy = bbox.y + bbox.h * 0.5
        kf = _Kalman2D(cx, cy,
                       q_pos=self.cfg.kf_q_pos,
                       q_vel=self.cfg.kf_q_vel,
                       r_meas=self.cfg.kf_r_meas)
        from common.frames import ClassificationResult, TargetClass
        det = ThermalDetection(
            bbox=BBox(x=int(bbox.x), y=int(bbox.y),
                      w=int(bbox.w), h=int(bbox.h)),
            area_px=int(bbox.w * bbox.h),
            contrast=0.0,
            classification=ClassificationResult(
                target_class=TargetClass.UNKNOWN,
                confidence=1.0,
                classifier_used="user",
            ),
            synthetic=True,
        )
        of_pts = None
        if agc8 is not None and _HAS_CV2:
            of_pts = _sample_features(agc8, det.bbox, self.cfg.of_max_features)
        new_id = self._next_id
        self._next_id += 1
        trk = _Track(
            det=det,
            kf=kf,
            hits=max(1, self.cfg.min_hits),  # confirmed from birth
            id=new_id,
            of_pts=of_pts,
            synthetic=True,
            announced_confirmed=True,
        )
        self._tracks.append(trk)
        log.info(
            "synthetic-track BORN id=%d bbox=(%d,%d,%d,%d)",
            new_id, bbox.x, bbox.y, bbox.w, bbox.h,
        )
        return new_id

    def clear_synthetic(self) -> int:
        """Remove all synthetic tracks. Returns the count removed."""
        before = len(self._tracks)
        self._tracks = [t for t in self._tracks if not t.synthetic]
        removed = before - len(self._tracks)
        if removed:
            log.info("synthetic-track CLEARED (%d removed)", removed)
        return removed

    def has_synthetic(self) -> bool:
        return any(t.synthetic for t in self._tracks)

    def update(self,
               detections: List[ThermalDetection],
               agc8: Optional[np.ndarray] = None,
               ) -> List[ThermalDetection]:
        """Run one tick. Returns CONFIRMED, matched-this-frame detections.

        ``agc8`` is the current frame's grayscale image (display-space,
        same coordinate frame as the bboxes). When provided, the optical-
        flow bridge uses it to re-localize unmatched confirmed tracks.
        If None, the tracker degrades to Kalman-only coasting.
        """
        if not self.cfg.enabled:
            return detections

        # 1. Kalman predict everyone forward one tick. Capture the
        #    pre-predict centroid BEFORE the predict call so the OF
        #    bridge below can use it directly as the "old position"
        #    anchor (no velocity-back-out math required — keeps the
        #    OF measurement derivation honest and readable).
        predicted: List[Tuple[float, float]] = []
        prev_centroids: List[Tuple[float, float]] = []
        for trk in self._tracks:
            prev_cx, prev_cy = _centroid(trk.det.bbox)
            prev_centroids.append((prev_cx, prev_cy))
            # If this is a synthetic track that was seeded without a
            # frame (of_pts is None), grab features now so OF can
            # start working from THIS tick forward. Without this the
            # track would coast at v=0 forever and the gimbal would
            # slew the scene out from under a frozen bbox.
            if (trk.synthetic and trk.of_pts is None
                    and agc8 is not None and _HAS_CV2):
                trk.of_pts = _sample_features(agc8, trk.det.bbox,
                                              self.cfg.of_max_features)
                log.info(
                    "synthetic track id=%d: late-sampled %d OF features",
                    trk.id,
                    0 if trk.of_pts is None else len(trk.of_pts),
                )
            px, py = trk.kf.predict()
            predicted.append((px, py))
            # Shift bbox to predicted position. EMA later replaces this
            # with a real measurement if one arrives.
            dx = px - prev_cx
            dy = py - prev_cy
            trk.det = _shift_bbox(trk.det, dx, dy)

        # 2. Match detections to predicted positions, greedy best-first.
        #    Synthetic (user-drawn) tracks are excluded from detection
        #    matching — they live on OF alone and are never re-bound to a
        #    heat blob that happens to wander under them.
        candidates: list[tuple[float, int, int]] = []
        for ti, (px, py) in enumerate(predicted):
            if self._tracks[ti].synthetic:
                continue
            for di, d in enumerate(detections):
                dcx, dcy = _centroid(d.bbox)
                dist = math.hypot(px - dcx, py - dcy)
                if dist <= self.cfg.max_dist_px:
                    candidates.append((dist, ti, di))
        candidates.sort()

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for _, ti, di in candidates:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            self._merge_detection(self._tracks[ti], detections[di], agc8)

        # 3. Optical-flow bridge for unmatched CONFIRMED tracks.
        #    Only bother if we have last frame, this frame, and the track
        #    was worth drawing. Pending tracks (hits < min_hits) don't
        #    get a bridge — they're cheap to respawn.
        of_shift_cap = self.cfg.max_dist_px * 0.5
        if (self.cfg.of_enabled and _HAS_CV2
                and self._prev_gray is not None and agc8 is not None
                and self._prev_gray.shape == agc8.shape):
            for ti, trk in enumerate(self._tracks):
                if ti in matched_t:
                    continue
                if not trk.synthetic and trk.hits < self.cfg.min_hits:
                    continue
                # Refuse to bridge if we've already OF-bridged this track
                # for max_of_bridges frames in a row — that way a truly
                # gone target dies instead of drifting forever. Synthetic
                # tracks skip this cap: they have no heat detection that
                # could ever reset the streak, so capping would kill the
                # user target after max_of_bridges frames regardless of
                # whether OF is still locked.
                if (not trk.synthetic
                        and trk.of_bridge_streak >= self.cfg.max_of_bridges):
                    continue
                shift = self._of_estimate_shift(trk, agc8)
                if shift is None:
                    continue
                dx, dy = shift
                # Per-bridge displacement cap: LK sometimes produces a
                # large "consistent" shift on pure texture. Anything
                # bigger than half the match budget is almost certainly
                # wrong.
                #
                # Synthetic tracks need a much larger cap: the moment
                # the user draws a box, the gimbal auto-locks and slews
                # toward the target, which drags the whole scene across
                # the image at up to ~35 px/frame (120°/s over a 75°
                # FOV at 30 Hz capture). The normal cap would reject
                # the legitimate OF match every frame during slew,
                # leaving the Kalman to coast at v=0 while the scene
                # runs out from under the bbox. Scale the cap with the
                # match budget itself rather than halving it.
                trk_shift_cap = (self.cfg.max_dist_px * 1.5
                                 if trk.synthetic else of_shift_cap)
                if math.hypot(dx, dy) > trk_shift_cap:
                    continue
                # Anchor on the pre-predict centroid captured in step 1.
                # Observation = old position + measured OF shift.
                old_cx, old_cy = prev_centroids[ti]
                meas_x = old_cx + dx
                meas_y = old_cy + dy

                # Warmth check: build the candidate bbox and make sure
                # it still contains something hotter than its surround.
                # Pure-texture drift fails this because there's no warm
                # blob at the new location.
                #
                # Skipped for synthetic tracks — the user targeted a
                # cold object on purpose (parked car, tree). The
                # displacement cap + inlier spread still guard drift.
                cur_cx, cur_cy = _centroid(trk.det.bbox)
                cand_bbox = _shift_bbox(trk.det,
                                        meas_x - cur_cx,
                                        meas_y - cur_cy).bbox
                if (not trk.synthetic
                        and not _roi_still_warm(agc8, cand_bbox,
                                                self.cfg.of_min_warmth_contrast)):
                    continue

                trk.kf.update(meas_x, meas_y, r_scale=self.cfg.of_r_scale)
                # Snap bbox to the OF-derived position (EMA would lag
                # too much on a coast). Re-sample features from the
                # new location for the next tick.
                nx, ny = trk.kf.pos
                cur_cx, cur_cy = _centroid(trk.det.bbox)
                trk.det = _shift_bbox(trk.det, nx - cur_cx, ny - cur_cy)
                trk.of_pts = _sample_features(agc8, trk.det.bbox,
                                              self.cfg.of_max_features)
                matched_t.add(ti)  # counts as a (weak) match
                # Do NOT reset misses to 0: the OF bridge is a weak
                # match and we still want max_misses to eventually kill
                # a track that only ever gets OF updates. We decrement
                # age a tick but leave misses ticking in step 4 below
                # by NOT adding ti to matched_t... wait, we did. So
                # instead: increment the bridge streak here and let
                # step 4 treat this as matched. Streak cap (above) is
                # the real guardrail.
                trk.of_bridge_streak += 1
                trk.age += 1

        # 4. Age / drop unmatched tracks.
        kept: List[_Track] = []
        for ti, trk in enumerate(self._tracks):
            if ti in matched_t:
                trk.coasted_last_tick = False
                kept.append(trk)
                continue
            trk.misses += 1
            trk.age += 1
            trk.coasted_last_tick = True
            # Synthetic (user-drawn) tracks are NEVER dropped by max_misses.
            # They persist on Kalman coast until the user hits Clear Target
            # or seeds a new one. The user owns the target lifecycle; the
            # tracker just carries the box.
            if trk.synthetic or trk.misses <= self.cfg.max_misses:
                kept.append(trk)
            else:
                log.info(
                    "heat-track DROPPED id=%d pos=(%.1f,%.1f) reason=max_misses "
                    "(hits=%d age=%d of_streak=%d)",
                    trk.id, trk.kf.pos[0], trk.kf.pos[1],
                    trk.hits, trk.age, trk.of_bridge_streak,
                )

        # 5. Birth new tracks for unmatched detections.
        for di, d in enumerate(detections):
            if di in matched_d:
                continue
            cx, cy = _centroid(d.bbox)
            kf = _Kalman2D(cx, cy,
                           q_pos=self.cfg.kf_q_pos,
                           q_vel=self.cfg.kf_q_vel,
                           r_meas=self.cfg.kf_r_meas)
            of_pts = _sample_features(agc8, d.bbox,
                                      self.cfg.of_max_features) if agc8 is not None else None
            new_id = self._next_id
            self._next_id += 1
            kept.append(_Track(det=d, kf=kf, id=new_id, of_pts=of_pts))
            log.info(
                "heat-track BORN id=%d pos=(%.1f,%.1f) area=%d",
                new_id, cx, cy, d.area_px,
            )

        self._tracks = kept

        # 5b. Confirmation announcements. Log once when a track crosses
        #     min_hits for the first time; this is what the user reads
        #     tomorrow morning to diagnose "why didn't my track lock?"
        for trk in self._tracks:
            if (not trk.announced_confirmed
                    and trk.hits >= self.cfg.min_hits):
                trk.announced_confirmed = True
                log.info(
                    "heat-track CONFIRMED id=%d pos=(%.1f,%.1f) hits=%d age=%d",
                    trk.id, trk.kf.pos[0], trk.kf.pos[1],
                    trk.hits, trk.age,
                )

        # Save this frame for the next tick's optical flow. Copy so
        # external mutation of the source buffer doesn't corrupt us.
        if agc8 is not None and _HAS_CV2:
            self._prev_gray = agc8.copy()

        # 6. Emit confirmed tracks matched this frame (real OR OF bridge).
        #    Synthetic (user-drawn) tracks are always emitted — they're
        #    rendered by the GUI from the detections list regardless of
        #    whether OF matched this tick, and their bbox is carried
        #    forward on Kalman coast.
        out: List[ThermalDetection] = []
        for trk in self._tracks:
            if trk.synthetic:
                out.append(trk.det)
            elif trk.hits >= self.cfg.min_hits and trk.misses == 0:
                out.append(trk.det)
        return out

    # -- internals -----------------------------------------------------

    def _merge_detection(self,
                         trk: _Track,
                         d: ThermalDetection,
                         agc8: Optional[np.ndarray]) -> None:
        """Matched pair: Kalman update, EMA bbox, refresh OF features."""
        mcx, mcy = _centroid(d.bbox)
        trk.kf.update(mcx, mcy, r_scale=1.0)

        # EMA the bbox shape, but snap center to Kalman estimate so the
        # prediction dominates when detection centroids jitter.
        a = self.cfg.ema
        b_prev = trk.det.bbox
        new_w = int(round(a * b_prev.w + (1 - a) * d.bbox.w))
        new_h = int(round(a * b_prev.h + (1 - a) * d.bbox.h))
        kx, ky = trk.kf.pos
        smoothed = BBox(
            x=int(round(kx - new_w * 0.5)),
            y=int(round(ky - new_h * 0.5)),
            w=new_w,
            h=new_h,
        )
        cls = d.classification if d.classification is not None else trk.det.classification
        trk.det = ThermalDetection(
            bbox=smoothed,
            area_px=d.area_px,
            contrast=d.contrast,
            classification=cls,
        )
        trk.hits += 1
        trk.misses = 0
        trk.of_bridge_streak = 0  # real detection resets the bridge streak
        trk.age += 1

        # Refresh optical-flow feature points inside the (smoothed)
        # bbox for next frame's bridge attempt.
        if agc8 is not None and _HAS_CV2:
            trk.of_pts = _sample_features(agc8, smoothed, self.cfg.of_max_features)

    def _of_estimate_shift(self,
                           trk: _Track,
                           curr_gray: np.ndarray,
                           ) -> Optional[Tuple[float, float]]:
        """Lucas-Kanade the previous bbox's features into the current
        frame and return the median (dx, dy). None if not enough features
        survive or the point cloud is too scattered (unreliable)."""
        if trk.of_pts is None or len(trk.of_pts) < self.cfg.of_min_features:
            return None
        prev_gray = self._prev_gray
        if prev_gray is None:
            return None

        try:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, trk.of_pts, None,
                winSize=(self.cfg.of_win_size, self.cfg.of_win_size),
                maxLevel=self.cfg.of_max_level,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
        except cv2.error:
            return None
        if nxt is None or status is None:
            return None

        good_prev = trk.of_pts[status.flatten() == 1]
        good_nxt = nxt[status.flatten() == 1]
        if len(good_nxt) < self.cfg.of_min_features:
            return None

        deltas = (good_nxt - good_prev).reshape(-1, 2)
        med = np.median(deltas, axis=0)
        spread = np.max(np.linalg.norm(deltas - med, axis=1))
        if float(spread) > self.cfg.of_max_inlier_spread:
            # Points scattered -> probably tracking different things.
            # Bail rather than drag the track somewhere random.
            return None
        return float(med[0]), float(med[1])


# --------------------------------- helpers ----------------------------

def _shift_bbox(d: ThermalDetection, dx: float, dy: float) -> ThermalDetection:
    """Return a new ThermalDetection with the bbox translated by (dx, dy)."""
    b = d.bbox
    new_bbox = BBox(
        x=int(round(b.x + dx)),
        y=int(round(b.y + dy)),
        w=b.w,
        h=b.h,
    )
    return ThermalDetection(
        bbox=new_bbox,
        area_px=d.area_px,
        contrast=d.contrast,
        classification=d.classification,
        synthetic=d.synthetic,
    )


def _roi_still_warm(gray: np.ndarray,
                    bbox: BBox,
                    min_contrast: float) -> bool:
    """True if the ROI is hotter than the surrounding pad by at least
    ``min_contrast`` mean-intensity counts.

    This is the OF bridge's "is there still a blob here?" check. The
    surround is a padded ring around the bbox; pure-texture drift shows
    roi_mean ~ surround_mean (contrast near 0) and gets rejected, while
    a real warm blob clears this easily.
    """
    if gray is None or gray.size == 0:
        return False
    H, W = gray.shape[:2]
    # Inner ROI
    x0 = max(0, bbox.x)
    y0 = max(0, bbox.y)
    x1 = min(W, bbox.x + bbox.w)
    y1 = min(H, bbox.y + bbox.h)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return False
    roi = gray[y0:y1, x0:x1]
    # Surround: bbox padded by half its size on each side, minus the ROI.
    pad_x = max(2, bbox.w // 2)
    pad_y = max(2, bbox.h // 2)
    sx0 = max(0, bbox.x - pad_x)
    sy0 = max(0, bbox.y - pad_y)
    sx1 = min(W, bbox.x + bbox.w + pad_x)
    sy1 = min(H, bbox.y + bbox.h + pad_y)
    if sx1 - sx0 < 2 or sy1 - sy0 < 2:
        return False
    surround = gray[sy0:sy1, sx0:sx1]
    roi_mean = float(roi.mean())
    # Subtract ROI contribution from the surround mean. Cheap
    # approximation: weighted mean removal.
    sur_total = float(surround.sum())
    sur_px = int(surround.size)
    roi_total = float(roi.sum())
    roi_px = int(roi.size)
    sur_only_px = max(1, sur_px - roi_px)
    sur_only_mean = (sur_total - roi_total) / sur_only_px
    return (roi_mean - sur_only_mean) >= float(min_contrast)


def _sample_features(gray: np.ndarray,
                     bbox: BBox,
                     max_features: int) -> Optional[np.ndarray]:
    """goodFeaturesToTrack constrained to the bbox interior.

    Returns a (N,1,2) float32 array suitable for calcOpticalFlowPyrLK,
    or None if no features found / cv2 unavailable / ROI empty.
    """
    if not _HAS_CV2 or gray is None:
        return None
    H, W = gray.shape[:2]
    x0 = max(0, bbox.x)
    y0 = max(0, bbox.y)
    x1 = min(W, bbox.x + bbox.w)
    y1 = min(H, bbox.y + bbox.h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    roi = gray[y0:y1, x0:x1]
    try:
        pts = cv2.goodFeaturesToTrack(
            roi,
            maxCorners=int(max_features),
            qualityLevel=0.01,
            minDistance=3,
            blockSize=5,
        )
    except cv2.error:
        return None
    if pts is None or len(pts) == 0:
        return None
    # Translate back to full-frame coordinates.
    pts = pts.astype(np.float32)
    pts[:, 0, 0] += float(x0)
    pts[:, 0, 1] += float(y0)
    return pts

"""
Temporal tracker for thermal detections.

Pipeline (per frame):

    1. Kalman predict — every existing track extrapolates (x, y) forward
       one tick from its own velocity. Matching is done against the
       PREDICTED centroid, not the last observation. This is what makes
       fast gimbal slews survivable: even if the blob jumps 80 px in
       one frame, the Kalman predicts ~80 px because last tick's
       velocity already said so, and the match still succeeds.

    2. Detection ↔ track matching by centroid distance, greedy best-first.
       Distance is predicted-centroid ↔ detection-centroid, so the
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


# ─────────────────────────────── Kalman ───────────────────────────────
#
# 4-state constant-velocity model per track:
#
#     state  x = [px, py, vx, vy]ᵀ
#     F = [[1 0 dt 0],
#          [0 1 0 dt],
#          [0 0 1  0],
#          [0 0 0  1]]
#     H = [[1 0 0 0],
#          [0 1 0 0]]
#
# dt = 1 tick (we run at a fixed ~20 Hz, no need to carry wall time).
# Process noise Q is tuned so the filter reacts quickly to acceleration
# (targets + gimbal jerks) without thrashing on detector jitter. Q/R
# ratio picked empirically — override via TrackerConfig if needed.


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


# ─────────────────────────────── data types ───────────────────────────

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
    # Optical-flow bridge state: last frame's feature points (Nx1x2 float32)
    # sampled inside the bbox, used by cv2.calcOpticalFlowPyrLK to locate
    # the track when the detector drops a frame. Refreshed on every
    # real-detection match.
    of_pts: Optional[np.ndarray] = field(default=None, repr=False)


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
    # 6 ticks ≈ 300 ms of Kalman coast — enough to bridge a single
    # detector hiccup without producing "ghost" boxes that visibly drift
    # across the screen on longer dropouts.
    max_misses: int = 6
    ema: float = 0.4             # bbox smoothing: 0 = raw, 1 = frozen

    # Kalman noise knobs (in px² for pos / (px/tick)² for vel / px² for meas).
    # Higher q → filter trusts motion model less, reacts faster to changes.
    # Higher r → filter trusts measurements less, smooths harder.
    kf_q_pos: float = 4.0
    kf_q_vel: float = 9.0
    kf_r_meas: float = 4.0

    # Optical-flow bridge. DEFAULT OFF: in early testing it latches onto
    # background texture on noisy thermal scenes, produces "consistent"
    # false shifts, and (because we zero `misses` on OF success) the
    # ghost tracks never expire — they just drift across the frame as
    # purple coasting boxes forever. Will be re-enabled once the bridge
    # is gated on detector quality / confidence.
    of_enabled: bool = False
    of_r_scale: float = 25.0     # how much to inflate R for OF measurements
    of_max_features: int = 20    # max corners to track per bbox
    of_min_features: int = 3     # min surviving corners to trust OF
    of_win_size: int = 21        # LK window (larger = handles more motion)
    of_max_level: int = 3        # LK pyramid levels
    # Maximum allowed displacement between the median feature shift and
    # any individual feature, in px, before we consider the OF result
    # unreliable (scene change, occlusion) and discard it.
    of_max_inlier_spread: float = 15.0


# ─────────────────────────────── tracker ─────────────────────────────

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
            out.append(HeatTrackSnapshot(
                id=trk.id,
                bbox=trk.det.bbox,
                hits=trk.hits,
                misses=trk.misses,
                age=trk.age,
                confirmed=(trk.hits >= self.cfg.min_hits),
                coasting=(trk.misses > 0),
            ))
        return out

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

        # 1. Kalman predict everyone forward one tick. Shift the stored
        #    bbox by the predicted delta so downstream logic (OF, snapshot)
        #    sees a reasonable current position even if the detection
        #    doesn't arrive.
        predicted: List[Tuple[float, float]] = []
        for trk in self._tracks:
            prev_cx, prev_cy = _centroid(trk.det.bbox)
            px, py = trk.kf.predict()
            predicted.append((px, py))
            # Shift bbox to predicted position. EMA later replaces this
            # with a real measurement if one arrives.
            dx = px - prev_cx
            dy = py - prev_cy
            trk.det = _shift_bbox(trk.det, dx, dy)

        # 2. Match detections to predicted positions, greedy best-first.
        candidates: list[tuple[float, int, int]] = []
        for ti, (px, py) in enumerate(predicted):
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
        if (self.cfg.of_enabled and _HAS_CV2
                and self._prev_gray is not None and agc8 is not None
                and self._prev_gray.shape == agc8.shape):
            for ti, trk in enumerate(self._tracks):
                if ti in matched_t:
                    continue
                if trk.hits < self.cfg.min_hits:
                    continue
                shift = self._of_estimate_shift(trk, agc8)
                if shift is None:
                    continue
                dx, dy = shift
                # Apply as a Kalman measurement with inflated noise.
                # Use the track's pre-predict bbox position + shift as
                # the measurement (we already predicted above, so add
                # shift to the OLD centroid to form the observation).
                old_cx, old_cy = _centroid(_shift_bbox(trk.det,
                                                       -(trk.kf.vel[0]),
                                                       -(trk.kf.vel[1])))
                # Simpler: the previous frame's feature-points centroid
                # is stored implicitly — the "before OF" position is
                # the Kalman state *before* this tick's predict. We
                # approximate by using the current predict position
                # minus one-tick velocity.
                # Compute observation directly from feature-point
                # median in the new frame (done inside _of_estimate_shift).
                meas_x = old_cx + dx
                meas_y = old_cy + dy
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
                trk.misses = 0
                trk.age += 1

        # 4. Age / drop unmatched tracks.
        kept: List[_Track] = []
        for ti, trk in enumerate(self._tracks):
            if ti in matched_t:
                kept.append(trk)
                continue
            trk.misses += 1
            trk.age += 1
            if trk.misses <= self.cfg.max_misses:
                kept.append(trk)

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
            kept.append(_Track(det=d, kf=kf, id=self._next_id, of_pts=of_pts))
            self._next_id += 1

        self._tracks = kept

        # Save this frame for the next tick's optical flow. Copy so
        # external mutation of the source buffer doesn't corrupt us.
        if agc8 is not None and _HAS_CV2:
            self._prev_gray = agc8.copy()

        # 6. Emit confirmed tracks matched this frame (real OR OF bridge).
        out: List[ThermalDetection] = []
        for trk in self._tracks:
            if trk.hits >= self.cfg.min_hits and trk.misses == 0:
                out.append(trk.det)
        return out

    # ── internals ──────────────────────────────────────────────────────

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
            # Points scattered → probably tracking different things.
            # Bail rather than drag the track somewhere random.
            return None
        return float(med[0]), float(med[1])


# ─────────────────────────────── helpers ──────────────────────────────

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
    )


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

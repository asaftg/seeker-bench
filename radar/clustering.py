"""DBSCAN clustering + Kalman tracker for the AWR2944P point cloud.

The AWR2944P mmw_demoDDM firmware emits the raw point cloud but does
NOT ship TI's Group Tracker, so this module is the PRIMARY path for
producing stable bounding boxes — not a fallback.

Pipeline per frame:
    1. DBSCAN over (x, y, z, doppler) to group physically-close
       points that are also moving similarly.
    2. Compute each cluster's centroid and half-extents.
    3. Associate each cluster to an existing tracklet (nearest-
       neighbour, gate widens while the tracklet is coasting).
    4. Constant-velocity Kalman update on matched tracklets;
       unmatched tracklets coast (predict, don't update) until
       they time out.

Two stabilisation layers sit on top of DBSCAN:

  * M-of-N confirmation — a new tracklet must accumulate
    ``confirm_min_hits`` hits in the last ``confirm_window`` frames
    before it's published. Kills single-frame false positives that
    would otherwise flash as a bbox on the display.

  * Coast-on-miss — a confirmed tracklet with no association this
    frame is predicted forward using its Kalman-estimated velocity
    and still published, flagged ``coasting=True`` so the GUI can
    render it dashed / dim. Dropped only after ``coast_max_frames``
    consecutive misses — this is the dead-reckoning the operator
    sees as a steady bbox through brief signal dropouts.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

from common.frames import RadarDetection, RadarTarget


@dataclass
class ClusterParams:
    """DBSCAN + tracker knobs.

    DBSCAN:
        ``eps_pos_m`` / ``eps_dop_mps`` together define DBSCAN's
        distance metric — a point is a neighbour if its Euclidean-
        over-position distance is within eps_pos_m AND its doppler
        difference is within eps_dop_mps.
        ``min_samples`` = minimum points to seed a cluster.
        ``min_size_m`` / ``max_size_m`` clamp reported bbox half-
        extents so tiny clusters don't flicker and giant ones don't
        dominate.

    Tracker:
        ``assoc_gate_m`` = nearest-neighbour gate at dt=0 (fresh hit).
        Gate grows at ``gate_growth_m_per_s`` per second of coast
        to reflect the widening uncertainty ball around a coasting
        track.

        ``coast_max_frames`` = drop a tracklet after this many
        consecutive missed associations. At 10 Hz, 10 frames ≈ 1 s
        of dead-reckoning — long enough to bridge packet dropouts,
        short enough that a target that really left doesn't linger
        as a ghost.

        ``confirm_min_hits`` / ``confirm_window`` — new tracklets
        must score ``min_hits`` hits in the trailing ``window``
        frames before their RadarTarget is published. 2-of-3 is the
        usual sweet spot.

        ``q_accel_mps2`` — process-noise RMS acceleration for the
        constant-velocity model. Higher = trusts measurement more,
        quicker response but more jitter. Lower = smoother but lags
        maneuvers.

        ``r_pos_m`` — measurement-noise std on position. DBSCAN
        centroids jitter ~10-20 cm frame-to-frame on a rock-still
        target, so ~0.3-0.5 m is realistic.
    """
    # DBSCAN
    eps_pos_m: float = 8.0
    eps_dop_mps: float = 3.0
    min_samples: int = 2
    min_size_m: float = 0.25
    max_size_m: float = 3.0
    # Tracker association
    assoc_gate_m: float = 5.0
    gate_growth_m_per_s: float = 2.0
    # Absorb-orphan radius: if a DBSCAN cluster finds no free track but
    # sits within this distance of an already-matched track's predicted
    # centroid, drop it instead of spawning a new ID. Kills the "ghost"
    # secondary IDs that appear when a single long / bright object
    # (vehicle, wall front) fragments into two DBSCAN clusters — the
    # first cluster claims the real track, the orphan is its split
    # sibling, not a separate object.
    merge_overlap_m: float = 5.0
    # Track-level merge: after per-frame association, merge
    # confirmed tracks whose centroids are within this gate.
    # Catches split-siblings that already established separate
    # tracks (DBSCAN splits that self-perpetuate because each
    # cluster matches its own track every frame).
    track_merge_base_m: float = 5.0
    track_merge_per_meter: float = 0.04  # +4cm/m of range
    # Tracker persistence
    coast_max_frames: int = 30          # ~2.3 s at 13 Hz — bridges long dropouts
    # Proportional coast: a track must accumulate this many hits to
    # earn the full coast budget.  Immature tracks (hits < this)
    # get a proportionally shorter budget:
    #   effective = max(3, coast_max_frames * hits / coast_maturity_hits)
    # This kills false-confirmation ghosts: a clutter cluster that
    # scored 2/3 hits coasts ~5 frames instead of 30.
    coast_maturity_hits: int = 10
    # Moving-target coast floor: if the track's 2D ground speed
    # exceeds this threshold, it gets at least this many frames of
    # coast even with few hits. Distinguishes walking humans
    # (~1 m/s) from static clutter (~0 m/s) which both have low
    # hit counts early in life.
    coast_moving_speed_mps: float = 0.5
    coast_moving_floor_frames: int = 15  # ~1.2s at 13 Hz
    confirm_min_hits: int = 2
    confirm_window: int = 3
    # Velocity half-life during coast (seconds). The Kalman's velocity
    # stays latched to whatever it was at last measurement, so a target
    # that actually slowed down would have its predicted position fly
    # past the real location, pushing the re-acquisition cluster out of
    # the association gate → new ID spawns. Damping the predicted
    # velocity toward zero during coast (exp decay) keeps the predicted
    # position near where the target actually is when it reappears.
    coast_vel_halflife_s: float = 1.0
    # Kalman
    q_accel_mps2: float = 6.0           # higher = tracks pivots faster (was 3.0, bumped for pan-drift)
    r_pos_m: float = 0.4
    # Track graveyard — reaped tracks are stashed for this many seconds
    # before being fully forgotten. When a would-be new track's centroid
    # lands within ``resurrect_radius_m`` of a graveyard entry, the old
    # ID is resurrected instead of minting a fresh one. Handles both
    # "target lost behind occlusion then reappears" and "target pivots
    # sharply out of the gate, KF prediction overshoots, reacquires
    # nearby" — classic sources of ID churn that coast budget alone
    # can't fix.
    graveyard_ttl_s: float = 4.0
    resurrect_radius_m: float = 12.0
    # 2026-05-12 fixes for "targets fading + new IDs born":
    #
    # bury_unconfirmed: also save UNCONFIRMED-but-coasting tracks
    #   to graveyard with a shorter TTL. The original "only confirmed"
    #   policy was an over-correction against noise — in practice it
    #   means a cluster that fragmented (FedEx truck case) and never
    #   reached confirm_min_hits before re-fragmenting just dies
    #   without ID preservation. The shorter TTL keeps the graveyard
    #   from clogging with true noise.
    bury_unconfirmed: bool = True
    graveyard_unconfirmed_ttl_s: float = 1.5
    # resurrect_radius_per_meter: scale the resurrect radius with the
    # buried track's last slant range. At long range a small angular
    # error projects to a large physical distance, so a tight 12m
    # gate misses re-acquires. 0.05 -> +5cm per meter of range -> at
    # 200m the radius becomes 12 + 10 = 22m. Set 0 to keep flat
    # `resurrect_radius_m` only.
    resurrect_radius_per_meter: float = 0.05

    # 2026-05-12 — post-DBSCAN merge pass for "multiple bboxes per
    # physical target" (FedEx truck case). DBSCAN can split one big
    # target into 2-4 clusters when surfaces have different micro-
    # doppler (truck body translates at v, wheels spin at v ± wheel_v).
    # Each fragment becomes its own track in the GUI. This pass merges
    # clusters whose centroids are close AND whose mean doppler agrees,
    # after DBSCAN runs.
    #
    # Merge gate = base + per_meter * range_a. Default 4m base +
    # 0.04 m/m at 50m -> 6m gate -> handles a typical big truck.
    # max_doppler_diff_mps protects against merging traffic moving
    # in opposite directions.
    post_dbscan_merge_enabled: bool = True
    cluster_merge_dist_m: float = 4.0
    cluster_merge_dist_per_meter: float = 0.04
    cluster_merge_max_doppler_diff_mps: float = 3.0


class _Tracklet:
    """Per-target Kalman state carried across frames.

    6D CV state: [px, py, pz, vx, vy, vz]. Measurement is position
    only; radial-doppler is used as an init hint and mixed into the
    velocity EMA for the published ``vel_*`` fields, but not as a
    Kalman observation (would need a non-linear H at each step).
    """

    __slots__ = (
        "tid", "x", "P", "size_half", "hits", "misses",
        "hit_history", "confirmed", "last_hit_t", "peak_snr",
    )

    def __init__(
        self,
        tid: int,
        centroid: np.ndarray,
        vel: np.ndarray,
        size_half: np.ndarray,
        now_t: float,
        r_pos_m: float,
    ) -> None:
        self.tid = tid
        # State [px,py,pz,vx,vy,vz]
        self.x = np.zeros(6, dtype=np.float64)
        self.x[0:3] = centroid
        self.x[3:6] = vel
        # Covariance: position uncertainty ~ R, velocity large (unknown).
        self.P = np.eye(6, dtype=np.float64)
        self.P[0:3, 0:3] *= (r_pos_m * r_pos_m)
        self.P[3:6, 3:6] *= 25.0  # (5 m/s)^2 — we really don't know yet
        self.size_half = size_half.astype(np.float64)
        self.hits = 1
        self.misses = 0
        self.hit_history: List[bool] = [True]
        self.confirmed = False
        self.peak_snr = 0.0
        self.last_hit_t = now_t

    def predict(
        self,
        dt: float,
        q_accel_mps2: float,
        coast_vel_halflife_s: float = 0.0,
    ) -> None:
        """Advance state by dt using CV model; inflate P by process noise.

        If this tracklet is coasting (``misses > 0``) and a positive
        ``coast_vel_halflife_s`` is provided, the velocity component of
        the state is exponentially decayed toward zero before propagation.
        That prevents a coasting predicted position from overshooting when
        the real target has slowed or stopped — the common cause of a
        reacquired target spawning a fresh ID because the KF prediction
        landed outside the association gate.
        """
        if dt <= 0:
            return
        if self.misses > 0 and coast_vel_halflife_s > 0.0:
            # exp decay: v *= 0.5 ** (dt / halflife)
            decay = 0.5 ** (dt / coast_vel_halflife_s)
            self.x[3:6] *= decay
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        self.x = F @ self.x
        # Discrete white-noise acceleration Q.
        # Per-axis: [[dt^4/4, dt^3/2],[dt^3/2, dt^2]] * q_accel^2
        q = float(q_accel_mps2) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        Qblk_pp = 0.25 * dt4 * q
        Qblk_pv = 0.5 * dt3 * q
        Qblk_vv = dt2 * q
        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] += Qblk_pp
            Q[i + 3, i + 3] += Qblk_vv
            Q[i, i + 3] += Qblk_pv
            Q[i + 3, i] += Qblk_pv
        self.P = F @ self.P @ F.T + Q

    def update(self, z_pos: np.ndarray, r_pos_m: float) -> None:
        """Kalman position measurement update."""
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0
        H[1, 1] = 1.0
        H[2, 2] = 1.0
        R = np.eye(3, dtype=np.float64) * (r_pos_m * r_pos_m)
        y = z_pos - H @ self.x             # innovation
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I6 = np.eye(6, dtype=np.float64)
        self.P = (I6 - K @ H) @ self.P

    @property
    def centroid(self) -> np.ndarray:
        return self.x[0:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    def reconfirm_on_hit(self, window: int, min_hits: int) -> None:
        self.hit_history.append(True)
        if len(self.hit_history) > window:
            self.hit_history = self.hit_history[-window:]
        self.hits += 1
        self.misses = 0
        if not self.confirmed and sum(self.hit_history) >= min_hits:
            self.confirmed = True

    def miss_tick(self, window: int) -> None:
        self.hit_history.append(False)
        if len(self.hit_history) > window:
            self.hit_history = self.hit_history[-window:]
        self.misses += 1


@dataclass
class _GraveyardEntry:
    """Buried tracklet awaiting potential resurrection.

    2026-05-12: replaced the legacy 3-tuple (centroid, vel, t) with
    this dataclass so the graveyard can carry confirmation state
    + range (needed for separate TTLs and range-aware resurrect
    radius).
    """
    centroid: np.ndarray
    velocity: np.ndarray
    buried_t: float
    was_confirmed: bool
    hits_at_burial: int
    last_range_m: float


class RadarClusterer:
    """Stateful DBSCAN + Kalman tracker for a point-cloud stream.

    One instance per RadarManager — don't share across sensors.
    """

    def __init__(self, params: Optional[ClusterParams] = None) -> None:
        self.params = params or ClusterParams()
        self._tracks: Dict[int, _Tracklet] = {}
        self._next_tid: int = 0
        self._last_step_t: Optional[float] = None
        # Graveyard: tid → _GraveyardEntry. Entries live for
        # graveyard_ttl_s (confirmed) or graveyard_unconfirmed_ttl_s
        # (unconfirmed) then are forgotten. Used to resurrect IDs
        # when a cluster spawns near a recently-dead track.
        self._graveyard: Dict[int, "_GraveyardEntry"] = {}

    # ──────────────────────── public API ────────────────────────
    def step(
        self, detections: List[RadarDetection],
    ) -> Tuple[List[RadarDetection], List[RadarTarget]]:
        """Cluster the current frame and return (points-with-tid, targets).

        Returned targets include confirmed tracklets that were either
        updated this frame OR are coasting within budget. Unconfirmed
        tracklets are kept internally but NOT emitted — that's the M-of-N
        filter in action.
        """
        now_t = time.time()
        dt = 0.1 if self._last_step_t is None else max(1e-3, now_t - self._last_step_t)
        self._last_step_t = now_t

        # Predict every tracklet forward to "now" before association,
        # so the assoc gate is measured against the predicted position
        # (important when a target moves fast between frames).
        for trk in self._tracks.values():
            trk.predict(
                dt,
                self.params.q_accel_mps2,
                self.params.coast_vel_halflife_s,
            )

        if not detections:
            self._miss_all_and_reap()
            return detections, self._publish_coasting_only()

        # ── DBSCAN ──
        # Decimate pathological frames (garbage injection, dust, jitter)
        # to a sane cap. Real targets emit ~20-50 points; >300 is noise.
        # Keep top-by-SNR so real targets survive.
        if len(detections) > 300:
            dets_sorted = sorted(
                detections,
                key=lambda d: (
                    float(d.snr_db) if d.snr_db is not None
                    and not (d.snr_db != d.snr_db) else -1000.0
                ),
                reverse=True,
            )
            detections = dets_sorted[:300]

        pts = np.array(
            [[d.x_m, d.y_m, d.z_m, d.doppler_mps] for d in detections],
            dtype=np.float32,
        )
        labels = _dbscan(
            pts,
            eps_pos=self.params.eps_pos_m,
            eps_dop=self.params.eps_dop_mps,
            min_samples=self.params.min_samples,
        )

        clusters: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            if lbl < 0:
                continue
            clusters.setdefault(int(lbl), []).append(i)

        # 2026-05-12 post-DBSCAN merge — collapses same-target
        # fragments before they each spawn their own track. See
        # `_merge_adjacent_clusters` for the merge condition.
        if self.params.post_dbscan_merge_enabled and len(clusters) > 1:
            clusters = self._merge_adjacent_clusters(clusters, pts)

        # ── Per-cluster stats in sensor coords ──
        frame_clusters: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
        for cid, idxs in clusters.items():
            xs = pts[idxs, 0]
            ys = pts[idxs, 1]
            zs = pts[idxs, 2]
            ds = pts[idxs, 3]
            centroid = np.array([xs.mean(), ys.mean(), zs.mean()], dtype=np.float64)
            half = np.array([
                1.5 * float(xs.std() if len(xs) > 1 else self.params.min_size_m),
                1.5 * float(ys.std() if len(ys) > 1 else self.params.min_size_m),
                1.5 * float(zs.std() if len(zs) > 1 else self.params.min_size_m),
            ], dtype=np.float64)
            np.clip(half, self.params.min_size_m, self.params.max_size_m, out=half)

            # Radial-doppler → 3D velocity hint along the sensor ray.
            r = max(float(np.linalg.norm(centroid)), 1e-3)
            dir_hat = centroid / r
            vel = dir_hat * float(ds.mean())
            # Max SNR of constituent detections for coast budget.
            _snrs = [float(detections[ii].snr_db) for ii in idxs
                     if detections[ii].snr_db is not None
                     and detections[ii].snr_db == detections[ii].snr_db]  # skip NaN
            max_snr = max(_snrs) if _snrs else 0.0
            frame_clusters.append((cid, centroid, half, vel, len(idxs), max_snr))

        # Associate largest clusters first (greedy).
        frame_clusters.sort(key=lambda t: -t[4])

        matched_tracks: set[int] = set()
        cid_to_tid: Dict[int, int] = {}

        for cid, centroid, half, vel, n_pts, max_snr in frame_clusters:
            tid = self._find_best_track(centroid, exclude=matched_tracks)
            if tid is None:
                # Before creating a new track, check if this cluster is
                # a DBSCAN-split sibling of an already-matched track —
                # same object, two clusters this frame. Dropping it
                # avoids the "ghost second ID" the operator sees parked
                # next to the real one on long/bright targets.
                if self._overlaps_matched(centroid, matched_tracks):
                    continue
                # Try to resurrect a recently-reaped track ID if this
                # cluster lands near one. Happens when a target
                # reappears after a long occlusion, or when a human
                # pivots sharply and the KF prediction overshot the
                # gate — same object, just looked briefly lost to the
                # association step.
                resurrected_tid = self._try_resurrect(centroid)
                if resurrected_tid is not None:
                    tid = resurrected_tid
                else:
                    tid = self._next_tid
                    self._next_tid += 1
                new_trk = _Tracklet(
                    tid=tid,
                    centroid=centroid,
                    vel=vel,
                    size_half=half,
                    now_t=now_t,
                    r_pos_m=self.params.r_pos_m,
                )
                # Resurrected IDs skip M-of-N re-warmup — only confirmed
                # tracks get stashed in the graveyard, so by the time we
                # pull one out we know the operator already saw this ID
                # as a real target. Making them wait 2 frames to be
                # visible again would show a brief "gap" in the trail.
                if resurrected_tid is not None:
                    new_trk.confirmed = True
                new_trk.peak_snr = max_snr
                self._tracks[tid] = new_trk
            else:
                trk = self._tracks[tid]
                trk.update(centroid, self.params.r_pos_m)
                # Size: EMA the measured half-extents so clusters that
                # shrink/grow by a point or two don't pulsate.
                alpha = 0.5
                trk.size_half = alpha * half + (1 - alpha) * trk.size_half
                trk.reconfirm_on_hit(
                    self.params.confirm_window,
                    self.params.confirm_min_hits,
                )
                trk.last_hit_t = now_t
                trk.peak_snr = max(0.3 * trk.peak_snr + 0.7 * max_snr, trk.peak_snr)
                matched_tracks.add(tid)

            cid_to_tid[cid] = tid

        # Merge confirmed tracks that are too close to be separate
        # objects. Catches self-perpetuating DBSCAN split-siblings.
        self._merge_close_tracks(matched_tracks)

        # Age un-matched tracklets.
        for tid, trk in list(self._tracks.items()):
            if tid not in matched_tracks:
                trk.miss_tick(self.params.confirm_window)

        # Reap dead.
        self._reap()

        # Back-fill detection→tid.
        for i, lbl in enumerate(labels):
            if lbl >= 0 and lbl in cid_to_tid:
                detections[i].target_id = cid_to_tid[lbl]
            else:
                detections[i].target_id = 255

        # Publish confirmed tracks (hit this frame OR coasting).
        targets = self._publish_all()
        return detections, targets

    # ──────────────────────── internals ────────────────────────
    def _find_best_track(
        self, centroid: np.ndarray, exclude: set[int]
    ) -> Optional[int]:
        """Greedy gated nearest-neighbour. The gate scales with *both*
        time-since-last-hit AND the track's own speed, so a 20 m/s
        target coasting half a second gets a ~10 m catch radius instead
        of the base-gate 3 m. Without this, re-acquired tracks spawn
        fresh IDs mid-coast — operator sees "same box" jumping numbers.
        """
        base_gate = self.params.assoc_gate_m
        growth = self.params.gate_growth_m_per_s
        best_tid: Optional[int] = None
        best_d2 = float("inf")
        now_t = self._last_step_t or time.time()
        for tid, trk in self._tracks.items():
            if tid in exclude:
                continue
            dt_since = max(0.0, now_t - trk.last_hit_t)
            speed = float(np.linalg.norm(trk.velocity))
            # Gate = base + (const growth + per-track speed) * dt.
            # At rest: just the linear growth term. Fast movers widen
            # their own gate proportionally to how far they could have
            # travelled since last measurement.
            gate = base_gate + (growth + speed) * dt_since
            gate2 = gate * gate
            d = trk.centroid - centroid
            d2 = float(d @ d)
            if d2 < gate2 and d2 < best_d2:
                best_d2 = d2
                best_tid = tid
        return best_tid

    def _merge_close_tracks(self, active_tids: set) -> None:
        """Merge confirmed tracks that are too close to be separate objects.

        After the per-frame association loop, two DBSCAN-split siblings
        can each match their own existing track — the overlap check
        never fires. This pass catches that case: for each pair of
        active (hit-this-frame) confirmed tracks, if their centroids
        are within a range-adaptive gate (base + per_m * range),
        absorb the weaker one. At 200m the gate is ~13m,
        matching the angular resolution spread.
        """
        base_gate = self.params.track_merge_base_m
        per_m = self.params.track_merge_per_meter
        tids = [t for t in active_tids
                if t in self._tracks and self._tracks[t].confirmed]
        absorbed: set = set()
        for i_idx in range(len(tids)):
            a_tid = tids[i_idx]
            if a_tid in absorbed:
                continue
            trk_a = self._tracks.get(a_tid)
            if trk_a is None:
                continue
            for j_idx in range(i_idx + 1, len(tids)):
                b_tid = tids[j_idx]
                if b_tid in absorbed:
                    continue
                trk_b = self._tracks.get(b_tid)
                if trk_b is None:
                    continue
                d = trk_a.centroid - trk_b.centroid
                d2 = float(d @ d)
                avg_range = 0.5 * (float(np.linalg.norm(trk_a.centroid))
                                   + float(np.linalg.norm(trk_b.centroid)))
                gate = base_gate + per_m * avg_range
                if d2 < gate * gate:
                    # Merge: keep the track with more hits
                    if trk_a.hits >= trk_b.hits:
                        winner, loser_tid = trk_a, b_tid
                    else:
                        winner, loser_tid = trk_b, a_tid
                    loser = self._tracks[loser_tid]
                    # Transfer peak SNR
                    winner.peak_snr = max(winner.peak_snr, loser.peak_snr)
                    # Bury the loser for ID resurrection later
                    now_t = self._last_step_t or time.time()
                    self._graveyard[loser_tid] = _GraveyardEntry(
                        centroid=loser.centroid.copy(),
                        velocity=loser.velocity.copy(),
                        buried_t=now_t,
                        was_confirmed=bool(loser.confirmed),
                        hits_at_burial=int(loser.hits),
                        last_range_m=float(np.linalg.norm(loser.centroid)),
                    )
                    del self._tracks[loser_tid]
                    absorbed.add(loser_tid)
                    break  # trk_a state may have changed

    def _overlaps_matched(
        self, centroid: np.ndarray, matched: set[int]
    ) -> bool:
        """True if centroid sits within merge_overlap_m of any already-
        matched track's predicted position. Used to drop DBSCAN-split
        siblings instead of spawning new IDs for them."""
        gate2 = self.params.merge_overlap_m ** 2
        for tid in matched:
            trk = self._tracks.get(tid)
            if trk is None:
                continue
            d = trk.centroid - centroid
            if float(d @ d) < gate2:
                return True
        return False

    def _miss_all_and_reap(self) -> None:
        for trk in self._tracks.values():
            trk.miss_tick(self.params.confirm_window)
        self._reap()

    def _merge_adjacent_clusters(
        self,
        clusters: Dict[int, List[int]],
        pts: np.ndarray,
    ) -> Dict[int, List[int]]:
        """Iterative pairwise merge of DBSCAN clusters that look like
        fragments of the same physical target.

        For each pair (a, b) of remaining clusters, merge b into a if:
          * |centroid_a - centroid_b| < (base + per_m * range_a), OR
          * Bounding boxes overlap on all 3 axes (full 3D bbox overlap)
        AND
          * |mean_doppler_a - mean_doppler_b| < max_doppler_diff_mps
            (protects against merging traffic moving in opposite
             directions through the same angular bin)

        Iterates until no merges happen, with a hard cap of 32 passes
        to bound worst-case compute on pathological scenes. Each merge
        emits a `radar_cluster_merged` event for post-hoc forensics.

        Range-aware gate: closer clusters get a tighter gate than
        farther ones because the same angular error projects to more
        meters at range. Default base=4m, per_m=0.04 -> at 50m gate=6m.

        Returns the merged cluster dict. Cluster IDs preserved: the
        winner (lower id) keeps its id, the loser's points are
        absorbed and the loser id is removed from the dict.
        """
        if len(clusters) < 2:
            return clusters

        def _cluster_stats(idxs):
            cp = pts[np.asarray(idxs)]
            centroid = cp[:, :3].mean(axis=0)
            if len(idxs) > 1:
                half = 1.5 * cp[:, :3].std(axis=0)
            else:
                half = np.array([self.params.min_size_m] * 3,
                                 dtype=np.float64)
            half = np.clip(half, self.params.min_size_m,
                           self.params.max_size_m)
            mean_dop = float(cp[:, 3].mean())
            return centroid, half, mean_dop

        base = float(self.params.cluster_merge_dist_m)
        per_m = float(self.params.cluster_merge_dist_per_meter)
        max_dop = float(self.params.cluster_merge_max_doppler_diff_mps)

        try:
            from common.events import emit as _emit
        except Exception:
            _emit = None  # type: ignore

        for _pass in range(32):
            cids = sorted(clusters.keys())
            stats = {c: _cluster_stats(clusters[c]) for c in cids}
            merged_any = False
            for i, cid_a in enumerate(cids):
                if cid_a not in clusters:
                    continue
                ca, ha, da = stats[cid_a]
                for cid_b in cids[i + 1:]:
                    if cid_b not in clusters:
                        continue
                    cb, hb, db = stats[cid_b]
                    if abs(da - db) > max_dop:
                        continue
                    bbox_overlap = bool(np.all(
                        np.abs(ca - cb) < (ha + hb)))
                    range_a = float(np.linalg.norm(ca))
                    gate = base + per_m * range_a
                    dist = float(np.linalg.norm(ca - cb))
                    if bbox_overlap or dist < gate:
                        clusters[cid_a] = (clusters[cid_a]
                                            + clusters[cid_b])
                        del clusters[cid_b]
                        if _emit is not None:
                            try:
                                _emit("radar_cluster_merged", {
                                    "winner": int(cid_a),
                                    "loser": int(cid_b),
                                    "n_points_after": len(clusters[cid_a]),
                                    "distance_m": round(dist, 2),
                                    "gate_m": round(gate, 2),
                                    "bbox_overlap": bbox_overlap,
                                    "doppler_diff_mps": round(
                                        abs(da - db), 2),
                                    "range_m": round(range_a, 2),
                                })
                            except Exception:
                                pass
                        merged_any = True
                        break  # ca/ha/da stale — restart pass
            if not merged_any:
                break
        return clusters

    def _reap(self) -> None:
        now_t = self._last_step_t or time.time()
        def _effective_coast(trk) -> int:
            """Proportional coast + velocity floor."""
            mh = self.params.coast_maturity_hits
            if trk.hits >= mh:
                return self.params.coast_max_frames
            base = max(3, self.params.coast_max_frames * trk.hits // mh)
            speed = float(np.linalg.norm(trk.velocity[:2]))
            if speed > self.params.coast_moving_speed_mps:
                base = max(base, self.params.coast_moving_floor_frames)
            # High-SNR targets are strong returns (vehicles) — give extra coast
            if trk.peak_snr > 15.0:  # strong return
                base = max(base, 20)
            return base
        dead = [tid for tid, trk in self._tracks.items()
                if trk.misses > _effective_coast(trk)]
        for tid in dead:
            trk = self._tracks[tid]
            # 2026-05-12: bury BOTH confirmed and unconfirmed tracks.
            # Original policy was "confirmed only" but in practice that
            # meant a cluster that fragmented (FedEx truck case) and
            # never reached confirm_min_hits before re-fragmenting just
            # died without ID preservation. Operator-reported as
            # "targets fading + new IDs". Unconfirmed entries get a
            # shorter TTL (graveyard_unconfirmed_ttl_s) so true noise
            # doesn't clog the dictionary.
            should_bury = trk.confirmed or self.params.bury_unconfirmed
            if should_bury:
                last_range_m = float(np.linalg.norm(trk.centroid))
                self._graveyard[tid] = _GraveyardEntry(
                    centroid=trk.centroid.copy(),
                    velocity=trk.velocity.copy(),
                    buried_t=now_t,
                    was_confirmed=bool(trk.confirmed),
                    hits_at_burial=int(trk.hits),
                    last_range_m=last_range_m,
                )
                # Diagnostic — visible in the JSONL events stream so
                # post-hoc forensics can correlate burial -> resurrect
                # cycles with operator-perceived ID churn.
                try:
                    from common.events import emit as _emit
                    _emit("radar_track_buried", {
                        "tid": int(tid),
                        "confirmed": bool(trk.confirmed),
                        "hits": int(trk.hits),
                        "misses": int(trk.misses),
                        "last_range_m": round(last_range_m, 2),
                        "last_x": round(float(trk.centroid[0]), 2),
                        "last_y": round(float(trk.centroid[1]), 2),
                        "last_z": round(float(trk.centroid[2]), 2),
                    })
                except Exception:
                    pass
            del self._tracks[tid]
        # Age out stale graveyard entries — separate TTL for confirmed
        # vs unconfirmed so noise doesn't clog the table.
        ttl_c = self.params.graveyard_ttl_s
        ttl_u = self.params.graveyard_unconfirmed_ttl_s
        stale = []
        for tid, entry in self._graveyard.items():
            ttl = ttl_c if entry.was_confirmed else ttl_u
            if (now_t - entry.buried_t) > ttl:
                stale.append(tid)
        for tid in stale:
            try:
                from common.events import emit as _emit
                _emit("radar_track_graveyard_expired", {
                    "tid": int(tid),
                    "was_confirmed": bool(self._graveyard[tid].was_confirmed),
                    "age_s": round(now_t - self._graveyard[tid].buried_t, 2),
                })
            except Exception:
                pass
            del self._graveyard[tid]

    def _try_resurrect(self, centroid: np.ndarray) -> Optional[int]:
        """Return a graveyard tid whose last position is closest to
        ``centroid`` and within the (range-aware) resurrect radius —
        or None.

        Range-aware radius: each entry's effective gate is
            resurrect_radius_m + resurrect_radius_per_meter * last_range_m
        So a long-range track (where small angular shifts project to
        large physical distances) gets a wider catchment than a
        close-range one. Set resurrect_radius_per_meter=0 in YAML
        to keep the flat radius.

        The entry is removed on success so two fresh clusters can't
        both claim the same dead ID in the same frame.
        """
        if not self._graveyard:
            return None
        base_r = float(self.params.resurrect_radius_m)
        per_m = float(self.params.resurrect_radius_per_meter)
        best_tid: Optional[int] = None
        best_d2 = float("inf")
        best_gate = 0.0
        for tid, entry in self._graveyard.items():
            gate = base_r + per_m * entry.last_range_m
            gate2 = gate * gate
            d = entry.centroid - centroid
            d2 = float(d @ d)
            if d2 < gate2 and d2 < best_d2:
                best_d2 = d2
                best_tid = tid
                best_gate = gate
        if best_tid is not None:
            entry = self._graveyard[best_tid]
            try:
                from common.events import emit as _emit
                _emit("radar_track_resurrected", {
                    "tid": int(best_tid),
                    "was_confirmed": bool(entry.was_confirmed),
                    "distance_m": round(float(best_d2 ** 0.5), 2),
                    "gate_m": round(best_gate, 2),
                    "age_s": round(
                        (self._last_step_t or time.time()) - entry.buried_t,
                        2),
                    "last_range_m": round(entry.last_range_m, 2),
                })
            except Exception:
                pass
            del self._graveyard[best_tid]
        return best_tid

    def _publish_all(self) -> List[RadarTarget]:
        """Return confirmed tracklets — hit this frame or coasting."""
        out: List[RadarTarget] = []
        for trk in self._tracks.values():
            if not trk.confirmed:
                continue
            coasting = trk.misses > 0
            out.append(self._to_target(trk, coasting))
        return out

    def _publish_coasting_only(self) -> List[RadarTarget]:
        """No-detection path: confirmed tracks that are still within
        coast budget get published as coasting."""
        return self._publish_all()

    def _to_target(self, trk: _Tracklet, coasting: bool) -> RadarTarget:
        c = trk.centroid
        v = trk.velocity
        sz = trk.size_half
        conf = min(1.0, trk.hits / 10.0)
        if coasting:
            # Decay confidence while coasting so the GUI can dim it.
            conf *= max(
                0.3,
                1.0 - trk.misses / max(1, self.params.coast_max_frames),
            )
        return RadarTarget(
            tid=int(trk.tid),
            pos_x_m=float(c[0]),
            pos_y_m=float(c[1]),
            pos_z_m=float(c[2]),
            vel_x_mps=float(v[0]),
            vel_y_mps=float(v[1]),
            vel_z_mps=float(v[2]),
            size_x_m=float(sz[0]),
            size_y_m=float(sz[1]),
            size_z_m=float(sz[2]),
            confidence=float(conf),
            source="kalman",
            num_points=int(trk.hits),
            coasting=bool(coasting),
            hits=int(trk.hits),
            misses=int(trk.misses),
            snr_db=float(trk.peak_snr),
        )


# ─────────────────────────────────────────────────────────────────
# DBSCAN — hand-rolled, numpy-only, ~40 LOC. Faster than sklearn for
# the N ≤ few hundred points we get from mmw_demo, and avoids dragging
# sklearn into the requirements.
# ─────────────────────────────────────────────────────────────────

def _dbscan(
    pts: np.ndarray,
    eps_pos: float,
    eps_dop: float,
    min_samples: int,
) -> np.ndarray:
    """Return an int32 label per point. -1 = noise, 0..k-1 = cluster.

    Distance gate: ``(dx² + dy² + dz²) ≤ eps_pos²`` AND
                   ``|doppler_i - doppler_j| ≤ eps_dop``.
    """
    n = pts.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    if n == 0:
        return labels
    visited = np.zeros(n, dtype=bool)

    xyz = pts[:, :3]
    dop = pts[:, 3]

    # cKDTree neighbor lists: O(N log N) instead of O(N^2). For N=500
    # this is ~5 ms on Xavier vs ~50-100 ms for the dense matrix.
    # Critical fix for thermal FPS drop when radar emits many points
    # (garbage injection, busy scene, dust returns).
    tree = cKDTree(xyz)
    spatial_nbrs = tree.query_ball_tree(tree, r=float(eps_pos))

    # Doppler gate is applied as a filter on the spatial neighbor list,
    # not as a full N x N matrix.
    def _dop_filter(i, neighbors):
        d_i = dop[i]
        return [j for j in neighbors if abs(dop[j] - d_i) <= eps_dop]

    next_label = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbour_idxs = _dop_filter(i, spatial_nbrs[i])
        if len(neighbour_idxs) < min_samples:
            continue
        labels[i] = next_label
        queue = list(neighbour_idxs)
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            if labels[j] == -1:
                labels[j] = next_label
            if visited[j]:
                continue
            visited[j] = True
            j_nbrs = _dop_filter(j, spatial_nbrs[j])
            if len(j_nbrs) >= min_samples:
                for k in j_nbrs:
                    if labels[k] == -1:
                        labels[k] = next_label
                        queue.append(int(k))
            elif labels[j] == -1:
                labels[j] = next_label
        next_label += 1

    return labels

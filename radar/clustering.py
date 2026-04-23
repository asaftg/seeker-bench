"""DBSCAN clustering + tracklet association for the AWR2944P point cloud.

The AWR2944P mmw_demoDDM firmware emits the raw point cloud but does
NOT ship TI's Group Tracker, so this module is the PRIMARY path for
producing stable bounding boxes — not a fallback. See Ticket 5a plan.

Pipeline per frame:
    1. DBSCAN over (x, y, z, doppler) to group physically-close
       points that are also moving similarly.
    2. Compute each cluster's centroid and half-extents.
    3. Greedy nearest-neighbour association to last frame's clusters
       (gated by position distance) to carry forward a persistent
       target ID — so the GUI can colour box 4 the same red every
       frame even though DBSCAN has no notion of memory.

Targets produced here are the class-less ``"radar_detection"``
geometry agreed in the Ticket 5a design — semantic labelling (person
vs. vehicle) happens in fusion against EO + thermal, never here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from common.frames import RadarDetection, RadarTarget


@dataclass
class ClusterParams:
    """DBSCAN + tracklet knobs.

    ``eps_pos_m`` / ``eps_dop_mps`` together define DBSCAN's distance
    metric — a point is a neighbour if its Euclidean-over-position
    distance is within eps_pos_m AND its doppler difference is within
    eps_dop_mps. Keeping these independent matters: two pedestrians
    standing close together have similar position but different
    velocity (when one is moving), and we want them as separate
    boxes. Pure-position DBSCAN would merge them.

    ``min_samples`` = minimum points to seed a cluster. Below this the
    point is noise (target_id = 255).

    ``assoc_gate_m`` = nearest-neighbour gate (metres) for cross-frame
    association. Set slightly larger than the largest plausible
    per-frame motion at our radar FPS (20 Hz → ~1 m at 20 m/s).

    ``min_size_m`` / ``max_size_m`` clamp the reported bbox half-
    extents. Tiny clusters would flicker as single pixels; giant ones
    usually mean two objects got merged and we don't want to show a
    confusingly-huge box.
    """
    eps_pos_m: float = 0.6
    eps_dop_mps: float = 1.5
    min_samples: int = 3
    assoc_gate_m: float = 1.5
    min_size_m: float = 0.25
    max_size_m: float = 3.0


@dataclass
class _Tracklet:
    """Per-target state carried across frames for ID persistence."""
    tid: int
    last_centroid: np.ndarray   # shape (3,), metres
    last_velocity: np.ndarray   # shape (3,), m/s
    misses: int                 # frames since last matched
    hits: int                   # frames matched so far


class RadarClusterer:
    """Stateful DBSCAN + tracklet tracker for a point-cloud stream.

    Stateful because we carry tracklet IDs across frames. One instance
    per RadarManager — don't share across sensors.
    """

    def __init__(self, params: Optional[ClusterParams] = None) -> None:
        self.params = params or ClusterParams()
        self._tracks: Dict[int, _Tracklet] = {}
        self._next_tid: int = 0
        self._max_misses: int = 5  # ≈250 ms at 20 Hz — reap dead tracklets

    # ──────────────────────── public API ────────────────────────
    def step(
        self, detections: List[RadarDetection],
    ) -> Tuple[List[RadarDetection], List[RadarTarget]]:
        """Cluster the current frame and return (points-with-tid, targets).

        The returned detections are the same objects passed in, with
        their ``target_id`` field mutated to reflect cluster membership
        (255 = noise / unassigned). No copy — the caller may treat it
        as a returned list for readability.
        """
        if not detections:
            # No points this frame — age existing tracklets so they
            # time out properly during a brief signal dropout.
            self._age_and_reap()
            return detections, []

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

        # Build per-cluster stats in positional coordinates.
        clusters: Dict[int, List[int]] = {}
        for i, lbl in enumerate(labels):
            if lbl < 0:
                continue
            clusters.setdefault(int(lbl), []).append(i)

        # Frame-local cluster → (centroid, size, mean_doppler, vel).
        frame_clusters: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray, int]] = []
        for cid, idxs in clusters.items():
            xs = pts[idxs, 0]
            ys = pts[idxs, 1]
            zs = pts[idxs, 2]
            ds = pts[idxs, 3]
            centroid = np.array([xs.mean(), ys.mean(), zs.mean()], dtype=np.float32)
            # Half-extents = 1.5 × std, clamped. std alone under-sells
            # small clusters; 3×std over-sells when the cloud is
            # spiky. 1.5 is the visually-honest compromise.
            half = np.array([
                1.5 * float(xs.std() if len(xs) > 1 else self.params.min_size_m),
                1.5 * float(ys.std() if len(ys) > 1 else self.params.min_size_m),
                1.5 * float(zs.std() if len(zs) > 1 else self.params.min_size_m),
            ], dtype=np.float32)
            np.clip(half, self.params.min_size_m, self.params.max_size_m, out=half)

            # Velocity: doppler is radial (along the ray from sensor).
            # We don't have the full 3D vel here; project the mean
            # doppler back onto the (unit-centroid) ray. It's an
            # approximation but it's what the gimbal / fuser actually
            # wants — "which way is this target heading relative to
            # our boresight".
            r = max(float(np.linalg.norm(centroid)), 1e-3)
            dir_hat = centroid / r
            vel = dir_hat * float(ds.mean())
            frame_clusters.append((cid, centroid, half, vel, len(idxs)))

        # ── associate to existing tracklets (nearest neighbour, gated) ──
        targets: List[RadarTarget] = []
        matched_tracks: set[int] = set()

        # Sort by largest cluster first so big persistent objects win
        # when two competes for the same tracklet.
        frame_clusters.sort(key=lambda t: -t[4])

        # Remember which point index → target tid so we can back-fill
        # detections in a second pass.
        cid_to_tid: Dict[int, int] = {}

        for cid, centroid, half, vel, n_pts in frame_clusters:
            tid = self._find_best_track(centroid, exclude=matched_tracks)
            if tid is None:
                tid = self._next_tid
                self._next_tid += 1
                self._tracks[tid] = _Tracklet(
                    tid=tid,
                    last_centroid=centroid,
                    last_velocity=vel,
                    misses=0,
                    hits=1,
                )
            else:
                trk = self._tracks[tid]
                # EMA smoothing — DBSCAN centroids jitter by 10-20 cm
                # frame to frame even on a rock-still target. Low EMA
                # alpha keeps boxes from quivering in the GUI.
                alpha = 0.5
                trk.last_centroid = alpha * centroid + (1 - alpha) * trk.last_centroid
                trk.last_velocity = alpha * vel + (1 - alpha) * trk.last_velocity
                trk.misses = 0
                trk.hits += 1
                centroid = trk.last_centroid
                vel = trk.last_velocity
                matched_tracks.add(tid)

            cid_to_tid[cid] = tid

            targets.append(RadarTarget(
                tid=int(tid),
                pos_x_m=float(centroid[0]),
                pos_y_m=float(centroid[1]),
                pos_z_m=float(centroid[2]),
                vel_x_mps=float(vel[0]),
                vel_y_mps=float(vel[1]),
                vel_z_mps=float(vel[2]),
                size_x_m=float(half[0]),
                size_y_m=float(half[1]),
                size_z_m=float(half[2]),
                confidence=min(1.0, n_pts / 10.0),  # more points → higher conf, capped
                source="dbscan",
                num_points=int(n_pts),
            ))

        # Age un-matched tracklets; reap ones past timeout.
        self._age_and_reap(matched=matched_tracks)

        # Back-fill each detection's target_id from its DBSCAN label
        # through the cluster → tid map (noise stays at 255).
        for i, lbl in enumerate(labels):
            if lbl >= 0 and lbl in cid_to_tid:
                detections[i].target_id = cid_to_tid[lbl]
            else:
                detections[i].target_id = 255

        return detections, targets

    # ──────────────────────── internals ────────────────────────
    def _find_best_track(
        self, centroid: np.ndarray, exclude: set[int]
    ) -> Optional[int]:
        best_tid: Optional[int] = None
        best_d2 = self.params.assoc_gate_m * self.params.assoc_gate_m
        for tid, trk in self._tracks.items():
            if tid in exclude:
                continue
            d = trk.last_centroid - centroid
            d2 = float(d @ d)
            if d2 < best_d2:
                best_d2 = d2
                best_tid = tid
        return best_tid

    def _age_and_reap(self, matched: Optional[set[int]] = None) -> None:
        matched = matched or set()
        dead: List[int] = []
        for tid, trk in self._tracks.items():
            if tid not in matched:
                trk.misses += 1
                if trk.misses > self._max_misses:
                    dead.append(tid)
        for tid in dead:
            del self._tracks[tid]


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
    Doppler acts as a hard gate rather than a distance component so
    two objects at the same position but different velocities split
    into separate clusters.
    """
    n = pts.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    visited = np.zeros(n, dtype=bool)
    eps_pos2 = float(eps_pos) * float(eps_pos)

    xyz = pts[:, :3]
    dop = pts[:, 3]

    # Precompute pairwise position distances (vectorised, N×N). Safe
    # for N up to ~1k — our mmw_demoDDM emits at most ~300 points.
    diff = xyz[:, None, :] - xyz[None, :, :]
    d2 = (diff * diff).sum(axis=-1)
    dop_ok = np.abs(dop[:, None] - dop[None, :]) <= eps_dop
    nbrs = (d2 <= eps_pos2) & dop_ok

    next_label = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbour_idxs = np.flatnonzero(nbrs[i])
        if len(neighbour_idxs) < min_samples:
            # labels[i] stays -1 — may get re-assigned as a border
            # point later if another core point expands into it.
            continue

        # Start a new cluster.
        labels[i] = next_label
        # Seed queue with i's neighbours.
        queue = list(neighbour_idxs)
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            if labels[j] == -1:
                labels[j] = next_label  # border point → current cluster
            if visited[j]:
                continue
            visited[j] = True
            j_nbrs = np.flatnonzero(nbrs[j])
            if len(j_nbrs) >= min_samples:
                # j is also a core point — extend the frontier.
                # Only add neighbours we haven't already queued/visited.
                queue.extend(int(k) for k in j_nbrs if labels[k] == -1)
                # Assign immediately so we don't queue duplicates.
                for k in j_nbrs:
                    if labels[k] == -1:
                        labels[k] = next_label
            elif labels[j] == -1:
                labels[j] = next_label

        next_label += 1

    return labels

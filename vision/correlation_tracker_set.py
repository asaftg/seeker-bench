"""Per-ID MOSSE tracker pool keyed by detector track ID.

Pattern: detector (YOLO + ByteTrack) runs at low Hz (5-10 Hz) and
produces detections with persistent track IDs. Between detector
ticks, this pool runs MOSSE on each ID at frame rate (30+ Hz),
keeping the published bbox alive even when the detector misses
(motion blur during slew, brief occlusion, low-confidence dip).

Lifecycle per track ID:
    new detection from ByteTrack with id=K, bbox=B
        -> if K not in pool: spawn MosseTracker(frame, B)
        -> if K in pool:     reseed(frame, B)
                              # snaps back to YOLO ground truth, kills drift
    NO detection this frame for id=K (typical between detector ticks)
        -> pool.update(K, frame)
                              # MOSSE finds the target by image correlation,
                              # updates bbox, returns PSR
        -> if PSR < lost threshold for `lost_frames` consecutive frames,
           prune the tracker (caller's downstream tracker takes over the
           ID's persistence — fusion's max_misses, ByteTrack's lost age)

The pool DOES NOT decide the target ID. ByteTrack assigns IDs; we
just key off them. Pool DOES NOT touch the detection's class or
confidence — we only edit the bbox. ID continuity comes from
ByteTrack; bbox liveness comes from us.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vision.mosse_tracker import MosseTracker, MosseUpdate, PSR_LOST_DEFAULT


@dataclass
class _TrackEntry:
    tracker: MosseTracker
    last_psr: float = 0.0
    lost_streak: int = 0
    last_seed_frame_id: int = -1   # frame id of last reseed from detector


@dataclass
class CorrelationTrackerSetConfig:
    """Knobs the pool reads. Mirrors YAML eo.correlation_tracker."""
    enabled: bool = True
    psr_lost: float = PSR_LOST_DEFAULT
    # Number of consecutive frames PSR can be below psr_lost before we
    # prune the tracker. 5 frames at 30 Hz EO = 167 ms — long enough
    # to bridge typical motion-blur dropouts, short enough that a
    # truly-lost target doesn't keep emitting stale bboxes.
    lost_frames: int = 5
    # MOSSE learning rate (online filter blending). Lower = more
    # stable on appearance change, slower to adapt. Bolme's default
    # is 0.125; reduce to 0.05 for slow-appearance targets like a
    # car under steady lighting; raise to 0.20 for fast-rotating
    # drones where the patch flips.
    learning_rate: float = 0.125
    # Gaussian peak sigma in the target response. Bigger = more
    # tolerance to small misalignments. 2.0 is the textbook default.
    sigma: float = 2.0
    # Cap the FFT working size per axis. Pure-numpy FFT cost scales
    # ~O(W*H*log(W*H)). A close-range vehicle in EO can fill a
    # 300×200 bbox; without a cap each tracker eats 30+ ms per
    # update and the publisher loop falls below 10 Hz. We resize
    # the patch to fit inside this dim before the FFT and scale the
    # peak position back to frame coords. 96 keeps EO at frame
    # rate even with multiple close targets; raise to 128 if a
    # particular scene has identity ambiguity from too-coarse
    # patches.
    max_patch_dim: int = 96


@dataclass
class _DetectorHit:
    """One ByteTrack detection passed in by the caller."""
    track_id: int
    bbox_xywh: Tuple[int, int, int, int]


@dataclass
class TrackerPoolUpdate:
    """Pool's per-frame output: which IDs are alive, their current
    bbox + PSR, and which were pruned this tick."""
    bboxes: Dict[int, Tuple[int, int, int, int]] = field(default_factory=dict)
    psr: Dict[int, float] = field(default_factory=dict)
    pruned: List[int] = field(default_factory=list)


class CorrelationTrackerSet:
    """Pool of MOSSE trackers, keyed by an external (ByteTrack) ID.

    Two entry points:

        on_detector_tick(frame, hits, frame_id)
            Called when YOLO+ByteTrack produced fresh detections.
            Reseeds existing trackers from the new bboxes, spawns
            trackers for new IDs, and prunes IDs the detector has
            stopped emitting for too long.

        on_frame(frame, frame_id)
            Called on every frame between detector ticks. Updates
            every tracker's bbox via MOSSE and prunes losers. Caller
            consumes the returned dict to publish bboxes to fusion.

    The class is intentionally NOT a manager-thread; it's a pure
    data structure operated by EOManager / ThermalManager on their
    process thread, so we don't introduce another thread crossing.
    """

    def __init__(self, cfg: Optional[CorrelationTrackerSetConfig] = None) -> None:
        self._cfg = cfg or CorrelationTrackerSetConfig()
        self._tracks: Dict[int, _TrackEntry] = {}

    # ── public ────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._cfg.enabled

    def reset(self) -> None:
        """Drop all trackers — used when the source pipeline restarts
        (e.g., camera reconnect, manager shutdown)."""
        self._tracks.clear()

    def on_detector_tick(self,
                          frame: np.ndarray,
                          hits: List[_DetectorHit],
                          frame_id: int) -> TrackerPoolUpdate:
        """Detector produced fresh detections. Reseed existing trackers
        for matching IDs, spawn new trackers for new IDs, and prune
        IDs the detector has stopped emitting for `lost_frames` frames.
        """
        out = TrackerPoolUpdate()
        if not self._cfg.enabled:
            return out
        seen_ids = set()
        for h in hits:
            seen_ids.add(int(h.track_id))
            if h.track_id in self._tracks:
                # Existing tracker: reseed on the YOLO ground truth.
                # Kills any MOSSE drift accumulated between detector ticks.
                try:
                    self._tracks[h.track_id].tracker.reseed(frame, h.bbox_xywh)
                    self._tracks[h.track_id].last_seed_frame_id = frame_id
                    self._tracks[h.track_id].lost_streak = 0
                except ValueError:
                    # Bbox went off-frame at reseed; drop the tracker.
                    out.pruned.append(h.track_id)
                    self._tracks.pop(h.track_id, None)
                    continue
            else:
                # New ID: spawn a tracker.
                try:
                    t = MosseTracker(
                        frame, h.bbox_xywh,
                        learning_rate=self._cfg.learning_rate,
                        sigma=self._cfg.sigma,
                        psr_lost=self._cfg.psr_lost,
                        max_patch_dim=self._cfg.max_patch_dim,
                    )
                    self._tracks[h.track_id] = _TrackEntry(
                        tracker=t,
                        last_psr=0.0,
                        lost_streak=0,
                        last_seed_frame_id=frame_id,
                    )
                except ValueError:
                    # Seed bbox out of frame — skip this ID.
                    continue
        # Output current bboxes (post-reseed) for the caller.
        for tid, entry in self._tracks.items():
            out.bboxes[tid] = entry.tracker.bbox_xywh
            out.psr[tid] = entry.last_psr
        return out

    def on_frame(self, frame: np.ndarray, frame_id: int) -> TrackerPoolUpdate:
        """Run MOSSE on every active tracker and prune IDs that have
        been below PSR threshold for too many consecutive frames.
        """
        out = TrackerPoolUpdate()
        if not self._cfg.enabled:
            return out
        to_prune: List[int] = []
        for tid, entry in self._tracks.items():
            upd: MosseUpdate = entry.tracker.update(frame)
            entry.last_psr = float(upd.psr)
            if not upd.locked:
                entry.lost_streak += 1
                if entry.lost_streak >= self._cfg.lost_frames:
                    to_prune.append(tid)
                    continue
            else:
                entry.lost_streak = 0
            out.bboxes[tid] = upd.bbox_xywh
            out.psr[tid] = float(upd.psr)
        for tid in to_prune:
            self._tracks.pop(tid, None)
            out.pruned.append(tid)
        return out

    def has(self, track_id: int) -> bool:
        return int(track_id) in self._tracks

    def __len__(self) -> int:
        return len(self._tracks)


# Re-export for callers that prefer importing from this module
DetectorHit = _DetectorHit

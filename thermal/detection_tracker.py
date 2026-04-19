"""
Temporal tracker for thermal detections.

The raw heat detector produces per-frame blob lists. On noisy indoor
scenes those blobs flicker: a warm spot appears on one frame, is gone
the next, reappears one pixel over, etc. Feeding that straight to the
GUI produces visually distracting dancing boxes.

This tracker sits between `HeatDetector.detect()` and `BUS.publish()`.
It solves three problems at once:

    1. **Flicker suppression.** A track has to appear in `min_hits`
       consecutive frames before it is confirmed and emitted. One-off
       noise spikes never get a box drawn.
    2. **Persistence across drops.** A confirmed track survives up to
       `max_misses` frames without a match, so one-frame detector
       dropouts don't make the box blink.
    3. **Position smoothing.** Matched bboxes are blended with the
       previous position via EMA so edges stop quivering.

Matching is by **centroid distance** (not IoU). Thermal blobs change
shape unpredictably between frames — a pixel-level IoU threshold is
too brittle. Centroid distance is stable: if the blob's center stays
within `max_dist_px` of a previous track's center, it's the same
target regardless of how the edges wobble.

Classification stickiness: when a track is matched to a new detection,
the new detection inherits the PREVIOUS track's classification if the
new one is None. This matters because the classifier runs every Nth
frame — we don't want the label to disappear on the N-1 in-between.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from common.frames import BBox, ThermalDetection


def _centroid(b: BBox) -> tuple[float, float]:
    return (b.x + b.w * 0.5, b.y + b.h * 0.5)


def _cdist(a: BBox, b: BBox) -> float:
    """Euclidean distance between bbox centers."""
    ax, ay = _centroid(a)
    bx, by = _centroid(b)
    return math.hypot(ax - bx, ay - by)


@dataclass
class _Track:
    det: ThermalDetection
    hits: int = 1
    misses: int = 0
    age: int = 1


@dataclass
class TrackerConfig:
    enabled: bool = True
    max_dist_px: float = 40.0    # max centroid distance to match
    min_hits: int = 5            # frames before a track is emitted
    max_misses: int = 5          # frames a confirmed track survives without a hit
    ema: float = 0.5             # bbox smoothing: 0 = raw, 1 = frozen


class DetectionTracker:
    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.cfg = config or TrackerConfig()
        self._tracks: List[_Track] = []

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, detections: List[ThermalDetection]) -> List[ThermalDetection]:
        """Run one tick of the tracker. Returns CONFIRMED detections only."""
        if not self.cfg.enabled:
            return detections

        # 1. Greedy centroid-distance matching: build all pairs that
        #    are within max_dist_px, sort by distance ascending (best
        #    matches first), and assign greedily.
        candidates: list[tuple[float, int, int]] = []
        for ti, trk in enumerate(self._tracks):
            for di, d in enumerate(detections):
                dist = _cdist(trk.det.bbox, d.bbox)
                if dist <= self.cfg.max_dist_px:
                    candidates.append((dist, ti, di))
        candidates.sort()  # ascending distance = best first

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for _, ti, di in candidates:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            self._merge(self._tracks[ti], detections[di])

        # 2. Age unmatched tracks; drop those past max_misses.
        kept: List[_Track] = []
        for ti, trk in enumerate(self._tracks):
            if ti in matched_t:
                kept.append(trk)
                continue
            trk.misses += 1
            trk.age += 1
            if trk.misses <= self.cfg.max_misses:
                kept.append(trk)

        # 3. Birth new tracks for unmatched detections.
        for di, d in enumerate(detections):
            if di not in matched_d:
                kept.append(_Track(det=d))

        self._tracks = kept

        # 4. Emit only confirmed tracks that got a hit this frame.
        #    Coasting tracks (misses > 0) stay in memory for future
        #    matching but are NOT rendered — otherwise a stuck box
        #    lingers after a real target leaves the frame.
        out: List[ThermalDetection] = []
        for trk in self._tracks:
            if trk.hits >= self.cfg.min_hits and trk.misses == 0:
                out.append(trk.det)
        return out

    def _merge(self, trk: _Track, d: ThermalDetection) -> None:
        """Update an existing track with a matched new detection."""
        a = self.cfg.ema
        b = trk.det.bbox
        nb = d.bbox
        smoothed = BBox(
            x=int(round(a * b.x + (1 - a) * nb.x)),
            y=int(round(a * b.y + (1 - a) * nb.y)),
            w=int(round(a * b.w + (1 - a) * nb.w)),
            h=int(round(a * b.h + (1 - a) * nb.h)),
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

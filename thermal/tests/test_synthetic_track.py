"""
Unit tests for the synthetic "Draw Target" user-seeded track path.

Synthetic tracks are born confirmed from a user-drawn bbox, skip the
heat-detection matching stage entirely, and propagate purely via the
optical-flow bridge with the warmth check disabled (they're allowed to
be cold objects like parked cars). These tests seed a synthetic track,
feed a grayscale sequence with a moving contrast blob inside the
seeded bbox, and assert the track follows via OF.
"""
from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from common.frames import BBox, TargetClass
from thermal.detection_tracker import DetectionTracker, TrackerConfig


# ---- helpers -----------------------------------------------------------

def _gray_with_blob(h: int, w: int, cx: int, cy: int,
                    blob_r: int = 6, intensity: int = 220, bg: int = 30,
                    noise: int = 2) -> np.ndarray:
    """Reproducible grayscale image with a single contrast blob.

    Intensity-only; the synthetic path doesn't care about "warmth" per
    se — it just needs a textured region for LK to lock on.
    """
    rng = np.random.default_rng(cx * 13 + cy)
    img = (rng.normal(loc=bg, scale=noise, size=(h, w))
              .clip(0, 255).astype(np.uint8))
    cv2.circle(img, (cx, cy), blob_r, int(intensity), -1)
    return img


def _centroid(bbox: BBox) -> tuple[float, float]:
    return (bbox.x + bbox.w * 0.5, bbox.y + bbox.h * 0.5)


# ---- tests -------------------------------------------------------------

def test_seed_synthetic_emits_confirmed_immediately():
    """A freshly-seeded synthetic track is born confirmed and shows up in
    the tracker output on its very first update — the user expects to
    see the USER TARGET box the instant they release the mouse."""
    trk = DetectionTracker(TrackerConfig(min_hits=5, of_enabled=True))
    H, W = 120, 200
    # Prime the previous-frame gray buffer so OF has a reference.
    gray0 = _gray_with_blob(H, W, 60, 60)
    trk.update([], gray0)

    tid = trk.seed_synthetic(BBox(x=40, y=40, w=40, h=40), agc8=gray0)
    assert tid is not None

    # No heat detections — but synthetic track still emits.
    out = trk.update([], _gray_with_blob(H, W, 60, 60))
    assert len(out) == 1
    assert out[0].synthetic is True
    assert out[0].classification is not None
    assert out[0].classification.classifier_used == "user"
    assert out[0].classification.target_class == TargetClass.UNKNOWN

    snap = trk.snapshot()
    assert any(s.synthetic and s.confirmed for s in snap)


def test_synthetic_track_follows_moving_blob_via_of():
    """Seed a bbox around a blob, then slide the blob across the frame
    one pixel per tick. The synthetic track must follow via the OF
    bridge even though NO heat detections are ever provided."""
    trk = DetectionTracker(TrackerConfig(
        min_hits=5, of_enabled=True, max_dist_px=60.0,
    ))
    H, W = 120, 320
    cx, cy = 60, 60

    # Prime prev-frame buffer and seed.
    trk.update([], _gray_with_blob(H, W, cx, cy))
    tid = trk.seed_synthetic(
        BBox(x=cx - 15, y=cy - 15, w=30, h=30),
        agc8=_gray_with_blob(H, W, cx, cy),
    )
    assert tid is not None

    # Move the blob and feed only grayscale frames (no detections).
    for step in range(1, 25):
        cx += 2
        out = trk.update([], _gray_with_blob(H, W, cx, cy))
        assert len(out) == 1, f"synthetic track should always emit, step={step}"
        assert out[0].synthetic is True

    # The synthetic track's bbox centroid should have followed the blob
    # within a reasonable tolerance. We allow Kalman lag (~a few px) but
    # require it to be well past the seed position and near the current
    # blob position.
    out_cx, out_cy = _centroid(out[0].bbox)
    assert out_cx > 75, (
        f"synthetic track failed to follow blob via OF: cx={out_cx}, blob={cx}"
    )
    assert abs(out_cx - cx) < 15, (
        f"synthetic track drifted too far from blob: cx={out_cx}, blob={cx}"
    )
    assert abs(out_cy - cy) < 10


def test_clear_synthetic_removes_track_and_replaces():
    """clear_synthetic() drops the seeded track; re-seeding while one
    exists replaces it (single user target at a time)."""
    trk = DetectionTracker(TrackerConfig(min_hits=5, of_enabled=True))
    H, W = 100, 160
    trk.update([], _gray_with_blob(H, W, 50, 50))

    tid1 = trk.seed_synthetic(BBox(x=30, y=30, w=40, h=40),
                              agc8=_gray_with_blob(H, W, 50, 50))
    assert tid1 is not None
    assert trk.has_synthetic()

    # Re-seed: old one replaced, new id issued.
    tid2 = trk.seed_synthetic(BBox(x=80, y=40, w=30, h=30),
                              agc8=_gray_with_blob(H, W, 95, 55))
    assert tid2 is not None and tid2 != tid1
    snaps = trk.snapshot()
    synthetic_snaps = [s for s in snaps if s.synthetic]
    assert len(synthetic_snaps) == 1
    assert synthetic_snaps[0].id == tid2

    # Clear: no synthetic tracks left, tracker keeps running.
    removed = trk.clear_synthetic()
    assert removed == 1
    assert not trk.has_synthetic()
    out = trk.update([], _gray_with_blob(H, W, 95, 55))
    assert out == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

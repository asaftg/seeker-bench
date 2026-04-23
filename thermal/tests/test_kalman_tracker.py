"""
Unit tests for the Kalman + optical-flow temporal tracker.

Covers:
  - Constant-velocity blob: track stays alive across many frames with
    no false ID churn.
  - Single-frame detection dropout: track survives and re-matches.
  - Two blobs crossing: IDs should NOT swap. See note below — plain
    constant-velocity centroid matching has a known weakness at
    near-zero-separation crossings; we assert the weaker but still
    useful property that both IDs persist without being dropped, and
    we document the swap weakness for future Hungarian / appearance
    work.
  - OF bridge on noise: enabling the OF bridge on a pure-noise scene
    must not produce lingering ghost tracks.
"""
from __future__ import annotations

import numpy as np
import pytest

from common.frames import BBox, ThermalDetection
from thermal.detection_tracker import DetectionTracker, TrackerConfig


# ---- helpers -----------------------------------------------------------

def _det(cx: float, cy: float, w: int = 10, h: int = 10) -> ThermalDetection:
    return ThermalDetection(
        bbox=BBox(x=int(round(cx - w / 2)), y=int(round(cy - h / 2)), w=w, h=h),
        area_px=w * h,
        contrast=100.0,
    )


def _gray_with_blob(h: int, w: int, cx: int, cy: int,
                    blob_r: int = 5, warm: int = 220, bg: int = 30,
                    noise: int = 3) -> np.ndarray:
    import cv2
    rng = np.random.default_rng(cx * 7 + cy)  # reproducible per-position noise
    img = (rng.normal(loc=bg, scale=noise, size=(h, w))
              .clip(0, 255).astype(np.uint8))
    cv2.circle(img, (cx, cy), blob_r, int(warm), -1)
    return img


def _pure_noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(loc=60, scale=15, size=(h, w))
              .clip(0, 255).astype(np.uint8))


# ---- tests -------------------------------------------------------------

def test_constant_velocity_blob_stays_locked():
    """Blob moves 10 px/frame for 20 frames. Track should confirm and
    keep the same ID the whole time."""
    cfg = TrackerConfig(min_hits=3, max_misses=4, of_enabled=False)
    trk = DetectionTracker(cfg)

    H, W = 120, 320
    ids_seen: set[int] = set()
    confirmed_frames = 0

    x = 20.0
    for _ in range(20):
        gray = _gray_with_blob(H, W, int(x), 60)
        out = trk.update([_det(x, 60)], gray)
        snap = trk.snapshot()
        assert len(snap) == 1, "should never spawn a second track on a clean scene"
        ids_seen.add(snap[0].id)
        if out:
            confirmed_frames += 1
        x += 10.0

    assert len(ids_seen) == 1, f"ID churned: saw {ids_seen}"
    # After min_hits=3, we should be emitting on basically every frame.
    assert confirmed_frames >= 15


def test_single_frame_dropout_survives():
    """Track should still be alive and re-match when the detection
    returns after a single-frame drop."""
    cfg = TrackerConfig(min_hits=3, max_misses=4, of_enabled=False)
    trk = DetectionTracker(cfg)

    H, W = 120, 320
    # 5 frames of steady detections — confirm the track.
    for i in range(5):
        x = 50 + i * 5
        trk.update([_det(x, 60)], _gray_with_blob(H, W, x, 60))

    snap = trk.snapshot()
    assert len(snap) == 1 and snap[0].confirmed
    surviving_id = snap[0].id

    # Frame with NO detections.
    x = 50 + 5 * 5
    out_drop = trk.update([], _gray_with_blob(H, W, x, 60))
    snap = trk.snapshot()
    assert len(snap) == 1, "track must coast, not die, on one-frame dropout"
    assert snap[0].id == surviving_id
    assert snap[0].coasting

    # Detection returns — should re-match, not spawn new.
    x = 50 + 6 * 5
    trk.update([_det(x, 60)], _gray_with_blob(H, W, x, 60))
    snap = trk.snapshot()
    assert len(snap) == 1
    assert snap[0].id == surviving_id
    assert not snap[0].coasting


def test_two_blobs_crossing_paths_both_survive():
    """Two blobs moving toward each other, crossing, then continuing.

    Known limitation: a plain constant-velocity Kalman with greedy
    centroid matching CAN swap IDs at the crossing point when the two
    predicted centroids momentarily fall within max_dist_px of each
    other's measurement. We therefore assert the weaker — and still
    operationally useful — property that BOTH tracks continue to
    exist across the crossing without being dropped. A future pass
    (Hungarian matching + appearance / size features) would eliminate
    the swap; filed as future work.
    """
    cfg = TrackerConfig(min_hits=3, max_misses=6, max_dist_px=25.0,
                        of_enabled=False)
    trk = DetectionTracker(cfg)

    H, W = 120, 320
    import cv2

    a_x = 40.0
    b_x = 280.0
    y = 60
    for _ in range(25):
        img = _pure_noise(H, W, seed=0)
        cv2.circle(img, (int(a_x), y), 5, 220, -1)
        cv2.circle(img, (int(b_x), y), 5, 220, -1)
        trk.update([_det(a_x, y), _det(b_x, y)], img)
        a_x += 10.0
        b_x -= 10.0

    snap = trk.snapshot()
    # Two tracks must still exist (neither was dropped through the crossing).
    confirmed = [s for s in snap if s.confirmed]
    assert len(confirmed) >= 2, (
        f"lost a track through the crossing: {snap}")


def test_of_bridge_disabled_no_ghost_on_noise():
    """Sanity baseline: with OF disabled, on a pure-noise scene we should
    never spawn a confirmed track (there are no detections)."""
    cfg = TrackerConfig(min_hits=3, max_misses=4, of_enabled=False)
    trk = DetectionTracker(cfg)
    H, W = 96, 160
    for i in range(30):
        out = trk.update([], _pure_noise(H, W, seed=i))
        assert out == []
    assert trk.snapshot() == []


def test_of_bridge_enabled_does_not_spawn_ghosts_on_noise():
    """The failure mode from the field: enable OF, feed pure noise (no
    detections, no warm blob), ensure no phantom confirmed track
    materializes or drifts forever.

    This exercises all three new guards (no detections -> no track to
    bridge; warmth check would reject even if there were one; streak
    cap would kill any that slipped through).
    """
    cfg = TrackerConfig(min_hits=3, max_misses=4, of_enabled=True,
                        max_of_bridges=3)
    trk = DetectionTracker(cfg)
    H, W = 96, 160
    for i in range(40):
        out = trk.update([], _pure_noise(H, W, seed=i))
        assert out == []
    assert trk.snapshot() == [], (
        "OF bridge must not materialize ghost tracks from nothing")


def test_of_bridge_kills_track_after_streak_cap():
    """Seed a real track, then stop feeding detections AND switch to
    pure-noise imagery. The post-shift warmth check should reject the
    bridge every frame (or the streak cap should fire), and the track
    must die within max_of_bridges + max_misses frames — never drift
    forever."""
    cfg = TrackerConfig(min_hits=3, max_misses=4, of_enabled=True,
                        max_of_bridges=3)
    trk = DetectionTracker(cfg)
    H, W = 96, 200

    # Confirm a track at (60, 48).
    for _ in range(5):
        trk.update([_det(60, 48)], _gray_with_blob(H, W, 60, 48))
    assert any(s.confirmed for s in trk.snapshot())

    # Now remove the blob entirely — pure noise, no detections.
    for i in range(20):
        trk.update([], _pure_noise(H, W, seed=100 + i))

    assert trk.snapshot() == [], "track must die — no ghost drift"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

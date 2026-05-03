"""Pool-level behaviour tests for CorrelationTrackerSet."""
from __future__ import annotations

import numpy as np
import pytest

from vision.correlation_tracker_set import (
    CorrelationTrackerSet,
    CorrelationTrackerSetConfig,
    DetectorHit,
)


_BG_CACHE: dict = {}
_TARGET_PATCH_CACHE: dict = {}


def _bg_noise(h: int, w: int) -> np.ndarray:
    key = (h, w)
    if key not in _BG_CACHE:
        rng = np.random.default_rng(42)
        _BG_CACHE[key] = (
            rng.normal(80, 5, (h, w)).clip(0, 255).astype(np.uint8))
    return _BG_CACHE[key].copy()


def _target_patch(size: int, seed: int = 7) -> np.ndarray:
    key = (size, seed)
    if key not in _TARGET_PATCH_CACHE:
        rng = np.random.default_rng(seed)
        _TARGET_PATCH_CACHE[key] = (
            rng.normal(180, 40, (size, size)).clip(0, 255).astype(np.uint8))
    return _TARGET_PATCH_CACHE[key].copy()


def _make_frame(*targets,
                frame_h: int = 240,
                frame_w: int = 320,
                size: int = 30) -> np.ndarray:
    """Deterministic BG + fixed-pattern textured target per (cx, cy).
    Each target gets a different noise seed so they have distinct
    appearances (so MOSSE doesn't lock the wrong target)."""
    f = _bg_noise(frame_h, frame_w)
    half = size // 2
    for i, (cx, cy) in enumerate(targets):
        patch = _target_patch(size, seed=7 + i)
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half
        f[y0:y1, x0:x1] = patch
    return f


def test_pool_starts_empty_and_disabled_no_op():
    p = CorrelationTrackerSet(
        CorrelationTrackerSetConfig(enabled=False))
    f = _make_frame((100, 100))
    out = p.on_detector_tick(f, [DetectorHit(1, (85, 85, 30, 30))], 0)
    assert len(p) == 0
    assert out.bboxes == {}


def test_pool_spawns_on_detector_hit():
    """Detector hit creates a tracker entry keyed by the detector ID
    and returns its initial bbox in the update."""
    p = CorrelationTrackerSet()
    f = _make_frame((100, 120))
    out = p.on_detector_tick(
        f, [DetectorHit(1, (85, 105, 30, 30))], 0)
    assert 1 in out.bboxes
    assert len(p) == 1
    # Initial bbox matches the detector hit (modulo even-rounding
    # snap MOSSE applies internally for FFT-friendly sizes).
    bx, by, bw, bh = out.bboxes[1]
    assert abs(bx - 85) <= 1
    assert abs(by - 105) <= 1
    assert bw == 30 and bh == 30


def test_pool_reseed_corrects_drift_on_new_detector_hit():
    """Reseeding on a fresh detection snaps the tracker back to the
    detector's bbox, regardless of drift."""
    p = CorrelationTrackerSet()
    f0 = _make_frame((100, 120))
    p.on_detector_tick(f0, [DetectorHit(1, (85, 105, 30, 30))], 0)

    # Skip ahead several frames worth of dead reckoning (no fresh
    # frames between detector hits — pool has nothing to update).
    # Then detector lands a fresh detection at a new position; reseed
    # should make pool snap to it.
    f_seed = _make_frame((150, 120))
    out = p.on_detector_tick(
        f_seed, [DetectorHit(1, (135, 105, 30, 30))], 4)
    bx, by, bw, bh = out.bboxes[1]
    assert abs((bx + bw // 2) - 150) <= 3


def test_pool_prunes_after_lost_streak():
    """Track gets a hit, then we force the lost-streak counter past the
    threshold and confirm the pool removes the entry on the next tick.
    Tests the pruning state machine directly without depending on the
    PSR-vs-real-noise correlation behavior (which is data-dependent)."""
    cfg = CorrelationTrackerSetConfig(lost_frames=3, psr_lost=999.0)
    # psr_lost=999 forces every update to report not-locked
    p = CorrelationTrackerSet(cfg)
    f0 = _make_frame((100, 120))
    p.on_detector_tick(f0, [DetectorHit(1, (85, 105, 30, 30))], 0)
    assert p.has(1)

    out_pruned: list = []
    for i in range(1, 6):
        out = p.on_frame(f0, i)
        out_pruned.extend(out.pruned)
        if 1 in out.pruned:
            break
    assert 1 in out_pruned
    assert not p.has(1)


def test_pool_two_simultaneous_targets():
    """Two distinct detector IDs spawn two trackers, each keyed by ID."""
    p = CorrelationTrackerSet()
    f0 = _make_frame((80, 100), (220, 140))
    p.on_detector_tick(f0, [
        DetectorHit(1, (65, 85, 30, 30)),
        DetectorHit(2, (205, 125, 30, 30)),
    ], 0)
    assert len(p) == 2
    assert p.has(1)
    assert p.has(2)
    assert not p.has(3)


def test_pool_reset_drops_everything():
    p = CorrelationTrackerSet()
    f = _make_frame((100, 120), (200, 80))
    p.on_detector_tick(f, [
        DetectorHit(1, (85, 105, 30, 30)),
        DetectorHit(2, (185, 65, 30, 30)),
    ], 0)
    assert len(p) == 2
    p.reset()
    assert len(p) == 0

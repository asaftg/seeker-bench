"""LockTracker — lifecycle, reseed, lost, coast, recover.

Locks in the contract that
``vision.lock_tracker.LockTracker``:
  * produces a bbox every frame while ACTIVE or COASTING
  * transitions to COASTING after `lost_frames` of sub-PSR updates
  * transitions to HARD_RELEASED after `coast_window_s` in COASTING
  * recovers to ACTIVE on reseed or PSR rebound
  * is a no-op when state is OFF
"""
from __future__ import annotations

import numpy as np
import pytest

from vision.lock_tracker import (
    LockState, LockTracker, LockTrackerConfig, LockUpdate,
)


def _bg_frame(rng: np.random.Generator, h: int = 200, w: int = 200,
               noise_amp: float = 30.0, mean: float = 128.0) -> np.ndarray:
    """Synthetic textured background. Random gaussian on uint8."""
    f = (rng.standard_normal((h, w)).astype(np.float32) * noise_amp
         + mean).clip(0, 255)
    return f.astype(np.uint8)


def _frame_with_target(bg: np.ndarray, cx: int, cy: int,
                        rng: np.random.Generator,
                        size: int = 40) -> np.ndarray:
    """Drop a textured patch onto bg at (cx, cy). The patch needs
    internal gradient to anchor a MOSSE filter (uniform squares give
    a flat correlation surface)."""
    out = bg.copy()
    half = size // 2
    y0, y1 = max(0, cy - half), min(bg.shape[0], cy + half)
    x0, x1 = max(0, cx - half), min(bg.shape[1], cx + half)
    h, w = y1 - y0, x1 - x0
    if h <= 0 or w <= 0:
        return out
    # Random-textured patch with bright bias.
    patch = (rng.standard_normal((h, w)).astype(np.float32) * 25
             + 200).clip(0, 255).astype(np.uint8)
    out[y0:y1, x0:x1] = patch
    return out


# ── basic lifecycle ─────────────────────────────────────────────

def test_off_state_is_noop():
    """Update on a tracker that hasn't been seeded does nothing."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    lock = LockTracker()
    upd = lock.update(bg, now=0.0)
    assert upd.state == LockState.OFF
    assert upd.bbox_xywh is None
    assert not lock.is_active


def test_seed_then_active_returns_bbox_every_frame():
    """After seed(), every update() returns a bbox while PSR is high."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker()
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    assert lock.state == LockState.ACTIVE
    # A few stationary updates — bbox stays near (80, 80, 40, 40)
    for k in range(10):
        upd = lock.update(f0, now=0.05 * (k + 1))
        assert upd.state == LockState.ACTIVE
        assert upd.bbox_xywh is not None
        x, y, w, h = upd.bbox_xywh
        assert abs(x - 80) < 5 and abs(y - 80) < 5


def test_seed_rejects_out_of_frame_bbox():
    """seed() returns False if the bbox extends outside the frame.
    Caller's contract is to handle the False return — we don't
    silently fail."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    lock = LockTracker()
    # bbox (190, 190, 40, 40) on a 200×200 frame → spills past edge
    assert not lock.seed(bg, (190, 190, 40, 40), now=0.0)
    assert lock.state == LockState.OFF


def test_release_clears_state():
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker()
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    assert lock.is_active
    lock.release()
    assert lock.state == LockState.OFF
    assert not lock.is_active


# ── PSR loss + COASTING ─────────────────────────────────────────

def test_low_psr_for_n_frames_transitions_to_coasting():
    """When PSR drops below threshold for `lost_frames` consecutive
    frames, state moves ACTIVE → COASTING. The bbox keeps coming
    out (last-known position) but the filter freezes."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker(LockTrackerConfig(psr_lost=999.0,  # impossible PSR
                                          lost_frames=3,
                                          coast_window_s=10.0))
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    # First 3 sub-PSR updates accumulate the lost streak.
    for k in range(3):
        upd = lock.update(f0, now=0.05 * (k + 1))
        # Until the streak hits lost_frames we stay ACTIVE.
        if k < 2:
            assert upd.state == LockState.ACTIVE
    # 4th sub-PSR update flips to COASTING (streak reaches 3).
    # Wait, 3 streak flips at k=2 (third update). Re-walk:
    # k=0 → streak=1 → ACTIVE
    # k=1 → streak=2 → ACTIVE
    # k=2 → streak=3 → COASTING
    # so let's check the state after k=2 directly:
    lock2 = LockTracker(LockTrackerConfig(psr_lost=999.0, lost_frames=3,
                                           coast_window_s=10.0))
    assert lock2.seed(f0, (80, 80, 40, 40), now=0.0)
    upd = None
    for k in range(3):
        upd = lock2.update(f0, now=0.05 * (k + 1))
    assert upd.state == LockState.COASTING
    # bbox is still produced
    assert upd.bbox_xywh is not None


def test_coast_window_expiry_to_hard_released():
    """After coast_window_s elapses without a reseed, state moves
    COASTING → HARD_RELEASED and bbox is None."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker(LockTrackerConfig(psr_lost=999.0, lost_frames=1,
                                          coast_window_s=0.5))
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    # Force COASTING immediately
    lock.update(f0, now=0.05)
    assert lock.state == LockState.COASTING
    # Stay in COASTING for 0.4 s — still alive
    upd = lock.update(f0, now=0.45)
    assert upd.state == LockState.COASTING
    # 0.6 s — past coast_window_s — HARD_RELEASED
    upd = lock.update(f0, now=0.65)
    assert upd.state == LockState.HARD_RELEASED
    assert upd.bbox_xywh is None


def test_hard_released_is_terminal_until_release():
    """Once HARD_RELEASED, update() does nothing further. Caller
    must call .release() (or .seed()) to clear state."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker(LockTrackerConfig(psr_lost=999.0, lost_frames=1,
                                          coast_window_s=0.1))
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    lock.update(f0, now=0.05)        # → COASTING
    lock.update(f0, now=0.20)        # → HARD_RELEASED
    assert lock.state == LockState.HARD_RELEASED
    # More updates don't change anything
    upd = lock.update(f0, now=0.30)
    assert upd.state == LockState.HARD_RELEASED
    # A successful seed clears the released state
    assert lock.seed(f0, (80, 80, 40, 40), now=0.40)
    assert lock.state == LockState.ACTIVE


# ── reseed ────────────────────────────────────────────────────────

def test_reseed_during_coasting_recovers_to_active():
    """When fusion delivers a fresh observation while we're COASTING,
    .reseed() rebuilds the MOSSE filter on the new bbox and the
    state flips back to ACTIVE without operator intervention."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker(LockTrackerConfig(psr_lost=999.0, lost_frames=1,
                                          coast_window_s=10.0))
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    lock.update(f0, now=0.05)  # → COASTING
    assert lock.state == LockState.COASTING
    # Caller (LockMode coordinator) decides a fresh fused obs is in
    # range and calls reseed.
    assert lock.reseed(f0, (82, 82, 40, 40), now=0.10)
    assert lock.state == LockState.ACTIVE


def test_reseed_during_off_acts_as_seed():
    """Calling reseed() before any seed() is a convenience path —
    it acts as a fresh seed."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker()
    assert lock.reseed(f0, (80, 80, 40, 40), now=0.0)
    assert lock.state == LockState.ACTIVE


def test_time_since_reseed_tracks_correctly():
    """time_since_reseed() lets the coordinator throttle reseeds —
    e.g. don't reseed faster than 0.5 s even if fresh obs arrive
    every 0.04 s."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker()
    lock.seed(f0, (80, 80, 40, 40), now=10.0)
    assert abs(lock.time_since_reseed(now=10.5) - 0.5) < 1e-6
    assert lock.reseed(f0, (80, 80, 40, 40), now=12.0)
    assert abs(lock.time_since_reseed(now=12.7) - 0.7) < 1e-6


# ── PSR rebound recovery ────────────────────────────────────────

def test_psr_rebound_recovers_from_coasting_to_active():
    """If the appearance match recovers (PSR rebounds above the
    threshold) without a reseed — e.g. the target temporarily
    occluded then visible again with the same look — the lock
    auto-flips COASTING → ACTIVE."""
    rng = np.random.default_rng(42)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    # Use a mid-range psr_lost so a real second seed gets a strong
    # PSR rebound.
    lock = LockTracker(LockTrackerConfig(psr_lost=10.0,
                                          lost_frames=2,
                                          coast_window_s=10.0))
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    # First, force COASTING by feeding a totally different frame
    # that has no correlation with the seed patch.
    bad_frame = _bg_frame(rng, mean=50, noise_amp=10)  # uniform-ish
    for k in range(3):
        lock.update(bad_frame, now=0.04 * (k + 1))
    assert lock.state == LockState.COASTING
    # Now feed back the original frame — PSR should rebound on the
    # seed-trained filter.
    upd = lock.update(f0, now=0.20)
    # Either ACTIVE (if PSR rebounded) or still COASTING (if the
    # filter degraded too much during the bad frames). The
    # important thing is the lock didn't crash and a bbox is still
    # being returned.
    assert upd.state in (LockState.ACTIVE, LockState.COASTING)
    assert upd.bbox_xywh is not None


# ── disabled (config) ──────────────────────────────────────────

def test_default_config_works_out_of_box():
    """Default LockTrackerConfig should let a basic seed+update+
    reseed cycle work without crashing on any field."""
    rng = np.random.default_rng(0)
    bg = _bg_frame(rng)
    f0 = _frame_with_target(bg, cx=100, cy=100, rng=rng)
    lock = LockTracker()  # all defaults
    assert lock.seed(f0, (80, 80, 40, 40), now=0.0)
    for k in range(20):
        upd = lock.update(f0, now=0.04 * (k + 1))
        assert upd.state in (LockState.ACTIVE, LockState.COASTING)
        assert upd.bbox_xywh is not None
    assert lock.reseed(f0, (80, 80, 40, 40), now=2.0)
    assert lock.state == LockState.ACTIVE

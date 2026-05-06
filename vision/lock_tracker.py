"""Per-sensor "lock mode" persistent tracker.

Operator-engaged TRACK is supposed to be rock-solid. Today the
visible bbox is classifier-driven: every frame's box comes from
YOLO + ByteTrack on EO, or heat-detector + HV YOLO on thermal,
joined via fusion. When any of those drops the target — motion blur
on a slew, contrast loss into shadow, occlusion behind foliage,
class flicker — the bbox vanishes even though the operator is still
locked on the same physical target. Recordings show 400-600 ms
gaps at engagement and 17-second mid-track gaps where the operator
never released the lock.

LockTracker decouples the visible bbox from the classifier chain.
It wraps ``vision.mosse_tracker.MosseTracker`` with a small state
machine that keeps producing a bbox every frame while the engagement
is active, regardless of YOLO/heat/fusion state. Periodic re-seeding
from fresh fused observations corrects MOSSE drift; PSR-based loss
detection plus a coast window distinguishes "classifier transient"
from "target genuinely lost."

Lifecycle::

    OFF
     │  caller calls .seed(frame, bbox) on operator engagement
     ▼
    ACTIVE  ─────────── PSR drops <psr_lost ─────────────► COASTING
     │                                                       │
     │   fresh fused obs near locked pos (caller calls       │
     │   .reseed) — appearance template refreshed,           │
     │   stays ACTIVE                                        │
     │                                                       │
     │                                                       │
     ▼                              caller calls .reseed()   │
   .release()                       within coast_window_s ◄──┘
   (operator dropped lock)                  │
     │                                       ▼
     ▼                                    ACTIVE again
    OFF                                        │
                                  coast_window_s passes
                                  with no reseed
                                            │
                                            ▼
                                       HARD_RELEASED
                                            │
                                            ▼
                                           OFF

The class is sensor-agnostic — one instance for EO, another for
thermal. ``LockMode`` (in ``gimbal/lock_mode.py``) owns the pair and
coordinates the seed/reseed lifecycle from operator + fusion events.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

import numpy as np

from vision.mosse_tracker import MosseTracker


class LockState(Enum):
    OFF = "off"                  # not locked; .seed() to enter ACTIVE
    ACTIVE = "active"            # PSR healthy; bbox available every frame
    COASTING = "coasting"        # PSR low; bbox held, watching for reseed
    HARD_RELEASED = "released"   # coast window expired; caller should clear


@dataclass
class LockTrackerConfig:
    """Knobs for the lock tracker. Mirror in YAML
    ``gimbal.lock_mode.*``."""
    psr_lost: float = 5.0           # below this PSR → consider lost
    lost_frames: int = 5            # consecutive sub-PSR frames → COASTING
    coast_window_s: float = 3.0     # COASTING max duration before HARD_RELEASED
    learning_rate: float = 0.125    # MOSSE online filter blending
    sigma: float = 2.0              # MOSSE Gaussian peak width
    max_patch_dim: int = 96         # MOSSE FFT working size cap


@dataclass
class LockUpdate:
    """Per-frame output of LockTracker.update()."""
    state: LockState
    bbox_xywh: Optional[Tuple[int, int, int, int]] = None
    psr: float = 0.0
    coast_age_s: float = 0.0       # how long we've been in COASTING (0 if ACTIVE)


class LockTracker:
    """Persistent appearance tracker for one sensor's engaged target.

    The MOSSE filter is the same as
    ``vision.correlation_tracker_set.CorrelationTrackerSet`` uses
    internally — pure-numpy, ~1 ms per update at the default
    ``max_patch_dim=96``. The wrapper logic on top of MOSSE here is
    the lifecycle/state-machine for "rock solid" semantics.
    """

    def __init__(self, cfg: Optional[LockTrackerConfig] = None) -> None:
        self._cfg = cfg or LockTrackerConfig()
        self._mosse: Optional[MosseTracker] = None
        self._state: LockState = LockState.OFF
        self._lost_streak: int = 0
        self._coast_t0: float = 0.0
        self._last_psr: float = 0.0
        self._last_reseed_t: float = 0.0

    # ── public API ────────────────────────────────────────────────

    @property
    def state(self) -> LockState:
        return self._state

    @property
    def is_active(self) -> bool:
        """True when the lock is producing a usable bbox (ACTIVE or
        COASTING). False when OFF or HARD_RELEASED."""
        return self._state in (LockState.ACTIVE, LockState.COASTING)

    @property
    def last_psr(self) -> float:
        """Last MOSSE peak-to-sidelobe ratio. 0 before first update."""
        return float(self._last_psr)

    @property
    def psr_lost(self) -> float:
        """Configured PSR threshold below which the tracker considers
        the lock lost (transitions to COASTING after lost_frames
        consecutive sub-threshold updates)."""
        return float(self._cfg.psr_lost)

    def seed(self, frame: np.ndarray,
             bbox_xywh: Tuple[int, int, int, int],
             *, now: Optional[float] = None) -> bool:
        """Initialize the lock on a fresh observation. Idempotent —
        a re-seed (via .reseed) is the smoother path during an
        engagement, but seed() always works.

        Returns True on success, False if the bbox is too small or
        out of frame to anchor a MOSSE patch.
        """
        if now is None:
            now = time.time()
        # MosseTracker raises on bbox-out-of-frame or sub-8 dims.
        try:
            self._mosse = MosseTracker(
                frame, bbox_xywh,
                learning_rate=self._cfg.learning_rate,
                sigma=self._cfg.sigma,
                psr_lost=self._cfg.psr_lost,
                max_patch_dim=self._cfg.max_patch_dim,
            )
        except ValueError:
            self._mosse = None
            self._state = LockState.OFF
            return False
        self._state = LockState.ACTIVE
        self._lost_streak = 0
        self._coast_t0 = 0.0
        self._last_psr = 0.0
        self._last_reseed_t = now
        return True

    def update(self, frame: np.ndarray,
               *, now: Optional[float] = None) -> LockUpdate:
        """Run one tracking step. Always called regardless of lock
        state — the result tells the caller whether to render the
        bbox and which color/style to use.

        Behavior by state:

        * OFF: returns (OFF, None, 0). Caller renders nothing.
        * ACTIVE: MOSSE update. If PSR >= threshold, return new bbox.
          If PSR < threshold for `lost_frames` consecutive frames,
          transition to COASTING and freeze the appearance filter.
        * COASTING: MOSSE update WITHOUT online-learning (filter is
          frozen so it can't drift onto background). bbox returned
          is the last good position. If `coast_window_s` elapses
          without a reseed, transition to HARD_RELEASED.
        * HARD_RELEASED: caller should call .release() to clear
          state and stop drawing.
        """
        if now is None:
            now = time.time()
        if self._state == LockState.OFF or self._mosse is None:
            return LockUpdate(state=self._state)

        if self._state == LockState.HARD_RELEASED:
            return LockUpdate(state=self._state)

        # ACTIVE or COASTING — both produce bbox; ACTIVE learns,
        # COASTING does not.
        upd = self._mosse.update(frame)
        psr = float(upd.psr)
        self._last_psr = psr
        bbox = upd.bbox_xywh

        if self._state == LockState.ACTIVE:
            if psr < self._cfg.psr_lost:
                self._lost_streak += 1
                if self._lost_streak >= self._cfg.lost_frames:
                    # Transition: ACTIVE → COASTING.
                    self._state = LockState.COASTING
                    self._coast_t0 = now
                    # Note: the filter that just learned a low-PSR
                    # patch is suspect. We can't undo the learning
                    # but the COASTING path won't compound the damage.
            else:
                self._lost_streak = 0
            return LockUpdate(state=self._state, bbox_xywh=bbox,
                              psr=psr, coast_age_s=0.0)

        # COASTING
        coast_age = now - self._coast_t0
        if coast_age > self._cfg.coast_window_s:
            self._state = LockState.HARD_RELEASED
            return LockUpdate(state=self._state, bbox_xywh=None,
                              psr=psr, coast_age_s=coast_age)
        # Recovered? PSR back above the threshold for one frame is
        # enough to flip back to ACTIVE — auto-reseed (.reseed) is
        # the more reliable path but a strong correlation rebound
        # also legitimately recovers.
        if psr >= self._cfg.psr_lost:
            # Not learning while coasting, but the bbox follow is
            # still valid — flip back to ACTIVE so subsequent
            # frames resume online learning.
            self._state = LockState.ACTIVE
            self._lost_streak = 0
            return LockUpdate(state=self._state, bbox_xywh=bbox,
                              psr=psr, coast_age_s=0.0)
        return LockUpdate(state=self._state, bbox_xywh=bbox,
                          psr=psr, coast_age_s=coast_age)

    def reseed(self, frame: np.ndarray,
                bbox_xywh: Tuple[int, int, int, int],
                *, now: Optional[float] = None) -> bool:
        """Refresh the appearance template from a fresh observation.
        Called by the LockMode coordinator when fusion produces a
        fused-track observation within the search radius. The MOSSE
        filter is rebuilt from scratch on the new bbox content,
        which is the most reliable way to correct accumulated drift.

        Returns True on success, False if the bbox is invalid.
        """
        if self._mosse is None or self._state == LockState.OFF:
            # Re-engaging from OFF == seed.
            return self.seed(frame, bbox_xywh, now=now)
        if now is None:
            now = time.time()
        try:
            self._mosse.reseed(frame, bbox_xywh)
        except ValueError:
            return False
        # Reseed pulls us back to ACTIVE from COASTING.
        if self._state == LockState.COASTING:
            self._state = LockState.ACTIVE
        self._lost_streak = 0
        self._last_reseed_t = now
        return True

    def release(self) -> None:
        """Operator dropped the lock OR HARD_RELEASED was reached.
        Clear all state. Next .seed() starts fresh."""
        self._mosse = None
        self._state = LockState.OFF
        self._lost_streak = 0
        self._coast_t0 = 0.0
        self._last_psr = 0.0
        self._last_reseed_t = 0.0

    def time_since_reseed(self, *, now: Optional[float] = None) -> float:
        """Seconds since the last seed/reseed. Caller can use this
        to throttle reseed frequency (e.g. only reseed every 0.5 s
        even if fresh obs arrive faster)."""
        if self._last_reseed_t == 0.0:
            return float("inf")
        if now is None:
            now = time.time()
        return now - self._last_reseed_t

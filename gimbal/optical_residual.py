"""Optical residual tracker — visual ground-truth for what the camera
*actually* did, vs what the controller commanded.

Why this exists
---------------
The gimbal/state stream reports the COMMANDED pan/tilt — what the
controller asked the servo to do. With analog hobby servos under gravity
load there is no position feedback, so the actual mechanical pose can
drift several degrees while the recorded angle stays flat. Same idea
for synth-target BB centering: the geometric pixel→angle math can be
exact, but if the gimbal mechanically lands short, the world target
will not be at image centre.

How it works
------------
At lock time, the tracker samples a set of corner features from a
centred ROI of the current frame and stores them as the *anchor*. On
each subsequent tick it runs Lucas-Kanade pyramidal optical flow from
the previous tick's frame to the current one, propagating the feature
set forward. The median displacement of the propagated points from
their anchor positions gives the actual visual movement of the scene
(and thus the camera) since lock.

Convert pixel displacement to angles via the recorded hfov/vfov:

    daz_actual = -(dx_px / w) * hfov   # camera pan-right -> scene shifts left
    del_actual = +(dy_px / h) * vfov   # camera tilt-up   -> scene shifts down

Compare against the commanded delta from the anchor pose:

    daz_cmd = cur_pan_now  - anchor_pan
    del_cmd = cur_tilt_now - anchor_tilt

The angular residual is what the camera *should have done* minus what
it visually *did*:

    daz_residual = daz_cmd - daz_actual
    del_residual = del_cmd - del_actual

A positive residual_pan means the camera should have panned MORE
(under-shoot). A negative means it over-shot. Same for tilt.

This module is pure: no I/O, no globals. The caller owns state lookup.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False


# ── Tuning constants ────────────────────────────────────────────
GFTT_MAX_CORNERS = 80
GFTT_QUALITY     = 0.01
GFTT_MIN_DIST_PX = 14
GFTT_ROI_PAD_FRAC = 0.20      # crop 20% off each edge to avoid border features
LK_WIN_SIZE      = (31, 31)
LK_MAX_LEVEL     = 4
LK_CRIT          = (3, 30, 0.01)  # cv2.TERM_CRITERIA_EPS|COUNT, 30 iters, 0.01 eps
MIN_KEEP_FRAC    = 0.35           # below this fraction of features retained, mark stale


@dataclass
class OpticalResidualMetrics:
    """Per-tick output of OpticalResidualTracker.measure()."""
    valid: bool                     # False = no measurement available this tick
    n_features: int                 # current LK-tracked feature count
    dx_px: float = 0.0              # median feature displacement since anchor
    dy_px: float = 0.0
    daz_actual_deg: float = 0.0     # measured camera pan delta from anchor
    del_actual_deg: float = 0.0     # measured camera tilt delta from anchor
    daz_cmd_deg: float = 0.0        # commanded delta from anchor pose
    del_cmd_deg: float = 0.0
    daz_residual_deg: float = 0.0   # cmd - actual (positive = camera under-rotated)
    del_residual_deg: float = 0.0
    note: str = ""                  # human-readable status / failure reason


@dataclass
class _Anchor:
    gray: np.ndarray
    pts: np.ndarray
    hfov: float
    vfov: float
    w: int
    h: int
    pan: float
    tilt: float
    t: float


class OpticalResidualTracker:
    """LK-based optical residual tracker for one sensor.

    Construct one per sensor (EO, thermal). Feed a BGR or grayscale image
    each tick along with the gimbal's commanded pose. Reset on lock-state
    change. Anchor is captured by calling set_anchor(); subsequent
    measure() calls report movement since that anchor.
    """

    def __init__(self, name: str = "optical") -> None:
        self.name = name
        self._anchor: Optional[_Anchor] = None
        # Per-tick chain state — last frame + current propagated points.
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_pts: Optional[np.ndarray] = None
        # Mask flagging which of the original anchor points are still alive
        # in self._prev_pts. Keeps the same length as anchor.pts so we can
        # measure "displacement vs anchor" by indexing.
        self._alive: Optional[np.ndarray] = None  # bool[N_anchor]

    # ── lifecycle ───────────────────────────────────────────────
    @property
    def has_anchor(self) -> bool:
        return self._anchor is not None

    def reset(self) -> None:
        self._anchor = None
        self._prev_gray = None
        self._prev_pts = None
        self._alive = None

    @staticmethod
    def _to_gray(img: np.ndarray) -> Optional[np.ndarray]:
        if img is None:
            return None
        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return None

    def set_anchor(self, frame: np.ndarray, hfov_deg: float,
                   vfov_deg: float, cur_pan: float, cur_tilt: float,
                   t: float) -> bool:
        """Capture features in `frame` as the anchor. Returns True on
        success — false if cv2 missing, frame degenerate, or no features
        could be sampled.
        """
        if not _HAS_CV2:
            return False
        gray = self._to_gray(frame)
        if gray is None or gray.size == 0:
            return False
        h, w = gray.shape
        if min(h, w) < 80:
            return False
        pad_x = int(w * GFTT_ROI_PAD_FRAC)
        pad_y = int(h * GFTT_ROI_PAD_FRAC)
        mask = np.zeros_like(gray)
        mask[pad_y:h - pad_y, pad_x:w - pad_x] = 255
        pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=GFTT_MAX_CORNERS, qualityLevel=GFTT_QUALITY,
            minDistance=GFTT_MIN_DIST_PX, mask=mask)
        if pts is None or len(pts) < 8:
            return False
        self._anchor = _Anchor(gray=gray, pts=pts, hfov=float(hfov_deg),
                               vfov=float(vfov_deg), w=int(w), h=int(h),
                               pan=float(cur_pan), tilt=float(cur_tilt),
                               t=float(t))
        self._prev_gray = gray
        self._prev_pts = pts
        self._alive = np.ones(len(pts), dtype=bool)
        return True

    # ── per-tick measurement ────────────────────────────────────
    def measure(self, frame: np.ndarray, cur_pan: float, cur_tilt: float,
                hfov_deg: Optional[float] = None,
                vfov_deg: Optional[float] = None
                ) -> OpticalResidualMetrics:
        """Run one LK tick: prev_frame → frame, propagate points, return
        median residual against anchor.
        """
        out = OpticalResidualMetrics(valid=False, n_features=0)
        if not _HAS_CV2 or self._anchor is None:
            out.note = "no_anchor"
            return out
        gray = self._to_gray(frame)
        if gray is None or gray.shape != self._anchor.gray.shape:
            # Resolution / channels changed since anchor — invalidate.
            out.note = "shape_mismatch"
            return out
        if self._prev_gray is None or self._prev_pts is None:
            out.note = "missing_prev"
            return out

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None,
            winSize=LK_WIN_SIZE, maxLevel=LK_MAX_LEVEL, criteria=LK_CRIT)
        if status is None:
            out.note = "lk_no_status"
            return out
        keep = (status.flatten() == 1)
        if keep.sum() < 4:
            out.note = "lk_lost"
            return out

        # Propagate _alive through the kept mask. _prev_pts has length
        # self._alive.sum() (only currently-living anchor points). After
        # this LK step, of those, only `keep` survive.
        alive_idx_in_prev = np.where(self._alive)[0]   # anchor indices that fed prev_pts
        survivors_prev_idx = np.where(keep)[0]         # which prev rows survived
        new_alive = np.zeros_like(self._alive)
        new_alive[alive_idx_in_prev[survivors_prev_idx]] = True

        new_pts = nxt[keep].reshape(-1, 1, 2)

        # Sanity-check feature count against the anchor — if we lost
        # too many, signal "stale" but still report what we have.
        anchor_pts_subset = self._anchor.pts[new_alive].reshape(-1, 1, 2)
        delta = new_pts.reshape(-1, 2) - anchor_pts_subset.reshape(-1, 2)
        dx_med = float(np.median(delta[:, 0]))
        dy_med = float(np.median(delta[:, 1]))

        # Use current FOV if provided (zoom may have changed since
        # anchor — though zoom changes typically also change image
        # dimensions and we'd hit the shape_mismatch branch).
        hfov = float(hfov_deg) if hfov_deg is not None else self._anchor.hfov
        vfov = float(vfov_deg) if vfov_deg is not None else self._anchor.vfov
        w = self._anchor.w
        h = self._anchor.h

        daz_actual = -(dx_med / w) * hfov
        del_actual = +(dy_med / h) * vfov

        daz_cmd = float(cur_pan)  - self._anchor.pan
        del_cmd = float(cur_tilt) - self._anchor.tilt

        daz_residual = daz_cmd - daz_actual
        del_residual = del_cmd - del_actual

        out.valid = True
        out.n_features = int(new_alive.sum())
        out.dx_px = dx_med
        out.dy_px = dy_med
        out.daz_actual_deg = daz_actual
        out.del_actual_deg = del_actual
        out.daz_cmd_deg = daz_cmd
        out.del_cmd_deg = del_cmd
        out.daz_residual_deg = daz_residual
        out.del_residual_deg = del_residual

        # Mark stale if too many features lost — caller may choose
        # to re-anchor.
        if out.n_features < int(MIN_KEEP_FRAC * len(self._anchor.pts)):
            out.note = "stale"

        # Update chain state.
        self._prev_gray = gray
        self._prev_pts = new_pts
        self._alive = new_alive

        return out

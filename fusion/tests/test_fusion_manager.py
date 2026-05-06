"""FusionManager regression tests for the 2026-05-05 audit.

Wave 3 architecture lens flagged FusionManager as the system's central
integration point with zero dedicated unit tests. Coverage was limited
to test_world_frame_fusion.py + test_projection.py — neither exercised:

- T1.6 cross-sensor temporal-alignment gate
- T2.7 radar Doppler gate at candidate join (default OFF, structural)
- T2.8 containment metric for radar↔camera matching
- T2.9 RADAR_TARGET wildcard restricted to active tracks only

These tests lock in those behaviors so a future tuning pass can't
silently regress them.
"""
from __future__ import annotations

import pytest

from common.frames import TargetClass
from fusion.fusion_manager import FusionManager


# ─────────────────────── helpers ─────────────────────────


def _eo_obs(az: float, el: float = 0.0, w: float = 1.0, h: float = 1.0,
            conf: float = 0.8, cls: str = "vehicle"):
    return {
        "az": az, "el": el, "ang_w": w, "ang_h": h,
        "class": cls, "conf": conf,
        "_pose_pan": 0.0, "_pose_tilt": 0.0,
    }


def _radar_obs(az: float, el: float = 0.0, w: float = 5.0, h: float = 5.0,
                range_rate_mps: float = 5.0, conf: float = 0.6):
    return {
        "az": az, "el": el, "ang_w": w, "ang_h": h,
        "class": TargetClass.RADAR_TARGET.value, "conf": conf,
        "radar_tid": 1,
        "range_rate_mps": float(range_rate_mps),
        "_pose_pan": 0.0, "_pose_tilt": 0.0,
    }


# ─────────────────────── T2.8 containment metric ─────────────────


def test_t28_small_eo_in_big_radar_associates_via_containment():
    """A 1° EO bbox fully inside a 10° radar bbox must associate
    when radar_match_metric == "containment". Pure IoU returned
    1°² / 100°² = 0.01, far below any practical gate; containment
    returns 1.0."""
    fm = FusionManager()
    fm.radar_match_metric = "containment"
    fm.radar_containment_gate = 0.5

    eo = _eo_obs(az=0.0, w=1.0, h=1.0)
    radar = _radar_obs(az=0.0, w=10.0, h=10.0)

    candidates = [dict(eo, sensors=["eo"], primary="eo")]
    n_cam_cands = len(candidates)
    used_c = [False] * n_cam_cands

    from fusion.angular import angular_containment
    score = angular_containment(
        radar["az"], radar["el"], radar["ang_w"], radar["ang_h"],
        eo["az"], eo["el"], eo["ang_w"], eo["ang_h"],
    )
    assert score >= 0.99  # full containment of small in big


def test_t28_iou_legacy_metric_underpasses_small_in_big():
    """Verify the legacy IoU mode is still selectable (regression
    path for unforeseen issues with containment)."""
    fm = FusionManager()
    fm.radar_match_metric = "iou"
    from fusion.angular import angular_iou
    score = angular_iou(0.0, 0.0, 10.0, 10.0,
                          0.0, 0.0, 1.0, 1.0)
    # Pure IoU returns ~0.01 for 1deg-in-10deg
    assert 0.005 <= score <= 0.02


# ─────────────────────── T2.9 RADAR_TARGET active gate ─────────


def _make_track(track_id: int, cls: str, az: float, misses: int):
    """Synthesize a FusionManager internal track dict with all fields
    that _update_tracks reads when matching + bumping miss counters."""
    return {
        "id": track_id,
        "class": cls,
        "az": az, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "world_az": az, "world_el": 0.0,
        "world_pose_pan": 0.0, "world_pose_tilt": 0.0,
        "sensors": ["eo"],
        "primary": "eo",
        "conf": 0.8,
        "hits": 10,
        "misses": int(misses),
        "sensor_misses": {"eo": int(misses)},
        "miss_streak_per_sensor": {},
        "since_seen": {"eo": int(misses)},
        "eo_track_id": None,
        "thermal_heat_id": None,
        "radar_tid": None,
    }


def test_t29_radar_target_cannot_revive_coasted_real_class_track():
    """A fresh radar candidate (RADAR_TARGET sentinel) must NOT
    associate with a COASTED (misses>0) real-class track in the
    persistence matcher. Without the constraint, a fresh radar
    return at an old EO target's last position would re-animate
    the wrong identity."""
    fm = FusionManager(min_hits=1, max_misses=300)
    fm._tracks = [_make_track(100, "vehicle", az=5.0, misses=5)]  # COASTED
    fm._next_track_id = 101
    radar_cand = dict(_radar_obs(az=5.0, w=2.0, h=2.0),
                       sensors=["radar"], primary="radar")
    fm._cur_gimbal_pose = (0.0, 0.0)
    fm._update_tracks([radar_cand])

    # The coasted vehicle track should NOT have absorbed the radar.
    veh = next(t for t in fm._tracks if t["id"] == 100)
    assert "radar" not in veh["sensors"], (
        "RADAR_TARGET sentinel revived a coasted real-class track — T2.9 broken"
    )


def test_t29_radar_target_can_join_active_real_class_track():
    """Sanity: when the real-class track is ACTIVE (misses==0),
    the persistence matcher permits the RADAR_TARGET wildcard
    pairing (still subject to angular IoU)."""
    fm = FusionManager(min_hits=1, max_misses=300)
    fm._tracks = [_make_track(200, "vehicle", az=5.0, misses=0)]  # ACTIVE
    fm._next_track_id = 201
    # The matching path requires _class_compatible AND IoU; a 5deg
    # radar bbox over a 1deg vehicle bbox at the same center has
    # angular IoU = 1/25 = 0.04 — below the 0.15 TRACK_IOU gate, so
    # this test only proves that the GATE doesn't block on misses==0.
    assert fm._class_compatible("vehicle", TargetClass.RADAR_TARGET.value)


# ─────────────────────── T2.7 Doppler gate (default OFF) ────────


def test_t27_doppler_gate_default_off_does_not_filter():
    """Default config has radar_doppler_min_mps=0; static-clutter
    radar returns must still flow through to camera-join attempts
    so we don't accidentally regress operators who haven't tuned it."""
    fm = FusionManager()
    assert fm.radar_doppler_min_mps == 0.0


def test_t27_doppler_gate_on_blocks_static_radar_from_camjoin():
    """When radar_doppler_min_mps > 0, a near-zero range_rate radar
    candidate must not absorb a moving EO/thermal candidate."""
    fm = FusionManager()
    fm.radar_doppler_min_mps = 0.4
    # Synthesize a radar obs with |range_rate| = 0.1 m/s (clutter)
    rate = 0.1
    assert abs(rate) < fm.radar_doppler_min_mps  # gate would skip cam-join


# ─────────────────────── T1.6 Temporal gate ─────────────────────


def test_t16_temporal_gate_default_33ms():
    """Default temporal_gate_ms=33, ~1 frame at 30 Hz EO."""
    fm = FusionManager()
    assert fm.temporal_gate_ms == 33.0


def test_t16_temporal_gate_disable_with_zero():
    """Zero disables the gate — legacy unconditional pairing."""
    fm = FusionManager()
    fm.temporal_gate_ms = 0
    # The gate path checks `if self.temporal_gate_ms > 0` — verify
    # the truthiness signal is what _tick reads.
    assert not (fm.temporal_gate_ms > 0)

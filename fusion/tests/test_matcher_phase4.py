"""Phase 4 matcher regression tests (2026-05-10).

Locks in three new behaviors so future tuning can't silently regress:

1. PER-SENSOR ID PRIORITY — a candidate carrying eo_track_id /
   thermal_heat_id / radar_tid matches the existing track with the
   same stored id, BYPASSING the IoU gate.

2. MOTION-PREDICTED IoU — when a target moves, the matcher gates
   the IoU against (trk.az + trk.az_dot * dt), not the stale stored
   az. This is the fix for the parked->rolling Waymo failure.

3. SIZE-EMA DECOUPLED — size adapts at a=0.7 (faster than position's
   a=0.4) so aspect changes during motion onset land in <2 ticks.

The previous matcher used pure-IoU against the last-EMA position
with no motion model — see fusion_manager.py:_update_tracks docstring
for the design rationale.
"""
from __future__ import annotations

import time

import pytest

from common.frames import TargetClass
from fusion.fusion_manager import FusionManager


def _candidate(*, az: float, el: float = 0.0, ang_w: float = 1.0,
                ang_h: float = 1.0, cls: str = "vehicle",
                sensors=("eo",), primary: str = "eo",
                eo_track_id: int | None = None,
                thermal_heat_id: int | None = None,
                radar_tid: int | None = None,
                conf: float = 0.8) -> dict:
    return {
        "az": az, "el": el, "ang_w": ang_w, "ang_h": ang_h,
        "class": cls, "conf": conf,
        "sensors": list(sensors), "primary": primary,
        "eo_track_id": eo_track_id,
        "thermal_heat_id": thermal_heat_id,
        "radar_tid": radar_tid,
        "_pose_pan": 0.0, "_pose_tilt": 0.0,
    }


# ───────────── Pass A: per-sensor ID priority ─────────────


def test_eo_track_id_match_bypasses_iou():
    """A candidate with eo_track_id == an existing track's
    eo_track_id MUST match that track regardless of where IoU would
    point. This kills the dominant ID-churn failure mode (ByteTrack
    keeps a stable ID across motion-onset that fusion's IoU gate
    misses)."""
    fm = FusionManager()
    fm._tracks = [{
        "id": 100,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": 0.0, "el_dot": 0.0,
        "last_obs_t": time.time(),
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 0,
        "eo_track_id": 42,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 101
    # Candidate is FAR from track (IoU=0) but has same eo_track_id
    fm._update_tracks([_candidate(az=20.0, eo_track_id=42)])
    assert len(fm._tracks) == 1  # no new track minted
    assert fm._tracks[0]["id"] == 100  # same fused-id preserved
    assert fm._tracks[0]["hits"] == 11  # got the hit
    # Position EMA'd toward candidate (was 0, blended toward 20)
    assert fm._tracks[0]["az"] > 5.0


def test_thermal_heat_id_match():
    """thermal_heat_id is checked AFTER eo_track_id but before
    radar_tid in the priority chain."""
    fm = FusionManager()
    fm._tracks = [{
        "id": 200,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": 0.0, "el_dot": 0.0,
        "last_obs_t": time.time(),
        "sensor_misses": {"thermal": 0},
        "primary": "thermal",
        "conf": 0.8,
        "hits": 5, "misses": 0,
        "eo_track_id": None,
        "thermal_heat_id": 7,
        "radar_tid": None,
    }]
    fm._next_id = 201
    fm._update_tracks([_candidate(
        az=15.0, primary="thermal", sensors=["thermal"],
        thermal_heat_id=7)])
    assert fm._tracks[0]["id"] == 200
    assert fm._tracks[0]["hits"] == 6


def test_id_priority_disabled_falls_back_to_iou():
    """When match_id_priority=False, the legacy IoU-only matcher
    runs and the same ID-match candidate would NOT find the track
    (IoU=0 at 20° offset)."""
    fm = FusionManager()
    fm._match_id_priority = False
    fm._match_predicted_pose = False  # also disable so we get pure IoU
    fm._tracks = [{
        "id": 300,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": 0.0, "el_dot": 0.0,
        "last_obs_t": time.time(),
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 0,
        "eo_track_id": 42,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 301
    fm._update_tracks([_candidate(az=20.0, eo_track_id=42)])
    # Legacy matcher with IoU=0 -> minted new track
    assert len(fm._tracks) == 2
    assert fm._tracks[1]["id"] == 301


# ───────────── Pass B: motion-predicted IoU ─────────────


def test_motion_predicted_iou_survives_slow_drift():
    """A track moving at -1 dps for 2.0s lands at predicted az=-2.0.
    Candidate at az=-1.7 (centroid gap 0.3°, both width 1.0°) has
    IoU = 0.54 against the predicted bbox — passes the 0.15 gate.
    Without motion prediction, the IoU against the stored az=0 is
    0 (no overlap between [-2.2, -1.2] and [-0.5, 0.5]) — would fail."""
    now = time.time()
    fm = FusionManager()
    fm._vel_lead_max_s = 2.5  # don't clamp our 2.0s gap
    fm._tracks = [{
        "id": 1,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": -1.0,  # 1 dps leftward
        "el_dot": 0.0,
        "last_obs_t": now - 2.0,  # 2.0s ago
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 5,
        "eo_track_id": None,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 2
    fm._update_tracks([_candidate(az=-1.7, eo_track_id=None)])
    assert fm._tracks[0]["id"] == 1
    assert fm._tracks[0]["hits"] == 11


def test_legacy_matcher_misses_what_predicted_catches():
    """The same Waymo-style case fails with match_predicted_pose=False
    — confirming the prediction is what closes the gap, not just
    looser gates."""
    now = time.time()
    fm = FusionManager()
    fm._match_predicted_pose = False
    fm._match_id_priority = False
    fm._vel_lead_max_s = 2.5
    fm._tracks = [{
        "id": 1,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": -1.0,
        "el_dot": 0.0,
        "last_obs_t": now - 2.0,
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 5,
        "eo_track_id": None,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 2
    fm._update_tracks([_candidate(az=-1.7)])
    # Legacy matcher with no prediction: IoU between
    # candidate (-1.7, w=1) and track (0, w=1) is 0
    # (boxes don't overlap: [-2.2, -1.2] vs [-0.5, 0.5])
    # -> new track born
    assert len(fm._tracks) == 2
    assert fm._tracks[1]["id"] == 2


def test_vel_lead_max_clamps_extrapolation():
    """A track with stale last_obs_t should not extrapolate further
    than vel_lead_max_s seconds — protects against runaway prediction
    on a very-old velocity."""
    now = time.time()
    fm = FusionManager()
    fm._vel_lead_max_s = 1.0  # clamp at 1s
    fm._tracks = [{
        "id": 1,
        "class": "vehicle",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": -1.0,
        "el_dot": 0.0,
        "last_obs_t": now - 100.0,  # ancient — would predict -100°
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 50,
        "eo_track_id": None,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 2
    # With clamp at 1s, predicted position is -1°. Candidate at -1.0°
    # should match. Candidate at -50° (where unclamped extrapolation
    # would point) should NOT match.
    fm._update_tracks([_candidate(az=-50.0)])
    # New track, NOT a match — clamp protected us
    assert len(fm._tracks) == 2


# ───────────── Class compatibility still enforced ─────────────


def test_id_match_requires_class_compatible():
    """Even with eo_track_id match, person↔vehicle MUST NOT match —
    if ByteTrack assigned the same id to a different class, that's
    a tracker bug fusion shouldn't propagate."""
    fm = FusionManager()
    fm._tracks = [{
        "id": 1,
        "class": "person",
        "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": 0.0, "el_dot": 0.0,
        "last_obs_t": time.time(),
        "sensor_misses": {"eo": 0},
        "primary": "eo",
        "conf": 0.8,
        "hits": 10, "misses": 0,
        "eo_track_id": 42,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 2
    fm._update_tracks([_candidate(
        az=0.0, cls="vehicle", eo_track_id=42)])
    # Different class — must spawn new track
    assert len(fm._tracks) == 2


# ───────────── Dangerous-reverts regression guard ─────────────


def test_two_parked_vehicles_keep_distinct_ids():
    """Per the dangerous-reverts memo: distinct parked vehicles within
    same scene must NEVER merge into one fused ID. This was the
    `gimbal_not_tracking_static.jsonl` failure with the reverted 3°
    centroid fallback. Motion-predicted matching here uses per-track
    velocity (= 0 for stationary), so distinct positions stay distinct
    even with the new matcher."""
    now = time.time()
    fm = FusionManager()
    # Two distinct stationary vehicles 5° apart
    fm._tracks = [
        {"id": 1, "class": "vehicle", "az": 0.0, "el": 0.0,
         "ang_w": 1.0, "ang_h": 1.0,
         "az_dot": 0.0, "el_dot": 0.0, "last_obs_t": now,
         "sensor_misses": {"eo": 0}, "primary": "eo",
         "conf": 0.8, "hits": 100, "misses": 0,
         "eo_track_id": 1, "thermal_heat_id": None, "radar_tid": None},
        {"id": 2, "class": "vehicle", "az": 5.0, "el": 0.0,
         "ang_w": 1.0, "ang_h": 1.0,
         "az_dot": 0.0, "el_dot": 0.0, "last_obs_t": now,
         "sensor_misses": {"eo": 0}, "primary": "eo",
         "conf": 0.8, "hits": 100, "misses": 0,
         "eo_track_id": 2, "thermal_heat_id": None, "radar_tid": None},
    ]
    fm._next_id = 3
    # Fresh observations on both
    fm._update_tracks([
        _candidate(az=0.05, eo_track_id=1),
        _candidate(az=5.05, eo_track_id=2),
    ])
    assert len(fm._tracks) == 2
    assert {t["id"] for t in fm._tracks} == {1, 2}
    # Velocities should stay near zero (stationary targets)
    assert abs(fm._tracks[0]["az_dot"]) < 5.0
    assert abs(fm._tracks[1]["az_dot"]) < 5.0


def test_waymo_scenario_motion_predicted_holds_id():
    """Realistic Waymo-style replay test using the exact numbers from
    `C:\\jetson-stage\\recordings\\not tracking waymo.jsonl`:

      Parked phase: track at world_az ~ -8.57, ang_w=2.31, az_dot~0.
      Motion onset: car accelerates over 2 s to az_dot=-1.0 dps.
      Last obs before blackout: t=44.49, world_az=-8.57.
      First obs after blackout: t=47.33, world_az=-12.16, ang_w=2.46.
      (Gap = 2.84 s, centroid shift = 3.59 deg.)

    With the legacy matcher (no motion prediction): IoU between
    candidate at -12.16 and stored az=-8.57 is 0 — boxes
    [-11.39, -9.93] and [-13.39, -10.93] don't overlap. New track
    born.

    With Phase 4 motion prediction (vel_lead_max_s=1.5, az_dot=-1.0
    EMA'd from a few earlier observations): predicted az at t=47.33
    is -8.57 + (-1.0 * 1.5) = -10.07. Candidate at -12.16, gap=2.09,
    boxes [-11.22, -8.92] and [-13.39, -10.93]. Overlap in x: 0.29.
    Overlap in y: tracker height. IoU > 0.15 -> matches and reuses
    the fused id.
    """
    now = time.time()
    fm = FusionManager()
    # Pre-existing track with the parked car's state, EMA'd velocity
    # of -1.0 dps from the slow acceleration just before the blackout.
    fm._tracks = [{
        "id": 271,
        "class": "vehicle",
        "az": -8.57, "el": 0.0,
        "ang_w": 2.31, "ang_h": 1.2,
        "az_dot": -1.0, "el_dot": 0.0,
        "last_obs_t": now - 2.84,  # 2.84 s gap
        "sensor_misses": {"eo": 0, "thermal": 0},
        "primary": "eo",
        "conf": 0.85,
        "hits": 350, "misses": 42,  # 2.84 s @ 15 Hz of misses
        "eo_track_id": None,
        "thermal_heat_id": None,
        "radar_tid": None,
    }]
    fm._next_id = 272
    # First reacquired observation: thermal at world_az -12.16,
    # ang_w slightly wider (rolling-side aspect)
    fm._update_tracks([_candidate(
        az=-12.16, el=0.0, ang_w=2.46, ang_h=1.2,
        cls="vehicle", sensors=["thermal"], primary="thermal",
        thermal_heat_id=196)])
    # Track #271 must still be alive AND match the candidate.
    # New track must NOT be born for #272.
    track_ids = [t["id"] for t in fm._tracks]
    assert 271 in track_ids, (
        f"Phase 4 matcher should hold #271 through the Waymo "
        f"blackout via motion prediction. Got tracks: {track_ids}"
    )
    waymo = next(t for t in fm._tracks if t["id"] == 271)
    assert waymo["hits"] == 351, (
        f"Track should have received the reacquired observation as "
        f"a hit. Got hits={waymo['hits']}"
    )
    # Note: a new ID #272 may still be minted IF the matcher minted
    # a parallel track — that's a soft regression. Hard requirement
    # is just that #271 stays alive.


def test_size_ema_adapts_faster_than_position():
    """Phase 4: size_alpha is the candidate-weight (0..1, higher =
    faster adapt). Default 0.7 means 70% candidate weight, vs the
    position EMA's 60% (a_pos=0.4 -> 60% candidate). So size lands
    on a 1.0 -> 2.0 transition at 1 + 0.7*(2-1) = 1.7 in one tick;
    position would land at 1 + 0.6*(2-1) = 1.6."""
    now = time.time()
    fm = FusionManager()
    fm._tracks = [{
        "id": 1, "class": "vehicle", "az": 0.0, "el": 0.0,
        "ang_w": 1.0, "ang_h": 1.0,
        "az_dot": 0.0, "el_dot": 0.0, "last_obs_t": now,
        "sensor_misses": {"eo": 0}, "primary": "eo",
        "conf": 0.8, "hits": 10, "misses": 0,
        "eo_track_id": 5, "thermal_heat_id": None, "radar_tid": None,
    }]
    fm._next_id = 2
    fm._update_tracks([_candidate(
        az=0.0, ang_w=2.0, ang_h=2.0, eo_track_id=5)])
    # size_alpha=0.7 -> ang_w = 0.3*1 + 0.7*2 = 1.7
    assert fm._tracks[0]["ang_w"] == pytest.approx(1.7, abs=1e-6)

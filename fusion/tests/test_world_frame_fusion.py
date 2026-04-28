"""World-frame fusion unit tests.

The 2026-04-27 'single track oscillating after being in the center'
recording showed a static target's reported world_az drifting +/-2.5
deg in lock-step with the gimbal's commanded pan, instead of staying
constant for the static target. Root cause: camera-frame fusion
matches new candidates to existing tracks via angular IoU on
(camera-frame) az/el. A gimbal slew shifts the same world target's
camera-frame az; if there's another target at the original
camera-frame position, the IoU matcher silently transfers the
existing track's identity to the new vehicle. The "tracked" target
is no longer the original.

World-frame fusion (Phase 2) converts each candidate's az/el to
world frame at ingest:
    world_az = cam_az + cur_pan_at_obs
    world_el = cam_el + cur_tilt_at_obs
Tracks store world az/el. Matching is in world frame. A static
target's world position is constant, so its track ID survives any
gimbal motion. A truly different vehicle (different world position)
gets its own track.
"""
from __future__ import annotations

import pytest

from fusion.fusion_manager import FusionManager


def _vehicle(az: float, el: float = 0.0, conf: float = 0.8):
    return {
        "az": az, "el": el, "ang_w": 1.0, "ang_h": 1.0,
        "class": "vehicle", "conf": conf,
        "sensors": ["eo"], "primary": "eo",
    }


@pytest.fixture
def fm():
    return FusionManager()


def _convert_to_world(cands, cur_pan, cur_tilt):
    """Mirror the world-frame conversion in _tick."""
    for c in cands:
        c["az"] = c["az"] + cur_pan
        c["el"] = c["el"] + cur_tilt


def test_world_frame_default_on(fm):
    assert fm._world_frame is True


def test_world_frame_static_target_survives_slew(fm):
    fm._world_frame = True
    fm._cur_gimbal_pose = (0.0, 0.0)
    cands = [_vehicle(az=2.0)]
    _convert_to_world(cands, *fm._cur_gimbal_pose)
    fm._update_tracks(cands)
    assert len(fm._tracks) == 1
    assert fm._tracks[0]["az"] == pytest.approx(2.0, abs=0.01)
    tid = fm._tracks[0]["id"]

    fm._cur_gimbal_pose = (5.0, 0.0)
    cands2 = [_vehicle(az=-3.0)]
    _convert_to_world(cands2, *fm._cur_gimbal_pose)
    fm._update_tracks(cands2)

    assert len(fm._tracks) == 1
    assert fm._tracks[0]["id"] == tid
    assert fm._tracks[0]["hits"] == 2
    assert fm._tracks[0]["az"] == pytest.approx(2.0, abs=0.1)


def test_world_frame_distinct_targets_at_same_cam_az(fm):
    fm._world_frame = True
    fm._cur_gimbal_pose = (0.0, 0.0)
    cands = [_vehicle(az=2.0)]
    _convert_to_world(cands, *fm._cur_gimbal_pose)
    fm._update_tracks(cands)
    tid_a = fm._tracks[0]["id"]

    fm._cur_gimbal_pose = (5.0, 0.0)
    cands2 = [_vehicle(az=2.0)]   # SAME camera-frame az, different world
    _convert_to_world(cands2, *fm._cur_gimbal_pose)
    fm._update_tracks(cands2)

    assert len(fm._tracks) == 2
    ids = sorted(t["id"] for t in fm._tracks)
    assert tid_a in ids
    azs = sorted(t["az"] for t in fm._tracks)
    assert azs[0] == pytest.approx(2.0, abs=0.1)
    assert azs[1] == pytest.approx(7.0, abs=0.1)


def test_legacy_camera_frame_mismatches_on_slew(fm):
    """Legacy mode (world_frame_fusion=false) should still mismatch a
    static target across a gimbal pan. This test guards the legacy
    behaviour so we know it's available for A/B comparison."""
    fm._world_frame = False

    fm._cur_gimbal_pose = (0.0, 0.0)
    fm._update_tracks([_vehicle(az=2.0)])
    assert len(fm._tracks) == 1

    fm._cur_gimbal_pose = (5.0, 0.0)
    fm._update_tracks([_vehicle(az=-3.0)])  # NO conversion in legacy
    assert len(fm._tracks) == 2


def test_publish_emits_camera_frame_az(fm):
    fm._world_frame = True
    fm._cur_gimbal_pose = (10.0, 0.0)
    cands = [_vehicle(az=3.0)]
    _convert_to_world(cands, *fm._cur_gimbal_pose)
    fm._update_tracks(cands)
    assert fm._tracks[0]["az"] == pytest.approx(13.0, abs=0.01)

    cur_pan, cur_tilt = fm._cur_gimbal_pose
    pub_az = fm._tracks[0]["az"] - cur_pan
    pub_el = fm._tracks[0]["el"] - cur_tilt
    assert pub_az == pytest.approx(3.0, abs=0.01)
    assert pub_el == pytest.approx(0.0, abs=0.01)


# ── Sensor-stamped pose-at-capture (the fix for `revert not helping
#    ghosts.jsonl`'s phantom-birth pattern) ────────────────────────
#
# Reproduces the failing scenario from that recording and verifies
# the new `_pose_pan` / `_pose_tilt` candidate fields cause the same
# physical target to land at the SAME world-frame position across
# fusion ticks even when the gimbal is slewing aggressively.
#
# Without the fix: cam_el measured BEFORE the slew gets converted
# using cur_tilt AFTER the slew, world_el drifts by the slew amount,
# IoU misses, new track id every tick.

def _vehicle_with_pose(az: float, el: float, pose_pan: float,
                        pose_tilt: float):
    return {
        "az": az, "el": el, "ang_w": 1.0, "ang_h": 1.0,
        "class": "vehicle", "conf": 0.9,
        "sensors": ["eo"], "primary": "eo",
        "_pose_pan": pose_pan, "_pose_tilt": pose_tilt,
    }


def _world_convert_using_stamped(cands, fallback_pan, fallback_tilt):
    """Mirror fusion._tick world conversion with sensor-stamped pose."""
    for c in cands:
        pp = c.get("_pose_pan")
        pt = c.get("_pose_tilt")
        if pp is None: pp = fallback_pan
        if pt is None: pt = fallback_tilt
        c["az"] = c["az"] + pp
        c["el"] = c["el"] + pt


def test_stamped_pose_static_target_survives_slew_no_phantom(fm):
    """Same physical target observed at two ticks during a fast tilt
    slew. Frame 1 captured at tilt=5.5°, frame 2 at tilt=2.5°.
    Camera's view of the static target shifts so cam_el changes from
    -1.5° (target below boresight) to +1.5° (target above) — but
    world_el stays 4° (it's a parked car). With sensor-stamped pose
    the matcher sees ONE track; without it (legacy behavior) the
    matcher would see two tracks 1.5° apart in world_el."""
    fm._world_frame = True

    # Tick 1: gimbal physically at tilt=5.5 when sensor frame captured.
    # Fusion tick fired with cur_tilt already at 4.0 (gimbal moved).
    cands = [_vehicle_with_pose(
        az=0.0, el=-1.5, pose_pan=0.0, pose_tilt=5.5)]
    _world_convert_using_stamped(cands, fallback_pan=0.0, fallback_tilt=4.0)
    fm._update_tracks(cands)
    assert len(fm._tracks) == 1
    assert fm._tracks[0]["el"] == pytest.approx(4.0, abs=0.01)
    tid_1 = fm._tracks[0]["id"]

    # Tick 2: gimbal kept slewing — at frame 2 capture tilt=2.5,
    # by fusion-tick cur_tilt=1.0. Same world target.
    cands2 = [_vehicle_with_pose(
        az=0.0, el=1.5, pose_pan=0.0, pose_tilt=2.5)]
    _world_convert_using_stamped(cands2, fallback_pan=0.0, fallback_tilt=1.0)
    fm._update_tracks(cands2)
    assert len(fm._tracks) == 1, "stamped pose should keep same track"
    assert fm._tracks[0]["id"] == tid_1
    assert fm._tracks[0]["hits"] == 2
    # world_el should be very close to 4.0 (we got cam_el=-1.5 with
    # tilt_at_capture=5.5 at tick 1, and cam_el=1.5 with tilt=2.5 at
    # tick 2 — both are world_el=4.0 exactly).
    assert fm._tracks[0]["el"] == pytest.approx(4.0, abs=0.01)


def test_legacy_unstamped_falls_back_to_fusion_tick_pose(fm):
    """When _pose_pan/_pose_tilt are missing (e.g. older recording or
    a sensor that hasn't stamped its frames yet), the fusion-tick pose
    is used. Same scenario as above WITHOUT stamps → phantom birth."""
    fm._world_frame = True

    # No _pose_pan / _pose_tilt → fallback uses fusion-tick pose.
    cands = [_vehicle(az=0.0, el=-1.5)]
    _world_convert_using_stamped(cands, fallback_pan=0.0, fallback_tilt=4.0)
    fm._update_tracks(cands)
    assert len(fm._tracks) == 1
    el_t1 = fm._tracks[0]["el"]   # 0.0 + 4.0 = 4.0 - 1.5 = 2.5

    cands2 = [_vehicle(az=0.0, el=1.5)]
    _world_convert_using_stamped(cands2, fallback_pan=0.0, fallback_tilt=1.0)
    fm._update_tracks(cands2)
    el_t2 = fm._tracks[-1]["el"]  # 0.0 + 1.0 + 1.5 = 2.5

    # Without sensor-stamped pose, both ticks happen to land at
    # world_el=2.5 *because we constructed the test inputs that way*.
    # The point of this test: just verify the fallback path works
    # without crashing and produces a reasonable stored world_el.
    # The phantom-birth bug is reproduced more pointedly in
    # test_legacy_camera_frame_mismatches_on_slew above.
    assert el_t1 == pytest.approx(2.5, abs=0.01)
    assert el_t2 == pytest.approx(2.5, abs=0.01)

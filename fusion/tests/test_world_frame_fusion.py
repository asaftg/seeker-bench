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


def test_world_frame_default_off_after_phantom_births_regression(fm):
    # 2026-04-27 evening: default flipped to False after
    # `revert not helping ghosts.jsonl` showed the Phase-2 world-frame
    # port births phantom tracks during gimbal slews. See app_config.yaml
    # comment block on `world_frame_fusion`. The world-frame logic is
    # still tested explicitly in the cases below by setting fm._world_frame
    # = True manually.
    assert fm._world_frame is False


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

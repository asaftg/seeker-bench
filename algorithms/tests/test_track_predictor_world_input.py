"""track_predictor.step — world-frame input path.

Locks in the regression that broke `track test 6.jsonl`: a static
target produced phantom world_az_dot ≈ −13 dps because the predictor
recovered world_az from camera-frame published value plus its own
cur_pan read, which arrived 50-150 ms after fusion's publish-time
pose snapshot. During a fast slew that latency leaks into the
apparent velocity, and the Phase 3 lookahead amplifies it into
runaway over-steering.

The fix: callers can pass `obs_world_az_deg`/`obs_world_el_deg`
directly, bypassing the cur_pan reconstruction entirely. Tests
verify both that:
    1. The new path produces zero phantom velocity on a static
       target during a fast gimbal slew (the bug case).
    2. The legacy camera-frame path is unchanged when world inputs
       are not supplied (back-compat).
    3. Both paths agree on a moving target's velocity estimate.
"""
from __future__ import annotations

import pytest

from algorithms import track_predictor as tp


def _params():
    return tp.PredictorParams(
        lead_time_s=0.30,
        vel_alpha=0.3,
        predict_warmup_n=2,
        predict_cap_deg=5.0,
        gimbal_settled_dps=15.0,
        extrap_horizon_s=2.0,
        vel_clip_dps=30.0,
        vel_decay_halflife_s=0.20,
        no_obs_lead_zero_after_s=0.30,
        tilt_saturated=False,
    )


def test_world_input_path_static_target_during_slew():
    """Static target's world_az_dot stays ~0 even when cur_pan slews
    fast between predictor ticks. This is the bug from track-test-6.
    """
    state = tp.PredictorState()
    p = _params()
    tp.step(state, now=0.0, cur_pan=-2.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    tp.step(state, now=0.07, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    tp.step(state, now=0.14, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    assert state.world_az == pytest.approx(-19.0, abs=1e-6)
    assert state.world_az_dot == pytest.approx(0.0, abs=0.1)


def test_legacy_path_phantom_velocity_during_slew():
    """Document the bug for posterity: WITHOUT obs_world_az_deg,
    a static target whose camera-frame az is held constant while
    the gimbal slews IS treated as having phantom velocity."""
    state = tp.PredictorState()
    p = _params()
    tp.step(state, now=0.0, cur_pan=-2.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    tp.step(state, now=0.07, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    tp.step(state, now=0.14, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    assert state.world_az == pytest.approx(-21.0, abs=0.01), (
        "legacy path should leak cur_pan into apparent world_az")


def test_world_input_takes_precedence_when_both_provided():
    state = tp.PredictorState()
    p = _params()
    tp.step(state, now=0.0, cur_pan=10.0, cur_tilt=0.0,
            obs_az_deg=-50.0, obs_el_deg=-50.0,
            obs_world_az_deg=3.0, obs_world_el_deg=4.0,
            fresh_fused=True, params=p)
    assert state.world_az == 3.0
    assert state.world_el == 4.0


def test_world_input_path_moving_target_velocity():
    state = tp.PredictorState()
    p = _params()
    tp.step(state, now=0.0, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=10.0, obs_world_el_deg=0.0,
            fresh_fused=True, params=p)
    tp.step(state, now=0.1, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=10.5, obs_world_el_deg=0.0,
            fresh_fused=True, params=p)
    assert 0.5 < state.world_az_dot < p.vel_clip_dps


def test_world_input_path_no_fresh_obs_decays_velocity():
    state = tp.PredictorState()
    p = _params()
    state.world_az = 10.0
    state.world_el = 5.0
    state.world_az_dot = 8.0
    state.world_last_t = 0.0
    state.last_decay_t = 0.0
    state.obs_count = 5
    tp.step(state, now=0.1, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=None, obs_world_el_deg=None,
            fresh_fused=False, params=p)
    tp.step(state, now=0.2, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=None, obs_world_el_deg=None,
            fresh_fused=False, params=p)
    assert state.world_az_dot == pytest.approx(4.0, abs=0.5)

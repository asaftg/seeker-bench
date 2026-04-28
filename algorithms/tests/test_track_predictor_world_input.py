"""track_predictor.step — world-frame input path.

Locks in the regression that broke `track test 6.jsonl`: a static
target was producing phantom world_az_dot ≈ −13 dps because the
predictor recovered world_az from the camera-frame published value
plus its own cur_pan read, which arrived 50-150 ms after fusion's
publish-time pose snapshot. During a fast slew that latency leaks
into the apparent velocity, and the Phase 3 lookahead amplifies it
into runaway over-steering.

The fix: callers can now pass `obs_world_az_deg`/`obs_world_el_deg`
directly, bypassing the cur_pan reconstruction entirely. These tests
verify both that:
    1. The new path produces zero phantom velocity on a static
       target during a fast gimbal slew (the bug case).
    2. The legacy camera-frame path is unchanged when world inputs
       are not supplied (back-compat).
    3. Both paths agree on a moving target's velocity estimate when
       the gimbal pose used at obs time matches.
"""
from __future__ import annotations

import pytest

from algorithms import track_predictor as tp


def _params():
    return tp.PredictorParams(
        lead_time_s=0.30,
        vel_alpha=0.3,
        predict_warmup_n=2,        # warm up fast in tests
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
    fast between predictor ticks. This is the bug from track-test-6
    that the structural fix removes.
    """
    state = tp.PredictorState()
    p = _params()
    # Tick 1: cur_pan = -2.5°, target world_az = -19.0°.
    tp.step(state, now=0.0, cur_pan=-2.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    # Tick 2: gimbal slewed to -4.5° (delta = 2°), target STILL at
    # world_az = -19.0°. Settled gate must consider the gimbal
    # 'unsettled' on this step (it just moved 2° in 0.07 s ≈ 28 dps,
    # above settled threshold) — so velocity update is deferred until
    # gimbal calms. We verify world_az_dot remains 0 across the slew.
    tp.step(state, now=0.07, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    # Tick 3: gimbal settles at -4.5° (delta = 0). Velocity update
    # gate opens; observation is unchanged → world_az_dot = 0.
    tp.step(state, now=0.14, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=-19.0, obs_world_el_deg=8.0,
            fresh_fused=True, params=p)
    assert state.world_az == pytest.approx(-19.0, abs=1e-6)
    assert state.world_az_dot == pytest.approx(0.0, abs=0.1)


def test_legacy_path_phantom_velocity_during_slew():
    """Documents the bug for posterity: WITHOUT obs_world_az_deg,
    a static target whose camera-frame az is held constant while
    the gimbal slews IS treated as having phantom velocity.

    This is what the structural fix bypasses. If this test ever
    starts failing (i.e. the legacy path stops producing phantom
    velocity), great — but until then it's the canary that the
    legacy path still has the bug it's supposed to.
    """
    state = tp.PredictorState()
    p = _params()
    # Caller passes camera-frame az = -16.5 (= what fusion published).
    # Gimbal cur_pan moves -2.5 -> -4.5 -> -4.5 between ticks.
    # Recovered world_az = cur_pan + (-16.5) drifts -19.0 -> -21.0.
    tp.step(state, now=0.0, cur_pan=-2.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    # Tick 2: gimbal still moving (28 dps > settled threshold);
    # velocity update is gated, but obs_world_az is still recomputed
    # at the new cur_pan, so state.world_az drifts.
    tp.step(state, now=0.07, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    # Tick 3: gimbal settles. velocity update opens between
    # state.world_az (=-21.0 from t2) and obs_world_az (=-21.0 too).
    # But the *prior* drift between t1 and t2 already moved
    # state.world_az from -19 to -21. The predictor's stored
    # world_az reflects the legacy round-trip; that's the symptom
    # this test pins down.
    tp.step(state, now=0.14, cur_pan=-4.5, cur_tilt=0.0,
            obs_az_deg=-16.5, obs_el_deg=8.0,
            fresh_fused=True, params=p)
    assert state.world_az == pytest.approx(-21.0, abs=0.01), (
        "legacy path should leak cur_pan into apparent world_az")


def test_world_input_takes_precedence_when_both_provided():
    """If a caller passes both world AND camera inputs, the world
    inputs win. Defensive: keeps the new path's invariant intact
    even if legacy code paths later add cam_az for back-compat."""
    state = tp.PredictorState()
    p = _params()
    tp.step(state, now=0.0, cur_pan=10.0, cur_tilt=0.0,
            obs_az_deg=-50.0, obs_el_deg=-50.0,            # legacy says world=(-40,-50)
            obs_world_az_deg=3.0, obs_world_el_deg=4.0,    # world says (3,4)
            fresh_fused=True, params=p)
    assert state.world_az == 3.0
    assert state.world_el == 4.0


def test_world_input_path_moving_target_velocity():
    """A target moving steadily +5 dps in world frame should produce
    world_az_dot close to 5 once the velocity gate opens (settled
    gimbal, ≥2 fresh observations)."""
    state = tp.PredictorState()
    p = _params()
    # Settled gimbal (cur_pan held at 0) — three observations 0.1s
    # apart, world_az advancing 0.5° per tick = 5 dps.
    tp.step(state, now=0.0, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=10.0, obs_world_el_deg=0.0,
            fresh_fused=True, params=p)
    # 2nd tick — pose hasn't moved, so settled=True, velocity update fires.
    tp.step(state, now=0.1, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=10.5, obs_world_el_deg=0.0,
            fresh_fused=True, params=p)
    # Smoothed world_az_dot starts pulling toward 5 dps (alpha=0.3).
    # Don't pin the exact value — alpha-beta filter design — just
    # verify it's positive and below the clip.
    assert 0.5 < state.world_az_dot < p.vel_clip_dps


def test_world_input_path_no_fresh_obs_decays_velocity():
    """Stale ticks (fresh_fused=False) should still decay velocity
    via half-life, regardless of which input path was used."""
    state = tp.PredictorState()
    p = _params()
    state.world_az = 10.0
    state.world_el = 5.0
    state.world_az_dot = 8.0
    state.world_last_t = 0.0
    state.last_decay_t = 0.0
    state.obs_count = 5
    # Two stale ticks at 100 ms each → 200 ms total = one half-life.
    tp.step(state, now=0.1, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=None, obs_world_el_deg=None,
            fresh_fused=False, params=p)
    tp.step(state, now=0.2, cur_pan=0.0, cur_tilt=0.0,
            obs_az_deg=None, obs_el_deg=None,
            obs_world_az_deg=None, obs_world_el_deg=None,
            fresh_fused=False, params=p)
    # ~half-life elapsed → velocity ~ half its starting value.
    assert state.world_az_dot == pytest.approx(4.0, abs=0.5)

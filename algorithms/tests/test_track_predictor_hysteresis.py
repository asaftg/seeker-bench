"""track_predictor.step — settled-gate hysteresis.

Locks in the fix for the centered-target hunting observed on
`night tracking a bit flickery.jsonl` track #6: a noisy gimbal_dps
signal hovering near the single-threshold settled gate flipped the
gate at ~4 Hz, producing 20 settled<->unsettled transitions in 5 s.
The hysteresis splits the gate into upper (flip-out) and lower
(flip-in) thresholds so noise near the boundary doesn't toggle.

The settled gate is purely a function of gimbal_dps (not |err|)
because it gates the predictor's velocity-update path — i.e. when
the gimbal is moving fast enough to have unreliable observations,
freeze velocity learning. Hysteresis preserves that intent while
removing single-tick noise sensitivity.
"""
from __future__ import annotations

from algorithms import track_predictor as tp


def _params(hysteresis_ratio: float = 0.6, settled_dps: float = 8.0):
    return tp.PredictorParams(
        lead_time_s=0.30,
        vel_alpha=0.3,
        predict_warmup_n=3,
        gimbal_settled_dps=settled_dps,
        settled_hysteresis_ratio=hysteresis_ratio,
    )


def _step(state, params, *, now, cur_pan, cur_tilt, fresh=False,
          obs_az=None, obs_el=None):
    return tp.step(
        state, now=now, cur_pan=cur_pan, cur_tilt=cur_tilt,
        obs_az_deg=None, obs_el_deg=None,
        obs_world_az_deg=obs_az, obs_world_el_deg=obs_el,
        fresh_fused=fresh, params=params,
    )


def test_legacy_no_hysteresis_flips_at_threshold():
    """ratio=1.0 collapses to the legacy single-threshold gate."""
    state = tp.PredictorState()
    params = _params(hysteresis_ratio=1.0, settled_dps=8.0)

    # Walk gimbal_dps right at the threshold from a settled start.
    # Single-threshold means each tick decides independently.
    cur = 0.0
    fastpan = 0.30   # 0.30 deg / 0.030 s = 10 dps  -> unsettled
    slowpan = 0.18   # 0.18 deg / 0.030 s =  6 dps  -> settled
    seq = [fastpan, slowpan, fastpan, slowpan, fastpan, slowpan]
    settled_states = []
    for i, dpan in enumerate(seq):
        cur += dpan
        _, _, diag = _step(state, params, now=0.030 * (i + 1),
                           cur_pan=cur, cur_tilt=0.0)
        settled_states.append(diag["settled"])
    # Each tick flips: True/False/True/False/...  (after the first
    # tick which has no prior pose, gimbal_dps=0 so settled=True)
    transitions = sum(1 for a, b in zip(settled_states, settled_states[1:])
                      if a != b)
    # First tick: gimbal_dps=0 (no prior). Subsequent ticks alternate.
    # That gives at least 4 transitions in 6 ticks.
    assert transitions >= 4, f"legacy single-threshold expected to flip a lot; got {transitions}"


def test_hysteresis_holds_settled_through_borderline_noise():
    """ratio=0.6 means once settled, gimbal_dps must drop below
    8.0 * 0.6 = 4.8 dps to flip back; more importantly, transient
    spikes through the upper threshold while we're already settled
    cause exactly ONE flip-out, not a chain of flips."""
    state = tp.PredictorState()
    params = _params(hysteresis_ratio=0.6, settled_dps=8.0)

    # Borderline noise: gimbal_dps oscillates 6 dps ↔ 10 dps.
    # Single-threshold would flip every tick; hysteresis flips
    # once on the first 10 dps (8→10 crosses 8 upper), then must
    # drop below 4.8 dps to flip back. 6 dps doesn't cross 4.8.
    cur = 0.0
    seq_dps = [6.0, 10.0, 6.0, 10.0, 6.0, 10.0, 6.0]
    settled_states = []
    for i, dps in enumerate(seq_dps):
        # Move cur_pan by dps × 0.030 s per tick.
        cur += dps * 0.030
        _, _, diag = _step(state, params, now=0.030 * (i + 1),
                           cur_pan=cur, cur_tilt=0.0)
        settled_states.append(diag["settled"])
    transitions = sum(1 for a, b in zip(settled_states, settled_states[1:])
                      if a != b)
    # First tick gimbal_dps=0 → settled=True. Second tick 10 dps >
    # 8 → unsettled. Then 6 dps < 8 but NOT < 4.8 → still unsettled.
    # 10 dps → still unsettled. 6 dps → still unsettled. So 1 flip total.
    assert transitions <= 1, (
        f"hysteresis expected to suppress oscillation; got {transitions} "
        f"transitions in {settled_states}"
    )


def test_hysteresis_flips_back_when_truly_quiet():
    """If gimbal_dps drops well below the lower threshold, the
    gate must flip back to settled — hysteresis isn't a one-way
    latch, it's just a debounce."""
    state = tp.PredictorState()
    params = _params(hysteresis_ratio=0.6, settled_dps=8.0)

    cur = 0.0
    # Step 1: small motion, settled=True
    cur += 0.10  # 3.3 dps
    _, _, diag = _step(state, params, now=0.030, cur_pan=cur, cur_tilt=0.0)
    # Step 2: big motion, flip to unsettled
    cur += 0.45  # 15 dps
    _, _, diag = _step(state, params, now=0.060, cur_pan=cur, cur_tilt=0.0)
    assert not diag["settled"]
    # Step 3: still some motion above the lower threshold (5 dps > 4.8)
    cur += 0.16  # 5.3 dps
    _, _, diag = _step(state, params, now=0.090, cur_pan=cur, cur_tilt=0.0)
    assert not diag["settled"], "5.3 dps still above lower threshold; should stay unsettled"
    # Step 4: drop well below lower threshold
    cur += 0.05  # 1.7 dps < 4.8
    _, _, diag = _step(state, params, now=0.120, cur_pan=cur, cur_tilt=0.0)
    assert diag["settled"], "below lower threshold should flip back to settled"


def test_hysteresis_clamp_handles_out_of_range():
    """Negative or >1 ratios are config bugs. Implementation
    clamps to [0, 1] rather than crashing or producing weirdness."""
    state = tp.PredictorState()
    params = tp.PredictorParams(gimbal_settled_dps=8.0,
                                settled_hysteresis_ratio=-0.5)
    # Tick 0: no prior pose → gimbal_dps=0 → settled=True
    _, _, _ = _step(state, params, now=0.030,
                    cur_pan=0.0, cur_tilt=0.0)
    # Tick 1: 15 dps motion → unsettled (above upper threshold)
    _, _, diag = _step(state, params, now=0.060,
                       cur_pan=0.45, cur_tilt=0.0)
    assert not diag["settled"]
    # Tick 2: tiny motion (≈ 0.03 dps). Negative ratio clamped to 0
    # means lower threshold = 0 → flip-back requires gimbal_dps < 0
    # which is impossible. Verify we stay unsettled (== legacy
    # one-way latch) rather than crash.
    _, _, diag = _step(state, params, now=0.090,
                       cur_pan=0.451, cur_tilt=0.0)
    assert not diag["settled"]


def test_hysteresis_ratio_one_collapses_to_legacy_threshold():
    """Setting ratio=1.0 should be byte-equivalent to a single-
    threshold gate. Verify gimbal_dps = settled_dps - epsilon
    settles, gimbal_dps = settled_dps + epsilon doesn't, and
    we don't accidentally introduce a one-tick lag."""
    state = tp.PredictorState()
    params = tp.PredictorParams(gimbal_settled_dps=8.0,
                                settled_hysteresis_ratio=1.0)
    # Tick 0 establishes pose history
    _, _, _ = _step(state, params, now=0.030,
                    cur_pan=0.0, cur_tilt=0.0)
    # Tick 1: 7.5 dps (below 8) → settled
    _, _, diag = _step(state, params, now=0.060,
                       cur_pan=0.225, cur_tilt=0.0)
    assert diag["settled"]
    # Tick 2: 9 dps (above 8) → unsettled
    _, _, diag = _step(state, params, now=0.090,
                       cur_pan=0.225 + 0.27, cur_tilt=0.0)
    assert not diag["settled"]
    # Tick 3: 7.5 dps again (just below 8). With ratio=1.0
    # the lower threshold equals the upper, so this MUST flip
    # back to settled.
    _, _, diag = _step(state, params, now=0.120,
                       cur_pan=0.225 + 0.27 + 0.225, cur_tilt=0.0)
    assert diag["settled"]

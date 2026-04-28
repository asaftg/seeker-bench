"""
Pure-function tracking predictor.

This is the alpha-beta state estimator extracted from
``gimbal_manager._tick``. It is called by:

* the live manager every tick while a fused track is locked, and
* ``scripts/replay_algo.py`` against the captured JSONL stream.

Both paths must compute identical setpoints when fed identical
inputs — that's the parity test in the recorder verification plan.
The function is therefore deliberately pure: no module globals,
no I/O, no time.time() (the caller passes ``now``).

State is a small mutable dict the caller owns. ``step`` returns
``(new_sp_pan, new_sp_tilt, diag)`` where ``diag`` mirrors the
fields of the ``track_predictor_step`` event so the manager can
emit it directly.

Why this matters: live tracking failures repro within ~30 s of
physical setup but cost ~30 min per iteration; replay is millisec.
Iterating predictor knobs against captured data is the loop we're
buying with this whole recorder system.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


# ──────────────────────────────────────────────────────────────
# Tunable parameters. Match the names + defaults in
# config.gimbal so a YAML reload Just Works for both live + replay.
# ──────────────────────────────────────────────────────────────
@dataclass
class PredictorParams:
    lead_time_s: float = 0.30
    vel_alpha: float = 0.3              # alpha-beta beta; higher = snappier
    predict_warmup_n: int = 5
    predict_cap_deg: float = 5.0        # hard cap on per-axis predictive shift
    gimbal_settled_dps: float = 15.0    # below this dps = "settled"
    extrap_horizon_s: float = 2.0       # hold position past this gap
    # Sanity clip on raw observed velocity. Tightened 2026-04-26 from
    # 60→30: a single fresh observation arriving on a brief settle
    # between fast slews previously produced a 49 dps raw_dot that
    # smoothed into +17 dps and pushed the setpoint past the cap →
    # gimbal hunted blindly until grace expired. 30 dps is well above
    # any legitimate human/vehicle target rate at our typical ranges
    # but well below the latency-induced spike envelope.
    vel_clip_dps: float = 30.0
    # Half-life for velocity decay when fresh observations stop
    # arriving. Once `fresh_fused` is False for a tick, world_*_dot is
    # multiplied by exp(-dt / vel_decay_halflife_s) so the cached
    # velocity fades fast and the setpoint falls back to "hold last
    # observed position" rather than coasting on a stale spike.
    # Set to 0 (or negative) to disable decay (legacy behaviour).
    vel_decay_halflife_s: float = 0.20
    # Above this age (s since the last fresh observation), the
    # predictor's lead-time extrapolation is hard-zeroed. Decay alone
    # leaves a residual sp = world + vel*lead*decay term that visibly
    # hunts on the bench rig. Zeroing lead past this threshold makes
    # the gimbal hold the last observed position cleanly. Combined
    # with the decay above, the result is: lead is full + decaying
    # for the first 0.3 s, then OFF entirely until either a fresh
    # obs arrives or extrap_horizon_s ends the lock.
    no_obs_lead_zero_after_s: float = 0.30
    # Per-axis tilt saturation flags. The caller sets these True when
    # the gimbal is mechanically against the floor or ceil and the
    # observed el-axis error would push it further into the wall.
    # When set, the predictor zeroes el-velocity contributions and
    # holds sp_tilt at the saturated edge, while pan tracking
    # continues independently. Without this, the el-error keeps
    # accumulating in world_el_dot from observations the gimbal
    # physically can't follow, eventually contaminating the predict.
    # The fused-track caller in gimbal_manager toggles these every
    # tick based on cur_tilt + the incoming observation.
    tilt_saturated: bool = False


@dataclass
class PredictorState:
    """Mutable per-track state. The manager owns one; replay owns one."""
    world_az: Optional[float] = None
    world_el: Optional[float] = None
    world_az_dot: float = 0.0
    world_el_dot: float = 0.0
    world_last_t: Optional[float] = None
    obs_count: int = 0
    cur_pan_prev: Optional[float] = None
    cur_tilt_prev: Optional[float] = None
    cur_pose_prev_t: Optional[float] = None
    last_sp_pan: Optional[float] = None
    last_sp_tilt: Optional[float] = None
    # Wall-clock of the most recent velocity-decay tick. Distinct from
    # world_last_t (which only advances on fresh observations) — this
    # advances every step() call so the decay applies an *incremental*
    # multiplier per tick rather than re-applying the full age-factor
    # cumulatively.
    last_decay_t: Optional[float] = None

    def reset(self) -> None:
        self.world_az = None
        self.world_el = None
        self.world_az_dot = 0.0
        self.world_el_dot = 0.0
        self.world_last_t = None
        self.obs_count = 0
        self.cur_pan_prev = None
        self.cur_tilt_prev = None
        self.cur_pose_prev_t = None
        self.last_sp_pan = None
        self.last_sp_tilt = None
        self.last_decay_t = None


def step(
    state: PredictorState,
    *,
    now: float,
    cur_pan: float,
    cur_tilt: float,
    obs_az_deg: Optional[float],   # camera-frame az; None when no fresh obs
    obs_el_deg: Optional[float],
    fresh_fused: bool,
    params: PredictorParams,
) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """One predictor tick.

    Inputs:
        cur_pan/cur_tilt: gimbal pose right now (deg, world-frame)
        obs_az/el_deg: camera-frame angle of the target on the latest
            fused track. May be None on a dropped tick.
        fresh_fused: True when this tick has a NEW fused observation
            (i.e. the (id, hits) tuple changed since last call).
        params: PredictorParams snapshot — caller may swap variants
            between calls for A/B replay.

    Returns:
        (sp_pan, sp_tilt, diag) — sp_* are the world-frame setpoint
        the gimbal should slew toward. Either may be None when there's
        no track state yet (caller should fall back to current pose).
        ``diag`` is an event-shaped dict (see channel ``events`` /
        type ``track_predictor_step`` in recording/README.md).
    """
    # Estimate the gimbal's own angular velocity. Used for the velocity
    # gate that prevents latency-induced phantom velocity.
    gimbal_dps = 0.0
    if (state.cur_pan_prev is not None
            and state.cur_pose_prev_t is not None):
        pose_dt = max(0.001, now - state.cur_pose_prev_t)
        dpan_dt  = (cur_pan  - state.cur_pan_prev)  / pose_dt
        dtilt_dt = (cur_tilt - (state.cur_tilt_prev or cur_tilt)) / pose_dt
        gimbal_dps = (dpan_dt * dpan_dt + dtilt_dt * dtilt_dt) ** 0.5
    state.cur_pan_prev = cur_pan
    state.cur_tilt_prev = cur_tilt
    state.cur_pose_prev_t = now

    settled = gimbal_dps < params.gimbal_settled_dps

    obs_world_az: Optional[float] = None
    obs_world_el: Optional[float] = None
    if fresh_fused and obs_az_deg is not None and obs_el_deg is not None:
        obs_world_az = float(cur_pan)  + float(obs_az_deg)
        obs_world_el = float(cur_tilt) + float(obs_el_deg)
        # Velocity update — gated on settled.
        if (settled and state.world_az is not None
                and state.world_last_t is not None):
            dt = max(0.05, now - state.world_last_t)
            if dt < 1.0:
                new_az_dot = (obs_world_az - state.world_az) / dt
                new_el_dot = (obs_world_el - state.world_el) / dt
                clip = params.vel_clip_dps
                new_az_dot = max(-clip, min(clip, new_az_dot))
                new_el_dot = max(-clip, min(clip, new_el_dot))
                a = params.vel_alpha
                state.world_az_dot = ((1.0 - a) * state.world_az_dot
                                      + a * new_az_dot)
                # When tilt is saturated against a stop, ignore the
                # el-axis observation for velocity purposes — the
                # gimbal can't follow it, so accumulating world_el_dot
                # would contaminate the next non-saturated extrapolation.
                if not params.tilt_saturated:
                    state.world_el_dot = ((1.0 - a) * state.world_el_dot
                                          + a * new_el_dot)
                else:
                    state.world_el_dot = 0.0
        state.world_az = obs_world_az
        state.world_el = obs_world_el
        state.world_last_t = now
        state.obs_count += 1
    else:
        # No fresh observation this tick. Decay the cached velocity
        # toward 0 with the configured half-life so the predictor
        # doesn't coast on a stale spike (see vel_decay_halflife_s
        # docstring). Position estimate (state.world_az/el) is
        # unchanged — that's still the last-known location.
        # Decay is applied as an INCREMENT per step (not cumulative
        # against world_last_t) so 60 Hz manager ticks compound
        # correctly to the configured half-life.
        if (params.vel_decay_halflife_s > 0
                and state.last_decay_t is not None):
            dtick = max(0.0, now - state.last_decay_t)
            if dtick > 0.0:
                k = 0.5 ** (dtick / max(1e-6, params.vel_decay_halflife_s))
                state.world_az_dot *= k
                state.world_el_dot *= k
    # Update decay clock every tick (fresh or stale) so the next
    # stale tick computes its delta against this one.
    state.last_decay_t = now

    sp_pan: Optional[float] = None
    sp_tilt: Optional[float] = None
    age = None
    lead = 0.0
    confidence = 0.0
    shift_az = 0.0
    shift_el = 0.0
    if state.world_az is not None and state.world_last_t is not None:
        age = now - state.world_last_t
        if age < params.extrap_horizon_s:
            warmup_n = max(1, params.predict_warmup_n)
            confidence = min(1.0, state.obs_count / float(warmup_n))
            # Three gates on the lead-time extrapolation:
            #   * gimbal must be settled (else latency-induced phantom
            #     velocity from the world_az = cur_pan + obs_az frame
            #     transformation contaminates the predict)
            #   * obs must be fresh-ish (no_obs_lead_zero_after_s) — past
            #     that, the velocity estimate is too stale to trust
            #     and we hold the last observed position cleanly
            #   * confidence ramp via warmup_n
            if settled and age <= params.no_obs_lead_zero_after_s:
                lead = (age + params.lead_time_s) * confidence
            else:
                lead = 0.0
            cap = params.predict_cap_deg
            shift_az = max(-cap, min(cap, state.world_az_dot * lead))
            shift_el = max(-cap, min(cap, state.world_el_dot * lead))
            sp_pan  = state.world_az + shift_az
            sp_tilt = state.world_el + shift_el
        else:
            # Lock effectively dead — caller will hold the last setpoint.
            sp_pan  = state.last_sp_pan
            sp_tilt = state.last_sp_tilt

    if sp_pan is not None:
        state.last_sp_pan = sp_pan
    if sp_tilt is not None:
        state.last_sp_tilt = sp_tilt

    diag = {
        "now": float(now),
        "cur_pan": float(cur_pan),
        "cur_tilt": float(cur_tilt),
        "gimbal_dps": float(gimbal_dps),
        "settled": bool(settled),
        "fresh_fused": bool(fresh_fused),
        "obs_world_az": (float(obs_world_az)
                          if obs_world_az is not None else None),
        "obs_world_el": (float(obs_world_el)
                          if obs_world_el is not None else None),
        "world_az": (float(state.world_az)
                     if state.world_az is not None else None),
        "world_el": (float(state.world_el)
                     if state.world_el is not None else None),
        "world_az_dot": float(state.world_az_dot),
        "world_el_dot": float(state.world_el_dot),
        "obs_count": int(state.obs_count),
        "age": (float(age) if age is not None else None),
        "confidence": float(confidence),
        "lead": float(lead),
        "shift_az": float(shift_az),
        "shift_el": float(shift_el),
        "sp_pan":  (float(sp_pan)  if sp_pan  is not None else None),
        "sp_tilt": (float(sp_tilt) if sp_tilt is not None else None),
    }
    return sp_pan, sp_tilt, diag

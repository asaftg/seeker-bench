"""Gimbal orchestrator thread.

Mirrors the ThermalManager / EOManager pattern:

    loop at ~20 Hz:
        if tracked_target_id is set and alive → setpoint = (az, el) of that
            fused track (converted through any tilt offset)
        else                                   → setpoint = manual (pan, tilt)
        → controller.step() (clamp + slew)
        → driver.set_target_us() on pan + tilt channels
        → publish GimbalState on FrameBus

Never crashes:
    • no hardware → ``connected=False`` state published every tick,
      GUI greys out the dpad, rest of app unaffected.
    • bad config  → falls back to safe defaults with a WARN log.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional, Tuple

from algorithms import track_predictor
from common.config import load_config
from common.events import emit as emit_event
from common.frame_bus import BUS
from common.frames import EOFrame, GimbalState, ThermalFrame, Topic
from common.logging_setup import get_logger
from gimbal.gimbal_controller import (
    GimbalController,
    GimbalLimits,
    ServoCalibration,
)
from gimbal.bus_servo_calibration import BusServoCalibration
from gimbal.bus_servo_driver import BusServoDriver
from gimbal.maestro_driver import MaestroDriver
from gimbal.optical_residual import OpticalResidualTracker

log = get_logger(__name__)


@dataclass
class _HeatObs:
    """Internal az/el observation derived from a heat-blob bbox."""
    az_deg: float
    el_deg: float
    synthetic: bool = False  # True if the source is a user-drawn target
    # Raw thermal-pixel centre of the bbox this obs was computed from.
    # The heat-track control loop gates its setpoint update on whether
    # this centre actually moved between ticks — without that gate, a
    # stale (constant) cx/cy lets the manager apply kp*az multiple
    # times per OF update and the gimbal hunts.
    cx_px: Optional[float] = None
    cy_px: Optional[float] = None


def _clip(v: float, lim: float) -> float:
    """Symmetric clamp: constrain v to [-lim, +lim]."""
    if lim <= 0.0:
        return v
    if v >  lim: return  lim
    if v < -lim: return -lim
    return v


def _bbox_iou(a: Tuple[int, int, int, int],
              b: Tuple[int, int, int, int]) -> float:
    """Intersection-over-union of two (x, y, w, h) bboxes.
    Used by the lock-mode auto-reseed gate to decide whether a fresh
    fused observation matches the current lock bbox closely enough
    to refresh the appearance template."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def _deadband(az: float, el: float, band_deg: float) -> tuple[float, float]:
    """Zero the error inside a small band around the boresight."""
    if band_deg <= 0.0:
        return az, el
    if abs(az) < band_deg:
        az = 0.0
    if abs(el) < band_deg:
        el = 0.0
    return az, el


def _hyst_deadband(az: float,
                   el: float,
                   enter_band: float,
                   exit_ratio: float,
                   was_inside: bool) -> tuple[float, float, bool]:
    """Hysteretic deadband (Schmitt trigger).

    If currently *outside* the band, we stay outside until ``|err| <
    enter_band`` — then snap to zero. If currently *inside*, we stay
    inside (commanding no motion) until ``|err| > enter_band * exit_ratio``
    — only then do we release and respond to the error. Returns the
    (possibly zeroed) error plus the updated inside-band state.

    This is what finally makes a backlashed hobby servo settle: once
    close, we commit to "close enough" and stop chasing the last
    fraction of a degree of noise.

    NOTE 2026-04-25: superseded by ``_smooth_proportional`` for the
    track path. Hyst-deadband is binary (full kp outside, zero inside)
    which causes hunting at the boundary on close, slow-drift targets.
    Kept here in case future code wants a hard-cutoff settle behaviour
    (it's also still wired to the dev-mode heat-track fallback).
    """
    if enter_band <= 0.0:
        return az, el, False
    exit_band = enter_band * max(1.0, exit_ratio)
    mag_az, mag_el = abs(az), abs(el)
    mag = max(mag_az, mag_el)
    if was_inside:
        if mag < exit_band:
            return 0.0, 0.0, True
        return az, el, False
    else:
        if mag < enter_band:
            return 0.0, 0.0, True
        return az, el, False


def _smooth_proportional(err: float,
                         kp: float,
                         zero_band_deg: float,
                         full_band_deg: float) -> float:
    """Soft-deadband proportional gain — smooth replacement for the
    binary hysteretic deadband used by earlier track loops.

    Three regimes, all on a single P-controller (no integrator, no
    derivative, no PID juggle):

      |err| <= zero_band_deg          → output 0 (truly stationary
                                        when at target)
      zero_band_deg < |err| < full_band_deg
                                      → output ramps linearly from
                                        0 to kp*err (gentle approach
                                        when close)
      |err| >= full_band_deg          → output = kp * err (full
                                        proportional gain when far)

    Why this beats hyst_deadband for tracking close foreground
    targets: the binary deadband had a cliff at the boundary —
    when the target's centroid jitters across the threshold, the
    output snaps between 0 and full kp*err, which the operator sees
    as "twitchy hunting around the centre". The linear ramp turns
    that cliff into a smooth gradient: tiny excursions across
    the inner band produce tiny commands (which then clip via
    max_step_deg or simply get absorbed by servo resolution),
    while real off-center errors still drive full-speed correction.

    Tunables (config: gimbal.track_zero_band_deg / track_full_band_deg):
      * zero_band_deg ≈ servo angular resolution + sensor jitter floor
        (~0.2° on this rig — below this, commanding motion just
        wastes effort).
      * full_band_deg ≈ "this much error means we should respond
        urgently" (~1.0° matches a small target visibly off-centre
        on the EO panel).

    No state across calls — pure function. The lp_filter upstream
    handles temporal smoothing; this handles spatial gain shaping.
    """
    if full_band_deg <= zero_band_deg:
        # Misconfigured — collapse to hard deadband.
        return 0.0 if abs(err) <= zero_band_deg else kp * err
    a = abs(err)
    if a <= zero_band_deg:
        return 0.0
    if a >= full_band_deg:
        return kp * err
    # Linear ramp: scale grows from 0 (at zero_band) to 1 (at full_band).
    span = full_band_deg - zero_band_deg
    scale = (a - zero_band_deg) / span
    return kp * err * scale


def _pan_only_if_tilt_saturated(cur_tilt: float,
                                d_tilt: float,
                                tilt_floor: float,
                                tilt_ceil: float,
                                eps: float) -> tuple[float, bool]:
    """Zero the tilt delta when the servo is at a mechanical stop and
    the controller wants to drive it further into that stop.

    Returns (d_tilt_out, saturated). When saturated, the caller keeps
    the pan command (visual servo still tracks horizontally) but stops
    spending control effort on tilt — which would otherwise produce
    micro-jitter via clamp-and-retry each frame.
    """
    at_floor = cur_tilt <= tilt_floor + eps
    at_ceil  = cur_tilt >= tilt_ceil  - eps
    if at_floor and d_tilt < 0.0:
        return 0.0, True
    if at_ceil and d_tilt > 0.0:
        return 0.0, True
    return d_tilt, False


class GimbalManager:
    """Pan/tilt servo manager.

    External API (all thread-safe):
        start() / stop()
        set_manual_delta(d_pan_deg, d_tilt_deg)   — GUI arrow buttons
        set_home()                                — center pan, middle tilt
        set_track_target(track_id | None)         — engage / release auto-track
    """

    def __init__(self, port: Optional[str] = None) -> None:
        cfg = load_config()
        gcfg = cfg.get("gimbal", {}) or {}

        # ── Driver selection ────────────────────────────────
        #   maestro            — V1 Pololu Maestro + hobby servos (PWM µs)
        #   waveshare_st3025   — V2 Waveshare bus-servo adapter + ST3025
        self._driver_kind = str(gcfg.get("driver", "maestro")).lower()
        self._is_v2 = (self._driver_kind == "waveshare_st3025")

        # ── Calibration (per-servo) ─────────────────────────
        if self._is_v2:
            ws_cfg = (gcfg.get("waveshare") or {})
            pcal_cfg = (ws_cfg.get("pan")  or {})
            tcal_cfg = (ws_cfg.get("tilt") or {})
            self._pan_cal = BusServoCalibration(
                servo_id=int(pcal_cfg.get("servo_id", 2)),
                zero_raw=int(pcal_cfg.get("zero_raw", 2048)),
                invert=bool(pcal_cfg.get("invert", False)),
                raw_min=int(pcal_cfg.get("raw_min", 1365)),
                raw_max=int(pcal_cfg.get("raw_max", 2731)),
            )
            self._tilt_cal = BusServoCalibration(
                servo_id=int(tcal_cfg.get("servo_id", 1)),
                zero_raw=int(tcal_cfg.get("zero_raw", 1024)),
                invert=bool(tcal_cfg.get("invert", False)),
                raw_min=int(tcal_cfg.get("raw_min", 853)),
                raw_max=int(tcal_cfg.get("raw_max", 2048)),
            )
            lim_pan_lo, lim_pan_hi = -60.0,  60.0
            lim_tilt_lo, lim_tilt_hi = -15.0, 90.0
        else:
            pcal_cfg = (gcfg.get("pan_calibration")  or {})
            tcal_cfg = (gcfg.get("tilt_calibration") or {})
            self._pan_cal = ServoCalibration(
                channel=int(pcal_cfg.get("channel", 0)),
                min_deg=float(pcal_cfg.get("min_deg", -90.0)),
                max_deg=float(pcal_cfg.get("max_deg",  90.0)),
                us_at_min_deg=float(pcal_cfg.get("us_at_min_deg", 500.0)),
                us_at_max_deg=float(pcal_cfg.get("us_at_max_deg", 2500.0)),
                invert=bool(pcal_cfg.get("invert", False)),
            )
            self._tilt_cal = ServoCalibration(
                channel=int(tcal_cfg.get("channel", 1)),
                min_deg=float(tcal_cfg.get("min_deg",  0.0)),
                max_deg=float(tcal_cfg.get("max_deg", 22.0)),
                us_at_min_deg=float(tcal_cfg.get("us_at_min_deg", 1500.0)),
                us_at_max_deg=float(tcal_cfg.get("us_at_max_deg", 2000.0)),
                invert=bool(tcal_cfg.get("invert", False)),
            )
            lim_pan_lo, lim_pan_hi = self._pan_cal.min_deg, self._pan_cal.max_deg
            lim_tilt_lo, lim_tilt_hi = self._tilt_cal.min_deg, self._tilt_cal.max_deg

        lims_cfg = (gcfg.get("limits") or {})
        limits = GimbalLimits(
            pan_min_deg=float(lims_cfg.get("pan_min_deg",  lim_pan_lo)),
            pan_max_deg=float(lims_cfg.get("pan_max_deg",  lim_pan_hi)),
            tilt_min_deg=float(lims_cfg.get("tilt_min_deg", lim_tilt_lo)),
            tilt_max_deg=float(lims_cfg.get("tilt_max_deg", lim_tilt_hi)),
            pan_slew_deg_per_s=float(lims_cfg.get("pan_slew_deg_per_s", 120.0)),
            tilt_slew_deg_per_s=float(lims_cfg.get("tilt_slew_deg_per_s", 60.0)),
        )

        home_pan  = float(gcfg.get("home_pan_deg", 0.0))
        # Middle of the 0..22 tilt envelope by default
        home_tilt = float(gcfg.get("home_tilt_deg",
                                   (limits.tilt_min_deg + limits.tilt_max_deg) / 2.0))

        # V1 needs pan_cal/tilt_cal for `angles_to_us`; V2 doesn't call
        # that helper but passing them through is harmless.
        self._controller = GimbalController(
            pan_cal=self._pan_cal if not self._is_v2 else None,
            tilt_cal=self._tilt_cal if not self._is_v2 else None,
            limits=limits,
            home_pan_deg=home_pan,
            home_tilt_deg=home_tilt,
        )

        # Stash pan limits for the pan_saturated event detection later.
        # Tilt floor/ceil already live on self in the existing tilt-sat
        # path; pan was previously silent — adding for symmetry so the
        # JSONL stream documents both axes.
        self._pan_floor = float(limits.pan_min_deg)
        self._pan_ceil  = float(limits.pan_max_deg)
        self._pan_saturated_logged = False

        self._home_pan  = home_pan
        self._home_tilt = home_tilt
        self._rate_hz   = float(gcfg.get("rate_hz", 20.0))

        # Mechanical tilt envelope — used by the pan-only saturation
        # guard so we stop feeding phantom tilt corrections when the
        # target is outside the reachable pitch range. Without this,
        # a ground target below tilt_min causes the control loop to
        # command tilt deltas every frame that get clamped at the
        # floor, producing visible jitter in pan via coupled error
        # readout.
        self._tilt_floor = float(limits.tilt_min_deg)
        self._tilt_ceil  = float(limits.tilt_max_deg)
        # Small epsilon so we treat "within 0.3° of the stop" as saturated.
        self._tilt_sat_eps_deg = 0.3

        # Mounting geometry for tracking math.
        #
        #   cameras_on_gimbal: true  → cameras move with the gimbal. Fused
        #     az/el is off-boresight error; command = current + error, so
        #     as the gimbal rotates the target pixel drifts to center and
        #     the loop settles.
        #
        #   cameras_on_gimbal: false → bench setup: cameras are stationary
        #     on a tripod, gimbal is a separate rig. Fused az/el is a
        #     fixed bench-frame angle; command = az/el directly (plus a
        #     tilt offset so el=0 corresponds to home_tilt). The previous
        #     formula *integrated* a non-decreasing error every tick and
        #     slewed straight into the mechanical limits.
        self._cameras_on_gimbal = bool(gcfg.get("cameras_on_gimbal", False))

        # Control-law tuning for the cameras_on_gimbal=true path.
        #
        # The loop is a visual-servo with real latency we can't hide:
        # camera capture + WebSocket + gimbal IO ≈ 50–100 ms, and the
        # Maestro servos themselves lag the command by another few tens
        # of ms. That means each tick's az/el error reflects the scene
        # BEFORE the last couple of commands have physically happened.
        # Any kp that doesn't account for that pipeline lag integrates
        # stale error → overshoot → ring. Empirically kp=0.3 still rings
        # hard; kp=0.1 settles.
        #
        #   kp_track      — proportional gain on the off-boresight error
        #   deadband_deg  — error band where setpoint is frozen; kills
        #                   the limit-cycle from backlash + stale feedback
        #   max_step_deg  — hard rate limit on setpoint delta per tick,
        #                   a second safety net on top of the controller
        #                   slew limit (belt + suspenders)
        # Tuning reality: hobby servo gearing has visible backlash
        # (~1–2° of slack before a direction reversal produces motion),
        # and the vision pipeline has 2–3 frames of latency. A tight
        # deadband with any non-trivial kp therefore limit-cycles
        # forever around center. Widening the deadband until it covers
        # the backlash, and keeping kp low enough that any single
        # correction is smaller than what the residual error can
        # tolerate, is what finally makes it settle.
        # max_step_deg: per-tick setpoint-delta cap. Exists to prevent
        # a huge initial error from commanding a servo slew so large
        # that the next 2–3 frames of visual feedback all describe an
        # in-flight correction and the loop pile-drives through center.
        # At 60 Hz loop + ~32 Hz feedback, a 2.0°/tick cap keeps the
        # in-flight queue short enough while closing a 20° error in
        # ~0.3 s (vs 0.8 s at 0.8°/tick). Still much slower than the
        # controller's own 120°/s slew limit — belt + suspenders.
        self._kp_track       = float(gcfg.get("kp_track", 0.15))
        self._deadband_deg   = float(gcfg.get("deadband_deg", 2.5))
        self._max_step_deg   = float(gcfg.get("max_step_deg", 2.0))
        # D term on measurement (encoder velocity). Subtracts kd × dpose/dt
        # from d_pan_cl / d_tilt_cl in the closed-loop, which acts as a
        # brake when the gimbal is moving fast toward target → smooth
        # deceleration, no overshoot. Operates on MEASUREMENT (encoder
        # velocity) rather than ERROR derivative so setpoint changes
        # don't kick the controller. 0.0 = pure-P (legacy). Sensible
        # range: 0.01 – 0.05 with V2 encoder; bigger values can fight
        # the P term and slow convergence.
        # `tracking 532026 try3.jsonl` track #1 transit showed 0.6° peak
        # overshoot + reverse — the canonical use case for this knob.
        self._kd_track       = float(gcfg.get("kd_track", 0.02))
        # World-target smoothing alpha for the absolute-target
        # closed-loop. Each fresh fused observation contributes this
        # weight to the running target; previous value gets (1−alpha).
        # Lower = smoother (less obs-jitter passes through to gimbal),
        # higher = snappier response to actual target motion.
        # Try-4 transit analysis showed YOLO obs jumping by 2.5° in
        # one fresh-obs interval (cluster centroid hopping between
        # detections); 0.30 attenuates that to ~0.75° per smoothed
        # update, then the controller's slew rate handles the rest
        # smoothly. For fast-moving targets the lookahead term
        # (lead_time × world_az_dot) compensates for the ~3-tick
        # smoothing lag.
        self._track_target_lp_alpha = float(
            gcfg.get("track_target_lp_alpha", 0.30))
        # Switch: legacy proportional closed-loop vs new absolute-target
        # closed-loop. Default to the new path for V2 (encoder feedback
        # makes the absolute path safe). Set false to fall back to
        # cur_pan + d_pan_cl + ff if the absolute path needs more tuning.
        self._track_use_absolute_target = bool(
            gcfg.get("track_use_absolute_target", True))

        # Error-signal low-pass: hot targets like vehicles don't have a
        # single crisp centroid. Headlights, grille, engine bay, wheel
        # wells all flicker as separate blobs that merge and split frame
        # to frame; the resulting bbox centroid can jitter several
        # degrees of off-boresight angle. Feeding that raw into the
        # controller multiplies the jitter by kp and pumps the servo.
        # A first-order IIR on (az, el) with alpha ~0.35 smooths
        # high-frequency centroid noise while still reacting in ~3–4
        # frames to real target motion. Zero disables filtering.
        self._err_lp_alpha   = float(gcfg.get("err_lp_alpha", 0.35))
        self._err_lp_az: Optional[float] = None
        self._err_lp_el: Optional[float] = None

        # Sticky deadband: once inside the band, require the error to
        # exceed a LARGER threshold before re-engaging. This is a
        # Schmitt-trigger-style hysteresis that prevents the edge case
        # where noise rattles the error value just above/below the
        # band limit every frame. Enter @ deadband_deg, leave @
        # deadband_deg * exit_ratio.
        self._deadband_exit_ratio = float(gcfg.get("deadband_exit_ratio", 1.5))
        self._in_deadband = False  # track hysteresis state across ticks

        # Smooth-proportional band parameters. Replace the binary
        # deadband on the fused-track path: tiny stationary error
        # → no output; small error → ramped output; large error →
        # full kp gain. See _smooth_proportional() docstring for the
        # full rationale and shape. Set track_zero_band_deg ==
        # track_full_band_deg to collapse back to a hard deadband.
        self._track_zero_band_deg = float(
            gcfg.get("track_zero_band_deg", 0.20))
        self._track_full_band_deg = float(
            gcfg.get("track_full_band_deg", 1.00))
        # Output minimum threshold: if the proposed delta after the
        # smooth-proportional law is smaller than this, treat it as 0.
        # ~0.05° matches the servo's effective angular resolution
        # under this calibration; commanding finer-than-that just
        # wastes a USB write and risks the motor's PWM dither
        # turning into audible buzz on a static target.
        self._track_min_step_deg = float(
            gcfg.get("track_min_step_deg", 0.05))

        # Visual-servo gating: the control loop runs at 60 Hz but the
        # camera only feeds ~30 Hz. If we re-command every tick we end
        # up firing corrections twice per feedback sample → guaranteed
        # oscillation. Track the last-seen frame timestamp and only
        # recompute the tracking setpoint when a NEW frame has arrived;
        # stale-frame ticks just let the servo keep slewing toward the
        # last setpoint undisturbed.
        self._last_track_ts: Optional[float] = None
        self._last_sp_pan:   Optional[float] = None
        self._last_sp_tilt:  Optional[float] = None
        # Fused-track feedback freshness counter. Distinct from
        # _last_track_ts (which is keyed on the THERMAL frame at ~60 Hz).
        # FusionManager publishes at ~15 Hz, so 3-4 thermal ticks pass
        # between fused-track updates. If we re-applied kp*az on every
        # thermal tick we'd over-correct 3-4x per fusion cycle and the
        # gimbal would walk 20-40° past the target before the feedback
        # caught up (operator-reported 2026-04-25 "+30° overshoot").
        # Now we recompute the fused-track setpoint only when the fused
        # track's `hits` counter advances, i.e. when a sensor actually
        # contributed a new observation. Between updates we hold the
        # last setpoint and let the controller's slew limit finish the
        # in-flight motion. Tracks the (id, hits) tuple so a re-born
        # track with the same id and lower hits also counts as fresh.
        self._last_fused_track_hits: Optional[tuple[int, int]] = None

        # ── Track velocity / dead-reckoning state ──
        # Operator-reported (2026-04-26): "for a moving vehicle, the
        # moment I move the gimbal, the radar stops picking the target
        # because it's on the gimbal." Same applies to EO/thermal at
        # narrow zoom: the moment the camera slews aggressively, the
        # YOLO/heat detection breaks lock for a few hundred ms while
        # the new field of view stabilises. With pure proportional
        # control (sp = cur + kp*az) the gimbal stops at the last
        # observed position while the real target keeps moving --
        # by the time detection comes back the target is gone.
        #
        # Fix is a constant-velocity predictor (alpha-beta tracker):
        # we maintain the target's WORLD-frame position + velocity in
        # az/el space, updating both on every fresh fused track
        # observation. Between observations the predictor extrapolates
        # at the last-known velocity, and the setpoint always points a
        # `lead_time` seconds ahead of the predicted current position.
        # This is a state estimator, NOT a PID controller -- there is
        # no integral and no error-derivative term, so it doesn't
        # have the windup / noise-amplification issues operator
        # explicitly wanted to avoid.
        self._track_world_az: Optional[float] = None
        self._track_world_el: Optional[float] = None
        self._track_world_az_dot: float = 0.0
        self._track_world_el_dot: float = 0.0
        self._track_world_last_t: Optional[float] = None
        # Number of fresh observations seen since track engagement.
        # The predictor's effective lead time is scaled by this so
        # the first noisy velocity sample (heavily contaminated by
        # gimbal motion + sensor lag during the first slew) never
        # gets multiplied by the full lead. 0 -> no prediction;
        # ramps to 1.0 over `_track_predict_warmup_n` observations.
        self._track_obs_count: int = 0
        self._track_predict_warmup_n: int = int(
            gcfg.get("track_predict_warmup_n", 5))
        # Lookahead horizon: gimbal aims `lead_time` seconds ahead of
        # the current predicted target position. 0.3 s matches the
        # physical slew time for typical 5-15 deg corrections at our
        # ~120 deg/s pan slew rate, so the gimbal arrives roughly when
        # the target is there.
        self._track_lead_time_s: float = float(
            gcfg.get("track_lead_time_s", 0.30))
        # Velocity smoothing alpha (alpha-beta tracker beta term).
        # Lower = smoother (lags fast accelerations), higher = noisier
        # but more responsive. 0.3 = strong smoothing, picked because
        # the first 1-2 observations are heavily corrupted by gimbal
        # motion and radar lag; we'd rather trust accumulated history
        # than a fresh measurement. Bump to 0.5 for snappier response
        # on truly fast-accelerating targets if needed.
        self._track_vel_alpha: float = float(
            gcfg.get("track_vel_alpha", 0.3))
        # Hard cap on the predictive setpoint shift per axis (deg).
        # Even with full lead time and a high velocity estimate, we
        # never push the setpoint more than this far ahead of the
        # observed target position. Prevents runaway when a phantom
        # velocity slips past the smoothing.
        self._track_predict_cap_deg: float = float(
            gcfg.get("track_predict_cap_deg", 5.0))

        # Gimbal-velocity gate. The world-frame position estimate
        # `world_az = cur_pan + obs_az` is only correct when cur_pan
        # and obs_az are sampled at the same instant. In reality
        # `obs_az` is measured by the sensor at time t_sensor and
        # arrives at the manager at time t_now, with ~100-300 ms of
        # processing+fusion latency in between. During that gap the
        # gimbal has moved by `gimbal_velocity × latency`. At 60 deg/s
        # slew and 200 ms latency, that's 12 deg of error PER
        # observation, which feeds straight into the velocity
        # differentiator and produces a phantom 60 deg/s "target
        # velocity" that didn't exist. The hard cap and warmup ramp
        # only mitigate -- they don't break the positive-feedback
        # loop because the phantom velocity is SUSTAINED during the
        # slew (8+ cycles at the cap = 40 deg of drift, matches the
        # operator-reported runaway 2026-04-26).
        # Gate: only update the velocity estimate (and only apply
        # lead-time extrapolation) when the gimbal is approximately
        # stationary. Position update still runs freely (bounded
        # error during slew, converges as gimbal arrives). 15 deg/s
        # threshold is chosen so normal small tracking corrections
        # (typically 1-5 deg/s) still register as "settled" while
        # genuine slews (60-120 deg/s) freeze the predictor.
        self._track_gimbal_settled_dps: float = float(
            gcfg.get("track_gimbal_settled_dps", 15.0))
        # Previous gimbal pose snapshot, used to estimate the
        # gimbal's own angular velocity each tick.
        self._cur_pan_prev: Optional[float] = None
        self._cur_tilt_prev: Optional[float] = None
        self._cur_pose_prev_t: Optional[float] = None
        # How long to keep extrapolating after observations stop.
        # Beyond this we accept the lock is dead and hold position.
        # 2.0 s is enough for the gimbal to slew through ~240 deg of
        # accumulated motion (well past the mechanical envelope) and
        # for the radar/EO/thermal to re-acquire a new track id, but
        # short enough that we don't drift forever.
        self._track_extrap_horizon_s: float = float(
            gcfg.get("track_extrap_horizon_s", 2.0))
        # Velocity sanity clip (deg/sec). Targets faster than this in
        # angular space are likely a fusion ID swap (jumped from one
        # vehicle to another) -- don't propagate that velocity.
        self._track_vel_clip_dps: float = float(
            gcfg.get("track_vel_clip_dps", 30.0))
        # Velocity decay half-life for the no-fresh-obs case. Default
        # 0.2 s makes the cached predictor velocity fade fast once
        # observations stop, preventing the runaway-by-stale-spike
        # pattern observed 2026-04-26 on the radar+EO freak-out.
        self._track_vel_decay_halflife_s: float = float(
            gcfg.get("track_vel_decay_halflife_s", 0.20))
        # Hard-zero the predictor lead past this age-since-fresh-obs.
        # Replay (2026-04-27) showed decay alone leaves visible
        # hunting (~3° amplitude shrinking over 1+ s); zeroing lead
        # at 0.3 s collapses the hunt to ~0° immediately while still
        # giving the predictor 0.3 s of lead-aided convergence on a
        # genuinely fresh track.
        self._track_no_obs_lead_zero_after_s: float = float(
            gcfg.get("track_no_obs_lead_zero_after_s", 0.30))
        # Hysteresis ratio for the predictor's settled gate. See
        # PredictorParams.settled_hysteresis_ratio in
        # algorithms/track_predictor.py and YAML
        # gimbal.track_settled_hysteresis_ratio for the rationale.
        self._track_settled_hysteresis_ratio: float = float(
            gcfg.get("track_settled_hysteresis_ratio", 0.6))
        # Tracking-mode slew cap. When the manager is ENGAGED on a
        # fused track, the controller's per-tick step is clamped to
        # this dps regardless of how big d_pan_cl + ff_az_deg is. The
        # default 18 dps lands EO motion blur at ~0.7° per 40 ms
        # exposure (~80 px on the 1236-wide display), which YOLO
        # comfortably recovers from. Without this cap the V2 servo's
        # native ~35-40 dps mechanical max produces ~1.4° / 160 px
        # of EO blur per frame and YOLO loses the bbox mid-slew —
        # operator-reported in `gimbal moves too fast loses bbs.jsonl`.
        # 0 or negative disables (legacy behaviour: only the broader
        # `gimbal.controller.pan_slew_deg_per_s` limit applies).
        # Manual dpad / HOME slews are NOT capped — they don't run
        # through this branch.
        self._track_slew_cap_dps: float = float(
            gcfg.get("track_slew_cap_dps", 18.0))

        # Phase 3 — confidence gate for the velocity feed-forward in
        # the closed-loop fused-track tick branch. Only apply
        # `lead_time * world_dot` lookahead when the smoothed velocity
        # is above this threshold on that axis. Below threshold the
        # axis is treated as static and we rely on the closed-loop's
        # pixel-error correction alone (no lookahead noise on a static
        # target). Tuned to be just above the alpha-beta filter's
        # noise floor on YOLO bbox jitter (~0.5°/s on a 5 m target).
        self._track_lookahead_min_dps: float = float(
            gcfg.get("track_lookahead_min_dps", 1.0))

        # Pure-function predictor (algorithms.track_predictor) — both
        # the live tick AND scripts/replay_algo.py call into this one
        # implementation. The state object is owned by this manager;
        # the parameter snapshot is rebuilt every tick so a YAML reload
        # propagates without a restart.
        self._predictor_state = track_predictor.PredictorState()
        self._last_settled_state: Optional[bool] = None

        # World-frame angle a synthetic-target lock has committed to.
        # The synthetic_target / draw-target loop is fundamentally a
        # ONE-SHOT system: the user clicked on something, we compute
        # where in world space that something is (cur_pan + az at lock
        # time), and slew the gimbal to that absolute angle. The OF
        # tracker that's supposed to keep the bbox following the target
        # as the camera moves is unreliable in practice (lag, drift),
        # so we don't trust subsequent az readings to refine the
        # estimate — they'd just send the gimbal hunting. World-frame
        # az/el is cached here on the FIRST fresh observation and
        # never updated afterwards (the user can re-draw the target
        # for a new lock). Set to None when no synthetic lock is
        # active. Verified 2026-04-25 — replaces an iterative
        # closed-loop that produced unbounded oscillation when OF
        # lagged.
        self._synth_world_az_deg: Optional[float] = None
        self._synth_world_el_deg: Optional[float] = None
        # Wall-clock at which the synthetic-track auto-lock was
        # committed. Used to dampen the gimbal slew during the first
        # ~0.6 s post-draw so OF features have time to track without
        # the per-frame scene motion exceeding LK's tracking limit.
        # Without this, the bbox visibly drifts off the user's chosen
        # target during the initial slew (verified offline 2026-04-27
        # via NCC=−0.087 between draw-time and post-slew bbox content
        # in scripts/replay_of.py against the tree recording).
        self._synth_lock_t: Optional[float] = None
        self._synth_slew_window_s: float = float(
            gcfg.get("synth_slew_window_s", 0.6))
        self._synth_slew_dps: float = float(
            gcfg.get("synth_slew_dps", 25.0))

        # Optical-residual trackers. One per sensor. EO is the primary
        # source — higher resolution + more texture than thermal.
        # Thermal is the fallback when EO loses features (e.g. low-light,
        # target outside narrow EO FOV after a big slip). Anchors are
        # captured at synth-lock commit; reset on track release.
        #
        # Stage A (always on): per-tick measurement is emitted as the
        # `optical_residual` event. No effect on the gimbal.
        # Stage B (gated by `optical_correction_enabled`): the residual
        # drives an integrator that biases the synth-lock setpoint to
        # close the loop visually. Sign convention:
        #     residual = cmd - actual    (positive = camera fell short
        #                                  in the commanded direction)
        # so the correction is added to the target setpoint.
        self._opt_eo = OpticalResidualTracker(name="eo")
        self._opt_thermal = OpticalResidualTracker(name="thermal")
        self._opt_last_eo_frame_id: Optional[int] = None
        self._opt_last_thermal_frame_id: Optional[int] = None

        # Stage B config + state. Default is OFF so existing behaviour
        # is unchanged. Flip `optical_correction_enabled: true` in
        # config/app_config.yaml to A/B against the open-loop baseline.
        self._opt_corr_enabled: bool = bool(
            gcfg.get("optical_correction_enabled", False))
        self._opt_corr_alpha: float = float(
            gcfg.get("optical_correction_alpha", 0.2))
        self._opt_corr_max_deg: float = float(
            gcfg.get("optical_correction_max_deg", 5.0))
        self._opt_corr_min_features: int = int(
            gcfg.get("optical_correction_min_features", 12))
        self._opt_corr_warmup_s: float = float(
            gcfg.get("optical_correction_warmup_s", 1.0))
        # Per-tick step limit on the cumulative correction. Without
        # this, alpha * large_residual on the first post-warmup tick
        # can jump correction by several degrees instantly, which
        # the synth-OF tracker can't follow (visible bbox drift).
        self._opt_corr_step_max_deg: float = float(
            gcfg.get("optical_correction_step_max_deg", 0.3))
        # "Settled" gate: only update integrator when the controller
        # is close to its current setpoint (gimbal has substantially
        # completed the slew). Prevents the integrator from interpreting
        # in-progress slew motion as undershoot.
        self._opt_corr_settled_deg: float = float(
            gcfg.get("optical_correction_settled_deg", 0.7))

        # Stuck-servo safety. When a synth lock is active and the LK-
        # measured camera motion stays well below the commanded delta
        # for several seconds, the servo isn't physically responding
        # (operator hypothesis: PSU sag under simultaneous pan+tilt
        # load + servo internal current limit). In that state our SW
        # keeps issuing PWM commands the servo can't follow, which
        # could damage the servo. Release the servos to float and
        # clear the lock, emitting `servo_stuck` so the operator
        # sees what happened.
        self._stuck_enabled: bool = bool(
            gcfg.get("stuck_servo_protection", True))
        self._stuck_warmup_s: float = float(
            gcfg.get("stuck_servo_warmup_s", 3.0))
        self._stuck_required_min_deg: float = float(
            gcfg.get("stuck_servo_required_min_deg", 2.0))
        self._stuck_residual_frac: float = float(
            gcfg.get("stuck_servo_residual_frac", 0.7))
        self._stuck_consec_threshold: int = int(
            gcfg.get("stuck_servo_consec_threshold", 8))
        self._stuck_consec: int = 0
        self._stuck_released: bool = False

        # Cache the latest target-residual (where the world target
        # actually is in the current camera frame, per LK). Published
        # in GimbalState so sensor_bridge can render the synthetic
        # bbox at the target's true image position.
        self._latest_target_resid_az: Optional[float] = None
        self._latest_target_resid_el: Optional[float] = None

        # Fused-track unified control law. Default True (per operator
        # ask: same algorithm for static and dynamic targets). The
        # fused-track tick branch uses the same closed-loop pixel-error
        # proportional control as the real-heat-blob path (`fresh_heat`
        # branch in `_tick`):
        #     d_pan  = smooth_proportional(trk.az_deg)
        #     d_tilt = smooth_proportional(trk.el_deg)
        # Setpoint = cur_pan + d_pan, cur_tilt + d_tilt.
        # For static targets, az/el converges to 0 as the camera
        # centres → setpoint stops moving → identical end-state to
        # the synth-lock manual-BB path. For moving targets, every
        # fresh fused observation refreshes az/el and the loop
        # corrects toward the new position. No mode switching, no
        # velocity-based gating.
        # The alpha-beta predictor (track_predictor) still runs in
        # parallel for diagnostics and replay parity, but its
        # lookahead output (`world + lead_time*world_dot`) is no
        # longer fed to the controller in this mode. Lookahead helps
        # only for very fast targets (relative to gimbal latency)
        # and contributes lead-time*velocity-noise on slower ones —
        # the noise is what made TRACK feel less stable than the
        # manual BB.
        # Set to False to use the legacy predictor-driven setpoint.
        self._fused_closed_loop: bool = bool(
            gcfg.get("fused_track_closed_loop", True))
        # Cumulative correction applied to the synth-target world angle.
        # Reset on lock release.
        self._opt_corr_az: float = 0.0
        self._opt_corr_el: float = 0.0
        # Track which source contributed last update for the event log.
        self._opt_corr_last_source: str = ""

        # Heat-track (synthetic_target / draw-target) freshness signal.
        # Same shape of bug as fused tracks had, different cause:
        # the OF tracker that propagates a synthetic bbox is much
        # slower than the gimbal can slew, so cx/cy in the thermal
        # frame stays nearly constant for a few thermal ticks while
        # the gimbal rotates 5-10°. Manager runs at 60 Hz, applies
        # kp*az every tick using the SAME stale (cx, cy) — gimbal
        # walks far past the target before OF catches up, then
        # swings back. Operator reported "PID circles the target"
        # 2026-04-25.
        # Cache the last-seen bbox center in raw thermal pixels and
        # only recompute the heat-track setpoint when the centre
        # has moved by >= track_min_bbox_move_px since the last
        # update. Sub-pixel jitter is also gated. Default 2 px is
        # well above OF noise but small enough that real OF motion
        # registers within one update.
        self._last_heat_bbox_center: Optional[tuple[float, float]] = None
        self._track_min_bbox_move_px = float(
            gcfg.get("track_min_bbox_move_px", 2.0))

        # Pan-only saturation log throttle — we emit one INFO when we
        # enter saturation and another when we exit, but nothing in
        # between (would spam at rate_hz).
        self._tilt_saturated_logged = False

        # Driver — may or may not actually open.
        if self._is_v2:
            ws_cfg = (gcfg.get("waveshare") or {})
            # NOTE: do NOT fall back to gcfg["port"] — that's the legacy
            # Maestro Command Port pin (COM4) which is a different device
            # entirely. If ws_cfg.port is null, let BusServoDriver auto-
            # detect the CH343 by VID/PID instead.
            self._driver = BusServoDriver(
                port=port or ws_cfg.get("port"),
                baud=int(ws_cfg.get("baud", 1_000_000)),
            )
        else:
            # PWM-gating: skip Maestro writes when |new_us - last_sent_us|
            # < min_us_step. See MaestroDriver init docstring.
            min_us_step = float(gcfg.get("maestro_min_us_step", 5.0))
            self._driver = MaestroDriver(
                port=port or gcfg.get("port"),
                min_us_step=min_us_step,
            )
        self._connected = False
        # Counter for consecutive write failures (auto-reconnect logic).
        self._consec_write_fail = 0
        # Reconnect cooldown. _command_now runs at 60 Hz; without this,
        # every disconnected tick called driver.open() which produced a
        # warning log on systems with no Waveshare adapter — at 60 Hz the
        # log queue contention dragged thermal/EO publish rates from
        # 19-22 Hz down to ~6-9 Hz. Retry at most once every 5 s.
        self._next_reconnect_ts: float = 0.0
        # V2 only: last successful measured pose. Used as the published
        # pan/tilt and as the fallback when an encoder read times out so
        # the bus doesn't flap between measured and stale-commanded on a
        # single dropped reply.
        self._last_measured_pan: Optional[float] = None
        self._last_measured_tilt: Optional[float] = None
        # Previous-tick measured pose + timestamp for the closed-loop's
        # D-term-on-measurement. dpose/dt computed here is the encoder's
        # measured angular velocity; multiplied by kd_track and
        # subtracted from d_pan_cl / d_tilt_cl in the closed-loop to
        # damp overshoot.
        self._prev_pose_pan: Optional[float] = None
        self._prev_pose_tilt: Optional[float] = None
        self._prev_pose_t: Optional[float] = None
        # Smoothed world-frame target. Maintained across fresh
        # observations to absorb YOLO bbox jitter before commanding
        # the gimbal. EMA: new value contributes _target_lp_alpha,
        # previous (1 − _target_lp_alpha). Reset to None on a new
        # track-engagement so the first obs anchors the smoother.
        self._smooth_target_az: Optional[float] = None
        self._smooth_target_el: Optional[float] = None

        # ── Lock mode (vision/lock_tracker.py) ──────────────────────
        # Kill switch: gimbal.lock_mode.enabled in YAML. When False,
        # lock-mode code paths are no-ops and behaviour is byte-
        # equivalent to the good-baseline-v1 tag.
        from vision.lock_tracker import LockTracker, LockTrackerConfig
        lm_cfg = (gcfg.get("lock_mode") or {})
        self._lock_mode_enabled: bool = bool(lm_cfg.get("enabled", True))
        lt_cfg = LockTrackerConfig(
            psr_lost=float(lm_cfg.get("psr_lost", 5.0)),
            lost_frames=int(lm_cfg.get("lost_frames", 5)),
            coast_window_s=float(lm_cfg.get("coast_window_s", 3.0)),
            learning_rate=float(lm_cfg.get("learning_rate", 0.125)),
            sigma=float(lm_cfg.get("sigma", 2.0)),
            max_patch_dim=int(lm_cfg.get("max_patch_dim", 96)),
        )
        self._lock_eo = LockTracker(lt_cfg)
        self._lock_thermal = LockTracker(lt_cfg)
        # Engagement metadata captured at seed time. v2 uses ONLY
        # `_lock_target_id` for the reseed gate; class is no longer
        # checked (an ID match implies same physical target).
        self._lock_target_id: Optional[int] = None
        self._lock_target_class: Optional[str] = None
        self._lock_seed_pending: bool = False
        self._lock_reseed_min_period_s: float = float(
            lm_cfg.get("reseed_min_period_s", 0.30))
        # PSR margin for auto-reseed gate. Reseed only when state is
        # ACTIVE and last_psr >= psr_lost * margin. Default 1.5;
        # set 0 to disable (legacy behavior — reseed any time
        # is_active includes COASTING). See Wave 2 lock review.
        self._lock_reseed_psr_margin: float = float(
            lm_cfg.get("reseed_psr_margin", 1.5))
        # State-transition tracker for the JSONL recorder. We emit
        # one event per real transition (not per tick) so a future
        # replay can reconstruct the lock-state machine from the
        # event stream alone.
        self._lock_last_state_str: str = "off"
        # Seed-pending timeout. Without this, if the engaged fused-id
        # never appears in BUS.get_latest(Topic.FUSED) (engaged
        # against a track that died between engage and the next
        # fusion tick), seed_pending stays True forever and the
        # lock state machine never advances. Recordings showed 4/7
        # engagements with `lock_state="active"` published but zero
        # `lock_seeded` events (track worse.jsonl) — the seed-pending
        # path was the silent miss. We tag the wallclock when
        # seed_pending was armed; if 2 s pass without a successful
        # seed, abort and reset the state-machine string so the next
        # engagement emits a clean transition.
        self._lock_seed_pending_t0: Optional[float] = None
        self._lock_seed_pending_timeout_s: float = float(
            lm_cfg.get("seed_pending_timeout_s", 2.0))
        # Frame-id dedupe: skip MOSSE update() when the BUS-cached
        # frame is the same one we just processed. Gimbal tick runs
        # at ~35-60 Hz; EO at 17-25 Hz, thermal at 20-60 Hz. Without
        # dedupe, MOSSE re-runs on identical data 1-3 times per real
        # frame, burning CPU on no-op work — visible in
        # `recordings/lock test test.jsonl` as 15 FPS EO under load.
        # Dedupe also keeps the published lock_bbox_* coordinates
        # from "twitching" between identical-frame re-runs (the FFT
        # peak can land 1 px apart on identical input due to
        # learning-rate updates, which the operator sees as jitter).
        self._lock_last_eo_fid: Optional[int] = None
        self._lock_last_th_fid: Optional[int] = None
        # Cached last-update so we publish a stable bbox between
        # frames without re-running MOSSE.
        self._lock_last_eo_upd = None
        self._lock_last_th_upd = None

        # Manual setpoint (mutated by GUI dpad / WASD CLI)
        self._manual_pan  = home_pan
        self._manual_tilt = home_tilt

        # Track-lock state. Two mutually-exclusive lock modes:
        #   _tracked_id       — a fused track ID (production path; matched
        #                       against BUS.get_latest(Topic.FUSED))
        #   _tracked_heat_id  — a raw heat-blob tracker ID (dev-mode only;
        #                       matched against the latest ThermalFrame's
        #                       heat_tracks list and converted to az/el
        #                       from bbox center vs thermal FOV)
        # Setting one clears the other — never both at once.
        self._tracked_id: Optional[int] = None
        self._tracked_heat_id: Optional[int] = None

        # How many consecutive ticks to tolerate a tracked ID being
        # missing from the fused list before dropping the lock. At
        # 20 Hz, 20 ticks ≈ 1 s of grace — enough to survive a
        # classifier hiccup without losing the target.
        self._track_grace_ticks = int(gcfg.get("track_grace_ticks", 20))
        self._track_miss = 0

        # Thread
        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()

    def _axis_addrs(self) -> list:
        """Per-axis addressing list the active driver expects:
        Maestro channels for V1, bus-servo IDs for V2."""
        if self._is_v2:
            return [self._pan_cal.servo_id, self._tilt_cal.servo_id]
        return [self._pan_cal.channel, self._tilt_cal.channel]

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._connected = self._driver.open()
        if self._connected:
            # V2: write Acceleration register on each servo so per-tick
            # goal-position writes ramp instead of snapping. Default 0
            # is "max instant acceleration" which feels jerky at 60 Hz
            # update rates. ~50 is a noticeable smoothing without
            # killing responsiveness.
            if self._is_v2:
                ws_cfg = (load_config().get("gimbal", {}) or {}).get("waveshare", {}) or {}
                pan_cfg = ws_cfg.get("pan") or {}
                tilt_cfg = ws_cfg.get("tilt") or {}
                # Acceleration ramp on the motor commutation (0..255).
                self._driver.set_acceleration(self._pan_cal.servo_id,
                                               int(pan_cfg.get("acceleration", 50)))
                self._driver.set_acceleration(self._tilt_cal.servo_id,
                                               int(tilt_cfg.get("acceleration", 50)))
                # Position-Integral gain (0..255). Default firmware ships
                # this at 0, which leaves a steady-state error against
                # any constant load (gravity on the tilted-up gimbal
                # sits the axis a couple degrees below commanded). A
                # small I drives that residual to zero. Too large will
                # hunt. 2..3 is a conservative starting point.
                self._driver.set_position_i_gain(self._pan_cal.servo_id,
                                                  int(pan_cfg.get("ki", 2)))
                self._driver.set_position_i_gain(self._tilt_cal.servo_id,
                                                  int(tilt_cfg.get("ki", 3)))
            # Move gently to home instead of snapping — avoids a
            # startup slam when the servos wake up at a random pose.
            self._controller.reset_to(self._home_pan, self._home_tilt)
            self._command_now(self._home_pan, self._home_tilt)
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._run, name="GimbalManager", daemon=True,
        )
        self._thread.start()
        log.info("GimbalManager started (connected=%s, port=%s)",
                 self._connected, self._driver.port)

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._connected:
            # Release servos on shutdown so they don't keep holding
            # torque and overheat.
            self._driver.release_all(self._axis_addrs())
        self._driver.close()
        log.info("GimbalManager stopped")

    # ── GUI / CLI inputs ──────────────────────────────────────

    def set_manual_delta(self, d_pan_deg: float, d_tilt_deg: float) -> None:
        """Incremental nudge from the dpad. Implicitly releases any
        active TRACK lock (fused or heat) so the user's arrows always
        take priority.

        We clamp the accumulator to configured limits HERE, at the
        source, rather than relying solely on the controller's apply-
        time clamp. Otherwise repeated presses at the rail silently
        accrue past the limit (e.g. 22° tilt cap + 5 extra "up"
        presses parks the target at 47°), forcing the operator to
        unwind those 5 presses in the opposite direction before the
        servo visibly moves again. Clamping at accept-time means every
        press maps 1:1 to visible motion whenever motion is available.
        """
        with self._lock:
            if self._tracked_id is not None:
                log.info("Manual nudge → releasing fused track lock on #%d", self._tracked_id)
                self._tracked_id = None
            if self._tracked_heat_id is not None:
                log.info("Manual nudge → releasing heat track lock on H#%d", self._tracked_heat_id)
                self._tracked_heat_id = None
            new_pan  = self._manual_pan  + float(d_pan_deg)
            new_tilt = self._manual_tilt + float(d_tilt_deg)
            pan_lo  = float(self._pan_floor)
            pan_hi  = float(self._pan_ceil)
            tilt_lo = float(self._tilt_floor)
            tilt_hi = float(self._tilt_ceil)
            self._manual_pan  = max(pan_lo,  min(pan_hi,  new_pan))
            self._manual_tilt = max(tilt_lo, min(tilt_hi, new_tilt))

    def set_manual_absolute(self, pan_deg: float, tilt_deg: float) -> None:
        with self._lock:
            self._tracked_id  = None
            self._tracked_heat_id = None
            self._manual_pan  = float(pan_deg)
            self._manual_tilt = float(tilt_deg)

    def set_home(self) -> None:
        self.set_manual_absolute(self._home_pan, self._home_tilt)

    def set_track_target(self, track_id: Optional[int]) -> None:
        with self._lock:
            prev_id = self._tracked_id
            if track_id is None:
                self._tracked_id = None
                log.info("Fused track lock cleared → manual")
                return
            try:
                new_id = int(track_id)
            except (TypeError, ValueError):
                log.warning("Bad track_id: %r", track_id)
                return
            # Switching from one target to another (or engaging while
            # a previous lock was still alive with tracked_id never
            # passing through None) must reset ALL predictor and
            # closed-loop state. Without this, the prior track's
            # world_az_dot leaks into the new engagement and the
            # predictor's lookahead computes
            #     shift_az = lead_time × world_az_dot_inherited
            # at the very first tick — producing a 2°+ initial
            # setpoint error and the visible "aggressive at engage"
            # burst plus the bb-flicker on switch documented in
            # `still too aggresive i guess.jsonl` track #7 (entered
            # the predictor stream with world_az_dot=-6.93 dps before
            # ANY obs of #7 had been processed).
            #
            # The legacy reset path in _tick fires only when the else
            # branch (tracked_id is None) runs — i.e. between two
            # engagements that go through release. A direct switch
            # never hits that branch, so the reset never fired.
            if new_id != prev_id:
                self._track_world_az = None
                self._track_world_el = None
                self._track_world_az_dot = 0.0
                self._track_world_el_dot = 0.0
                self._track_world_last_t = None
                self._track_obs_count = 0
                self._cur_pan_prev = None
                self._cur_tilt_prev = None
                self._cur_pose_prev_t = None
                self._predictor_state.reset()
                self._last_settled_state = None
                self._smooth_target_az = None
                self._smooth_target_el = None
                self._last_sp_pan = None
                self._last_sp_tilt = None
                self._last_track_ts = None
                self._last_fused_track_hits = None
                # Lock mode: clear previous lock state and arm a
                # seed-pending flag. The actual seed happens on the
                # NEXT _tick once we have fresh EO + thermal frames
                # plus the fused track's bbox_eo / bbox_thermal
                # projections to anchor the MOSSE patches on.
                if self._lock_mode_enabled:
                    self._lock_eo.release()
                    self._lock_thermal.release()
                    self._lock_target_id = new_id
                    self._lock_target_class = None  # filled when seeding
                    self._lock_seed_pending = True
                    self._lock_seed_pending_t0 = time.time()
                    # Reset state-transition tracker + dedupe cache
                    # so the new engagement emits a clean `lock_seeded`
                    # event and runs a fresh MOSSE update on its first
                    # frame (mirrors the release-path reset in _tick's
                    # else branch).
                    self._lock_last_state_str = "off"
                    self._lock_last_eo_fid = None
                    self._lock_last_th_fid = None
                    self._lock_last_eo_upd = None
                    self._lock_last_th_upd = None
            self._tracked_id = new_id
            # Fused lock takes priority over any heat lock.
            self._tracked_heat_id = None
            log.info("Fused track lock engaged on #%d", self._tracked_id)

    def set_track_heat(self, heat_id: Optional[int]) -> None:
        """Lock the gimbal onto a raw heat-blob tracker ID (dev-mode path).

        Behaves like ``set_track_target`` but resolves the ID against
        the thermal frame's ``heat_tracks`` list rather than the fused
        tracker. Useful for debugging the heat detector/tracker without
        needing a classifier confirmation — the user can point the
        gimbal at anything warm.
        """
        with self._lock:
            if heat_id is None:
                self._tracked_heat_id = None
                log.info("Heat track lock cleared → manual")
            else:
                try:
                    self._tracked_heat_id = int(heat_id)
                    # Heat lock supersedes fused lock (they're mutually
                    # exclusive — the gimbal can only track one thing).
                    self._tracked_id = None
                    log.info("Heat track lock engaged on H#%d", self._tracked_heat_id)
                except (TypeError, ValueError):
                    log.warning("Bad heat_id: %r", heat_id)

    def _lp_filter_error(self, az: float, el: float) -> tuple[float, float]:
        """First-order IIR low-pass on the (az, el) error signal.

        Seeds on first call so we don't slam from 0 toward a large
        initial error. Returns the filtered (az, el) pair.
        """
        a = self._err_lp_alpha
        if a <= 0.0:
            return az, el
        if self._err_lp_az is None:
            self._err_lp_az = az
            self._err_lp_el = el
        else:
            self._err_lp_az = (1.0 - a) * self._err_lp_az + a * az
            self._err_lp_el = (1.0 - a) * self._err_lp_el + a * el
        return self._err_lp_az, self._err_lp_el

    def _reset_track_filter(self) -> None:
        """Clear the low-pass state so the next track acquisition starts
        clean and doesn't drag a stale error from the previous target."""
        self._err_lp_az = None
        self._err_lp_el = None

    # ── main loop ─────────────────────────────────────────────

    def _run(self) -> None:
        period = 1.0 / max(1.0, self._rate_hz)
        while not self._stop_evt.is_set():
            t0 = time.time()
            try:
                self._tick()
            except Exception:
                log.exception("Gimbal tick failed")
            # sleep out the remainder of the period
            sleep = period - (time.time() - t0)
            if sleep > 0:
                self._stop_evt.wait(sleep)

    def _tick(self) -> None:
        # Snapshot inputs
        with self._lock:
            tracked_id      = self._tracked_id
            tracked_heat_id = self._tracked_heat_id
            manual_pan      = self._manual_pan
            manual_tilt     = self._manual_tilt

        mode = "manual"
        sp_pan, sp_tilt = manual_pan, manual_tilt
        err: Optional[str] = None

        # ── Pose source for the predictor + closed-loop ──────────────
        # V2 (waveshare_st3025) has a 12-bit absolute encoder per servo.
        # Reading it here at the top of the tick gives us the SERVO'S
        # ACTUAL angular position (not the controller's commanded
        # rate-limited setpoint, which leads the encoder during slews).
        # Critical for two reasons:
        #   1. world_az = obs_az + cur_pan_at_capture_time. Using the
        #      commanded pose (which leads measured by 1-3°) injects
        #      that gap as observation noise, which the velocity filter
        #      then turns into phantom motion → noisy lookahead → jitter.
        #      `tracker 532026.jsonl` track #11: cam_az per-tick max
        #      delta 6.92° while world_az delta only 1.4° — the cam-frame
        #      jitter was almost entirely from pose-source mismatch.
        #   2. `gimbal_dps` (used by the predictor's settled gate)
        #      computed from MEASURED pose differences reflects real
        #      angular velocity. Computed from commanded pose, it
        #      flapped: 138 settled-flips in 451 ticks because the
        #      controller's commanded-pose jumps every tick when sp
        #      moves, even though the gimbal physically isn't moving
        #      at that rate.
        # V1 (Maestro PWM, no feedback): keep the historical commanded
        # path — `actual_pan/actual_tilt` is recovered from
        # get_last_written_us downstream of _command_now.
        pose_pan: Optional[float] = None
        pose_tilt: Optional[float] = None
        if self._is_v2 and self._connected:
            try:
                raw_p = self._driver.read_position(self._pan_cal.servo_id)
                raw_t = self._driver.read_position(self._tilt_cal.servo_id)
                if raw_p is not None:
                    self._last_measured_pan = self._pan_cal.units_to_angle(raw_p)
                if raw_t is not None:
                    self._last_measured_tilt = self._tilt_cal.units_to_angle(raw_t)
            except Exception as e:
                log.debug("V2 read_position at tick top failed: %r", e)
            pose_pan = self._last_measured_pan
            pose_tilt = self._last_measured_tilt
        if pose_pan is None or pose_tilt is None:
            pose_pan, pose_tilt = self._controller.current

        # Encoder-measured angular velocity for the closed-loop D term.
        # D-on-MEASUREMENT (not on error) so setpoint changes don't kick
        # the controller. Brakes proportional to how fast the gimbal is
        # physically moving — when approaching target fast, command
        # shrinks → smooth deceleration → no overshoot.
        import time as _t_dt
        _now_dt = _t_dt.time()
        meas_dpan_dps = 0.0
        meas_dtilt_dps = 0.0
        if (self._prev_pose_pan is not None
                and self._prev_pose_tilt is not None
                and self._prev_pose_t is not None):
            _dt = max(1e-3, _now_dt - self._prev_pose_t)
            if _dt < 0.5:   # ignore stale-prev (e.g. after a long pause)
                meas_dpan_dps  = (pose_pan  - self._prev_pose_pan)  / _dt
                meas_dtilt_dps = (pose_tilt - self._prev_pose_tilt) / _dt
        self._prev_pose_pan  = pose_pan
        self._prev_pose_tilt = pose_tilt
        self._prev_pose_t    = _now_dt

        # Visual feedback freshness gate. The camera supplies error
        # samples at ~30 Hz; the control loop ticks at 60 Hz. Commanding
        # a fresh P correction on every tick would fire 2 corrections
        # per feedback sample and oscillate forever. We check the
        # thermal frame timestamp and only advance the setpoint on a
        # new frame. Stale ticks just reuse the last setpoint so the
        # servo keeps slewing toward it (the controller's internal
        # slew/step limits still run every tick — only the *error-
        # driven* setpoint update is gated).
        tf_latest = BUS.get_latest(Topic.THERMAL)
        tf_ts = getattr(tf_latest, "timestamp", None) if tf_latest is not None else None
        fresh_frame = (tf_ts is not None and tf_ts != self._last_track_ts)

        # Resolve a heat-blob lock to a synthetic FusedTrack-shaped
        # record: (az_deg, el_deg). If the heat ID is gone this tick,
        # fall through into the same grace-tick logic used for fused
        # tracks so a brief detector hiccup doesn't drop the lock.
        heat_obs: Optional[_HeatObs] = None
        if tracked_heat_id is not None and tracked_id is None:
            heat_obs = self._resolve_heat_track(tracked_heat_id)

        if tracked_heat_id is not None and tracked_id is None:
            if heat_obs is not None:
                # Same rationale as fused-track path: use measured pose
                # (V2 encoder) for the predictor + closed-loop reference.
                cur_pan, cur_tilt = pose_pan, pose_tilt
                # Centroid-move gate. The OF tracker that propagates
                # synthetic targets advances cx/cy only every few
                # thermal frames. If we recompute the setpoint on
                # every fresh thermal tick, we apply kp*az 3-5 times
                # using the SAME stale (cx, cy) -- gimbal walks past
                # the target before OF catches up, then swings back
                # = "gimbal circles the target" pattern (operator-
                # reported 2026-04-25). Only update setpoint when the
                # bbox centre has actually moved by >= the configured
                # min pixels. Sub-pixel jitter is also gated.
                cur_center: Optional[tuple[float, float]] = None
                if heat_obs.cx_px is not None and heat_obs.cy_px is not None:
                    cur_center = (float(heat_obs.cx_px), float(heat_obs.cy_px))
                if cur_center is None:
                    centroid_moved = True       # no info -> default fresh
                elif self._last_heat_bbox_center is None:
                    centroid_moved = True       # first observation
                else:
                    dx = cur_center[0] - self._last_heat_bbox_center[0]
                    dy = cur_center[1] - self._last_heat_bbox_center[1]
                    centroid_moved = (dx*dx + dy*dy
                                      >= self._track_min_bbox_move_px ** 2)
                fresh_heat = (centroid_moved and fresh_frame) \
                             or self._last_sp_pan is None
                if heat_obs.synthetic and self._cameras_on_gimbal:
                    # ── SYNTHETIC TARGET PATH (ONE-SHOT) ──
                    # Lock world-frame target angle on first observation
                    # and slew there. Don't trust subsequent OF updates
                    # to refine — they oscillate when OF lags.
                    if self._synth_world_az_deg is None:
                        # First observation: capture world-frame angle.
                        # World az = current gimbal pan + observed
                        # camera-frame az. Same for el.
                        self._synth_world_az_deg = (
                            float(cur_pan) + float(heat_obs.az_deg))
                        self._synth_world_el_deg = (
                            float(cur_tilt) + float(heat_obs.el_deg))
                        self._synth_lock_t = time.time()
                        log.info("Synthetic lock: world target "
                                 "az=%.2f el=%.2f (cur=%.1f,%.1f + obs=%.1f,%.1f)"
                                 " slew-dampened for %.1fs at %.0f dps",
                                 self._synth_world_az_deg,
                                 self._synth_world_el_deg,
                                 cur_pan, cur_tilt,
                                 heat_obs.az_deg, heat_obs.el_deg,
                                 self._synth_slew_window_s,
                                 self._synth_slew_dps)
                        # Capture optical-residual anchors on both
                        # sensors. Best-effort — failures are silent
                        # (no cv2, frame missing, no features).
                        self._opt_capture_anchors(cur_pan, cur_tilt)
                    # Stage B: bias the world-frame target by the
                    # cumulative optical correction. Closes the loop
                    # against mechanical residuals (servo backlash,
                    # gravity creep, dead-zone) that the controller is
                    # blind to. Zero-impact when correction is disabled
                    # (`_opt_corr_az/_el` stay at 0.0).
                    target_sp_pan  = (self._synth_world_az_deg
                                      + self._opt_corr_az)
                    target_sp_tilt = (self._synth_world_el_deg
                                      + self._opt_corr_el)
                    # Slew dampening window. During the first
                    # _synth_slew_window_s after lock commit, cap the
                    # per-tick setpoint advance so per-frame scene
                    # motion stays within the LK tracking limit. Once
                    # the window expires, normal slew rate (from the
                    # GimbalController) applies.
                    if (self._synth_lock_t is not None
                            and (time.time() - self._synth_lock_t)
                                < self._synth_slew_window_s):
                        # Per-tick step cap = dps × tick period.
                        max_step_deg = (self._synth_slew_dps
                                        / max(1e-3, self._rate_hz))
                        d_pan  = target_sp_pan  - cur_pan
                        d_tilt = target_sp_tilt - cur_tilt
                        if abs(d_pan)  > max_step_deg:
                            d_pan  = max_step_deg if d_pan  > 0 else -max_step_deg
                        if abs(d_tilt) > max_step_deg:
                            d_tilt = max_step_deg if d_tilt > 0 else -max_step_deg
                        sp_pan  = cur_pan  + d_pan
                        sp_tilt = cur_tilt + d_tilt
                    else:
                        sp_pan  = target_sp_pan
                        sp_tilt = target_sp_tilt
                    # Saturation log line uses the live error.
                    el_err = self._synth_world_el_deg - cur_tilt
                    _, tilt_sat = _pan_only_if_tilt_saturated(
                        cur_tilt, el_err,
                        self._tilt_floor, self._tilt_ceil,
                        self._tilt_sat_eps_deg)
                    if tilt_sat and not self._tilt_saturated_logged:
                        log.info("Tilt saturated at mechanical stop "
                                 "(cur=%.1f, el_err=%.2f) — pan-only",
                                 cur_tilt, el_err)
                        try:
                            emit_event("tilt_saturated_enter",
                                       {"cur_tilt": float(cur_tilt),
                                        "el_err": float(el_err)})
                        except Exception:
                            pass
                        self._tilt_saturated_logged = True
                    elif not tilt_sat:
                        if self._tilt_saturated_logged:
                            try:
                                emit_event("tilt_saturated_exit", {})
                            except Exception:
                                pass
                        self._tilt_saturated_logged = False
                    self._last_track_ts = tf_ts
                    self._last_heat_bbox_center = (
                        (float(heat_obs.cx_px), float(heat_obs.cy_px))
                        if heat_obs.cx_px is not None else None
                    )
                    self._last_sp_pan   = sp_pan
                    self._last_sp_tilt  = sp_tilt
                    # Stage A: measure visual residual against anchor
                    # and emit an event. No effect on the gimbal yet.
                    self._opt_measure_and_emit(cur_pan, cur_tilt,
                                               tracked_id=tracked_heat_id)
                elif fresh_heat:
                    # ── REAL HEAT-BLOB PATH (CLOSED-LOOP) ──
                    if self._cameras_on_gimbal:
                        # First-order low-pass on the raw error before
                        # deadband + gain. Smooths centroid jitter on
                        # multi-patch targets (vehicles, humans) without
                        # adding meaningful lag for real motion.
                        az_in, el_in = self._lp_filter_error(
                            float(heat_obs.az_deg), float(heat_obs.el_deg))
                        # Real heat blob — re-detection-based tracker,
                        # centroid genuinely updates as the camera
                        # moves, so closed-loop smooth-proportional gain
                        # converges naturally.
                        kp_eff = self._kp_track
                        d_pan  = _smooth_proportional(
                            az_in, kp_eff,
                            self._track_zero_band_deg,
                            self._track_full_band_deg)
                        d_tilt = _smooth_proportional(
                            el_in, kp_eff,
                            self._track_zero_band_deg,
                            self._track_full_band_deg)
                        if abs(d_pan)  < self._track_min_step_deg: d_pan  = 0.0
                        if abs(d_tilt) < self._track_min_step_deg: d_tilt = 0.0
                        d_pan  = _clip(d_pan,  self._max_step_deg)
                        d_tilt = _clip(d_tilt, self._max_step_deg)
                        # `el` here is just for the saturation log.
                        el = el_in
                        d_tilt, tilt_sat = _pan_only_if_tilt_saturated(
                            cur_tilt, d_tilt,
                            self._tilt_floor, self._tilt_ceil,
                            self._tilt_sat_eps_deg)
                        if tilt_sat and not self._tilt_saturated_logged:
                            log.info("Tilt saturated at mechanical stop "
                                     "(cur=%.1f, el_err=%.2f) — pan-only",
                                     cur_tilt, el)
                            try:
                                emit_event("tilt_saturated_enter",
                                           {"cur_tilt": float(cur_tilt),
                                            "el_err": float(el)})
                            except Exception:
                                pass
                            self._tilt_saturated_logged = True
                        elif not tilt_sat:
                            if self._tilt_saturated_logged:
                                try:
                                    emit_event("tilt_saturated_exit", {})
                                except Exception:
                                    pass
                            self._tilt_saturated_logged = False
                        sp_pan  = cur_pan  + d_pan
                        sp_tilt = cur_tilt + d_tilt
                    else:
                        sp_pan  = self._home_pan  + float(heat_obs.az_deg)
                        sp_tilt = self._home_tilt + float(heat_obs.el_deg)
                    self._last_track_ts = tf_ts
                    self._last_heat_bbox_center = cur_center
                    self._last_sp_pan   = sp_pan
                    self._last_sp_tilt  = sp_tilt
                else:
                    # Centroid hasn't moved yet -- hold last setpoint
                    # so the gimbal finishes its in-flight motion
                    # rather than re-applying kp*az on stale data.
                    sp_pan  = self._last_sp_pan
                    sp_tilt = self._last_sp_tilt
                mode = "auto"
                self._track_miss = 0
                with self._lock:
                    self._manual_pan  = cur_pan
                    self._manual_tilt = cur_tilt
            else:
                self._track_miss += 1
                mode = "auto"
                if self._track_miss >= self._track_grace_ticks:
                    err = f"heat id H#{tracked_heat_id} lost after {self._track_miss} ticks"
                    log.info("Dropping heat track lock on H#%d (lost)", tracked_heat_id)
                    try:
                        emit_event("track_grace_expired", {
                            "kind": "heat",
                            "tracked_id": int(tracked_heat_id),
                            "miss_ticks": int(self._track_miss),
                        })
                    except Exception:
                        pass
                    with self._lock:
                        if self._tracked_heat_id == tracked_heat_id:
                            self._tracked_heat_id = None
                    self._track_miss = 0
                    mode = "manual"
                sp_pan, sp_tilt = self._controller.current

        elif tracked_id is not None:
            # Pull the latest fused list and find our target
            fused = BUS.get_latest(Topic.FUSED)
            trk = None
            if fused:
                for t in fused:
                    if getattr(t, "id", None) == tracked_id:
                        trk = t
                        break
            if trk is not None:
                # Use measured pose (encoder) on V2 instead of the
                # controller's commanded pose. See pose_pan/pose_tilt
                # docstring at the top of _tick — leads to clean
                # gimbal_dps, stable settled gate, and correct
                # world_az = obs_az + cur_pan_at_capture conversion.
                cur_pan, cur_tilt = pose_pan, pose_tilt
                # Fused-track freshness gate. Only feed the predictor
                # a "fresh" observation when the FUSED track has
                # actually been refreshed (id+hits tuple changed) --
                # NOT on every thermal tick.
                fused_key: Optional[tuple[int, int]] = None
                try:
                    fused_key = (int(trk.id), int(trk.hits))
                except Exception:
                    fused_key = None
                fresh_fused = (
                    fused_key is not None
                    and fused_key != self._last_fused_track_hits
                )
                # Pure-function predictor — see algorithms/track_predictor.
                # Both this live tick AND scripts/replay_algo.py call
                # the same `step` against the same captured inputs;
                # parity is the verification gate before iterating
                # variants offline.
                # Tilt-saturation gate. The fused-track predictor path
                # historically bypassed _pan_only_if_tilt_saturated (the
                # heat-track path uses it). Result: tracking a target
                # at el<0 (e.g. a person below boresight when home_tilt
                # is 0°) walked the gimbal pan correctly but commanded
                # negative tilt every tick — which the controller
                # silently clamped at the floor. Predictor kept
                # accumulating world_el_dot from observations the
                # gimbal physically couldn't follow, contaminating the
                # next non-saturated extrapolation.
                # Now: detect saturation BEFORE calling step(), tell
                # the predictor to zero el-velocity contributions, then
                # clip sp_tilt to the floor/ceil after step() returns.
                # Pan tracking continues independently — exactly the
                # "below 0° → pan only" behavior we agreed.
                # Determine el-error for saturation. When fresh, use
                # the latest fused observation. When stale, use the
                # predictor's last-known world_el (= last_observed_el)
                # so the saturation state doesn't flicker every other
                # tick — the previous code passed 0.0 on stale ticks
                # which made tilt_sat oscillate True↔False at 60 Hz
                # and emitted hundreds of pointless enter/exit events
                # (visible in the 2026-04-27 mistrack recording: 12
                # enter/exit pairs in 0.4 s on track #42).
                if fresh_fused:
                    el_obs_for_sat = float(trk.el_deg)
                elif self._predictor_state.world_el is not None:
                    el_obs_for_sat = (float(self._predictor_state.world_el)
                                      - float(cur_tilt))
                else:
                    el_obs_for_sat = 0.0
                _, tilt_sat_now = _pan_only_if_tilt_saturated(
                    cur_tilt, el_obs_for_sat,
                    self._tilt_floor, self._tilt_ceil,
                    self._tilt_sat_eps_deg)
                import time as _t
                now = _t.time()
                params = track_predictor.PredictorParams(
                    lead_time_s=self._track_lead_time_s,
                    vel_alpha=self._track_vel_alpha,
                    predict_warmup_n=self._track_predict_warmup_n,
                    predict_cap_deg=self._track_predict_cap_deg,
                    gimbal_settled_dps=self._track_gimbal_settled_dps,
                    extrap_horizon_s=self._track_extrap_horizon_s,
                    vel_clip_dps=self._track_vel_clip_dps,
                    vel_decay_halflife_s=self._track_vel_decay_halflife_s,
                    no_obs_lead_zero_after_s=self._track_no_obs_lead_zero_after_s,
                    settled_hysteresis_ratio=self._track_settled_hysteresis_ratio,
                    tilt_saturated=tilt_sat_now,
                )
                # Prefer world-frame angles published directly by fusion
                # (timing-invariant). Fall back to camera-frame for back-
                # compat when running legacy fusion mode that leaves
                # world_az_deg/world_el_deg unset.
                trk_world_az = getattr(trk, "world_az_deg", None)
                trk_world_el = getattr(trk, "world_el_deg", None)
                use_world = (fresh_fused
                              and trk_world_az is not None
                              and trk_world_el is not None)
                sp_pan_pred, sp_tilt_pred, diag = track_predictor.step(
                    self._predictor_state,
                    now=now,
                    cur_pan=cur_pan, cur_tilt=cur_tilt,
                    obs_az_deg=(float(trk.az_deg) if fresh_fused else None),
                    obs_el_deg=(float(trk.el_deg) if fresh_fused else None),
                    obs_world_az_deg=(float(trk_world_az) if use_world else None),
                    obs_world_el_deg=(float(trk_world_el) if use_world else None),
                    fresh_fused=fresh_fused,
                    params=params,
                )

                # Closed-loop pixel-error control law (operator ask:
                # same algorithm for static and dynamic targets, no
                # velocity-based mode switching). Mirrors the
                # real-heat-blob path: smooth proportional gain on the
                # fused track's camera-frame az/el, soft deadband
                # around image centre, low-pass error filter, per-tick
                # max-step cap. Active when `fused_track_closed_loop`
                # is true (default). The predictor still runs above
                # for diagnostics; we just override its sp output here.
                if self._fused_closed_loop and self._cameras_on_gimbal:
                    if fresh_fused:
                        # Prefer world-frame error: trk.az_deg/el_deg is
                        # cam-frame relative to fusion's pose snapshot at
                        # publish time, not the gimbal's NOW pose. During
                        # a slew, cur_pan_now − cur_pan_at_fusion_publish
                        # leaks into the closed-loop error and the
                        # controller over-corrects. World-az error uses
                        # the gimbal's current pose and is timing-clean.
                        if (trk_world_az is not None
                                and trk_world_el is not None):
                            cam_err_az = float(trk_world_az) - cur_pan
                            cam_err_el = float(trk_world_el) - cur_tilt
                        else:
                            cam_err_az = float(trk.az_deg)
                            cam_err_el = float(trk.el_deg)
                        az_in, el_in = self._lp_filter_error(
                            cam_err_az, cam_err_el)
                        kp_eff = self._kp_track
                        d_pan_cl = _smooth_proportional(
                            az_in, kp_eff,
                            self._track_zero_band_deg,
                            self._track_full_band_deg)
                        d_tilt_cl = _smooth_proportional(
                            el_in, kp_eff,
                            self._track_zero_band_deg,
                            self._track_full_band_deg)
                        # D term on encoder velocity (kd * dpose/dt
                        # subtracted from the proportional output).
                        # Brakes the controller when the gimbal is
                        # already moving fast in the same direction as
                        # the commanded delta — kills the overshoot
                        # seen in `tracking 532026 try3.jsonl` track #1
                        # (transit at 36 dps, then bounce-back at 0.6°
                        # amplitude before settling). kd=0 (legacy
                        # pure-P) is still selectable via YAML.
                        # Per-axis "centered" gate: the brake is only
                        # meaningful during APPROACH. When |err| <
                        # zero_band the kp output is already zero
                        # (deadband), so subtracting kd×velocity
                        # would make the brake the dominant force on
                        # sp inside the deadband — driving the
                        # arrival-bounce documented in the absolute-
                        # target branch above. Same semantics here.
                        kd_brake_az_legacy = self._kd_track * meas_dpan_dps
                        kd_brake_el_legacy = self._kd_track * meas_dtilt_dps
                        if abs(az_in) < self._track_zero_band_deg:
                            kd_brake_az_legacy = 0.0
                        if abs(el_in) < self._track_zero_band_deg:
                            kd_brake_el_legacy = 0.0
                        d_pan_cl  -= kd_brake_az_legacy
                        d_tilt_cl -= kd_brake_el_legacy
                        if abs(d_pan_cl)  < self._track_min_step_deg:
                            d_pan_cl  = 0.0
                        if abs(d_tilt_cl) < self._track_min_step_deg:
                            d_tilt_cl = 0.0
                        d_pan_cl  = _clip(d_pan_cl,  self._max_step_deg)
                        d_tilt_cl = _clip(d_tilt_cl, self._max_step_deg)
                        d_tilt_cl, _tilt_sat_cl = _pan_only_if_tilt_saturated(
                            cur_tilt, d_tilt_cl,
                            self._tilt_floor, self._tilt_ceil,
                            self._tilt_sat_eps_deg)
                        # Phase 3: confidence-gated velocity feed-forward.
                        # The closed-loop above corrects the *current*
                        # observed pixel error. For a moving target the
                        # observation already lags reality by the
                        # capture/process pipeline (~50-200 ms), and the
                        # gimbal then takes more time to reach the
                        # commanded position — so the camera always
                        # arrives where the target *was*. Adding
                        # lead_time × predicted_world_velocity pre-empts
                        # the lag.
                        # Gating: skip lookahead when the predictor
                        # isn't confident in its velocity estimate
                        # (so noise doesn't perturb a static target):
                        #   - obs_count >= predict_warmup_n
                        #   - last fresh obs < no_obs_lead_zero_after_s ago
                        #   - |world_dot| >= track_lookahead_min_dps
                        # Per-axis: a target moving fast in pan but
                        # static in tilt gets pan lookahead only.
                        ff_az_deg = 0.0
                        ff_el_deg = 0.0
                        ps = self._predictor_state
                        if (ps.obs_count >= self._track_predict_warmup_n
                                and ps.world_last_t is not None
                                and (now - ps.world_last_t) <
                                    self._track_no_obs_lead_zero_after_s):
                            if abs(ps.world_az_dot) >= self._track_lookahead_min_dps:
                                ff_az_deg = self._track_lead_time_s * ps.world_az_dot
                            if abs(ps.world_el_dot) >= self._track_lookahead_min_dps:
                                ff_el_deg = self._track_lead_time_s * ps.world_el_dot
                            # Reuse predictor's existing extrap cap so
                            # a velocity spike can't slam the setpoint.
                            cap = self._track_predict_cap_deg
                            ff_az_deg = max(-cap, min(cap, ff_az_deg))
                            ff_el_deg = max(-cap, min(cap, ff_el_deg))
                        # Per-axis "centered" gate on the lookahead.
                        # When the proportional output is already zero
                        # (|err| < zero_band), the closed-loop is saying
                        # "this axis is on target." Adding ff here just
                        # feeds the predictor's velocity-estimate noise
                        # into the setpoint: ~1-4 dps of jitter from
                        # observation noise on a stationary target × 0.3 s
                        # lead = 0.3-1.2° of unwanted setpoint motion
                        # every tick. The controller chases that, the
                        # settled gate flips ~4 Hz, the operator sees it
                        # as "aggressive when centered" hunting (see
                        # `night tracking a bit flickery.jsonl` track #6).
                        # Per-axis decision so a target moving fast in
                        # pan but centered in tilt still gets pan
                        # lookahead. The lookahead re-engages cleanly
                        # the moment |err| crosses zero_band — i.e.
                        # when chasing actually pays off.
                        if abs(az_in) < self._track_zero_band_deg:
                            ff_az_deg = 0.0
                        if abs(el_in) < self._track_zero_band_deg:
                            ff_el_deg = 0.0
                        # Per-axis "centered" gate on the kd brake. The
                        # absolute-target setpoint below subtracts
                        # kd × encoder_velocity from the commanded
                        # position so the gimbal decelerates while
                        # APPROACHING the target. That intent is sound
                        # during chase but pathological inside the
                        # deadband: when |err| < zero_band the kp
                        # output is already zero, so the kd term
                        # becomes the dominant force on sp. With
                        # kd=0.02 and slew-cap-bound velocities of
                        # ~18 dps, the brake shifts sp by ~0.36°
                        # against motion — comparable to zero_band
                        # (0.5°). The gimbal arrives near target with
                        # velocity, the brake pushes sp PAST target
                        # in the opposite direction, gimbal reverses,
                        # brake flips, gimbal reverses again. Visible
                        # in `gimbal too aggresive 2.jsonl` track #137
                        # at t=6.59-6.97s as a +17 → -8.6 dps velocity
                        # reversal within 100 ms. Per-axis gate so a
                        # target moving fast in one axis but centered
                        # in the other gets the brake on the chasing
                        # axis only.
                        kd_brake_az = self._kd_track * meas_dpan_dps
                        kd_brake_el = self._kd_track * meas_dtilt_dps
                        if abs(az_in) < self._track_zero_band_deg:
                            kd_brake_az = 0.0
                        if abs(el_in) < self._track_zero_band_deg:
                            kd_brake_el = 0.0
                        if (self._track_use_absolute_target
                                and trk_world_az is not None
                                and trk_world_el is not None):
                            # ABSOLUTE-TARGET path. Treat the
                            # smoothed world-frame target as the
                            # setpoint directly — exactly like the
                            # Home button. Controller's slew rate
                            # limit handles smooth approach; servo
                            # arrives once and stays. No more
                            # cur_pan + d_pan_cl Zeno asymptote that
                            # caused the visible "stuck on the way"
                            # stutter in `tracker 532026 try 4`.
                            a = self._track_target_lp_alpha
                            if self._smooth_target_az is None:
                                self._smooth_target_az = float(trk_world_az)
                                self._smooth_target_el = float(trk_world_el)
                            else:
                                self._smooth_target_az = (
                                    (1.0 - a) * self._smooth_target_az
                                    + a * float(trk_world_az))
                                self._smooth_target_el = (
                                    (1.0 - a) * self._smooth_target_el
                                    + a * float(trk_world_el))
                            # D-on-encoder-velocity brake. The absolute-
                            # target setpoint above hands the controller a
                            # static target which it slews to at full slew
                            # rate (120 dps) — servo momentum then carries
                            # it past on arrival. Subtracting kd × actual
                            # angular velocity from the setpoint pulls the
                            # commanded position BEHIND the true target by
                            # an amount proportional to how fast the gimbal
                            # is currently moving. As the gimbal decelerates
                            # near target (encoder velocity drops), the brake
                            # shrinks and the commanded position converges
                            # to the true target. Result: smooth approach,
                            # no overshoot.
                            sp_pan_pred  = (self._smooth_target_az
                                             + ff_az_deg
                                             - kd_brake_az)
                            sp_tilt_pred = (self._smooth_target_el
                                             + ff_el_deg
                                             - kd_brake_el)
                        else:
                            # Legacy delta-from-current closed-loop.
                            # Kept as a fallback when world coords
                            # aren't available or operator wants A/B.
                            sp_pan_pred  = cur_pan  + d_pan_cl  + ff_az_deg
                            sp_tilt_pred = cur_tilt + d_tilt_cl + ff_el_deg
                    else:
                        # Stale tick — hold the previous setpoint so
                        # the gimbal finishes its in-flight motion
                        # rather than re-applying gain on stale data.
                        if (self._last_sp_pan is not None
                                and self._last_sp_tilt is not None):
                            sp_pan_pred  = self._last_sp_pan
                            sp_tilt_pred = self._last_sp_tilt
                # Clip sp_tilt at the saturated edge. Without this the
                # controller still clamps, but the predictor's "last
                # setpoint" memory carries an out-of-range value.
                if tilt_sat_now and sp_tilt_pred is not None:
                    sp_tilt_pred = max(self._tilt_floor,
                                       min(self._tilt_ceil, sp_tilt_pred))
                # Tilt-saturation transition events for the timeline.
                if tilt_sat_now and not self._tilt_saturated_logged:
                    log.info("Tilt saturated (fused track) cur=%.1f el_obs=%.2f"
                             " — pan-only", cur_tilt, el_obs_for_sat)
                    try:
                        emit_event("tilt_saturated_enter", {
                            "kind": "fused",
                            "cur_tilt": float(cur_tilt),
                            "el_err": float(el_obs_for_sat),
                            "tracked_id": int(tracked_id),
                        })
                    except Exception:
                        pass
                    self._tilt_saturated_logged = True
                elif not tilt_sat_now:
                    if self._tilt_saturated_logged:
                        try:
                            emit_event("tilt_saturated_exit",
                                       {"kind": "fused"})
                        except Exception:
                            pass
                    self._tilt_saturated_logged = False
                # Mirror the live state into the legacy fields a few
                # other places still read from. This is mechanical;
                # the predictor state is the source of truth.
                self._track_world_az      = self._predictor_state.world_az
                self._track_world_el      = self._predictor_state.world_el
                self._track_world_az_dot  = self._predictor_state.world_az_dot
                self._track_world_el_dot  = self._predictor_state.world_el_dot
                self._track_world_last_t  = self._predictor_state.world_last_t
                self._track_obs_count     = self._predictor_state.obs_count
                self._cur_pan_prev        = self._predictor_state.cur_pan_prev
                self._cur_tilt_prev       = self._predictor_state.cur_tilt_prev
                self._cur_pose_prev_t     = self._predictor_state.cur_pose_prev_t

                # Emit one diagnostic event per tick. The recorder will
                # capture this in the JSONL events stream — single
                # highest-leverage signal for offline tracking-debug.
                # Tagged with track id + name so a future agent can
                # filter quickly via `replay_inspect.py --grep predictor`.
                try:
                    diag_payload = dict(diag)
                    diag_payload["tracked_id"] = int(tracked_id)
                    emit_event("track_predictor_step", diag_payload)
                except Exception:
                    pass
                # Settled-state transition is a useful event too --
                # changes in `settled` mark the start/end of slew
                # windows in the timeline.
                if self._last_settled_state != bool(diag.get("settled")):
                    try:
                        emit_event("track_settled_change", {
                            "settled": bool(diag.get("settled")),
                            "gimbal_dps": float(diag.get("gimbal_dps", 0.0)),
                        })
                    except Exception:
                        pass
                    self._last_settled_state = bool(diag.get("settled"))

                if fresh_fused:
                    self._last_fused_track_hits = fused_key
                    self._last_track_ts = tf_ts
                    log.debug(
                        "Track #%d  gimbal=(%.1f,%.1f) dps=%.1f settled=%s "
                        "obs_world=(%s,%s) vel=(%.1f,%.1f)",
                        tracked_id, cur_pan, cur_tilt,
                        diag.get("gimbal_dps", 0.0), diag.get("settled"),
                        diag.get("obs_world_az"), diag.get("obs_world_el"),
                        self._track_world_az_dot, self._track_world_el_dot,
                    )

                # Predictor returns sp = observed_world + capped predictive
                # shift, OR (when extrap horizon exceeded) the last setpoint.
                # When there's no track state at all (first tick before
                # first fresh observation) sp_*_pred is None; hold pose.
                if sp_pan_pred is not None and sp_tilt_pred is not None:
                    sp_pan, sp_tilt = sp_pan_pred, sp_tilt_pred
                else:
                    sp_pan  = cur_pan
                    sp_tilt = cur_tilt
                self._last_sp_pan   = sp_pan
                self._last_sp_tilt  = sp_tilt
                mode = "auto"
                self._track_miss = 0
                # Keep the manual park position synced so that when
                # the user releases track, the servo stays where the
                # tracker put it instead of snapping back.
                with self._lock:
                    self._manual_pan  = cur_pan
                    self._manual_tilt = cur_tilt
            else:
                # Tracked ID not in this tick's fused list. Don't
                # drop the lock on the first miss — fusion hiccups
                # happen; tolerate `track_grace_ticks` of them before
                # giving up. While we wait, hold the last commanded
                # position.
                self._track_miss += 1
                mode = "auto"
                if self._track_miss >= self._track_grace_ticks:
                    err = f"tracked id #{tracked_id} lost after {self._track_miss} ticks"
                    log.info("Dropping track lock on #%d (lost)", tracked_id)
                    try:
                        emit_event("track_grace_expired", {
                            "kind": "fused",
                            "tracked_id": int(tracked_id),
                            "miss_ticks": int(self._track_miss),
                        })
                    except Exception:
                        pass
                    with self._lock:
                        if self._tracked_id == tracked_id:
                            self._tracked_id = None
                    self._track_miss = 0
                    mode = "manual"
                # Hold current position while waiting
                sp_pan, sp_tilt = self._controller.current
        else:
            self._track_miss = 0
            # Lock mode: operator dropped the lock OR grace expired.
            # Release the per-sensor lock trackers so the GUI stops
            # rendering the lock bbox.
            #
            # Race-fix 2026-05-05: re-acquire self._lock and re-check
            # that no engagement happened between this tick's
            # snapshot and now. Without this, the WS handler thread
            # could call set_track_target(N) (which atomically arms
            # _lock_target_id=N + _lock_seed_pending=True under the
            # same lock) AFTER the tick took the snapshot — and this
            # else branch would wipe the just-armed seed, so the
            # next _lock_mode_tick saw _lock_target_id=None and
            # quietly emitted `lock_state_change tracked_id=null
            # active→off` instead of `lock_seeded`. Visible in
            # `recordings/lock test test.jsonl`: 7 track_engaged
            # events, only 1 lock_seeded — 6 of 7 engagements lost
            # to this race.
            if self._lock_mode_enabled:
                with self._lock:
                    still_clear = (self._tracked_id is None)
                if still_clear:
                    self._lock_eo.release()
                    self._lock_thermal.release()
                    self._lock_target_id = None
                    self._lock_target_class = None
                    self._lock_seed_pending = False
                    self._lock_seed_pending_t0 = None
                    # Reset the state-transition tracker so the NEXT
                    # engagement emits a clean `lock_seeded` event when
                    # off → active. Without this, the tracker keeps the
                    # stale "active" / "coasting" string from the prior
                    # engagement (because _lock_mode_tick early-returns
                    # on tracked_id=None and never updates it). The
                    # next engagement's off → active transition would
                    # then read as active → active and emit nothing —
                    # the silent miss seen in `track worse.jsonl`
                    # (4 engagements, 0 lock_seeded events).
                    self._lock_last_state_str = "off"
                    # Also drop the dedupe-cache fids/upds so the
                    # next engagement always runs a fresh MOSSE update
                    # on its first frame, even if frame_id happens to
                    # equal the last value seen pre-release.
                    self._lock_last_eo_fid = None
                    self._lock_last_th_fid = None
                    self._lock_last_eo_upd = None
                    self._lock_last_th_upd = None
            # Not tracking — flush cached setpoints so next engage
            # starts fresh against whatever the new target's error is.
            self._last_track_ts = None
            self._last_sp_pan   = None
            self._last_sp_tilt  = None
            self._last_fused_track_hits = None
            self._last_heat_bbox_center = None
            self._synth_world_az_deg = None
            self._synth_world_el_deg = None
            self._synth_lock_t = None
            # Reset optical-residual anchors so the next lock starts
            # fresh against the new scene.
            self._opt_eo.reset()
            self._opt_thermal.reset()
            self._opt_last_eo_frame_id = None
            self._opt_last_thermal_frame_id = None
            # Reset Stage B correction integrator so the next lock
            # starts at zero bias.
            self._opt_corr_az = 0.0
            self._opt_corr_el = 0.0
            self._opt_corr_last_source = ""
            # Reset stuck-servo guard counters on lock release.
            self._stuck_consec = 0
            self._stuck_released = False
            # Clear cached target residual.
            self._latest_target_resid_az = None
            self._latest_target_resid_el = None
            # Reset alpha-beta tracker so the next engage starts
            # fresh, no stale velocity from a prior target.
            self._track_world_az = None
            self._track_world_el = None
            self._track_world_az_dot = 0.0
            self._track_world_el_dot = 0.0
            self._track_world_last_t = None
            self._track_obs_count = 0
            # Drop the gimbal-pose history too so the next engage
            # starts with gimbal_dps=0 (treated as settled), allowing
            # the FIRST observation to anchor world position cleanly.
            self._cur_pan_prev = None
            self._cur_tilt_prev = None
            self._cur_pose_prev_t = None
            self._predictor_state.reset()
            self._last_settled_state = None
            self._in_deadband   = False
            self._reset_track_filter()
            self._tilt_saturated_logged = False
            # Drop the smoothed-target memory too — the next engage
            # will anchor on its first fresh observation.
            self._smooth_target_az = None
            self._smooth_target_el = None

        # Pan-saturation detection (symmetric with the tilt-saturated
        # logic above). Detect when the desired sp_pan would push the
        # gimbal past its mechanical pan envelope, so the operator /
        # replay tool can see WHY the gimbal stopped responding to a
        # tracking target outside its travel.
        if sp_pan is not None:
            pan_sat = ((sp_pan <= self._pan_floor + 0.1
                        and sp_pan < self._controller.current[0])
                       or (sp_pan >= self._pan_ceil - 0.1
                           and sp_pan > self._controller.current[0]))
            if pan_sat and not self._pan_saturated_logged:
                try:
                    emit_event("pan_saturated_enter", {
                        "cur_pan": float(self._controller.current[0]),
                        "sp_pan": float(sp_pan),
                        "edge": ("min" if sp_pan <= self._pan_floor + 0.1
                                 else "max"),
                    })
                except Exception:
                    pass
                self._pan_saturated_logged = True
            elif not pan_sat and self._pan_saturated_logged:
                try:
                    emit_event("pan_saturated_exit", {})
                except Exception:
                    pass
                self._pan_saturated_logged = False

        # Slew + clamp.
        # When ENGAGED on a fused track, cap the per-tick slew at
        # _track_slew_cap_dps so EO motion blur stays within YOLO's
        # recoverability envelope. Manual dpad / HOME / synth-target
        # paths fall through with slew_cap_dps=None (no extra cap).
        # tracked_id is held under self._lock; read it once.
        with self._lock:
            _engaged = (self._tracked_id is not None)
        slew_cap = self._track_slew_cap_dps if _engaged else None
        if slew_cap is not None and slew_cap <= 0:
            slew_cap = None
        cmd_pan, cmd_tilt = self._controller.step(
            sp_pan, sp_tilt, slew_cap_dps=slew_cap)
        self._command_now(cmd_pan, cmd_tilt)

        # Recover the SERVO'S ACTUAL-LAST-WRITTEN pose from the Maestro
        # driver. `cmd_pan/cmd_tilt` is the controller's per-tick rate-
        # limited setpoint, but the PWM gate (min_us_step) suppresses
        # commands smaller than the servo's deadband — when that happens
        # cmd_pan advances but the servo doesn't move. Publishing
        # cmd_pan as the gimbal pose makes downstream world-frame
        # conversions assign each sensor frame to a pose the camera
        # never reached, and the same physical target ends up at
        # different world_az/el across consecutive ticks → phantom
        # track births. ('multiple bbs.jsonl' showed the EO image
        # CONTENT staying frozen across a 2.5° commanded tilt change
        # — the servo wasn't moving, but published tilt advanced
        # 0.5°/tick.)
        # Falls back to cmd_pan/tilt when nothing has been written yet.
        if self._is_v2:
            # V2: encoder was already read at the top of _tick (and the
            # measured pose cached in self._last_measured_*). Reuse the
            # cached value here — avoids a second pair of bus reads per
            # tick (saves ~3-6 ms of bus time) and ensures the published
            # GimbalState matches the pose the predictor + closed-loop
            # actually used this tick.
            actual_pan  = (self._last_measured_pan
                           if self._last_measured_pan is not None else cmd_pan)
            actual_tilt = (self._last_measured_tilt
                           if self._last_measured_tilt is not None else cmd_tilt)
        else:
            last_us_pan  = self._driver.get_last_written_us(self._pan_cal.channel)
            last_us_tilt = self._driver.get_last_written_us(self._tilt_cal.channel)
            actual_pan  = (self._pan_cal.us_to_angle(last_us_pan)
                            if last_us_pan is not None else cmd_pan)
            actual_tilt = (self._tilt_cal.us_to_angle(last_us_tilt)
                            if last_us_tilt is not None else cmd_tilt)

        # Lock-mode tick: seed-on-pending, per-frame MOSSE update,
        # auto-reseed from matching fused observations. No-op when
        # gimbal.lock_mode.enabled=false in YAML, or when no track
        # is engaged.
        try:
            lock_bbox_eo, lock_bbox_thermal, lock_state_str = self._lock_mode_tick()
        except Exception:
            log.exception("lock_mode_tick failed; emitting empty lock state")
            lock_bbox_eo, lock_bbox_thermal, lock_state_str = None, None, "off"
        # Snapshot the current lock target id under the lock so the
        # GUI knows which fused-track green box to suppress.
        with self._lock:
            lock_target_id_pub = self._lock_target_id

        # Publish state. `tracked_target_id` carries whichever lock is
        # live — fused id if that's set, else the heat id. The GUI only
        # uses this for display, and the two namespaces don't collide
        # visually (heat rows are tagged H#N in the list).
        published_track = tracked_id if tracked_id is not None else tracked_heat_id
        state = GimbalState(
            timestamp=time.time(),
            connected=self._connected,
            pan_deg=actual_pan,
            tilt_deg=actual_tilt,
            mode=mode,
            target_pan_deg=sp_pan,
            target_tilt_deg=sp_tilt,
            tracked_target_id=published_track,
            error=err,
            synth_world_az_deg=self._synth_world_az_deg,
            synth_world_el_deg=self._synth_world_el_deg,
            target_resid_az_deg=self._latest_target_resid_az,
            target_resid_el_deg=self._latest_target_resid_el,
            lock_state=lock_state_str,
            lock_bbox_eo=lock_bbox_eo,
            lock_bbox_thermal=lock_bbox_thermal,
            lock_target_id=lock_target_id_pub,
        )
        BUS.publish(Topic.GIMBAL, state)

    def _lock_mode_tick(self) -> Tuple[Optional["BBox"], Optional["BBox"], str]:
        """Run one lock-mode tick: seed-on-pending, per-frame update,
        auto-reseed from a matching fused observation. Returns
        ``(lock_bbox_eo, lock_bbox_thermal, lock_state)`` for publish
        on GimbalState. When lock mode is disabled OR not engaged,
        returns ``(None, None, "off")``.

        The two sensor locks have independent state machines —
        thermal can be COASTING while EO is ACTIVE, or vice versa.
        The published lock_state is the WORST of the two (most
        conservative for the GUI's amber/green decision).
        """
        from common.frames import BBox
        if not self._lock_mode_enabled or self._tracked_id is None:
            return None, None, "off"

        now = time.time()
        ef = BUS.get_latest(Topic.EO)
        tf = BUS.get_latest(Topic.THERMAL)
        ef_ok = (isinstance(ef, EOFrame) and ef.connected
                  and ef.bgr is not None)
        tf_ok = (isinstance(tf, ThermalFrame) and tf.connected
                  and tf.agc8 is not None)

        # ── Seed-pending: latch the engagement bbox from the fused
        # track on the first tick after operator engagement.
        if self._lock_seed_pending:
            # Timeout: if seed_pending has been armed for too long
            # without a successful seed (engaged track never appeared
            # in fused, or per-sensor seed bbox kept failing
            # validation), give up. Without this, _lock_last_state_str
            # stays at "off" forever AFTER it transitions to "active"
            # via the state-machine block below — but more importantly,
            # the publish layer reports lock_state="off" forever and
            # the operator sees no feedback.
            if (self._lock_seed_pending_t0 is not None
                    and (now - self._lock_seed_pending_t0)
                        >= self._lock_seed_pending_timeout_s):
                try:
                    emit_event("lock_seed_timeout", {
                        "tracked_id": self._lock_target_id,
                        "elapsed_s": round(
                            now - self._lock_seed_pending_t0, 3),
                    })
                except Exception:
                    pass
                self._lock_seed_pending = False
                self._lock_seed_pending_t0 = None
                # Reset state-string so next engagement re-emits the
                # off→active transition cleanly.
                self._lock_last_state_str = "off"
            fused = BUS.get_latest(Topic.FUSED) or []
            target = next((t for t in fused
                            if getattr(t, "id", None) == self._lock_target_id),
                           None)
            if target is None:
                # Fused track gone (race between engage and tick) —
                # leave seed pending, retry next tick within grace.
                pass
            else:
                self._lock_target_class = getattr(
                    getattr(target, "target_class", None), "value", None)
                # Seed bbox source priority:
                #   1. The per-sensor detection that contributed to this
                #      fused track (EOFrame.detections matching
                #      target.eo_track_id; ThermalFrame.detections
                #      matching target.thermal_heat_id). This is the
                #      RAW classifier/heat-detector bbox tightly fit to
                #      the visible target — what MOSSE wants.
                #   2. Fall back to the angular reprojection
                #      (_fused_to_eo_bbox) only when the sensor didn't
                #      contribute (radar-only track) or the detection
                #      lookup fails.
                # Recording `track worse.jsonl` showed seeded bboxes
                # 240×156 (target #3) and 190×420 (target #15) — way
                # bigger than the visible humans because the fused
                # angular extent (ang_w_deg/ang_h_deg) gets contaminated
                # by radar's coarse cluster bbox. MOSSE seeded on a
                # 3-4x oversized patch then "tracks" the union of
                # target + background, jumping wildly.
                seeded_any = False
                eo_src = "skip"
                th_src = "skip"
                if ef_ok:
                    bbox_eo_det = self._eo_detection_bbox(target, ef)
                    if bbox_eo_det is not None:
                        bbox_eo = bbox_eo_det
                        eo_src = "eo_track_id"
                    else:
                        bbox_eo = self._fused_to_eo_bbox(target, ef)
                        eo_src = "fused_angular" if bbox_eo else "no_bbox"
                    if bbox_eo is not None:
                        if self._lock_eo.seed(ef.bgr, bbox_eo, now=now):
                            seeded_any = True
                if tf_ok:
                    bbox_th_det = self._thermal_detection_bbox(target, tf)
                    if bbox_th_det is not None:
                        bbox_th = bbox_th_det
                        th_src = "thermal_heat_id"
                    else:
                        bbox_th = self._fused_to_thermal_bbox(target, tf)
                        th_src = "fused_angular" if bbox_th else "no_bbox"
                    if bbox_th is not None:
                        if self._lock_thermal.seed(tf.agc8, bbox_th, now=now):
                            seeded_any = True
                if seeded_any:
                    self._lock_seed_pending = False
                    self._lock_seed_pending_t0 = None
                    # Observability: record which seed-source path was
                    # taken per sensor (eo_track_id / thermal_heat_id =
                    # tight per-sensor classifier bbox, the fix from
                    # commit 5670246; fused_angular = legacy projection
                    # via FusedTrack.ang_w/h, which can be oversized
                    # when radar contributes). Lets a future audit tell
                    # whether the silent oversize-bbox bug regressed.
                    try:
                        emit_event("lock_seed_source", {
                            "tracked_id": self._lock_target_id,
                            "eo_source": eo_src,
                            "thermal_source": th_src,
                        })
                    except Exception:
                        pass

        # ── Per-frame lock updates.
        # Dedupe by frame_id: only run MOSSE when the BUS-cached
        # frame is actually new. Re-running on the same frame burns
        # CPU and produces noise (FFT peak landing 1 px apart on
        # identical input, plus the online learning-rate retraining
        # the filter on the same patch).
        #
        # When dedupe hits, we publish a POSE-SHIFTED version of the
        # cached LockUpdate's bbox: the gimbal tick runs at 35-60 Hz
        # while EO publishes at 17-25 Hz, so several gimbal ticks
        # land on the same EO frame. Without the pose shift, the
        # operator sees the lock bbox "freeze" at the cached frame's
        # image-pixel coords during a slew (the camera has panned
        # since the cached frame was captured, so the same image-px
        # bbox would land off-target in the LATEST frame the GUI
        # is about to render). The shift = (cur_pan - pan_at_cached_
        # frame_capture) × pixels-per-degree gives an honest
        # interpolation between real MOSSE updates. Wave 2 review of
        # this morning's frame-id dedupe (commit 5670246) flagged
        # the freeze as the "sluggish lock bbox" symptom.
        cur_pan_now = float(getattr(self, "_last_measured_pan", 0.0)
                              or self._controller.current[0])
        cur_tilt_now = float(getattr(self, "_last_measured_tilt", 0.0)
                              or self._controller.current[1])
        if ef_ok and self._lock_eo.is_active:
            ef_fid = getattr(ef, "frame_id", None)
            if ef_fid is not None and ef_fid == self._lock_last_eo_fid:
                eo_upd = self._pose_shift_lock_update(
                    self._lock_last_eo_upd, ef,
                    cur_pan_now, cur_tilt_now)
            else:
                eo_upd = self._lock_eo.update(ef.bgr, now=now)
                self._lock_last_eo_fid = ef_fid
                self._lock_last_eo_upd = eo_upd
        else:
            eo_upd = None
            self._lock_last_eo_upd = None

        if tf_ok and self._lock_thermal.is_active:
            tf_fid = getattr(tf, "frame_id", None)
            if tf_fid is not None and tf_fid == self._lock_last_th_fid:
                th_upd = self._pose_shift_lock_update(
                    self._lock_last_th_upd, tf,
                    cur_pan_now, cur_tilt_now)
            else:
                th_upd = self._lock_thermal.update(tf.agc8, now=now)
                self._lock_last_th_fid = tf_fid
                self._lock_last_th_upd = th_upd
        else:
            th_upd = None
            self._lock_last_th_upd = None

        # ── Auto-reseed (v2): STRICT ID-MATCH ONLY.
        #
        # v1 used (class match + IoU >= 0.20) to find a "matching"
        # fused track and reseed onto it. In dense same-class scenes
        # (5+ vehicles in `recordings/lock poorly.jsonl` at t≈9.6s)
        # that gate fired on a different vehicle that briefly
        # overlapped the lock bbox, swapping the lock identity to a
        # passing target. Operator: "additional locked bbs that were
        # mixing and confusing between targets on the yolo side."
        #
        # v2: only reseed when the fused-track stream contains a
        # track with id == self._lock_target_id. The original ID
        # assigned by fusion at engagement is the only thing that
        # ever refreshes the appearance template. Drift onto a
        # different physical target becomes structurally impossible
        # because the only path to overwrite the template requires
        # ID-equality with the engaged track. If fusion drops the
        # engaged ID permanently (max_misses), the lock keeps
        # COASTING on appearance MOSSE alone, then HARD_RELEASED at
        # the coast window — operator clicks TRACK on a new ID for
        # a fresh lock. Trade-off accepted per 2026-05-05 retro.
        fused = BUS.get_latest(Topic.FUSED) or []
        target = None
        for trk in fused:
            if getattr(trk, "id", None) == self._lock_target_id:
                target = trk
                break
        if target is not None:
            # PSR + ACTIVE gate: require state == ACTIVE (not COASTING)
            # AND last_psr above the safety margin before reseeding.
            # Without this, auto-reseed fires DURING coasting recovery
            # on a low-PSR frame, imprinting a partially-occluded
            # patch as the new appearance template — visible in
            # `lock test test.jsonl` engagement #57 with PSR resumes
            # 5.91 / 6.38 (just above psr_lost=5.0). The 1.5x margin
            # collapses ~95% of low-quality reseeds without
            # weakening the legitimate refresh path on stable
            # tracks (PSR routinely 30-60 there). Set the multiplier
            # to 0 in YAML to disable the gate (legacy behavior).
            from vision.lock_tracker import LockState
            margin = float(self._lock_reseed_psr_margin)
            def _ok_to_reseed(lt):
                if margin <= 0:
                    return lt.is_active
                return (lt.state == LockState.ACTIVE
                        and lt.last_psr >= lt.psr_lost * margin)
            if (ef_ok and _ok_to_reseed(self._lock_eo)
                    and self._lock_eo.time_since_reseed(now=now)
                        >= self._lock_reseed_min_period_s):
                bbox_eo = (self._eo_detection_bbox(target, ef)
                            or self._fused_to_eo_bbox(target, ef))
                if bbox_eo is not None:
                    self._lock_eo.reseed(ef.bgr, bbox_eo, now=now)
            if (tf_ok and _ok_to_reseed(self._lock_thermal)
                    and self._lock_thermal.time_since_reseed(now=now)
                        >= self._lock_reseed_min_period_s):
                bbox_th = (self._thermal_detection_bbox(target, tf)
                            or self._fused_to_thermal_bbox(target, tf))
                if bbox_th is not None:
                    self._lock_thermal.reseed(tf.agc8, bbox_th, now=now)

        # ── Compose published values.
        def to_bbox(upd):
            if upd is None or upd.bbox_xywh is None:
                return None
            x, y, w, h = upd.bbox_xywh
            return BBox(x=int(x), y=int(y), w=int(w), h=int(h))

        bbox_eo_pub = to_bbox(eo_upd)
        bbox_th_pub = to_bbox(th_upd)

        # State priority: HARD_RELEASED > COASTING > ACTIVE > OFF.
        # We want the GUI to see "coasting" if EITHER sensor is
        # coasting (so it goes amber), and "active" only when both
        # are healthy.
        from vision.lock_tracker import LockState
        states = []
        if eo_upd is not None: states.append(eo_upd.state)
        if th_upd is not None: states.append(th_upd.state)
        if not states:
            lock_state = "off"
        elif LockState.HARD_RELEASED in states:
            lock_state = "released"
        elif LockState.COASTING in states:
            lock_state = "coasting"
        elif LockState.ACTIVE in states:
            lock_state = "active"
        else:
            lock_state = "off"

        # ── Emit per-transition events for replay-debug. One event
        # per real edge in the state machine, never per-tick.
        if lock_state != self._lock_last_state_str:
            ev_payload = {
                "tracked_id": self._lock_target_id,
                "from_state": self._lock_last_state_str,
                "to_state": lock_state,
                "psr_eo": (round(float(eo_upd.psr), 2)
                            if eo_upd is not None else None),
                "psr_thermal": (round(float(th_upd.psr), 2)
                                  if th_upd is not None else None),
            }
            event_name = {
                ("off", "active"):       "lock_seeded",
                ("active", "coasting"):  "lock_coasting_enter",
                ("coasting", "active"):  "lock_active_resume",
                ("coasting", "released"):"lock_hard_released",
                ("active", "released"):  "lock_hard_released",
            }.get((self._lock_last_state_str, lock_state),
                  "lock_state_change")
            try:
                emit_event(event_name, ev_payload)
            except Exception:
                pass
            self._lock_last_state_str = lock_state

        # ── HARD_RELEASED side effects: drop engagement so gimbal
        # returns to manual. Done after event emission so the
        # transition is recorded.
        if lock_state == "released":
            with self._lock:
                if self._tracked_id == self._lock_target_id:
                    self._tracked_id = None
            self._lock_eo.release()
            self._lock_thermal.release()
            self._lock_target_id = None
            self._lock_target_class = None
            self._lock_seed_pending = False
            self._lock_seed_pending_t0 = None
            self._lock_last_state_str = "off"
            return None, None, "released"
        return bbox_eo_pub, bbox_th_pub, lock_state

    def _pose_shift_lock_update(self, upd, frame,
                                  cur_pan_deg: float,
                                  cur_tilt_deg: float):
        """Pose-shift the cached LockUpdate's bbox so the operator
        sees the lock follow the gimbal between real frame updates.

        EO publishes at 17-25 Hz, gimbal at 35-60 Hz — between two
        real EO frames the gimbal pose advances. The cached MOSSE
        bbox is in IMAGE coords of the cached frame; rendering it
        verbatim looks "frozen" because the GUI's overlay sits on
        top of the latest gimbal pose. Compute the pose delta from
        the cached frame's gimbal_*_at_capture to the current pose
        and shift the bbox by (delta * px_per_deg) so the bracket
        keeps tracking the target through the slew. When the next
        real frame arrives, MOSSE re-anchors on actual image data
        and any prediction error is corrected in one tick.

        Returns a NEW LockUpdate instance with the shifted bbox so
        we don't mutate the cached value.
        """
        if upd is None or upd.bbox_xywh is None or frame is None:
            return upd
        from vision.lock_tracker import LockUpdate
        cap_pan = getattr(frame, "gimbal_pan_at_capture", None)
        cap_tilt = getattr(frame, "gimbal_tilt_at_capture", None)
        bgr_or_agc = getattr(frame, "bgr", None)
        if bgr_or_agc is None:
            bgr_or_agc = getattr(frame, "agc8", None)
        if (cap_pan is None or cap_tilt is None
                or bgr_or_agc is None):
            return upd
        h, w = bgr_or_agc.shape[:2]
        hfov = float(getattr(frame, "hfov_deg", 0.0))
        vfov = float(getattr(frame, "vfov_deg", 0.0))
        if hfov <= 0 or vfov <= 0:
            return upd
        d_pan = float(cur_pan_deg) - float(cap_pan)
        d_tilt = float(cur_tilt_deg) - float(cap_tilt)
        # Skip if delta is sub-pixel — saves a copy and avoids
        # introducing micro-jitter from float rounding.
        px_per_deg_x = w / hfov
        px_per_deg_y = h / vfov
        # Sign convention: when the gimbal pans RIGHT (+pan), the
        # scene shifts LEFT in the image → bbox.x decreases. Same
        # for tilt: tilt UP (+tilt) → scene moves DOWN in image →
        # bbox.y increases. Verified against
        # eo/eo_manager.py:1490+ phase-correlate sign conventions.
        dx = -d_pan * px_per_deg_x
        dy = d_tilt * px_per_deg_y
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return upd
        x, y, bw, bh = upd.bbox_xywh
        new_x = int(round(x + dx))
        new_y = int(round(y + dy))
        # Clamp to frame so a runaway slew can't push the bbox off
        # screen (the brackets render correctly even when partially
        # clipped, but a fully-off bbox vanishes).
        new_x = max(-bw + 1, min(w - 1, new_x))
        new_y = max(-bh + 1, min(h - 1, new_y))
        return LockUpdate(
            state=upd.state,
            bbox_xywh=(new_x, new_y, int(bw), int(bh)),
            psr=upd.psr,
            coast_age_s=upd.coast_age_s,
        )

    def _eo_detection_bbox(self, trk: Any,
                             ef: "EOFrame") -> Optional[Tuple[int, int, int, int]]:
        """Return the EO classifier's RAW bbox for the EO detection
        that contributed to this fused track, or None if the fused
        track has no EO contributor in this frame.

        Why this exists: ``_fused_to_eo_bbox`` reprojects the fused
        track's angular extent (ang_w_deg × ang_h_deg) into pixel
        space. That extent is the cross-sensor union — for a track
        with both EO and radar contributing, ang_w/h is dominated by
        radar's coarse cluster span (~3°) instead of YOLO's tight
        per-pixel bbox (~0.5°). MOSSE seeded on a 3-4× oversized
        patch tracks the union of target + background and jumps
        wildly. For seeding a per-sensor MOSSE, the right bbox is
        the SENSOR'S OWN detection, not the fused projection.
        """
        eo_tid = getattr(trk, "eo_track_id", None)
        if eo_tid is None:
            return None
        for det in getattr(ef, "detections", None) or []:
            if getattr(det, "track_id", None) == eo_tid:
                bb = det.bbox
                if bb is None:
                    return None
                return (int(bb.x), int(bb.y), int(bb.w), int(bb.h))
        return None

    def _thermal_detection_bbox(self, trk: Any,
                                  tf: "ThermalFrame") -> Optional[Tuple[int, int, int, int]]:
        """Same as ``_eo_detection_bbox`` but for the thermal panel —
        looks up the ThermalDetection whose ``track_id`` matches the
        fused track's ``thermal_heat_id``."""
        th_tid = getattr(trk, "thermal_heat_id", None)
        if th_tid is None:
            return None
        for det in getattr(tf, "detections", None) or []:
            if getattr(det, "track_id", None) == th_tid:
                bb = det.bbox
                if bb is None:
                    return None
                return (int(bb.x), int(bb.y), int(bb.w), int(bb.h))
        return None

    def _fused_to_eo_bbox(self, trk: Any,
                           ef: "EOFrame") -> Optional[Tuple[int, int, int, int]]:
        """Project a fused track onto the EO image's pose-at-capture.
        Mirrors gui.sensor_bridge.fused_to_wire's per-panel
        re-projection so the lock seed lands on the same pixel
        position the operator clicked on."""
        from fusion.angular import angular_bbox_visible, angular_to_bbox
        wa = getattr(trk, "world_az_deg", None)
        we = getattr(trk, "world_el_deg", None)
        e_pan = getattr(ef, "gimbal_pan_at_capture", None)
        e_tilt = getattr(ef, "gimbal_tilt_at_capture", None)
        if (wa is not None and we is not None
                and e_pan is not None and e_tilt is not None):
            az = wa - float(e_pan)
            el = we - float(e_tilt)
        else:
            az = float(getattr(trk, "az_deg", 0.0))
            el = float(getattr(trk, "el_deg", 0.0))
        ang_w = float(getattr(trk, "ang_w_deg", 0.0))
        ang_h = float(getattr(trk, "ang_h_deg", 0.0))
        if ang_w <= 0 or ang_h <= 0:
            return None
        if ef.bgr is None:
            return None
        h, w = ef.bgr.shape[:2]
        hfov = float(getattr(ef, "hfov_deg", 11.05))
        vfov = float(getattr(ef, "vfov_deg", 9.23))
        if not angular_bbox_visible(az, el, ang_w, ang_h, hfov, vfov):
            return None
        x, y, bw, bh = angular_to_bbox(az, el, ang_w, ang_h,
                                         w, h, hfov, vfov)
        if bw <= 0 or bh <= 0:
            return None
        return (int(x), int(y), int(bw), int(bh))

    def _fused_to_thermal_bbox(self, trk: Any,
                                 tf: "ThermalFrame") -> Optional[Tuple[int, int, int, int]]:
        """Same as _fused_to_eo_bbox but for the thermal panel,
        accounting for the thermal extrinsic bias."""
        from fusion.angular import angular_bbox_visible, angular_to_bbox
        wa = getattr(trk, "world_az_deg", None)
        we = getattr(trk, "world_el_deg", None)
        t_pan = getattr(tf, "gimbal_pan_at_capture", None)
        t_tilt = getattr(tf, "gimbal_tilt_at_capture", None)
        bias_az = float(getattr(self, "_thermal_az_bias_deg", 0.0))
        bias_el = float(getattr(self, "_thermal_el_bias_deg", 0.0))
        if (wa is not None and we is not None
                and t_pan is not None and t_tilt is not None):
            az = wa - float(t_pan) - bias_az
            el = we - float(t_tilt) - bias_el
        else:
            az = float(getattr(trk, "az_deg", 0.0)) - bias_az
            el = float(getattr(trk, "el_deg", 0.0)) - bias_el
        ang_w = float(getattr(trk, "ang_w_deg", 0.0))
        ang_h = float(getattr(trk, "ang_h_deg", 0.0))
        if ang_w <= 0 or ang_h <= 0:
            return None
        if tf.agc8 is None:
            return None
        h, w = tf.agc8.shape[:2]
        hfov = float(getattr(tf, "hfov_deg", 37.0))
        vfov = float(getattr(tf, "vfov_deg", 30.0))
        if not angular_bbox_visible(az, el, ang_w, ang_h, hfov, vfov):
            return None
        x, y, bw, bh = angular_to_bbox(az, el, ang_w, ang_h,
                                         w, h, hfov, vfov)
        if bw <= 0 or bh <= 0:
            return None
        return (int(x), int(y), int(bw), int(bh))

    def _resolve_heat_track(self, heat_id: int) -> Optional[_HeatObs]:
        """Look up the current az/el of a heat-blob tracker ID.

        Pulls the freshest ThermalFrame from the bus, finds the
        `heat_tracks` entry matching ``heat_id``, and converts the bbox
        centroid into off-boresight angles using the frame's FOV (which
        reflects the active zoom preset). Returns None if the track
        has left the list or if we don't have the metadata needed to
        do the pixel → angle math.
        """
        tf = BUS.get_latest(Topic.THERMAL)
        if not isinstance(tf, ThermalFrame) or not tf.connected:
            return None
        if tf.agc8 is None:
            return None

        h, w = tf.agc8.shape[:2]
        if w <= 0 or h <= 0:
            return None

        hit = None
        for ht in (tf.heat_tracks or []):
            if int(ht.id) == int(heat_id):
                hit = ht
                break
        if hit is None:
            return None

        # Bbox center in display-pixel space (already scaled up from any
        # zoom crop by ThermalManager; tf.hfov_deg/vfov_deg describe the
        # currently-visible FOV, so this conversion is crop-aware).
        cx = hit.bbox.x + hit.bbox.w * 0.5
        cy = hit.bbox.y + hit.bbox.h * 0.5

        # Normalized offset from center: -0.5 .. +0.5
        nx = (cx / float(w)) - 0.5
        ny = (cy / float(h)) - 0.5

        az = nx * float(tf.hfov_deg)
        # y=0 is top of image; el positive = up → flip sign
        el = -ny * float(tf.vfov_deg)
        return _HeatObs(
            az_deg=float(az),
            el_deg=float(el),
            synthetic=bool(getattr(hit, "synthetic", False)),
            cx_px=float(cx),
            cy_px=float(cy),
        )

    # ── Optical residual (Stage A: diagnostic only) ─────────────
    def _opt_capture_anchors(self, cur_pan: float, cur_tilt: float) -> None:
        """Snapshot the latest EO + thermal frames as residual anchors.

        Called once when synth lock commits. Best-effort — logs but
        does not raise on missing frames or cv2.
        """
        now = time.time()
        ef = BUS.get_latest(Topic.EO)
        if isinstance(ef, EOFrame) and ef.connected and ef.bgr is not None:
            ok = self._opt_eo.set_anchor(
                ef.bgr, hfov_deg=float(ef.hfov_deg),
                vfov_deg=float(ef.vfov_deg),
                cur_pan=cur_pan, cur_tilt=cur_tilt, t=now)
            self._opt_last_eo_frame_id = int(ef.frame_id) if ok else None
            if ok:
                log.info("Optical anchor (EO) captured: %dx%d hfov=%.1f vfov=%.1f",
                         ef.bgr.shape[1], ef.bgr.shape[0],
                         ef.hfov_deg, ef.vfov_deg)
        tf = BUS.get_latest(Topic.THERMAL)
        if isinstance(tf, ThermalFrame) and tf.connected and tf.agc8 is not None:
            ok = self._opt_thermal.set_anchor(
                tf.agc8, hfov_deg=float(tf.hfov_deg),
                vfov_deg=float(tf.vfov_deg),
                cur_pan=cur_pan, cur_tilt=cur_tilt, t=now)
            self._opt_last_thermal_frame_id = int(tf.frame_id) if ok else None
            if ok:
                log.info("Optical anchor (thermal) captured: %dx%d hfov=%.1f vfov=%.1f",
                         tf.agc8.shape[1], tf.agc8.shape[0],
                         tf.hfov_deg, tf.vfov_deg)

    def _check_stuck_servo(self, source: str, m) -> None:
        """Detect when a commanded slew isn't being delivered by the
        servo (LK-measured actual motion stays well below required for
        several consecutive samples post-warmup) and release the
        servos to float so we don't keep pushing PWM at a stuck servo.

        Operator-driven safety per 2026-04-27 session: at high tilt the
        camera physically wasn't following commands (recordings/
        diag_high_tilt_tree.jsonl, BB#3: cmd -5.9 deg pan -4.5 deg tilt,
        actual -0.08 deg / 0.02 deg). Without this guard our SW would
        keep commanding and potentially overheat or damage the servo.
        """
        if not self._stuck_enabled:
            return
        if (self._synth_world_az_deg is None
                or self._synth_world_el_deg is None
                or self._synth_lock_t is None):
            return
        elapsed = time.time() - self._synth_lock_t
        if elapsed < self._stuck_warmup_s:
            return
        if source == "eo":
            anchor_pan = self._opt_eo.anchor_pan
            anchor_tilt = self._opt_eo.anchor_tilt
        else:
            anchor_pan = self._opt_thermal.anchor_pan
            anchor_tilt = self._opt_thermal.anchor_tilt
        if anchor_pan is None or anchor_tilt is None:
            return
        required_az = self._synth_world_az_deg - anchor_pan
        required_el = self._synth_world_el_deg - anchor_tilt
        # actual = LK-measured motion since anchor
        actual_az = float(m.daz_actual_deg)
        actual_el = float(m.del_actual_deg)
        # "Stuck" on an axis = required > min AND actual delivered
        # less than (1 - residual_frac) * required, i.e. we asked for
        # a real slew but most of it didn't happen.
        stuck_pan = (abs(required_az) >= self._stuck_required_min_deg
                     and (abs(actual_az) <
                          (1.0 - self._stuck_residual_frac)
                          * abs(required_az)))
        stuck_tilt = (abs(required_el) >= self._stuck_required_min_deg
                      and (abs(actual_el) <
                           (1.0 - self._stuck_residual_frac)
                           * abs(required_el)))
        if stuck_pan or stuck_tilt:
            self._stuck_consec += 1
        else:
            self._stuck_consec = 0
            return
        if self._stuck_consec < self._stuck_consec_threshold:
            return
        # Stuck condition confirmed. Release the servos to float (PWM=0)
        # and clear the synth lock so the gimbal stops trying.
        if self._stuck_released:
            return  # already released for this lock cycle
        self._stuck_released = True
        log.warning("SERVO STUCK detected — required=(%.2f, %.2f) "
                    "actual=(%.2f, %.2f) source=%s elapsed=%.1fs — "
                    "releasing servos + clearing synth lock",
                    required_az, required_el,
                    actual_az, actual_el, source, elapsed)
        try:
            emit_event("servo_stuck", {
                "source": source,
                "elapsed_s": float(elapsed),
                "required_az_deg": float(required_az),
                "required_el_deg": float(required_el),
                "actual_az_deg": float(actual_az),
                "actual_el_deg": float(actual_el),
                "stuck_pan": bool(stuck_pan),
                "stuck_tilt": bool(stuck_tilt),
            })
        except Exception:
            pass
        try:
            self._driver.release_all(self._axis_addrs())
        except Exception as e:
            log.warning("release_all failed: %s", e)
        # Clear synth lock so subsequent ticks don't re-engage.
        self._synth_world_az_deg = None
        self._synth_world_el_deg = None
        self._tracked_heat_id = None

    def _opt_pick_best(self, eo_metrics, thermal_metrics):
        """Choose which sensor's measurement to feed into the integrator.

        EO is primary when it has enough features and isn't stale AND
        the world target is within the EO FOV (otherwise the original
        anchor scene is off-frame and LK matches noise to noise,
        producing spurious data — observed live in the first A/B run
        where corr_az saturated at -8 because EO LK reported false
        small motion during a 25 deg slew that put the anchor scene
        completely outside the 11 deg EO frame).

        Thermal at 75/37.5/18.75/12.5 deg is the fallback. Even at
        the narrowest preset (12.5 deg), thermal sees ~6x more world
        than EO.

        Returns (label, metrics) or (None, None) when neither is usable.
        """
        def _ok(m) -> bool:
            if m is None or not m.valid:
                return False
            if m.n_features < self._opt_corr_min_features:
                return False
            if m.note == "stale":
                return False
            return True

        # FOV gate for EO. The required slew = (synth_world - anchor_pan)
        # if anchor exists. Anything bigger than ~40% of EO half-FOV
        # means the anchor scene is mostly off-frame in EO.
        eo_in_range = True
        if (_ok(eo_metrics) and self._synth_world_az_deg is not None
                and self._opt_eo.anchor_pan is not None):
            req_az = abs(self._synth_world_az_deg - self._opt_eo.anchor_pan)
            req_el = abs(self._synth_world_el_deg - self._opt_eo.anchor_tilt)
            # Use the metrics' anchored FOV (we stored hfov/vfov at
            # anchor time). Half-FOV * 0.4 = "comfortable" range.
            eo_hfov_half = self._opt_eo._anchor.hfov * 0.5 if self._opt_eo._anchor else 5.5
            eo_vfov_half = self._opt_eo._anchor.vfov * 0.5 if self._opt_eo._anchor else 4.6
            if (req_az > 0.4 * eo_hfov_half * 2  # i.e. 0.4 * full hfov
                    or req_el > 0.4 * eo_vfov_half * 2):
                eo_in_range = False

        if _ok(eo_metrics) and eo_in_range:
            return "eo", eo_metrics
        if _ok(thermal_metrics):
            return "thermal", thermal_metrics
        return None, None

    def _opt_update_correction(self, source: str, m) -> None:
        """Stage B integrator: cumulative correction += alpha * target_residual.

        We use the **target residual** (how much further the camera
        needs to move to land on the synth-locked world target), not
        the metrics' "cmd vs actual" residual.

            target_resid_az = (synth_world_az - anchor_pan) - daz_lk_actual

        The "cmd vs actual" residual reports the SERVO error (commanded
        but not delivered). It stays equal to the mechanical disturbance
        even when the camera has reached the world target with a built-
        up correction — so feeding it into the integrator would never
        let the integrator stop growing.

        target_resid goes to 0 exactly when camera lands at the world
        target. Integrator stops growing. Anti-windup cap (±max_deg)
        catches the case where the servo cannot physically reach the
        target despite the cumulative bias.
        """
        if (self._synth_world_az_deg is None
                or self._synth_world_el_deg is None):
            return
        if source == "eo":
            anchor_pan = self._opt_eo.anchor_pan
            anchor_tilt = self._opt_eo.anchor_tilt
        else:
            anchor_pan = self._opt_thermal.anchor_pan
            anchor_tilt = self._opt_thermal.anchor_tilt
        if anchor_pan is None or anchor_tilt is None:
            return

        required_az = self._synth_world_az_deg - anchor_pan
        required_el = self._synth_world_el_deg - anchor_tilt
        target_resid_az = required_az - float(m.daz_actual_deg)
        target_resid_el = required_el - float(m.del_actual_deg)

        cur_pan, cur_tilt = self._controller.current
        # Settled gate: don't run integrator while gimbal is still
        # slewing to the current setpoint. Otherwise the LK actual_delta
        # is mid-slew and "looks like" undershoot, kicking the integrator.
        cmd_pan = self._synth_world_az_deg + self._opt_corr_az
        cmd_tilt = self._synth_world_el_deg + self._opt_corr_el
        if (abs(cur_pan - cmd_pan) > self._opt_corr_settled_deg
                or abs(cur_tilt - cmd_tilt) > self._opt_corr_settled_deg):
            return  # still slewing — skip update this tick
        # Saturation-aware update. If the controller is at a mechanical
        # pan/tilt limit AND the residual would push further into the
        # limit, skip the update on that axis. Otherwise the integrator
        # winds up against the wall (observed first A/B run: world
        # target at -43 deg with pan limit -45 deg, integrator
        # saturated at -8 deg correction trying to push a camera that
        # was already clamped).
        pan_sat_lo = (cmd_pan <= self._pan_floor + 0.1
                      and target_resid_az < 0)
        pan_sat_hi = (cmd_pan >= self._pan_ceil - 0.1
                      and target_resid_az > 0)
        tilt_sat_lo = (cmd_tilt <= self._tilt_floor + 0.1
                       and target_resid_el < 0)
        tilt_sat_hi = (cmd_tilt >= self._tilt_ceil - 0.1
                       and target_resid_el > 0)

        a = self._opt_corr_alpha
        cap = self._opt_corr_max_deg
        step_cap = self._opt_corr_step_max_deg
        if not (pan_sat_lo or pan_sat_hi):
            delta = a * target_resid_az
            if   delta >  step_cap: delta =  step_cap
            elif delta < -step_cap: delta = -step_cap
            new_az = self._opt_corr_az + delta
            if   new_az >  cap: new_az =  cap
            elif new_az < -cap: new_az = -cap
            self._opt_corr_az = new_az
        if not (tilt_sat_lo or tilt_sat_hi):
            delta = a * target_resid_el
            if   delta >  step_cap: delta =  step_cap
            elif delta < -step_cap: delta = -step_cap
            new_el = self._opt_corr_el + delta
            if   new_el >  cap: new_el =  cap
            elif new_el < -cap: new_el = -cap
            self._opt_corr_el = new_el
        self._opt_corr_last_source = source

    def _opt_measure_and_emit(self, cur_pan: float, cur_tilt: float,
                              tracked_id: Optional[int]) -> None:
        """Measure visual residual on EO + thermal against anchors,
        update the Stage B integrator (when enabled), and emit the
        `optical_residual` event.

        Skips a tick when no fresh frame has arrived for the sensor
        (matched by frame_id) — avoids reprocessing the same frame at
        60 Hz when the camera publishes at ~20 Hz.
        """
        eo_metrics = None
        thermal_metrics = None
        # EO
        if self._opt_eo.has_anchor:
            ef = BUS.get_latest(Topic.EO)
            if (isinstance(ef, EOFrame) and ef.connected
                    and ef.bgr is not None
                    and int(ef.frame_id) != self._opt_last_eo_frame_id):
                self._opt_last_eo_frame_id = int(ef.frame_id)
                eo_metrics = self._opt_eo.measure(
                    ef.bgr, cur_pan=cur_pan, cur_tilt=cur_tilt,
                    hfov_deg=float(ef.hfov_deg),
                    vfov_deg=float(ef.vfov_deg))
        # Thermal
        if self._opt_thermal.has_anchor:
            tf = BUS.get_latest(Topic.THERMAL)
            if (isinstance(tf, ThermalFrame) and tf.connected
                    and tf.agc8 is not None
                    and int(tf.frame_id) != self._opt_last_thermal_frame_id):
                self._opt_last_thermal_frame_id = int(tf.frame_id)
                thermal_metrics = self._opt_thermal.measure(
                    tf.agc8, cur_pan=cur_pan, cur_tilt=cur_tilt,
                    hfov_deg=float(tf.hfov_deg),
                    vfov_deg=float(tf.vfov_deg))
        if eo_metrics is None and thermal_metrics is None:
            return

        # Stage B: drive the correction integrator from the best
        # available source. Skip during the warmup window to avoid
        # interpreting in-progress slew as undershoot.
        in_warmup = (self._synth_lock_t is not None
                     and (time.time() - self._synth_lock_t)
                         < self._opt_corr_warmup_s)
        # Pick the best source for both Stage B correction (when
        # enabled) and stuck-servo detection.
        src, picked = self._opt_pick_best(eo_metrics, thermal_metrics)
        if self._opt_corr_enabled and not in_warmup and picked is not None:
            self._opt_update_correction(src, picked)
        if picked is not None:
            self._check_stuck_servo(src, picked)
            # Also cache the target residual (world target's current
            # image-frame position in degrees) for publication.
            if (self._synth_world_az_deg is not None
                    and self._synth_world_el_deg is not None):
                if src == "eo":
                    anchor_pan = self._opt_eo.anchor_pan
                    anchor_tilt = self._opt_eo.anchor_tilt
                else:
                    anchor_pan = self._opt_thermal.anchor_pan
                    anchor_tilt = self._opt_thermal.anchor_tilt
                if anchor_pan is not None and anchor_tilt is not None:
                    req_az = self._synth_world_az_deg - anchor_pan
                    req_el = self._synth_world_el_deg - anchor_tilt
                    self._latest_target_resid_az = (
                        req_az - float(picked.daz_actual_deg))
                    self._latest_target_resid_el = (
                        req_el - float(picked.del_actual_deg))

        payload: dict = {"tracked_heat_id": tracked_id,
                         "cur_pan": float(cur_pan),
                         "cur_tilt": float(cur_tilt),
                         "corr_enabled": self._opt_corr_enabled,
                         "corr_az_deg": float(self._opt_corr_az),
                         "corr_el_deg": float(self._opt_corr_el),
                         "corr_source": self._opt_corr_last_source,
                         "in_warmup": bool(in_warmup)}
        if eo_metrics is not None and eo_metrics.valid:
            payload["eo"] = {
                "n_features": eo_metrics.n_features,
                "dx_px": eo_metrics.dx_px, "dy_px": eo_metrics.dy_px,
                "daz_actual_deg": eo_metrics.daz_actual_deg,
                "del_actual_deg": eo_metrics.del_actual_deg,
                "daz_cmd_deg": eo_metrics.daz_cmd_deg,
                "del_cmd_deg": eo_metrics.del_cmd_deg,
                "daz_residual_deg": eo_metrics.daz_residual_deg,
                "del_residual_deg": eo_metrics.del_residual_deg,
                "note": eo_metrics.note,
            }
        if thermal_metrics is not None and thermal_metrics.valid:
            payload["thermal"] = {
                "n_features": thermal_metrics.n_features,
                "dx_px": thermal_metrics.dx_px, "dy_px": thermal_metrics.dy_px,
                "daz_actual_deg": thermal_metrics.daz_actual_deg,
                "del_actual_deg": thermal_metrics.del_actual_deg,
                "daz_cmd_deg": thermal_metrics.daz_cmd_deg,
                "del_cmd_deg": thermal_metrics.del_cmd_deg,
                "daz_residual_deg": thermal_metrics.daz_residual_deg,
                "del_residual_deg": thermal_metrics.del_residual_deg,
                "note": thermal_metrics.note,
            }
        if "eo" in payload or "thermal" in payload:
            try:
                emit_event("optical_residual", payload)
            except Exception:
                pass

    def _command_now(self, pan_deg: float, tilt_deg: float) -> None:
        # Auto-reconnect on transient failures. The Pololu Maestro's
        # USB-CDC driver on Windows occasionally raises a "device
        # doesn't recognize the command" PermissionError 13 under
        # sustained 60 Hz writes — this is a known Windows USB-CDC
        # quirk, not a real fault. Permanently marking the gimbal
        # disconnected after one transient failure (the previous
        # behaviour) caused the open-loop gimbal to silently go offline
        # ~8 s into a session, with `_controller.current` continuing to
        # advance based on commanded setpoints, so gimbal/state lied
        # about reality. Confirmed in recordings/ab2_off_mid.jsonl:
        # connected=False for all 1363 gimbal/state samples while
        # commanded pan moved -15 -> -27.6, but thermal LK reported
        # zero scene shift.
        if not self._connected:
            # Throttle reconnect attempts. Without the gate this fires
            # at the 60 Hz tick rate, and driver.open() logs a warning
            # each time it can't find the adapter — tens of writes/sec
            # on the global logger queue stalls thermal/EO publishers.
            now = time.time()
            if now < self._next_reconnect_ts:
                return
            self._next_reconnect_ts = now + 5.0  # try again in 5 s
            self._connected = self._driver.open()
            if not self._connected:
                return
            log.info("Maestro re-connected after transient failure")
            self._consec_write_fail = 0
        if self._is_v2:
            raw_p = self._pan_cal.angle_to_units(pan_deg)
            raw_t = self._tilt_cal.angle_to_units(tilt_deg)
            ok1 = self._driver.set_target_units(self._pan_cal.servo_id, raw_p)
            ok2 = self._driver.set_target_units(self._tilt_cal.servo_id, raw_t)
        else:
            us_p, us_t = self._controller.angles_to_us(pan_deg, tilt_deg)
            ok1 = self._driver.set_target_us(self._pan_cal.channel,  us_p)
            ok2 = self._driver.set_target_us(self._tilt_cal.channel, us_t)
        if not (ok1 and ok2):
            self._consec_write_fail += 1
            if self._consec_write_fail >= 3:
                # Several consecutive failures: drop the handle so the
                # next tick will attempt a fresh open(). pyserial seems
                # to recover after a close+open cycle even when the
                # underlying USB-CDC driver is in a stuck state.
                log.warning("Maestro: %d consec write fails — dropping "
                            "handle for re-open on next tick",
                            self._consec_write_fail)
                try:
                    self._driver.close()
                except Exception:
                    pass
                self._connected = False
        else:
            self._consec_write_fail = 0

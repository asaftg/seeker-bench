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
from typing import Optional

from common.config import load_config
from common.frame_bus import BUS
from common.frames import GimbalState, ThermalFrame, Topic
from common.logging_setup import get_logger
from gimbal.gimbal_controller import (
    GimbalController,
    GimbalLimits,
    ServoCalibration,
)
from gimbal.maestro_driver import MaestroDriver

log = get_logger(__name__)


@dataclass
class _HeatObs:
    """Internal az/el observation derived from a heat-blob bbox."""
    az_deg: float
    el_deg: float
    synthetic: bool = False  # True if the source is a user-drawn target


def _clip(v: float, lim: float) -> float:
    """Symmetric clamp: constrain v to [-lim, +lim]."""
    if lim <= 0.0:
        return v
    if v >  lim: return  lim
    if v < -lim: return -lim
    return v


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

        # ── Calibration (per-servo linear map angle→µs) ──────
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

        lims_cfg = (gcfg.get("limits") or {})
        limits = GimbalLimits(
            pan_min_deg=float(lims_cfg.get("pan_min_deg",  self._pan_cal.min_deg)),
            pan_max_deg=float(lims_cfg.get("pan_max_deg",  self._pan_cal.max_deg)),
            tilt_min_deg=float(lims_cfg.get("tilt_min_deg", self._tilt_cal.min_deg)),
            tilt_max_deg=float(lims_cfg.get("tilt_max_deg", self._tilt_cal.max_deg)),
            pan_slew_deg_per_s=float(lims_cfg.get("pan_slew_deg_per_s", 120.0)),
            tilt_slew_deg_per_s=float(lims_cfg.get("tilt_slew_deg_per_s", 60.0)),
        )

        home_pan  = float(gcfg.get("home_pan_deg", 0.0))
        # Middle of the 0..22 tilt envelope by default
        home_tilt = float(gcfg.get("home_tilt_deg",
                                   (limits.tilt_min_deg + limits.tilt_max_deg) / 2.0))

        self._controller = GimbalController(
            pan_cal=self._pan_cal,
            tilt_cal=self._tilt_cal,
            limits=limits,
            home_pan_deg=home_pan,
            home_tilt_deg=home_tilt,
        )

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

        # Pan-only saturation log throttle — we emit one INFO when we
        # enter saturation and another when we exit, but nothing in
        # between (would spam at rate_hz).
        self._tilt_saturated_logged = False

        # Driver — may or may not actually open.
        self._driver = MaestroDriver(port=port or gcfg.get("port"))
        self._connected = False

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

    # ── lifecycle ─────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._connected = self._driver.open()
        if self._connected:
            # Move gently to home instead of snapping — avoids a
            # startup slam when the servos wake up at a random µs.
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
            self._driver.release_all([self._pan_cal.channel, self._tilt_cal.channel])
        self._driver.close()
        log.info("GimbalManager stopped")

    # ── GUI / CLI inputs ──────────────────────────────────────

    def set_manual_delta(self, d_pan_deg: float, d_tilt_deg: float) -> None:
        """Incremental nudge from the dpad. Implicitly releases any
        active TRACK lock (fused or heat) so the user's arrows always
        take priority."""
        with self._lock:
            if self._tracked_id is not None:
                log.info("Manual nudge → releasing fused track lock on #%d", self._tracked_id)
                self._tracked_id = None
            if self._tracked_heat_id is not None:
                log.info("Manual nudge → releasing heat track lock on H#%d", self._tracked_heat_id)
                self._tracked_heat_id = None
            self._manual_pan  = self._manual_pan  + float(d_pan_deg)
            self._manual_tilt = self._manual_tilt + float(d_tilt_deg)

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
            if track_id is None:
                self._tracked_id = None
                log.info("Fused track lock cleared → manual")
            else:
                try:
                    self._tracked_id = int(track_id)
                    # Fused lock takes priority over any heat lock.
                    self._tracked_heat_id = None
                    log.info("Fused track lock engaged on #%d", self._tracked_id)
                except (TypeError, ValueError):
                    log.warning("Bad track_id: %r", track_id)

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
                cur_pan, cur_tilt = self._controller.current
                if fresh_frame or self._last_sp_pan is None:
                    if self._cameras_on_gimbal:
                        # First-order low-pass on the raw error before
                        # deadband + gain. Smooths centroid jitter on
                        # multi-patch targets (vehicles, humans) without
                        # adding meaningful lag for real motion.
                        az_in, el_in = self._lp_filter_error(
                            float(heat_obs.az_deg), float(heat_obs.el_deg))
                        # Synthetic targets have extra loop delay: OF
                        # needs (prev, curr) + Kalman smoothing adds its
                        # own dynamics. Using the same kp/deadband as a
                        # direct heat-blob centroid pumps the servo. Soft
                        # gains here trade settling precision for
                        # stability — acceptable because a user-drawn
                        # box doesn't need pixel-perfect centering.
                        if heat_obs.synthetic:
                            kp_eff  = self._kp_track * 0.6
                            band    = self._deadband_deg * 1.6
                            step_eff= self._max_step_deg * 0.6
                        else:
                            kp_eff  = self._kp_track
                            band    = self._deadband_deg
                            step_eff= self._max_step_deg
                        az, el, self._in_deadband = _hyst_deadband(
                            az_in, el_in,
                            band, self._deadband_exit_ratio,
                            self._in_deadband)
                        d_pan  = _clip(kp_eff * az, step_eff)
                        d_tilt = _clip(kp_eff * el, step_eff)
                        d_tilt, tilt_sat = _pan_only_if_tilt_saturated(
                            cur_tilt, d_tilt,
                            self._tilt_floor, self._tilt_ceil,
                            self._tilt_sat_eps_deg)
                        if tilt_sat and not self._tilt_saturated_logged:
                            log.info("Tilt saturated at mechanical stop "
                                     "(cur=%.1f, el_err=%.2f) — pan-only",
                                     cur_tilt, el)
                            self._tilt_saturated_logged = True
                        elif not tilt_sat:
                            self._tilt_saturated_logged = False
                        sp_pan  = cur_pan  + d_pan
                        sp_tilt = cur_tilt + d_tilt
                    else:
                        sp_pan  = self._home_pan  + float(heat_obs.az_deg)
                        sp_tilt = self._home_tilt + float(heat_obs.el_deg)
                    self._last_track_ts = tf_ts
                    self._last_sp_pan   = sp_pan
                    self._last_sp_tilt  = sp_tilt
                else:
                    # Stale frame — hold last setpoint; servo keeps slewing.
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
                cur_pan, cur_tilt = self._controller.current
                if fresh_frame or self._last_sp_pan is None:
                    if self._cameras_on_gimbal:
                        az_in, el_in = self._lp_filter_error(
                            float(trk.az_deg), float(trk.el_deg))
                        az, el, self._in_deadband = _hyst_deadband(
                            az_in, el_in,
                            self._deadband_deg, self._deadband_exit_ratio,
                            self._in_deadband)
                        d_pan  = _clip(self._kp_track * az, self._max_step_deg)
                        d_tilt = _clip(self._kp_track * el, self._max_step_deg)
                        d_tilt, tilt_sat = _pan_only_if_tilt_saturated(
                            cur_tilt, d_tilt,
                            self._tilt_floor, self._tilt_ceil,
                            self._tilt_sat_eps_deg)
                        if tilt_sat and not self._tilt_saturated_logged:
                            log.info("Tilt saturated at mechanical stop "
                                     "(cur=%.1f, el_err=%.2f) — pan-only",
                                     cur_tilt, el)
                            self._tilt_saturated_logged = True
                        elif not tilt_sat:
                            self._tilt_saturated_logged = False
                        sp_pan  = cur_pan  + d_pan
                        sp_tilt = cur_tilt + d_tilt
                    else:
                        # Bench setup: cameras stationary. az/el is already
                        # absolute bench-frame; command directly.
                        sp_pan  = self._home_pan  + float(trk.az_deg)
                        sp_tilt = self._home_tilt + float(trk.el_deg)
                    self._last_track_ts = tf_ts
                    self._last_sp_pan   = sp_pan
                    self._last_sp_tilt  = sp_tilt
                else:
                    sp_pan  = self._last_sp_pan
                    sp_tilt = self._last_sp_tilt
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
                    with self._lock:
                        if self._tracked_id == tracked_id:
                            self._tracked_id = None
                    self._track_miss = 0
                    mode = "manual"
                # Hold current position while waiting
                sp_pan, sp_tilt = self._controller.current
        else:
            self._track_miss = 0
            # Not tracking — flush cached setpoints so next engage
            # starts fresh against whatever the new target's error is.
            self._last_track_ts = None
            self._last_sp_pan   = None
            self._last_sp_tilt  = None
            self._in_deadband   = False
            self._reset_track_filter()
            self._tilt_saturated_logged = False

        # Slew + clamp
        cmd_pan, cmd_tilt = self._controller.step(sp_pan, sp_tilt)
        self._command_now(cmd_pan, cmd_tilt)

        # Publish state. `tracked_target_id` carries whichever lock is
        # live — fused id if that's set, else the heat id. The GUI only
        # uses this for display, and the two namespaces don't collide
        # visually (heat rows are tagged H#N in the list).
        published_track = tracked_id if tracked_id is not None else tracked_heat_id
        state = GimbalState(
            timestamp=time.time(),
            connected=self._connected,
            pan_deg=cmd_pan,
            tilt_deg=cmd_tilt,
            mode=mode,
            target_pan_deg=sp_pan,
            target_tilt_deg=sp_tilt,
            tracked_target_id=published_track,
            error=err,
        )
        BUS.publish(Topic.GIMBAL, state)

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
        )

    def _command_now(self, pan_deg: float, tilt_deg: float) -> None:
        if not self._connected:
            return
        us_p, us_t = self._controller.angles_to_us(pan_deg, tilt_deg)
        ok1 = self._driver.set_target_us(self._pan_cal.channel,  us_p)
        ok2 = self._driver.set_target_us(self._tilt_cal.channel, us_t)
        if not (ok1 and ok2):
            # Mark disconnected on a write failure; a subsequent
            # open() attempt could be added here but it keeps things
            # simple to just sit idle.
            log.warning("Maestro write failed — marking disconnected")
            self._connected = False

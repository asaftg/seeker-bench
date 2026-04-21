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
from typing import Optional

from common.config import load_config
from common.frame_bus import BUS
from common.frames import GimbalState, Topic
from common.logging_setup import get_logger
from gimbal.gimbal_controller import (
    GimbalController,
    GimbalLimits,
    ServoCalibration,
)
from gimbal.maestro_driver import MaestroDriver

log = get_logger(__name__)


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

        # Driver — may or may not actually open.
        self._driver = MaestroDriver(port=port or gcfg.get("port"))
        self._connected = False

        # Manual setpoint (mutated by GUI dpad / WASD CLI)
        self._manual_pan  = home_pan
        self._manual_tilt = home_tilt

        # Track-lock state
        self._tracked_id: Optional[int] = None

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
        active TRACK lock so the user's arrows always take priority."""
        with self._lock:
            if self._tracked_id is not None:
                log.info("Manual nudge → releasing track lock on #%d", self._tracked_id)
                self._tracked_id = None
            self._manual_pan  = self._manual_pan  + float(d_pan_deg)
            self._manual_tilt = self._manual_tilt + float(d_tilt_deg)

    def set_manual_absolute(self, pan_deg: float, tilt_deg: float) -> None:
        with self._lock:
            self._tracked_id  = None
            self._manual_pan  = float(pan_deg)
            self._manual_tilt = float(tilt_deg)

    def set_home(self) -> None:
        self.set_manual_absolute(self._home_pan, self._home_tilt)

    def set_track_target(self, track_id: Optional[int]) -> None:
        with self._lock:
            if track_id is None:
                self._tracked_id = None
                log.info("Track lock cleared → manual")
            else:
                try:
                    self._tracked_id = int(track_id)
                    log.info("Track lock engaged on #%d", self._tracked_id)
                except (TypeError, ValueError):
                    log.warning("Bad track_id: %r", track_id)

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
            tracked_id  = self._tracked_id
            manual_pan  = self._manual_pan
            manual_tilt = self._manual_tilt

        mode = "manual"
        sp_pan, sp_tilt = manual_pan, manual_tilt
        err: Optional[str] = None

        if tracked_id is not None:
            # Pull the latest fused list and find our target
            fused = BUS.get_latest(Topic.FUSED)
            trk = None
            if fused:
                for t in fused:
                    if getattr(t, "id", None) == tracked_id:
                        trk = t
                        break
            if trk is not None:
                # az/el are already in sensor-boresight degrees. Pan
                # maps to az directly. Tilt maps to el but offset by
                # the home tilt (since 0 in tilt = horizon, the drone
                # camera's "el=0" is approximately at home_tilt mech-
                # anically). Simple enough for bench; a real mount
                # will want a mount-calibration transform.
                sp_pan  = manual_pan  + float(trk.az_deg)   # relative to manual pan park
                sp_tilt = self._home_tilt + float(trk.el_deg)
                mode = "auto"
            else:
                # Tracked ID vanished — drop the lock, stay put.
                err = f"tracked id #{tracked_id} lost"
                with self._lock:
                    if self._tracked_id == tracked_id:
                        self._tracked_id = None

        # Slew + clamp
        cmd_pan, cmd_tilt = self._controller.step(sp_pan, sp_tilt)
        self._command_now(cmd_pan, cmd_tilt)

        # Publish state
        state = GimbalState(
            timestamp=time.time(),
            connected=self._connected,
            pan_deg=cmd_pan,
            tilt_deg=cmd_tilt,
            mode=mode,
            target_pan_deg=sp_pan,
            target_tilt_deg=sp_tilt,
            tracked_target_id=tracked_id,
            error=err,
        )
        BUS.publish(Topic.GIMBAL, state)

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

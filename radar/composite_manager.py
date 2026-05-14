"""Composite radar backend — wraps RadarManager (TLV) and the DCA1000
raw-ADC pipeline so all three Phase-3 modes can run on a single chip
session.

Architecture (Phase 3, validated 2026-04-29):

  - The chip runs ONE cfg (``awr2944P_unified.cfg`` = stock cfg + the
    single ``lvdsStreamCfg -1 0 1 0`` line). Pushed once at Seeker
    startup. After that the chip emits BOTH the TLV point cloud
    (over UART, COM10) AND raw ADC samples (over LVDS to the DCA1000,
    forwarded as UDP to the host).

  - Two host-side consumers read those two planes concurrently:
      * ``RadarManager``   → TLV  → publishes on ``Topic.RADAR``
      * ``DCAPipeline``    → raw  → publishes on ``Topic.RADAR_AA``

  - Mode dispatch (stock / ag / aa) is purely a host-side filter
    setting. Switching modes:
      * does NOT push a new cfg
      * does NOT power-cycle the chip
      * does NOT pause streaming
      * does NOT swap any manager instance

    What changes per mode:
      * stock: RadarManager filters at full FoV (±90°), no
        speed-min cull. GUI displays Topic.RADAR.
      * ag:    RadarManager filters at ±25°/±15° + speed_min=0.5
        m/s. GUI displays Topic.RADAR.
      * aa:    Same chip cfg. GUI additionally displays Topic.RADAR_AA
        (PMM hits) on top of Topic.RADAR.

This module owns lifecycle of all three internal pieces. Callers
treat it as a single ``RadarBackend`` with .start() / .stop() /
.set_mode() and the same diagnostics() shape as RadarManager.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from common.logging_setup import get_logger
from radar.radar_manager import RadarManager
from radar_dca.dca_control import DCAControl, DCAControlError
from radar_dca.data_port import DataPortListener
from radar_dca.dca_pipeline import DCAPipeline, dims_from_cfg

log = get_logger(__name__)


# Per-mode RadarManager filter settings. These map directly onto
# RadarManager.set_tuning(...) — no cfg push, no chip change, just
# host-side filters on the TLV detection list.
_MODE_FILTERS: Dict[str, Dict[str, float]] = {
    "stock": {
        # Full FoV, no speed cull — same behaviour as pre-Phase-3 Seeker.
        "az_half_deg": 60.0,
        "speed_min_mps": 0.0,
    },
    "ag": {
        # Air-to-ground long range: slope halved (4.5 MHz/µs) to reach
        # 500m R_max. Needs a chip cfg push (awr2944P_ag.cfg).
        # Host-side filters stay at operator-tuned values — the mode
        # switch only changes max_range_m to open the 500m gate.
        "az_half_deg": 25.0,
        "speed_min_mps": 0.5,
        "max_range_m": 500.0,
    },
    "aa": {
        # Air-to-air: same TLV filter as stock (we still want full
        # coverage of the sky on TLV); the new behaviour comes from
        # also surfacing the Topic.RADAR_AA PMM stream.
        "az_half_deg": 60.0,
        "speed_min_mps": 0.0,
    },
}


class CompositeRadarBackend:
    """Owns RadarManager + DCAControl + DCAPipeline as one unit.

    The class shape mimics RadarManager just enough that the rest of
    Seeker can treat it as a drop-in radar backend (start, stop,
    set_extrinsic, set_tuning, diagnostics, profile_name). Mode
    switching is a new method on this class.

    Failure mode: if DCA setup fails (cable unplugged, FPGA wedged,
    etc.) we still bring up the TLV path so Seeker has SOMETHING to
    work with. The AA mode just won't have data — Topic.RADAR_AA
    stays silent. This matches the existing "missing chip" graceful
    degradation pattern.
    """

    def __init__(
        self,
        *,
        radar: RadarManager,
        dca_control: Optional[DCAControl] = None,
        dca_pipeline: Optional[DCAPipeline] = None,
        dca_listener: Optional[DataPortListener] = None,
        initial_mode: str = "stock",
        radar_firmware: str = "demoDDM",
    ) -> None:
        self._radar = radar
        self._dca_control = dca_control
        self._dca_pipeline = dca_pipeline
        self._dca_listener = dca_listener
        # Accepted for parity with main.py's launch wiring (commit b416885
        # added the kwarg at the call site but missed it here, so every
        # bench start since then crashed in __init__ and silently fell
        # back to TLV-only at 7 Hz). Stored for diagnostics; the actual
        # firmware-mode branching lives in the launcher itself.
        self._radar_firmware = str(radar_firmware or "demoDDM").lower()
        self._mode = (initial_mode or "stock").lower()
        if self._mode not in _MODE_FILTERS:
            self._mode = "stock"
        self._lock = threading.RLock()
        # Hand the pipeline back-refs so its LVDS stall watchdog can
        # do the FULL recovery dance: reset DCA FPGA + kick chip cfg.
        # The chip on this firmware emits LVDS in bursts then halts;
        # both sides need a reset to resume cleanly.
        if self._dca_pipeline is not None:
            self._dca_pipeline._radar_manager_ref = self._radar
            self._dca_pipeline._dca_control_ref = self._dca_control

        # Hook the chip-wedge recovery so the DCA1000 FPGA gets re-armed
        # after xds110reset.exe pulses the chip. Without this, the FPGA
        # keeps thinking it's recording but the LVDS source vanished
        # under it, and after the chip resumes streaming the FPGA
        # never resyncs — A/A is dead until the next full app launch.
        # See radar_manager._probe_and_recover for the firing point.
        try:
            self._radar.set_post_recovery_callback(
                self._rearm_dca_after_chip_reset
            )
        except AttributeError:
            # Older RadarManager without the hook — fall back silently.
            log.debug(
                "RadarManager has no set_post_recovery_callback; "
                "DCA re-arm after chip wedge will not happen automatically"
            )

    def _rearm_dca_after_chip_reset(self) -> None:
        """Bring the DCA1000 FPGA back online after xds110reset.

        Trigger: RadarManager just hard-reset the chip via xds110reset.exe
        (post-Ctrl+C wedge recovery). The FPGA was in start_record state
        and the LVDS source disappeared mid-stream. Re-run the same
        arm sequence as composite.start(): stop_record (clears any
        in-flight state) → reset_fpga → setup_capture → start_record.
        Also drops the pipeline's accumulator buffer because whatever
        bytes were sitting in there are pre-reset garbage.

        Called BEFORE _push_profile pushes the cfg + sensorStarts the
        chip — so the FPGA is freshly armed when LVDS resumes.
        """
        if self._dca_control is None:
            return
        log.info("Composite: re-arming DCA1000 after chip wedge recovery")
        try:
            self._dca_control.stop_record()
        except Exception as e:
            log.debug("re-arm: stop_record failed (FPGA may already be off): %s", e)
        time.sleep(0.1)
        try:
            self._dca_control.reset_fpga()
            self._dca_control.setup_capture()
            self._dca_control.start_record()
        except Exception as e:
            log.warning(
                "Composite: DCA re-arm failed: %s — A/A will stay dead "
                "until the next app launch", e,
            )
            return
        # Drop pre-reset bytes from the pipeline's accumulator. Whatever
        # was buffered when LVDS stopped is misaligned vs the new stream.
        if self._dca_pipeline is not None:
            try:
                self._dca_pipeline._buf.clear()
                self._dca_pipeline._lvds_stalled_warned = False
            except Exception:
                pass
        log.info(
            "Composite: DCA1000 re-armed; LVDS will resume when chip "
            "sensorStarts (cfg push happens next in _push_profile)"
        )

    # ─────────────────────── pass-through attributes ────────────────────
    @property
    def profile_name(self) -> str:
        return getattr(self._radar, "profile_name", "awr2944p_unified")

    @property
    def mode(self) -> str:
        return self._mode

    # Pass-throughs for attributes the GUI's _sender / sensor_bridge
    # reads directly off the radar manager. Without these the composite
    # raises AttributeError on every WS payload build.
    @property
    def az_bias_deg(self) -> float:
        return getattr(self._radar, "az_bias_deg", 0.0)

    @az_bias_deg.setter
    def az_bias_deg(self, v: float) -> None:
        try: self._radar.az_bias_deg = float(v)
        except Exception: pass

    @property
    def el_bias_deg(self) -> float:
        return getattr(self._radar, "el_bias_deg", 0.0)

    @el_bias_deg.setter
    def el_bias_deg(self, v: float) -> None:
        try: self._radar.el_bias_deg = float(v)
        except Exception: pass

    # Generic fall-through for any other attribute the GUI may probe
    # on the radar manager (snr_min_db, max_range_m, etc).
    def __getattr__(self, name: str):
        # __getattr__ is only called if normal lookup fails, so this
        # won't shadow the explicit properties + methods defined above.
        try:
            return getattr(object.__getattribute__(self, "_radar"), name)
        except AttributeError:
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )

    # Surface RadarManager's tuning hooks so the existing GUI
    # radar_tune handler keeps working without any indirection.
    def set_tuning(self, **kwargs: Any) -> None:
        self._radar.set_tuning(**kwargs)

    def set_extrinsic(self, **kwargs: Any) -> None:
        self._radar.set_extrinsic(**kwargs)

    # ─────────────────────── lifecycle ──────────────────────────────────
    def start(self) -> None:
        """Start the TLV path first (which pushes the unified cfg to
        the chip — that's what enables raw-ADC streaming). Then bring
        up the DCA control plane and start the raw-ADC consumer.

        Each subsystem failure is non-fatal: TLV alone is enough for
        stock + AG modes; AA is the only mode that needs DCA.
        """
        # 1. RadarManager — pushes unified cfg to chip, starts TLV
        # capture + processing threads. NEVER overwrites the YAML-loaded
        # tuning sliders — Stock keeps the operator's saved values.
        log.info("Composite: starting TLV path (RadarManager) ...")
        self._radar.start()

        # 2. DCA control plane — set up FPGA + start UDP recording.
        # Best-effort; AA mode is unavailable if this fails.
        if self._dca_control is not None and self._dca_pipeline is not None and self._dca_listener is not None:
            try:
                log.info("Composite: configuring DCA1000 + starting raw-ADC capture ...")
                self._dca_control.reset_fpga()
                self._dca_control.setup_capture()
                self._dca_control.start_record()
                # Start the UDP listener and the host-side parsing pipeline.
                self._dca_listener.start()
                self._dca_pipeline.start()
                log.info("Composite: raw-ADC path live (Topic.RADAR_AA)")
            except DCAControlError as e:
                log.warning("Composite: DCA setup failed (%s) — AA mode will be unavailable", e)
            except Exception as e:
                log.exception("Composite: DCA setup raised — AA mode will be unavailable: %s", e)
        else:
            log.info("Composite: no DCA pipeline configured — AA mode unavailable")

    def stop(self) -> None:
        """Stop everything in reverse order. Each step is best-effort —
        a failed teardown should not stop other teardowns."""
        # 1. Stop raw-ADC consumers FIRST so they don't keep reading
        # from a sensor that's about to stop emitting.
        if self._dca_pipeline is not None:
            try:
                self._dca_pipeline.stop()
            except Exception as e:
                log.warning("DCAPipeline stop failed: %s", e)
        if self._dca_listener is not None:
            try:
                self._dca_listener.stop()
            except Exception as e:
                log.warning("DCA listener stop failed: %s", e)
        if self._dca_control is not None:
            try:
                self._dca_control.stop_record()
            except Exception as e:
                log.warning("DCA stop_record failed: %s", e)
        # 2. Stop TLV — RadarManager's stop also issues sensorStop on
        # the chip, so this should be last.
        try:
            self._radar.stop()
        except Exception as e:
            log.warning("RadarManager stop failed: %s", e)

    # ─────────────────────── mode dispatch ──────────────────────────────
    def set_mode(self, mode: str) -> str:
        """Change the radar operating mode. Returns the mode that was
        actually applied (echoes the input on success, returns the
        previous mode on failure or invalid input).

        **Stock ↔ A/G transitions push a new chip cfg** because the two
        modes use different chirp slopes (8.883 vs 4.5 MHz/µs). The cfg
        file itself starts with sensorStop + flushCfg, so the push
        handles the chip state-machine transitions.

        Stock ↔ A/A and A/A ↔ Stock are still host-side-only (same chirp
        profile, different display pipeline).

        On cfg-push failure the mode stays unchanged and the previous
        cfg is restored — the chip never ends up in a half-configured
        state.

        Idempotent — calling with the current mode is a no-op.
        """
        target = (mode or "").lower()
        if target not in ("stock", "ag", "aa"):
            log.warning("set_mode: unknown mode %r — keeping %s", mode, self._mode)
            return self._mode
        with self._lock:
            if target == self._mode:
                return self._mode

            old_mode = self._mode
            entering_ag = (target == "ag" and old_mode != "ag")
            leaving_ag = (old_mode == "ag" and target != "ag")

            # ── chip cfg push for A/G transitions ──
            if entering_ag or leaving_ag:
                cfg_dir = Path(self._radar.cfg_path).parent
                if entering_ag:
                    new_cfg = cfg_dir / "awr2944P_ag.cfg"
                else:
                    new_cfg = cfg_dir / "awr2944P_unified.cfg"

                log.info(
                    "Composite: mode %s → %s — pushing chip cfg %s",
                    old_mode, target, new_cfg.name,
                )
                ok = self._radar.reconfigure(new_cfg)
                if not ok:
                    log.error(
                        "Composite: cfg push for %s failed — staying in %s",
                        new_cfg.name, old_mode,
                    )
                    return old_mode

                # Open / close the range gate to match the new chirp.
                if entering_ag:
                    self._stock_max_range_m = self._radar.max_range_m
                    self._radar.max_range_m = 500.0
                    log.info("Composite: max_range_m → 500 m (A/G long range)")
                else:
                    restored = getattr(self, "_stock_max_range_m", 250.0)
                    self._radar.max_range_m = restored
                    log.info("Composite: max_range_m → %.0f m (restored)", restored)

                # Re-arm DCA if we just re-enabled LVDS (leaving AG → unified cfg).
                if leaving_ag:
                    self._rearm_dca_after_chip_reset()
            else:
                log.info(
                    "Composite: mode %s → %s (host-side only, no chip change)",
                    old_mode, target,
                )

            self._mode = target

            # Forward mode to the DCA pipeline so its _publish gate works.
            if self._dca_pipeline is not None:
                try:
                    self._dca_pipeline.set_mode(target)
                except Exception:
                    log.exception("dca_pipeline.set_mode(%s) failed", target)

        return self._mode

    # ─────────────────────── live-tune knobs ────────────────────────────
    def update_aa_params(self, **kwargs: Any) -> None:
        """Update PMM detector parameters live without restarting the
        pipeline. Accepts pmm_band_low_hz, pmm_band_high_hz,
        pmm_threshold_db, pmm_slow_time_win, staggered_prf."""
        if self._dca_pipeline is None:
            return
        if "pmm_band_low_hz" in kwargs:
            self._dca_pipeline._pmm_band_low = float(kwargs["pmm_band_low_hz"])
        if "pmm_band_high_hz" in kwargs:
            self._dca_pipeline._pmm_band_high = float(kwargs["pmm_band_high_hz"])
        if "pmm_threshold_db" in kwargs:
            self._dca_pipeline._pmm_threshold = float(kwargs["pmm_threshold_db"])
        if "pmm_slow_time_win" in kwargs:
            self._dca_pipeline._pmm_slow_time_win = int(kwargs["pmm_slow_time_win"])
        if "staggered_prf" in kwargs:
            self._dca_pipeline._staggered_prf = bool(kwargs["staggered_prf"])

    def update_ag_params(self, **kwargs: Any) -> None:
        """Update A/G long-range processor knobs (host-side raw-ADC
        coherent-integration pipeline). Accepts integrate_chirps,
        cfar_algo, cfar_threshold_db, capon_bf."""
        if self._dca_pipeline is None:
            return
        for k in ("integrate_chirps", "cfar_algo", "cfar_threshold_db", "capon_bf"):
            if k in kwargs:
                setattr(self._dca_pipeline, "_ag_" + k, kwargs[k])

    def apply_ag_cfar(self, threshold_db: float) -> bool:
        """Modify the AG cfg's range-CFAR threshold and reconfigure
        the chip if currently in A/G mode.

        The cfarCfg line for range direction (procDirection=1) has the
        threshold at position 7 (0-indexed). We rewrite that field
        in the cfg file on disk, then trigger a reconfigure.

        Returns True if the reconfigure succeeded (or if not in AG mode,
        in which case the new threshold takes effect on next AG entry).
        """
        import re as _re
        from pathlib import Path as _Path

        cfg_dir = _Path(self._radar.cfg_path).parent
        ag_cfg = cfg_dir / "awr2944P_ag.cfg"

        if not ag_cfg.exists():
            log.error("apply_ag_cfar: %s not found", ag_cfg)
            return False

        text = ag_cfg.read_text()

        # Match range-direction cfarCfg (procDirection=1, second field)
        # cfarCfg -1 1 <mode> <noiseWin> ... <thresholdScale> ...
        def replace_range_cfar(m):
            fields = m.group(0).split()
            # fields[7] is thresholdScale for range CFAR
            fields[7] = f"{threshold_db:.1f}"
            return " ".join(fields)

        new_text = _re.sub(
            r"^cfarCfg\s+-1\s+1\s+.*$",
            replace_range_cfar,
            text,
            flags=_re.MULTILINE,
        )

        if new_text == text:
            log.warning("apply_ag_cfar: no range cfarCfg line matched")
            return False

        ag_cfg.write_text(new_text)
        log.info("apply_ag_cfar: wrote %.1f dB to %s", threshold_db, ag_cfg.name)

        # If in AG mode, reconfigure to apply immediately
        if self._mode == "ag":
            log.info("apply_ag_cfar: in AG mode — reconfiguring chip")
            ok = self._radar.reconfigure(str(ag_cfg))
            if ok and self._dca_pipeline is not None:
                # LVDS is off in AG; no DCA re-arm needed
                pass
            return ok

        # Not in AG — threshold stored for next AG entry
        return True

    # ─────────────────────── diagnostics ────────────────────────────────
    def diagnostics(self) -> Dict[str, Any]:
        """Return one dict covering both data planes so the GUI debug
        panel and the /api/radar/aa_diagnostics endpoint can show
        WHERE in the chain the A/A path is stuck. The chain is:

            chip LVDS  →  DCA FPGA UDP  →  DataPortListener (host)
                                   │
                                   ▼
                           DCAPipeline frame assembly
                                   │
                                   ▼
                              PMM detector
                                   │
                                   ▼
                            Topic.RADAR_AA  →  GUI

        Reading the dict from top to bottom:
          udp.packets_total == 0          → DCA isn't reaching the host
                                            (cable, IP route, firewall,
                                            FPGA forward not started).
          udp.packets_total > 0 but       → DCA is sending but our frame
            aa.frames_assembled == 0       parser can't reassemble frames
                                            (chirp/sample dims wrong, or
                                            seq drops too high).
          frames_assembled > 0 but        → Frames flow but the PMM
            drone_detections == 0          detector never matches the
                                            symmetric sideband test —
                                            band wrong, threshold too
                                            high, or no real prop signal.
        """
        d: Dict[str, Any] = {"mode": self._mode}
        # RadarManager doesn't expose diagnostics() — synthesize a
        # small dict from its public attributes so this branch is
        # informative instead of an opaque AttributeError.
        try:
            r = self._radar
            d["tlv"] = {
                "snr_min_db": getattr(r, "snr_min_db", None),
                "az_half_deg": getattr(r, "az_half_deg", None),
                "speed_min_mps": getattr(r, "speed_min_mps", None),
                "range_min_m": getattr(r, "range_min_m", None),
                "max_range_m": getattr(r, "max_range_m", None),
                "frame_id": getattr(r, "_frame_id", None),
                "profile_name": getattr(r, "profile_name", None),
            }
        except Exception as e:
            d["tlv"] = {"error": str(e)}

        # UDP listener — the most upstream point in the A/A chain.
        if self._dca_listener is not None:
            try:
                ds = self._dca_listener.stats()
                d["udp"] = {
                    "listening": ds.listening,
                    "bound_addr": ds.bound_addr,
                    "packets_total": ds.packets_total,
                    "bytes_total": ds.bytes_total,
                    "packets_per_s": round(ds.packets_per_s, 1),
                    "bytes_per_s": round(ds.bytes_per_s, 1),
                    "seq_drops_total": ds.seq_drops_total,
                    "last_packet_age_s": (
                        round(ds.last_packet_age_s, 2)
                        if ds.last_packet_age_s != float("inf") else None
                    ),
                }
            except Exception as e:
                d["udp"] = {"error": str(e)}
        else:
            d["udp"] = {"available": False}

        # Pipeline + PMM detector counters.
        if self._dca_pipeline is not None:
            try:
                stats = self._dca_pipeline.stats()
                d["aa"] = {
                    "publish_enabled": getattr(self._dca_pipeline, "_publish_enabled", None),
                    "frames_assembled": stats.frames_assembled,
                    "frames_dropped": stats.frames_dropped,
                    "drone_detections": stats.drone_detections,
                    "last_drone_range_m": stats.last_drone_range_m,
                    "last_drone_blade_freq_hz": stats.last_drone_blade_freq_hz,
                    # Echo the live PMM tuning so we can see what
                    # threshold the detector is actually using.
                    "pmm_band_low_hz": getattr(self._dca_pipeline, "_pmm_band_low", None),
                    "pmm_band_high_hz": getattr(self._dca_pipeline, "_pmm_band_high", None),
                    "pmm_threshold_db": getattr(self._dca_pipeline, "_pmm_threshold", None),
                    "pmm_slow_time_win": getattr(self._dca_pipeline, "_pmm_slow_time_win", None),
                }
            except Exception as e:
                d["aa"] = {"error": str(e)}
        else:
            d["aa"] = {"available": False}
        return d

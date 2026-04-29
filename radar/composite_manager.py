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
        # Air-to-ground: narrow FoV (matches AWR2944P antenna main lobe)
        # and dynamic-only filter (drops parked vehicles / standing
        # humans that we'll catch with EO+thermal anyway).
        "az_half_deg": 25.0,
        "speed_min_mps": 0.5,
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
    ) -> None:
        self._radar = radar
        self._dca_control = dca_control
        self._dca_pipeline = dca_pipeline
        self._dca_listener = dca_listener
        self._mode = (initial_mode or "stock").lower()
        if self._mode not in _MODE_FILTERS:
            self._mode = "stock"
        self._lock = threading.RLock()

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
        """Change the host-side mode filter. Returns the mode that was
        actually applied (echoes the input on success, returns the
        previous mode if the input was invalid).

        No chip interaction. No power-cycle. Idempotent — calling with
        the current mode is a no-op and returns the current mode."""
        target = (mode or "").lower()
        if target not in ("stock", "ag", "aa"):
            log.warning("set_mode: unknown mode %r — keeping %s", mode, self._mode)
            return self._mode
        with self._lock:
            if target == self._mode:
                return self._mode
            log.info("Composite: mode %s → %s (host-side, no chip change, sliders untouched)",
                     self._mode, target)
            self._mode = target
            # Tell the DCA pipeline whether to publish PMM hits (only
            # when mode == aa). In stock + ag modes the pipeline keeps
            # consuming raw-ADC bytes (so we don't lose data) but
            # suppresses publishes — Topic.RADAR_AA stays quiet.
            if self._dca_pipeline is not None:
                try:
                    self._dca_pipeline._publish_enabled = (target == "aa")
                except Exception:
                    pass
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

    # ─────────────────────── diagnostics ────────────────────────────────
    def diagnostics(self) -> Dict[str, Any]:
        """Return a single dict with both subsystems' diagnostics, for
        the GUI debug panel and unit tests."""
        d: Dict[str, Any] = {"mode": self._mode}
        try:
            d["tlv"] = self._radar.diagnostics()
        except Exception as e:
            d["tlv"] = {"error": str(e)}
        if self._dca_pipeline is not None:
            try:
                stats = self._dca_pipeline.stats()
                d["aa"] = {
                    "frames_assembled": stats.frames_assembled,
                    "frames_dropped": stats.frames_dropped,
                    "drone_detections": stats.drone_detections,
                    "last_drone_range_m": stats.last_drone_range_m,
                    "last_drone_blade_freq_hz": stats.last_drone_blade_freq_hz,
                }
            except Exception as e:
                d["aa"] = {"error": str(e)}
        else:
            d["aa"] = {"available": False}
        return d

"""DCAManager — host-side raw-ADC radar pipeline.

The A/A backend. Mirrors ``radar.RadarManager``'s external contract
so the factory and hot-swap path can construct either type with the
same kwargs bag:

- ``start()`` / ``stop()`` / ``set_extrinsic()`` / ``get_tuning()``
- publishes ``RadarFrame`` on ``Topic.RADAR`` (so EO+thermal fusion
  doesn't care which backend is live)

Lifecycle on ``start()`` (Phase 3 step 3, A/A bring-up):

1. Push the A/A cfg to the AWR over COM11 (lvdsStreamCfg enabled,
   compressionCfg disabled). Same ``send_cfg`` helper that
   RadarManager uses, so framing/pacing/ack handling is identical.
2. Open the host UDP listener on ``192.168.33.30:4098``.
3. Query DCA FPGA via ``DCAControl`` (UDP 4096): version + system
   status (= AWR's LVDS clock present or not).
4. Issue the DCA capture chain: ``setup_capture()`` (CONFIG_FPGA_GEN
   + CONFIG_PACKET_DATA) then ``start_record()``.
5. Heartbeat thread re-queries control state every 5 s and publishes
   a ``RadarFrame`` sentinel each tick. M4 (range-FFT etc.) flips
   that to a real point cloud once the data-plane FFT pipeline ships.

Future milestones (radar_dca/dca_pipeline.py):

- M4: range FFT + Doppler FFT + PMM detection → emit RadarTarget
  point cloud with class=DRONE on PMM hit.
- M5: track promotion via FusionManager Phase 2 path (already wired).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional

import serial

from common.frame_bus import BUS
from common.frames import RadarFrame, Topic
from common.logging_setup import get_logger
from radar.cfg_sender import send_cfg
from radar_dca.dca_control import DCAControl, DCAControlError, FpgaVersion, SystemStatus
from radar_dca.data_port import DataPortListener, DataPortStats
from radar_dca.dca_pipeline import DCAPipeline, dims_from_cfg

log = get_logger(__name__)


class DCAManager:
    """Host-side DCA1000 radar manager.

    See module docstring for the bring-up state.
    """

    def __init__(
        self,
        *,
        # CLI port + cfg path are USED now (A/A pushes its cfg before
        # listening). Were ignored in the M2/M3 scaffold.
        cli_port: Optional[str] = None,
        data_port: Optional[str] = None,
        cfg_path: Optional[str] = None,
        cli_baud: int = 115200,
        data_baud: int = 921600,
        snr_min_db: float = 12.0,
        max_range_m: float = 250.0,
        az_half_deg: float = 60.0,
        speed_min_mps: float = 0.0,
        range_min_m: float = 0.0,
        profile_name: str = "awr2944p_aa",
        cluster_params=None,
        # Software extrinsic (radar → EO alignment), live-tunable.
        az_bias_deg: float = 0.0,
        el_bias_deg: float = 0.0,
        # DCA networking.
        host_ip: str = "192.168.33.30",
        dca_ip: str = "192.168.33.180",
        config_port: int = 4096,
        data_udp_port: int = 4098,
        **_unused,
    ) -> None:
        self.profile_name = profile_name
        self.max_range_m = float(max_range_m)
        self.az_half_deg = float(az_half_deg)
        self.snr_min_db = float(snr_min_db)
        self.speed_min_mps = float(speed_min_mps)
        self.range_min_m = float(range_min_m)
        self.az_bias_deg = float(az_bias_deg)
        self.el_bias_deg = float(el_bias_deg)

        # AWR CLI access — used to push the A/A cfg.
        self.cli_port = cli_port
        self.cli_baud = int(cli_baud)
        self.cfg_path = cfg_path

        self.host_ip = str(host_ip)
        self.dca_ip = str(dca_ip)
        self.config_port = int(config_port)
        self.data_udp_port = int(data_udp_port)

        # Control plane wrapper + data plane listener.
        self._control = DCAControl(
            host_ip=self.host_ip,
            dca_ip=self.dca_ip,
            config_port=self.config_port,
            data_port=self.data_udp_port,
        )
        self._listener = DataPortListener(
            host_ip=self.host_ip,
            data_port=self.data_udp_port,
        )
        # Range-FFT + PMM detection pipeline. Subscribes to the listener
        # for raw ADC, emits RadarFrame on Topic.RADAR with PMM-detected
        # DRONE targets. M4 work: scaffolding lives in dca_pipeline.py;
        # the byte-layout parser is the remaining gap.
        self._pipeline = DCAPipeline(
            listener=self._listener,
            dims=dims_from_cfg(),  # default = awr2944P_aa.cfg's chirp shape
            profile_name=self.profile_name,
            max_range_m=self.max_range_m,
            az_half_deg=self.az_half_deg,
        )

        # Diagnostics — last-known FPGA version + system status. Refreshed
        # on start() and every status_refresh_s seconds in the loop.
        self._fpga: Optional[FpgaVersion] = None
        self._sys: Optional[SystemStatus] = None
        self._control_error: Optional[str] = None  # last CLI failure message, if any
        self.status_refresh_s = 5.0

        # Threading + lifecycle.
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._tune_lock = threading.Lock()
        self._frame_id = 0

    # ─────────────────────── live-tune (parity with RadarManager) ───────────
    def set_tuning(
        self,
        *,
        snr_min_db: Optional[float] = None,
        az_half_deg: Optional[float] = None,
        speed_min_mps: Optional[float] = None,
        range_min_m: Optional[float] = None,
        cluster_eps_pos_m: Optional[float] = None,
        cluster_eps_dop_mps: Optional[float] = None,
        cluster_min_samples: Optional[int] = None,
    ) -> None:
        """Stub: simple knobs only. DBSCAN params will be applied
        once M4 (range/doppler + clustering) lands."""
        with self._tune_lock:
            if snr_min_db is not None:
                self.snr_min_db = float(snr_min_db)
            if az_half_deg is not None:
                self.az_half_deg = float(az_half_deg)
            if speed_min_mps is not None:
                self.speed_min_mps = float(speed_min_mps)
            if range_min_m is not None:
                self.range_min_m = float(range_min_m)

    def set_extrinsic(
        self,
        *,
        az_bias_deg: Optional[float] = None,
        el_bias_deg: Optional[float] = None,
    ) -> None:
        with self._tune_lock:
            if az_bias_deg is not None:
                self.az_bias_deg = float(az_bias_deg)
            if el_bias_deg is not None:
                self.el_bias_deg = float(el_bias_deg)

    def get_tuning(self) -> dict:
        return {
            "snr_min_db": self.snr_min_db,
            "max_range_m": self.max_range_m,
            "az_half_deg": self.az_half_deg,
            "speed_min_mps": self.speed_min_mps,
            "range_min_m": self.range_min_m,
            "cluster_eps_pos_m": 0.5,
            "cluster_eps_dop_mps": 1.0,
            "cluster_min_samples": 2,
            "az_bias_deg": self.az_bias_deg,
            "el_bias_deg": self.el_bias_deg,
        }

    # ─────────────────────── DCA-specific diagnostics ───────────────────────
    def diagnostics(self) -> dict:
        """Snapshot of DCA control + data plane state for the GUI.

        Returned dict is JSON-serialisable and embedded in the WS
        payload alongside ``radar_tuning`` so the radar panel can
        show 'DCA: FPGA v2.9 · 0 pkts/s · idle' etc.
        """
        ds = self._listener.stats()
        ps = self._pipeline.stats() if self._pipeline_alive() else None
        return {
            "host_ip": self.host_ip,
            "dca_ip": self.dca_ip,
            "config_port": self.config_port,
            "data_udp_port": self.data_udp_port,
            "fpga_version": self._fpga.version if self._fpga else None,
            "fpga_flavor":  self._fpga.flavor  if self._fpga else None,
            "sys_connected": self._sys.connected if self._sys else False,
            "control_error": self._control_error,
            # Data plane (UDP listener)
            "data_listening":   ds.listening,
            "data_bound_addr":  ds.bound_addr,
            "packets_total":    ds.packets_total,
            "bytes_total":      ds.bytes_total,
            "seq_drops_total":  ds.seq_drops_total,
            "packets_per_s":    round(ds.packets_per_s, 2),
            "bytes_per_s":      round(ds.bytes_per_s, 2),
            "last_packet_age_s": round(ds.last_packet_age_s, 2)
                                 if ds.last_packet_age_s != float("inf") else None,
            # Range-FFT + PMM pipeline
            "pipeline_alive":   ps is not None,
            "frames_assembled": ps.frames_assembled if ps else 0,
            "frames_dropped":   ps.frames_dropped if ps else 0,
            "drone_detections": ps.drone_detections if ps else 0,
            "last_drone_range_m": (round(ps.last_drone_range_m, 1)
                                   if ps and not (ps.last_drone_range_m != ps.last_drone_range_m) else None),
            "last_drone_blade_freq_hz": (round(ps.last_drone_blade_freq_hz, 1)
                                         if ps and not (ps.last_drone_blade_freq_hz != ps.last_drone_blade_freq_hz) else None),
        }

    # ─────────────────────── AWR cfg push ──────────────────────────────────
    def _push_aa_cfg_to_awr(self) -> bool:
        """Open COM11, push the A/A cfg (lvdsStreamCfg + compressionCfg=0),
        close. Returns True on success.

        The cfg's first lines (sensorStop + flushCfg) reset the AWR's
        in-memory config; the chip re-runs RF calibration on the
        terminal sensorStart. If any line returns "Error" the response
        is logged and we return False — caller decides whether to
        continue with the listener anyway (useful for offline tests
        where the AWR isn't connected)."""
        if self.cli_port is None or self.cfg_path is None:
            log.warning("DCA cfg-push skipped: no cli_port or cfg_path")
            return False
        if not Path(self.cfg_path).exists():
            log.warning("DCA cfg-push skipped: cfg_path %r does not exist",
                        self.cfg_path)
            return False
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                responses = send_cfg(ser, self.cfg_path)
                tail = responses[-1] if responses else ""
                if "Error" in tail or "error" in tail:
                    log.warning("DCA cfg-push: last line returned %r", tail.strip())
                    return False
                log.info("DCA cfg-push: %d lines pushed from %s",
                         len(responses), self.cfg_path)
                return True
        except serial.SerialException as e:
            log.warning("DCA cfg-push: could not open %s: %s", self.cli_port, e)
            return False
        except Exception as e:
            log.warning("DCA cfg-push: unexpected error: %s", e)
            return False

    # ─────────────────────── lifecycle ─────────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        # 1. Push the A/A cfg to the AWR so it streams raw ADC over LVDS.
        #    Done first because the FPGA's "system_status" only goes
        #    "connected" once the AWR is clocking LVDS, which only
        #    happens after our sensorStart fires.
        cfg_ok = self._push_aa_cfg_to_awr()
        if not cfg_ok:
            log.warning("DCA: AWR cfg-push failed — DCA will still listen but no "
                        "raw-ADC packets will arrive. Power-cycle AWR if it's stuck.")
        # 2. Spin up the data-port listener immediately so packets that
        #    arrive during the first FPGA query don't get dropped.
        try:
            self._listener.start()
        except OSError as e:
            log.warning("DCA data port bind failed: %s — continuing without listener", e)
        # 3. Initial FPGA query — non-fatal if it errors.
        self._refresh_status()
        # 4. If the FPGA is reachable, configure it for capture and start
        #    forwarding LVDS → UDP.
        if self._fpga is not None:
            try:
                self._control.setup_capture()
                self._control.start_record()
                log.info("DCA: capture configured and start_record issued.")
            except DCAControlError as e:
                self._control_error = f"setup/start_record: {e}"
                log.warning("DCA capture setup failed: %s", e)
        # 5. Start the range-FFT + PMM pipeline. It subscribes to the
        #    listener and emits RadarFrames at the AWR's frame rate.
        #    Note: scaffold mode emits zero-filled frames until M4.1
        #    (raw-ADC byte parsing) is wired — see dca_pipeline.py.
        self._pipeline.start()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DCAManager", daemon=True
        )
        self._thread.start()
        if self._fpga is not None:
            log.info(
                "DCAManager started (cfg-push=%s, FPGA v%s [%s], sys=%s; data port: %s)",
                "OK" if cfg_ok else "FAIL",
                self._fpga.version, self._fpga.flavor,
                "connected" if (self._sys and self._sys.connected) else "idle",
                f"{self.host_ip}:{self.data_udp_port}",
            )
        else:
            log.info(
                "DCAManager started (cfg-push=%s, control: CLI unreachable [%s]; data port: %s)",
                "OK" if cfg_ok else "FAIL",
                self._control_error or "?",
                f"{self.host_ip}:{self.data_udp_port}",
            )

    def stop(self) -> None:
        self._stop.set()
        # Stop the range-FFT pipeline first so it stops emitting
        # RadarFrames before we tear down the listener it depends on.
        try:
            self._pipeline.stop()
        except Exception as e:
            log.warning("DCA pipeline stop failed: %s", e)
        # Take the FPGA out of recording mode so the next backend (or
        # the next start of this one) starts from a clean state.
        try:
            self._control.stop_record()
        except Exception as e:
            log.warning("DCA stop_record failed: %s", e)
        # Try to politely sensorStop the AWR so the next backend
        # (likely stock or A/G with a different cfg) doesn't have to
        # fight a streaming chip on first contact.
        if self.cli_port is not None:
            try:
                with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                    ser.write(b"sensorStop\n")
                    ser.flush()
                    time.sleep(0.5)
            except Exception as e:
                log.debug("DCA shutdown sensorStop failed: %s", e)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._listener.stop()
        log.info("DCAManager stopped")

    # ─────────────────────── publish loop ───────────────────────────────────
    def _loop(self) -> None:
        """Heartbeat: refresh DCA control-plane status every
        ``status_refresh_s``. RadarFrame publication is owned by the
        DCAPipeline thread (5 Hz at the AWR's frame rate); we only
        publish a sentinel here when the pipeline isn't running, so
        the bus topic is always populated even during scaffold mode."""
        last_status_refresh = time.time()
        while not self._stop.is_set():
            now = time.time()
            if now - last_status_refresh >= self.status_refresh_s:
                self._refresh_status()
                last_status_refresh = now
            # If the pipeline thread isn't alive (boot failure,
            # listener bind failed, etc.) we keep the bus topic warm
            # with a connected=False sentinel.
            if not self._pipeline_alive():
                self._publish_sentinel()
            self._stop.wait(1.0)

    def _pipeline_alive(self) -> bool:
        thr = getattr(self._pipeline, "_thread", None)
        return thr is not None and thr.is_alive()

    def _refresh_status(self) -> None:
        """Re-query FPGA version + system status. Both are quick UDP
        round-trips. Failures are logged but non-fatal — the manager
        continues running with stale diagnostics."""
        try:
            self._fpga = self._control.fpga_version()
            self._sys = self._control.query_sys_status()
            self._control_error = None
        except DCAControlError as e:
            self._control_error = str(e)
            log.warning("DCA control refresh failed: %s", e)

    def _publish_sentinel(self) -> None:
        self._frame_id += 1
        BUS.publish(Topic.RADAR, RadarFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=False,  # M4 will flip this to True once we emit point clouds
            profile=self.profile_name,
            max_range_m=self.max_range_m,
            fov_half_deg=self.az_half_deg,
        ))

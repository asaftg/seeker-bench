"""AWR2944P radar pipeline orchestrator.

Two-thread manager modelled on ``eo.eo_manager.EOManager``:

    RadarCapture  — opens the data UART, reads raw bytes into
                    ``TLVStream``, hands complete parsed packets to
                    the process thread.
    RadarProcess  — SNR-gates the point cloud, runs DBSCAN to
                    synthesise target boxes, publishes ``RadarFrame``
                    on ``Topic.RADAR``.

On connect the manager opens the CLI UART, pushes the .cfg profile
via ``cfg_sender.send_cfg`` (the TI mmw_demo firmware will not stream
anything until ``sensorStart`` is received), then closes the CLI port
and leaves the data port open for streaming. On shutdown it re-opens
the CLI briefly to send ``sensorStop`` so the chip doesn't keep
transmitting into a dead host.

``connected=False`` sentinels are published on open / stream timeouts
so the GUI can flip the radar panel's DISCONNECTED overlay without
crashing — same contract as ThermalFrame / EOFrame.

No YOLO, no semantic classifier — all targets carry the generic
class ``"radar_detection"`` per the Ticket 5a design agreement with
Asaf. Class labels (vehicle / person) are a fusion-layer job.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import serial

from common.frame_bus import BUS
from common.frames import RadarDetection, RadarFrame, RadarTarget, Topic
from common.logging_setup import get_logger
from radar.cfg_sender import send_cfg
from radar.clustering import ClusterParams, RadarClusterer
from radar.tlv_parser import RadarPacket, TLVStream

log = get_logger(__name__)


class RadarManager:
    """Drives the AWR2944P EVM and publishes RadarFrame on Topic.RADAR."""

    def __init__(
        self,
        cli_port: str,
        data_port: str,
        cfg_path: str | Path,
        cli_baud: int = 115200,
        data_baud: int = 921600,
        reconnect_interval_s: float = 2.0,
        snr_min_db: float = 12.0,
        max_range_m: float = 250.0,
        az_half_deg: float = 60.0,
        speed_min_mps: float = 0.0,
        range_min_m: float = 0.0,
        profile_name: str = "awr2944p_ddm",
        stream_timeout_s: float = 3.0,
        cluster_params: Optional[ClusterParams] = None,
        az_bias_deg: float = 0.0,
        el_bias_deg: float = 0.0,
    ) -> None:
        self.cli_port = cli_port
        self.data_port = data_port
        self.cfg_path = str(cfg_path)
        self.cli_baud = int(cli_baud)
        self.data_baud = int(data_baud)
        self.reconnect_interval_s = float(reconnect_interval_s)
        self.snr_min_db = float(snr_min_db)
        self.max_range_m = float(max_range_m)
        self.az_half_deg = float(az_half_deg)
        self.speed_min_mps = float(speed_min_mps)
        self.range_min_m = float(range_min_m)
        self.profile_name = str(profile_name)
        self._tune_lock = threading.Lock()
        self.stream_timeout_s = float(stream_timeout_s)
        # Software extrinsic — applied to each radar target's az/el before
        # projection onto EO/thermal panels. Tuned live from GUI sliders
        # against EO as ground truth. Zero = boresight matches cameras.
        self.az_bias_deg = float(az_bias_deg)
        self.el_bias_deg = float(el_bias_deg)

        self._stop = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None

        # Hand-off from capture → process: latest packet + seq number.
        self._latest_cond = threading.Condition()
        self._latest_pkt: Optional[RadarPacket] = None
        self._latest_seq: int = 0

        self._data_ser: Optional[serial.Serial] = None
        self._frame_id: int = 0
        self._clusterer = RadarClusterer(cluster_params)

    # ─────────────────────── live tuning ─────────────────────
    def set_tuning(
        self,
        *,
        snr_min_db: Optional[float] = None,
        max_range_m: Optional[float] = None,
        az_half_deg: Optional[float] = None,
        speed_min_mps: Optional[float] = None,
        range_min_m: Optional[float] = None,
        cluster_eps_pos_m: Optional[float] = None,
        cluster_eps_dop_mps: Optional[float] = None,
        cluster_min_samples: Optional[int] = None,
    ) -> None:
        """Hot-update filter + cluster knobs without a manager restart.

        Called from the WS ``radar_tune`` command so the operator can
        drag sliders in the DEV tab and see the effect next frame. All
        args optional — unspecified fields are left alone.
        """
        with self._tune_lock:
            if snr_min_db is not None:
                self.snr_min_db = float(snr_min_db)
            if max_range_m is not None:
                self.max_range_m = float(max_range_m)
            if az_half_deg is not None:
                self.az_half_deg = float(az_half_deg)
            if speed_min_mps is not None:
                self.speed_min_mps = float(speed_min_mps)
            if range_min_m is not None:
                self.range_min_m = float(range_min_m)
            cp = self._clusterer.params
            if cluster_eps_pos_m is not None:
                cp.eps_pos_m = float(cluster_eps_pos_m)
            if cluster_eps_dop_mps is not None:
                cp.eps_dop_mps = float(cluster_eps_dop_mps)
            if cluster_min_samples is not None:
                cp.min_samples = int(cluster_min_samples)

    def set_extrinsic(
        self,
        *,
        az_bias_deg: Optional[float] = None,
        el_bias_deg: Optional[float] = None,
    ) -> None:
        """Hot-update software extrinsic (az/el bias applied at projection).

        EO is the ground-truth reference; radar bias nudges projected
        radar bboxes to match EO detections without remounting.
        """
        with self._tune_lock:
            if az_bias_deg is not None:
                self.az_bias_deg = float(az_bias_deg)
            if el_bias_deg is not None:
                self.el_bias_deg = float(el_bias_deg)

    def get_tuning(self) -> dict:
        cp = self._clusterer.params
        return {
            "snr_min_db": self.snr_min_db,
            "max_range_m": self.max_range_m,
            "az_half_deg": self.az_half_deg,
            "speed_min_mps": self.speed_min_mps,
            "range_min_m": self.range_min_m,
            "cluster_eps_pos_m": cp.eps_pos_m,
            "cluster_eps_dop_mps": cp.eps_dop_mps,
            "cluster_min_samples": cp.min_samples,
            "az_bias_deg": self.az_bias_deg,
            "el_bias_deg": self.el_bias_deg,
        }

    # ─────────────────────── lifecycle ───────────────────────
    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="RadarCapture", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, name="RadarProcess", daemon=True
        )
        self._capture_thread.start()
        self._process_thread.start()
        log.info("RadarManager started (cli=%s @%d, data=%s @%d, cfg=%s)",
                 self.cli_port, self.cli_baud, self.data_port, self.data_baud,
                 self.cfg_path)

    def stop(self) -> None:
        self._stop.set()
        with self._latest_cond:
            self._latest_cond.notify_all()
        for t in (self._process_thread, self._capture_thread):
            if t is not None:
                t.join(timeout=3.0)
        self._capture_thread = None
        self._process_thread = None
        self._close_data_port()
        # Courtesy sensorStop — the chip keeps transmitting after we
        # exit otherwise. Best-effort; a user yanking USB mid-shutdown
        # is normal and shouldn't throw.
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                ser.write(b"sensorStop\n")
                ser.flush()
        except Exception as e:
            log.debug("sensorStop on shutdown skipped: %s", e)

    # ─────────────────────── connect helpers ─────────────────
    @staticmethod
    def _cli_send(ser: serial.Serial, cmd: str, wait_s: float = 1.5) -> str:
        # Drain any stale response bytes from the previous command before
        # writing — reset_input_buffer alone can race with still-arriving
        # bytes from the firmware and truncate the next write's response
        # in weird ways. Read with a tiny idle window until quiet.
        idle_deadline = time.monotonic() + 0.25
        while time.monotonic() < idle_deadline:
            n = ser.in_waiting
            if n:
                ser.read(n)
                idle_deadline = time.monotonic() + 0.05
            else:
                time.sleep(0.01)
        try:
            ser.reset_input_buffer()
        except Exception:
            pass

        ser.write((cmd + "\n").encode("ascii"))
        ser.flush()
        deadline = time.monotonic() + wait_s
        buf = bytearray()
        while time.monotonic() < deadline:
            n = ser.in_waiting
            if n:
                buf.extend(ser.read(n))
                text = buf.decode("ascii", errors="replace")
                if "Done" in text or "Error" in text or "not recognized" in text:
                    # Small drain window so the final prompt lands too.
                    end = time.monotonic() + 0.08
                    while time.monotonic() < end:
                        m = ser.in_waiting
                        if m:
                            buf.extend(ser.read(m))
                        else:
                            time.sleep(0.005)
                    break
            else:
                time.sleep(0.005)
        return buf.decode("ascii", errors="replace")

    def _query_sensor_state(self, ser: serial.Serial) -> Optional[int]:
        """Return integer sensor state (0=INIT, 2=STARTED, 3=STOPPED) or None."""
        resp = self._cli_send(ser, "queryDemoStatus", wait_s=1.0)
        for line in resp.splitlines():
            s = line.strip()
            if s.lower().startswith("sensor state"):
                try:
                    return int(s.split(":")[1].strip())
                except Exception:
                    return None
        return None

    def _push_profile(self) -> bool:
        """Open CLI UART, bring sensor to STARTED, close. True on success.

        The TI ``mmw_demo`` CLI only accepts a reconfig + fresh ``sensorStart``
        when the chip is in state INIT (first boot). Any subsequent start
        must be ``sensorStart 0`` from state STOPPED — the firmware will
        reject a plain ``sensorStart`` with "Invalid Sensor Start". So:

          - state INIT  → push full .cfg (which ends in sensorStart)
          - state STOPPED or STARTED → sensorStop, then sensorStart 0,
            reusing whatever profile was previously configured

        If the user edits the .cfg and wants it applied, they must power-cycle
        the EVM so the chip comes back up in state INIT.
        """
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                state = self._query_sensor_state(ser)
                log.info("Radar CLI reports sensor state=%s", state)

                if state == 0:
                    # Fresh boot — push full profile. .cfg ends in sensorStart.
                    responses = send_cfg(ser, self.cfg_path)
                    tail = responses[-1] if responses else ""
                    if "Error" in tail or "error" in tail:
                        log.warning("Profile push failed on last line: %r", tail)
                        return False
                    return True

                # Non-INIT: stop → restart without reconfig. The chip keeps
                # whatever .cfg it was loaded with last. Warn loudly if the
                # path we were asked to send doesn't match what's on-chip
                # — only a power cycle can recover that.
                log.info("Sensor not in INIT; using sensorStart 0 "
                         "(previously-loaded profile, not %s)", self.cfg_path)
                self._cli_send(ser, "sensorStop", wait_s=1.0)
                time.sleep(0.1)
                resp = self._cli_send(ser, "sensorStart 0", wait_s=2.0)
                if "Done" not in resp:
                    log.warning("sensorStart 0 did not ack: %r", resp.strip())
                    return False
                return True
        except serial.SerialException as e:
            log.warning("Could not open CLI port %s: %s", self.cli_port, e)
            return False

    def _open_data_port(self) -> bool:
        try:
            # Short read timeout so the capture loop can stay responsive
            # to self._stop while waiting for UART bytes.
            self._data_ser = serial.Serial(
                self.data_port,
                self.data_baud,
                timeout=0.2,
            )
            return True
        except serial.SerialException as e:
            log.warning("Could not open data port %s: %s", self.data_port, e)
            self._data_ser = None
            return False

    def _close_data_port(self) -> None:
        if self._data_ser is not None:
            try:
                self._data_ser.close()
            except Exception:
                pass
            self._data_ser = None

    # ─────────────────────── capture loop ────────────────────
    def _capture_loop(self) -> None:
        stream = TLVStream()
        last_pkt_time = 0.0
        ever_connected = False

        while not self._stop.is_set():
            # Connect / reconnect.
            if self._data_ser is None:
                if not self._push_profile():
                    self._publish_disconnected()
                    self._stop.wait(self.reconnect_interval_s)
                    continue
                if not self._open_data_port():
                    self._publish_disconnected()
                    self._stop.wait(self.reconnect_interval_s)
                    continue
                stream = TLVStream()   # fresh buffer per reconnect
                last_pkt_time = time.monotonic()
                ever_connected = True
                log.info("Radar connected (data=%s @%d)", self.data_port, self.data_baud)

            try:
                chunk = self._data_ser.read(4096)
            except serial.SerialException as e:
                log.warning("Data port read error: %s — reconnecting", e)
                self._close_data_port()
                self._publish_disconnected()
                self._stop.wait(self.reconnect_interval_s)
                continue

            now = time.monotonic()
            if chunk:
                for pkt in stream.feed(chunk):
                    with self._latest_cond:
                        self._latest_pkt = pkt
                        self._latest_seq += 1
                        self._latest_cond.notify_all()
                    last_pkt_time = now

            # Stream-timeout disconnect: if we've been connected but
            # haven't seen a valid packet in stream_timeout_s, assume
            # the chip died / USB glitch and reconnect.
            if ever_connected and (now - last_pkt_time) > self.stream_timeout_s:
                log.warning("No radar packets for %.1fs — reconnecting",
                            now - last_pkt_time)
                self._close_data_port()
                self._publish_disconnected()
                self._stop.wait(self.reconnect_interval_s)

        self._close_data_port()
        log.info("RadarCapture thread stopped")

    # ─────────────────────── process loop ────────────────────
    def _process_loop(self) -> None:
        last_seen_seq = 0
        while not self._stop.is_set():
            with self._latest_cond:
                while self._latest_seq == last_seen_seq and not self._stop.is_set():
                    self._latest_cond.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                pkt = self._latest_pkt
                last_seen_seq = self._latest_seq
            if pkt is None:
                continue
            try:
                self._process_and_publish(pkt)
            except Exception as e:
                log.exception("Radar process loop error: %s", e)
        log.info("RadarProcess thread stopped")

    def _process_and_publish(self, pkt: RadarPacket) -> None:
        # 1. SNR + range gate. Points whose SNR is NaN (this build of
        #    mmw_demoDDM does not emit the SideInfo TLV) pass the SNR
        #    test automatically — we can't reject on a signal we don't
        #    have, and the firmware has already CFAR-gated anyway. Real
        #    SNR-aware firmwares (TDM, or a rebuilt DDM with SideInfo)
        #    will populate the field and the gate takes effect.
        import math as _m
        # Azimuth gate: drop off-axis points outside the ±az_half_deg
        # wedge. The AWR2944P antenna radiates well past ±60° but at
        # long range most of that is sidelobe clutter, so concentrating
        # on the main lobe cleans the display AND lets DBSCAN cluster
        # the actual target returns without noise diluting them.
        az_gate = self.az_half_deg
        speed_gate = self.speed_min_mps
        gated: List[RadarDetection] = [
            d for d in pkt.detections
            if (_m.isnan(d.snr_db) or d.snr_db >= self.snr_min_db)
               and d.range_m <= self.max_range_m
               and d.range_m >= self.range_min_m
               and abs(d.az_deg) <= az_gate
               and abs(d.doppler_mps) >= speed_gate
        ]

        # 2. DBSCAN + tracklet association.
        gated, targets = self._clusterer.step(gated)

        self._frame_id += 1
        rf = RadarFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=True,
            detections=gated,
            targets=targets,
            profile=self.profile_name,
            num_points=len(gated),
            num_targets=len(targets),
            max_range_m=self.max_range_m,
            fov_half_deg=self.az_half_deg,
        )
        BUS.publish(Topic.RADAR, rf)

    # ─────────────────────── disconnected sentinel ───────────
    def _publish_disconnected(self) -> None:
        self._frame_id += 1
        BUS.publish(Topic.RADAR, RadarFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=False,
            profile=self.profile_name,
            max_range_m=self.max_range_m,
            fov_half_deg=self.az_half_deg,
        ))

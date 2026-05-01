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
from common.frames import (
    GimbalState,
    RadarDetection,
    RadarFrame,
    RadarTarget,
    Topic,
)
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
        # Mutex around CLI port (COM11). Without this, the capture
        # loop's _push_profile and the DCAPipeline's auto-kick via
        # kick_lvds() can both try to open COM11 at once → loser
        # gets PermissionError. Observed in field run 2026-04-30.
        self._cli_lock = threading.Lock()

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

        Field test 2026-04-29 nailed down a chip-firmware quirk in
        ``mmw_demoDDM`` (SDK 4.7.2.1): the ``sensorStart`` command
        when issued FROM STATE INIT accepts the command and emits a
        few TLV frames, then silently halts. The same chip emits
        continuous data at 49 KB/s for 30 s when started with
        ``sensorStart 0`` instead. Verified by ``scripts/radar_raw_test.py``.

        So the boot sequence we use is:

          1. Read sensor state.
          2. If state is INIT (0)  →  push full cfg. The cfg ends in
             ``sensorStart`` which the firmware rejects with "Invalid
             Sensor Start" because the cfg's own ``sensorStop`` +
             ``flushCfg`` lines moved the chip out of INIT before the
             final line ran. We CATCH that expected error and follow
             up with ``sensorStart 0`` — same warm-restart code path
             the test script uses, which we know works.
          2'. If state is STARTED → ``sensorStop`` then ``sensorStart 0``.
          2''. If state is STOPPED → ``sensorStart 0`` directly.

        Net effect: regardless of the chip's pre-boot state, we end
        up taking the ``sensorStart 0`` path that the field test
        proved works continuously. We never use plain ``sensorStart``
        from INIT — that's the path that emits a few frames and dies.
        """
        if not self._cli_lock.acquire(timeout=5.0):
            log.warning("_push_profile: could not acquire CLI lock")
            return False
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                state = self._query_sensor_state(ser)
                log.info("Radar CLI reports sensor state=%s", state)

                if state in (0, None):
                    # Fresh boot — push the full profile. On THIS chip
                    # the cfg's final `sensorStart` actually succeeds
                    # (chip moves INIT → STARTED), and a follow-up
                    # `sensorStart 0` would be rejected with "Invalid
                    # Sensor Start" because the chip is already STARTED.
                    # That rejection is harmless: the chip is streaming
                    # fine; if we reconnect-loop on this we just end up
                    # cycling sensorStop→sensorStart 0 forever and
                    # never letting the data flow. So: after a cfg
                    # push from INIT, treat "Done" or "Init Calibration
                    # Status" or even "Invalid Sensor Start" on the
                    # follow-up as success — chip is already started.
                    responses = send_cfg(ser, self.cfg_path)
                    tail = responses[-1] if responses else ""
                    cfg_started_chip = (
                        "Done" in tail
                        or "Init Calibration Status" in tail
                        or "Calibration Status = 0x" in tail
                    )
                    if cfg_started_chip:
                        # Chip is already STARTED via cfg's sensorStart.
                        # Skip the follow-up sensorStart 0 — it would
                        # only return Invalid and trip the reconnect
                        # cycle, which would re-send sensorStop and
                        # disrupt the actively-streaming chip.
                        return True

                # Final step on every path: sensorStart 0. This is the
                # ONLY sensorStart variant we trust on this firmware
                # build. Skips the buggy INIT-time path even when
                # state was 0 — even if the cfg's own sensorStart
                # somehow succeeded, sensorStart 0 is a no-op-ish
                # call (chip moves through STOPPED then back to
                # STARTED) that lands us in the known-good state.
                if state == 2:  # STARTED — must stop first
                    self._cli_send(ser, "sensorStop", wait_s=1.0)
                    time.sleep(0.05)
                resp = self._cli_send(ser, "sensorStart 0", wait_s=2.0)
                if "Done" in resp:
                    return True
                # "Invalid Sensor Start" from STARTED state means the
                # chip is already running — accept as success rather
                # than reconnect-cycling and breaking streaming.
                if "Invalid Sensor Start" in resp:
                    log.info("sensorStart 0 returned Invalid (chip already STARTED) — accepted")
                    return True
                log.warning("sensorStart 0 did not ack: %r", resp.strip())
                return False
        except serial.SerialException as e:
            log.warning("Could not open CLI port %s: %s", self.cli_port, e)
            return False
        finally:
            self._cli_lock.release()

    # ─────────────────────── chip kick (LVDS recovery) ─────────────────
    def kick_lvds(self) -> dict:
        """Force the AWR to re-start streaming via sensorStop +
        sensorStart 0 over the CLI UART. Used when LVDS halts but
        TLV is still alive — there's no TLV-based watchdog event
        to trigger automatic recovery, so the operator (or the
        DCAPipeline stall watchdog) can call this directly.

        Locks against the capture-loop's CLI access by closing the
        normal CLI port for the duration. Returns the CLI's
        responses so the caller can see whether the chip ack'd.
        """
        out: dict = {"sensorStop": None, "sensorStart_0": None}
        if not self._cli_lock.acquire(timeout=1.0):
            out["error"] = "CLI busy (reconnect in progress)"
            return out
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                # sensorStop is idempotent from any state; clears the
                # chip's LVDS DMA and frame counters.
                resp_stop = self._cli_send(ser, "sensorStop", wait_s=1.0)
                out["sensorStop"] = resp_stop.strip()
                time.sleep(0.1)
                # sensorStart 0 resumes the previously-loaded profile
                # (the unified.cfg we pushed at boot, including the
                # lvdsStreamCfg line). Same call that brings the chip
                # up cleanly at startup — see _push_profile docstring.
                resp_start = self._cli_send(ser, "sensorStart 0", wait_s=2.0)
                out["sensorStart_0"] = resp_start.strip()
                if "Done" not in resp_start:
                    out["ok"] = False
                    log.warning("kick_lvds: sensorStart 0 did not ack: %r",
                                resp_start.strip())
                    return out
            log.info("kick_lvds: chip kicked (sensorStop + sensorStart 0)")
            return out
        except serial.SerialException as e:
            log.warning("kick_lvds: could not open CLI port %s: %s",
                        self.cli_port, e)
            out["error"] = str(e)
            return out
        finally:
            self._cli_lock.release()

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
        # See ThermalManager._process_and_publish — bind gimbal pose
        # at frame ingest so fusion can convert to world-frame using
        # the pose at capture, not at fusion-tick time.
        gs_for_capture = BUS.get_latest(Topic.GIMBAL)
        if isinstance(gs_for_capture, GimbalState):
            gimbal_pan_at_capture = float(gs_for_capture.pan_deg)
            gimbal_tilt_at_capture = float(gs_for_capture.tilt_deg)
        else:
            gimbal_pan_at_capture = None
            gimbal_tilt_at_capture = None
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
            gimbal_pan_at_capture=gimbal_pan_at_capture,
            gimbal_tilt_at_capture=gimbal_tilt_at_capture,
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

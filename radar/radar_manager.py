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

import glob as _glob
import os as _os
import subprocess as _subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, List, Optional

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

        # Hand-off from capture → process: bounded packet queue.
        #
        # Was: single-slot `_latest_pkt` overwritten on every capture
        # notify. That looked safe (latest-wins) but interacted badly
        # with the data-port read pattern: read(4096) with timeout=0.2
        # batches ~4 small TLV packets per call (cfar=0 frames are
        # ~50 bytes each), capture emits 4 notify_all back-to-back, and
        # the process loop's `while seq == last_seen_seq` only wakes
        # ONCE per "no-update → update" transition — dropping 3 of
        # every 4 packets. Net publish rate capped at 5 Hz even when
        # the chip emits at 20 Hz. Confirmed by `tools/diag_tlv_rate.py`.
        #
        # Now: deque(maxlen=8) preserves the latest-wins overload
        # behaviour (oldest auto-evicted when process lags) but
        # operates at packet granularity. Process loop drains one
        # packet per wake → publish rate matches chip rate when
        # processing is fast (it is on cfar=0 frames).
        self._pkt_cond = threading.Condition()
        self._pkt_queue: "deque[RadarPacket]" = deque(maxlen=8)

        self._data_ser: Optional[serial.Serial] = None
        self._frame_id: int = 0
        self._clusterer = RadarClusterer(cluster_params)

        # Optional hook fired AFTER a successful xds110reset chip recovery
        # but BEFORE _push_profile pushes the cfg. Composite registers a
        # callback here that re-arms the DCA1000 FPGA so LVDS resumes
        # cleanly when the chip sensorStarts. Without it, A/A is dead
        # after every Ctrl+C → relaunch cycle.
        self._post_recovery_callback: Optional[Callable[[], None]] = None

    def set_post_recovery_callback(
        self, cb: Optional[Callable[[], None]]
    ) -> None:
        """Register a callback fired after xds110 chip recovery.

        Composite uses this to re-arm the DCA1000 FPGA after the chip
        is hard-reset out from under it. Called inside the recovery
        path of _probe_and_recover, after the chip is confirmed alive
        and BEFORE the cfg push restarts LVDS streaming. Any exception
        the callback raises is logged but doesn't abort recovery.
        """
        self._post_recovery_callback = cb

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
        use_doppler: Optional[bool] = None,
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
            if use_doppler is not None:
                cp.use_doppler = bool(use_doppler)

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

    def get_extrinsic(self) -> dict:
        """Return the live radar az/el biases.

        Symmetric with `set_extrinsic`. The GUI's `extrinsic_save`
        handler calls this to capture the operator's tuned values
        BEFORE writing them to `config/calibration.json`.

        Bug history (2026-05-12 fix): this method was missing for the
        entire life of RadarManager. `gui/app.py:1075` calls
        `rm.get_extrinsic()` inside a broad try/except — the
        `AttributeError` was silently caught, the payload sent to
        `calibration_store.save()` had no `radar_az`/`radar_el`
        keys, the store's "leave-untouched" semantics never wrote
        any radar bias to disk, and the operator's calibration
        evaporated on every restart. Visible in the existing
        `config/calibration.json`: `radar: {}` while `thermal` has
        values.
        """
        with self._tune_lock:
            return {
                "az_bias_deg": float(self.az_bias_deg),
                "el_bias_deg": float(self.el_bias_deg),
            }

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
            "use_doppler": cp.use_doppler,
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
        with self._pkt_cond:
            self._pkt_cond.notify_all()
        for t in (self._process_thread, self._capture_thread):
            if t is not None:
                t.join(timeout=3.0)
        self._capture_thread = None
        self._process_thread = None
        self._close_data_port()
        # ISSUE-2 FIX: NO sensorStop on this firmware — the patches
        # removed the only poster of DPMstopSemHandle, so sensorStop
        # makes MmwDemo_stopSensor pend WAIT_FOREVER and wedges the
        # CLI parser. Leave the chip running; next start adopts it.
        log.warning("[ISSUE2-STOP] Skipping sensorStop on shutdown "
                    "(would wedge chip CLI on this firmware).")

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

    # ─────────────────── chip wedge recovery (issue 2) ───────────────
    # When the host process exits via Ctrl+C, pyserial's port-close (and
    # the next port-open) toggles DTR/RTS on the XDS110 virtual UART.
    # Those modem-control lines route to chip-side GPIOs that
    # mmw_demoDDM (SDK 4.7.2.1) uses for host handshake; the toggle
    # deadlocks the CLI parser inside UART_writePolling and the chip
    # stops acking ANY command until 12 V power-cycle. Verified by the
    # 2026-05-06 research-agent and pyserial issues #124 / #488.
    #
    # Recovery design (per user-approved plan i-have-serious-issues-...):
    #   1. After CLI port open: ser.send_break(0.25) — 250 ms break
    #      resets the chip's UART RX state machine (TI E2E recommended,
    #      same effect as "close+reopen TeraTerm" in TI's SDK guide).
    #   2. Probe with queryDemoStatus at short timeout. If chip responds,
    #      we're cold-booted or already recovered — proceed normally.
    #   3. If probe times out: chip is wedged. Close the port, run
    #      xds110reset.exe to pulse nRST via XDS110 JTAG (CCS ships it),
    #      wait for chip re-enumeration, reopen, send_break, retry probe.
    #   4. Single reset attempt per _push_profile call. If a second probe
    #      also fails, surface the error and let the reconnect loop spin
    #      — there's no third recovery layer in software.
    _XDS110_RESET_CANDIDATES = (
        r"C:\ti\ccs\ccs\ccs_base\common\uscif\xds110\xds110reset.exe",
        r"C:\ti\ccs*\ccs\ccs_base\common\uscif\xds110\xds110reset.exe",
        r"C:\ti\ccs1*\ccs\ccs_base\common\uscif\xds110\xds110reset.exe",
        r"C:\ti\ccs2*\ccs\ccs_base\common\uscif\xds110\xds110reset.exe",
    )

    @classmethod
    def _find_xds110_reset(cls) -> Optional[str]:
        for pat in cls._XDS110_RESET_CANDIDATES:
            if "*" in pat or "?" in pat:
                hits = _glob.glob(pat)
                if hits:
                    return hits[0]
            elif _os.path.isfile(pat):
                return pat
        return None

    def _run_xds110_reset(self) -> bool:
        """Pulse nRST on the AWR via the XDS110 debug bridge. Returns True
        on success. Linux uses an in-tree pyusb-based pulser
        (``tools/xds110reset_linux.py``); Windows shells out to TI's
        ``xds110reset.exe`` shipped with CCS.

        On Linux the AWR2944P's XDS110 is fully software-resettable via
        libusb — no CCS install is required. The pyusb path was added
        2026-05-08 when seeker moved from Windows to the Jetson; the
        Windows-only path stayed in place for the bench laptop.
        """
        import sys as _sys
        if _sys.platform.startswith("linux"):
            try:
                import os as _os2
                _tools_dir = _os2.path.join(
                    _os2.path.dirname(_os2.path.dirname(_os2.path.abspath(__file__))),
                    "tools",
                )
                if _tools_dir not in _sys.path:
                    _sys.path.insert(0, _tools_dir)
                from xds110reset_linux import xds110_pulse_nrst
            except Exception as e:
                log.error(
                    "[ISSUE2-RECOVERY] xds110reset_linux import failed: %s "
                    "(install pyusb in the venv, then retry)", e,
                )
                return False
            log.warning("[ISSUE2-RECOVERY] Pulsing nRST via tools/xds110reset_linux.py (libusb)")
            try:
                xds110_pulse_nrst(hold_ms=80)
            except Exception as e:
                log.error("[ISSUE2-RECOVERY] xds110reset_linux failed: %s", e)
                return False
            log.info("[ISSUE2-RECOVERY] xds110reset_linux OK — chip rebooting")
            return True

        # ---- Windows path (legacy) ----
        exe = self._find_xds110_reset()
        if exe is None:
            log.error(
                "[ISSUE2-RECOVERY] xds110reset.exe NOT FOUND. Install CCS "
                "(it ships there). Tried: %s",
                ", ".join(self._XDS110_RESET_CANDIDATES),
            )
            return False
        log.warning("[ISSUE2-RECOVERY] Pulsing nRST via %s", exe)
        try:
            r = _subprocess.run(
                [exe], capture_output=True, text=True, timeout=10.0,
            )
        except _subprocess.TimeoutExpired:
            log.error("[ISSUE2-RECOVERY] xds110reset timed out (>10s)")
            return False
        except Exception as e:
            log.error("[ISSUE2-RECOVERY] xds110reset failed to run: %s", e)
            return False
        log.info(
            "[ISSUE2-RECOVERY] xds110reset rc=%s stdout=%r stderr=%r",
            r.returncode, (r.stdout or "").strip()[:200],
            (r.stderr or "").strip()[:200],
        )
        return r.returncode == 0

    def _open_cli_with_break(self) -> Optional[serial.Serial]:
        """Open CLI port and issue a 250 ms break to reset chip-side UART RX."""
        try:
            ser = serial.Serial(self.cli_port, self.cli_baud, timeout=0.5)
        except serial.SerialException as e:
            log.warning("Could not open CLI port %s: %s", self.cli_port, e)
            return None
        try:
            ser.send_break(0.25)
        except Exception as e:
            log.debug("send_break failed (driver may not support): %s", e)
        # Drain any garbage the break may have produced.
        time.sleep(0.05)
        try:
            n = ser.in_waiting
            if n:
                ser.read(n)
        except Exception:
            pass
        return ser

    def _probe_and_recover(self) -> Optional[serial.Serial]:
        """Open the CLI port and confirm the chip is responsive.

        Returns an OPEN ``serial.Serial`` ready for cfg push, or None if
        even after one xds110reset attempt the chip stays silent.
        Caller is responsible for closing the returned Serial.
        """
        ser = self._open_cli_with_break()
        if ser is None:
            return None
        # Probe with the same 1.0 s the legacy _query_sensor_state used —
        # cold-boot chips need ~1 s for the CLI task to come up after the
        # BSS calibration banner. A 0.5 s probe would time out on a
        # healthy fresh boot and trigger an unnecessary xds110reset.
        probe = self._cli_send(ser, "queryDemoStatus", wait_s=1.0)
        if probe.strip():
            log.info("Radar CLI alive on first probe (resp=%r)",
                     probe.strip()[:80])
            return ser
        # Wedged. Close, reset, retry once.
        log.warning("[ISSUE2-RECOVERY] CLI gave no response — assuming "
                    "post-Ctrl+C wedge, attempting xds110 nRST")
        try:
            ser.close()
        except Exception:
            pass
        if not self._run_xds110_reset():
            return None
        # Wait for chip + USB CDC to re-enumerate after reset.
        time.sleep(1.5)
        ser2 = self._open_cli_with_break()
        if ser2 is None:
            log.error("[ISSUE2-RECOVERY] CLI port did not reopen after reset")
            return None
        probe2 = self._cli_send(ser2, "queryDemoStatus", wait_s=2.0)
        if probe2.strip():
            log.info("[ISSUE2-RECOVERY] Chip recovered (resp=%r)",
                     probe2.strip()[:80])
            # Fire the post-recovery hook BEFORE returning. Composite
            # uses this to re-arm the DCA1000 FPGA so LVDS resumes
            # cleanly when the cfg push (next step in _push_profile)
            # sensorStarts the chip. Best-effort — don't abort recovery
            # if the callback explodes.
            if self._post_recovery_callback is not None:
                try:
                    self._post_recovery_callback()
                except Exception:
                    log.exception(
                        "[ISSUE2-RECOVERY] post-recovery callback raised; "
                        "continuing with chip recovery anyway"
                    )
            return ser2
        log.error("[ISSUE2-RECOVERY] Chip still silent after xds110 reset; "
                  "giving up this _push_profile cycle")
        try:
            ser2.close()
        except Exception:
            pass
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
        # Probe + auto-recover from post-Ctrl+C wedge before we push any
        # cfg. Returns an OPEN Serial we own — close it ourselves.
        ser = self._probe_and_recover()
        if ser is None:
            return False
        try:
            with ser:
                state = self._query_sensor_state(ser)
                log.info("Radar CLI reports sensor state=%s", state)

                if state in (0, None):
                    # Fresh boot or unknown state — push the full
                    # profile so the chip has our cfg loaded.
                    #
                    # The cfg's last line is ``sensorStart``. On THIS
                    # firmware build the cfg's sensorStart from INIT
                    # actually succeeds (chip moves INIT → STARTED).
                    # If we then send our own ``sensorStart 0`` below
                    # the chip rejects it ("Invalid Sensor Start" —
                    # it's already STARTED) and V1.0's downstream
                    # check returned False → reconnect cycle, which
                    # broke streaming. Detect cfg-started-chip via
                    # "Done" / "Init Calibration Status" in the cfg's
                    # last-line response and return True directly,
                    # skipping the redundant sensorStart 0.
                    responses = send_cfg(ser, self.cfg_path)
                    tail = responses[-1] if responses else ""
                    # Count rejected lines in the cfg push. The chip's
                    # state machine sometimes rejects critical lines
                    # (profileCfg, chirpCfg, frameCfg) with "Error:
                    # Configuration is valid only if DFE Output Mode
                    # is X" — a chip-side state-machine race we don't
                    # fully understand. When that happens the chip's
                    # cfg is incomplete; sensorStart will still emit
                    # a partial calibration status (0x11e instead of
                    # the full 0xffe) but no useful frames stream.
                    # Don't fool ourselves — if any critical line was
                    # rejected, treat the push as failed and let the
                    # reconnect loop retry (next attempt often works
                    # because the chip happens to be in a clean
                    # state).
                    n_rejected = sum(
                        1 for r in responses
                        if ("Error" in r or "error" in r)
                    )
                    full_cal = ("0xffe" in tail or "Done" in tail)
                    if n_rejected == 0 and full_cal:
                        log.info("Chip started by cfg's own sensorStart "
                                 "(tail=%r, rejected=0) — skipping "
                                 "redundant sensorStart 0",
                                 tail.strip()[:80])
                        return True
                    if n_rejected > 0:
                        log.warning("cfg push had %d rejected line(s); "
                                    "chip cfg is incomplete — will retry",
                                    n_rejected)
                        return False
                    log.warning("cfg push tail=%r unrecognized as success; "
                                "will retry", tail.strip()[:120])
                    return False

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
                if "Done" not in resp:
                    log.warning("sensorStart 0 did not ack: %r", resp.strip())
                    return False
                return True
        except serial.SerialException as e:
            log.warning("Could not open CLI port %s: %s", self.cli_port, e)
            return False

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
        # First try the cheap path: just sensorStop + sensorStart 0.
        # If chip is in STARTED but DMA stalled, this resets it.
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                resp_stop = self._cli_send(ser, "sensorStop", wait_s=1.0)
                out["sensorStop"] = resp_stop.strip()
                time.sleep(0.1)
                resp_start = self._cli_send(ser, "sensorStart 0", wait_s=2.0)
                out["sensorStart_0"] = resp_start.strip()
                if "Done" in resp_start:
                    log.info("kick_lvds: chip kicked (sensorStop + sensorStart 0)")
                    return out
                # Empty response or Invalid Sensor Start → chip in a
                # state where sensorStart 0 doesn't help (likely the
                # DMA halt state where Stop+Start 0 isn't enough on
                # this firmware build). Fall through to a full cfg
                # re-push, which forces the chip through INIT.
                log.warning("kick_lvds: sensorStart 0 didn't recover "
                            "(resp=%r); will re-push full cfg",
                            resp_start.strip()[:80])
        except serial.SerialException as e:
            log.warning("kick_lvds: could not open CLI port %s: %s",
                        self.cli_port, e)
            out["error"] = str(e)
            return out

        # Heavy path: full cfg re-push. The cfg starts with sensorStop
        # + flushCfg, which forces the chip back through its state
        # machine. The cfg's final sensorStart should then bring the
        # chip back into STARTED with LVDS DMA reset.
        log.info("kick_lvds: heavy path — re-pushing full cfg")
        try:
            with serial.Serial(self.cli_port, self.cli_baud, timeout=0.5) as ser:
                responses = send_cfg(ser, self.cfg_path)
                tail = responses[-1] if responses else ""
                n_rejected = sum(
                    1 for r in responses if ("Error" in r or "error" in r)
                )
                ok = (n_rejected == 0
                      and ("0xffe" in tail or "Done" in tail))
                if ok:
                    out["cfg_repush"] = "ok"
                    log.info("kick_lvds: cfg re-push succeeded "
                             "(0 rejected, tail had 0xffe/Done)")
                else:
                    out["cfg_repush"] = f"failed (rejected={n_rejected})"
                    log.warning("kick_lvds: cfg re-push failed: "
                                "%d rejected, tail=%r",
                                n_rejected, tail.strip()[:80])
            return out
        except serial.SerialException as e:
            log.warning("kick_lvds: cfg re-push could not open CLI: %s", e)
            out["error"] = str(e)
            return out

    def _open_data_port(self) -> bool:
        try:
            # Read timeout matches the chip's frame period (50 ms = 20 Hz).
            # With timeout=0.2, on cfar=0 frames the buffer accumulates
            # ~4 small TLV packets per timeout window before read returns;
            # the capture loop then notifies the process queue 4 times
            # in microseconds, which the deque drains in microseconds,
            # producing a 20 Hz publish rate that's BURSTY (4 frames
            # back-to-back, then 200 ms of nothing). Average is correct
            # but visually it looks like ~5 Hz with motion blur. With
            # timeout matched to the chip's frame period, each read
            # returns with ~1 packet — steady 20 Hz, no bursts. Also
            # 4× faster shutdown responsiveness for free.
            self._data_ser = serial.Serial(
                self.data_port,
                self.data_baud,
                timeout=0.05,
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
                    with self._pkt_cond:
                        # deque(maxlen=8) auto-evicts oldest on overflow
                        # — that's the back-pressure behaviour we want
                        # if process_and_publish ever lags chip rate.
                        self._pkt_queue.append(pkt)
                        self._pkt_cond.notify()
                    last_pkt_time = now

            # Stream-timeout disconnect (LEGACY): if we've been
            # connected but haven't seen a valid packet in
            # stream_timeout_s, V1.0 reconnected — which re-pushed
            # the cfg, which started with `sensorStop`, which killed
            # any active LVDS streaming to the DCA1000.
            #
            # On the unified Phase-3 cfg this reconnect is HARMFUL:
            # `lvdsStreamCfg -1 0 1 0` enables LVDS but appears to
            # disable UART TLV output on this firmware build, so the
            # 3 s timeout fires every cycle. The reconnect's
            # sensorStop then breaks the LVDS stream that A/A
            # depends on. Field log 2026-04-30 19:21 showed the
            # chip cycling cfg pushes every 3 s and LVDS stalling
            # right after each one (76,556 packets received then
            # halted, repeat).
            #
            # Fix: when stream_timeout_s is 0 or negative, skip the
            # reconnect entirely. The chip stays running, LVDS
            # streams continuously into the DCA pipeline. If the
            # operator needs A/A only, they set stream_timeout_s=0
            # and the manager becomes a one-shot cfg pusher.
            if (self.stream_timeout_s > 0
                    and ever_connected
                    and (now - last_pkt_time) > self.stream_timeout_s):
                log.warning("No radar packets for %.1fs — reconnecting",
                            now - last_pkt_time)
                self._close_data_port()
                self._publish_disconnected()
                self._stop.wait(self.reconnect_interval_s)

        self._close_data_port()
        log.info("RadarCapture thread stopped")

    # ─────────────────────── process loop ────────────────────
    def _process_loop(self) -> None:
        # Drain the packet queue one packet per wake. This is the fix
        # for the 5 Hz cap — see `_pkt_queue` doc in __init__.
        while not self._stop.is_set():
            with self._pkt_cond:
                while not self._pkt_queue and not self._stop.is_set():
                    self._pkt_cond.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                pkt = self._pkt_queue.popleft()
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

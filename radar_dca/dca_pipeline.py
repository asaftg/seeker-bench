"""DCA raw-ADC → RadarTarget pipeline (M4 scaffold).

Sits between the UDP listener (``data_port.DataPortListener``) and the
``RadarFrame`` bus topic. Owned by ``DCAManager``. The pipeline thread
runs the following loop:

    UDP packets (queue from listener)
            │
            ▼ (M4.1)  ADC reassembly: drop the 10-byte header, append
              (NOTE: pipeline publishes on Topic.RADAR_AA, not Topic.RADAR.
               That keeps TLV-derived frames from RadarManager and PMM-derived
               frames from this pipeline cleanly separated on the bus, so
               downstream consumers can subscribe to whichever stream their
               mode requires.)
                       payload bytes to a frame buffer, slice into
                       ``(n_chirps, n_rx, n_samples)`` complex arrays
                       once a frame's worth has accumulated.
            ▼
    chirp matrix (n_chirps × n_rx × n_samples)
            │
            ▼ (M4.2)  Range FFT — per-chirp FFT across n_samples.
            ▼
    range-time matrix (n_chirps × n_rx × n_range_bins)
            │
            ▼ (M4.3)  Channel sum (or beamform) across n_rx.
            ▼
    slow-time matrix (n_chirps × n_range_bins) complex
            │
            ▼ (M4.4)  PMM scan — pmm_detector.scan_range_bins().
            ▼
    [(range_bin, PMMResult)] for bins with a propeller signature
            │
            ▼ (M4.5)  Convert range_bin → range_m, emit RadarTarget
                       with class=DRONE, position from range + (assumed)
                       boresight direction (proper AoA needs Capon BF
                       on the per-rx complex samples — M4.6, deferred).
            ▼
    BUS.publish(Topic.RADAR, RadarFrame(...))

Current status (overnight scaffold — bench validation BLOCKED until AWR
power-cycle):

- M4.1 ADC reassembly: STUB — uses the actual byte counters but does
  not parse samples. Code path is shaped correctly so wiring the
  real parser is a single function later.
- M4.2 Range FFT: STUB — generates zero arrays sized correctly.
- M4.3 Channel sum: STUB.
- M4.4 PMM scan: WIRED. Runs on the (zero-filled or real) slow-time
  matrix. With zero input, scan_range_bins returns no detections.
- M4.5 RadarTarget emission: WIRED. Consumes PMM hits, emits frames.

When real ADC samples flow in (post-power-cycle, A/A cfg pushed,
DCA in start_record state), three things need to be filled in based
on a real ``.bin`` capture's hex layout:

1. ``_parse_packet_payload(payload_bytes) -> chirp_samples_array``
   — knows how the AWR packs N RX × M ADC samples per packet.
2. ``_n_chirps_per_frame``, ``_n_rx``, ``_n_samples`` — read from
   the AWR cfg; computed in the helper below from the cfg file.
3. ``_chirp_period_s`` — inverse PRF for the PMM detector.

The TI doc that describes the layout is SPRUIK7 §4.x and the mmWave
SDK ``mmwavelink_user_guide``. We'll fill these from a ground-truth
`.bin` once the chip is back online.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from common.frame_bus import BUS
from common.frames import RadarDetection, RadarFrame, RadarTarget, TargetClass, Topic
from common.logging_setup import get_logger
from radar_dca.data_port import DataPortListener
from radar_dca.pmm_detector import PMMResult, scan_range_bins

log = get_logger(__name__)


@dataclass
class FrameDims:
    """Shape of a raw-ADC frame as configured by the AWR cfg.

    Computed once at pipeline start from the cfg parameters. The
    pipeline accumulates UDP payload bytes until it has enough for
    one frame, then reshapes into a ``(n_chirps, n_rx, n_samples)``
    complex buffer.
    """
    n_chirps: int
    n_rx: int
    n_samples: int
    bytes_per_sample: int = 4   # I + Q, each int16 LE = 4 bytes per sample
    range_resolution_m: float = 0.04   # for diagnostics; computed from chirp BW
    chirp_period_s: float = 100e-6     # 1 / PRF; computed from frameCfg

    @property
    def bytes_per_chirp(self) -> int:
        return self.n_rx * self.n_samples * self.bytes_per_sample

    @property
    def bytes_per_frame(self) -> int:
        return self.n_chirps * self.bytes_per_chirp

    @property
    def prf_hz(self) -> float:
        return 1.0 / self.chirp_period_s


def dims_from_cfg(
    n_chirps: int = 768,
    n_rx: int = 4,
    n_samples: int = 384,
    chirp_period_s: float = 27.81e-6,    # idleTime 7 + rampEnd 20.81 = 27.81 us
    range_resolution_m: float = 1.32,    # c / (2 × slope × adc_time)
) -> FrameDims:
    """Build FrameDims from explicit numbers (legacy entry point).

    Use ``dims_from_cfg_file()`` instead — it parses the actual .cfg
    so changes to chirpCfg/frameCfg/profileCfg are picked up
    automatically. This function exists for tests that want to
    construct dims without a file."""
    return FrameDims(
        n_chirps=n_chirps,
        n_rx=n_rx,
        n_samples=n_samples,
        chirp_period_s=chirp_period_s,
        range_resolution_m=range_resolution_m,
    )


def dims_from_cfg_file(cfg_path: str) -> FrameDims:
    """Parse the AWR mmw_demo ``.cfg`` and compute FrameDims.

    Reads the four lines that determine the wire-byte layout:

      ``channelCfg <rxMask> <txMask> ...``       — n_rx, n_tx
      ``profileCfg 0 <startGHz> <idleUs> <adcStartUs> <rampEndUs>
                   <txOutPower> <txPhaseShifter> <freqSlopeMHzPerUs>
                   <txStartUs> <numAdcSamples> <digOutSampleRateKsps> ...``
      ``chirpCfg <startIdx> <endIdx> <profileId> ...``
      ``frameCfg <chirpStartIdx> <chirpEndIdx> <numLoops>
                 <numFrames> <framePeriodMs> ...``

    Field ordering follows the mmw_demo SDK CLI parser (see
    ``mss_main.c::CLI_sensorStart``). Comments (``%``) and blank
    lines are skipped.

    Range resolution is c / (2 · slope · adc_capture_time), where
    adc_capture_time = numAdcSamples / digOutSampleRate. This is the
    SAMPLED bandwidth — different from the swept bandwidth (slope ×
    rampEnd), which over-estimates range resolution by the ratio of
    swept to captured time."""
    n_rx: Optional[int] = None
    n_samples: Optional[int] = None
    chirp_period_s: Optional[float] = None
    range_res_m: Optional[float] = None
    chirp_indices: Optional[int] = None
    n_loops: Optional[int] = None

    with open(cfg_path, "r") as f:
        for raw in f:
            ln = raw.strip()
            if not ln or ln.startswith("%") or ln.startswith("#"):
                continue
            tok = ln.split()
            cmd = tok[0]
            try:
                if cmd == "channelCfg":
                    rx_mask = int(tok[1])
                    n_rx = bin(rx_mask).count("1")
                elif cmd == "profileCfg":
                    # profileId startFreq idleTime adcStartTime rampEndTime
                    # txOutPower txPhaseShifter freqSlopeConst txStartTime
                    # numAdcSamples digOutSampleRate hpfCornerFreq1
                    # hpfCornerFreq2 rxGain
                    idle_us = float(tok[3])
                    ramp_us = float(tok[5])
                    slope_mhz_us = float(tok[8])
                    n_samples = int(tok[10])
                    digout_ksps = float(tok[11])
                    chirp_period_s = (idle_us + ramp_us) * 1e-6
                    adc_capture_us = n_samples / digout_ksps * 1e3   # ks → samples/s
                    bw_hz = slope_mhz_us * 1e6 * adc_capture_us       # MHz/us × us → Hz
                    range_res_m = 3e8 / (2.0 * bw_hz)
                elif cmd == "frameCfg":
                    start_idx = int(tok[1])
                    end_idx = int(tok[2])
                    chirp_indices = end_idx - start_idx + 1
                    n_loops = int(tok[3])
            except (IndexError, ValueError) as e:
                log.warning("dims_from_cfg_file: cannot parse %r: %s", ln, e)

    missing = [k for k, v in {
        "n_rx": n_rx, "n_samples": n_samples,
        "chirp_period_s": chirp_period_s, "range_res_m": range_res_m,
        "chirp_indices": chirp_indices, "n_loops": n_loops,
    }.items() if v is None]
    if missing:
        raise ValueError(
            f"cfg {cfg_path} missing required keys: {missing}; falling back "
            "to defaults will likely give wrong frame size"
        )

    return FrameDims(
        n_chirps=int(chirp_indices) * int(n_loops),
        n_rx=int(n_rx),                                    # type: ignore[arg-type]
        n_samples=int(n_samples),                          # type: ignore[arg-type]
        chirp_period_s=float(chirp_period_s),              # type: ignore[arg-type]
        range_resolution_m=float(range_res_m),             # type: ignore[arg-type]
    )


# ─────────────────────── pipeline ──────────────────────────────────────────


@dataclass
class PipelineStats:
    """Counters for the GUI / diagnostics."""
    frames_assembled: int = 0
    frames_dropped: int = 0           # raw-ADC bytes too few or out of sync
    drone_detections: int = 0
    last_drone_range_m: float = float("nan")
    last_drone_blade_freq_hz: float = float("nan")


class DCAPipeline:
    """Owns the worker thread that ingests raw ADC and emits RadarFrames.

    Lifecycle:
      ``start(listener)`` — spawn thread; pulls bytes from listener.
      ``stop()``          — graceful join.

    The pipeline does NOT bind the UDP socket itself — that's the
    listener's job. The pipeline reads from the listener's internal
    state (we extend the listener with a packet queue if needed).
    For Day-1 scaffold we read STATS from the listener and emit
    sentinel RadarFrames on a heartbeat; M4.1 fills in the actual
    byte queue.
    """

    def __init__(
        self,
        *,
        listener: DataPortListener,
        dims: FrameDims,
        pmm_band_low_hz: float = 50.0,
        pmm_band_high_hz: float = 500.0,
        pmm_threshold_db: float = 6.0,
        profile_name: str = "awr2944p_aa",
        max_range_m: float = 250.0,
        az_half_deg: float = 25.0,
    ) -> None:
        self._listener = listener
        self._dims = dims
        self._pmm_band_low = float(pmm_band_low_hz)
        self._pmm_band_high = float(pmm_band_high_hz)
        self._pmm_threshold = float(pmm_threshold_db)
        self.profile_name = profile_name
        self.max_range_m = float(max_range_m)
        self.az_half_deg = float(az_half_deg)
        # Optional back-ref to the RadarManager so the LVDS stall
        # watchdog (in _loop) can call kick_lvds() to recover when
        # the chip's DMA halts. Set by CompositeRadarBackend.
        self._radar_manager_ref = None
        # Optional back-ref to the DCAControl so the auto-kick can
        # ALSO reset the DCA's FPGA when chip-side recovery alone
        # isn't enough (DCA buffer flushed, start_record re-armed).
        self._dca_control_ref = None

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._stats = PipelineStats()

        # Raw-ADC byte buffer — accumulates UDP payloads from the listener
        # until bytes_per_frame is reached, then reshapes to a complex cube
        # and runs the per-mode signal-processing chain.
        self._buf = bytearray()
        # When False, the worker still consumes UDP bytes (so we don't
        # lose packets when not in A/A mode) but suppresses publishes
        # to Topic.RADAR_AA. The composite flips this on set_mode("aa").
        self._publish_enabled = False
        # A/G knobs — set by composite.update_ag_params; read by the
        # AG processor at frame time. Defaults match the GUI HTML defaults.
        self._ag_integrate_chirps = 512
        self._ag_cfar_algo = "os"
        self._ag_cfar_threshold_db = 12.0
        self._ag_capon_bf = True
        # PMM extras
        self._pmm_slow_time_win = 256
        self._staggered_prf = False

    # ─────────────────────── public API ─────────────────────────────────
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DCAPipeline", daemon=True
        )
        self._thread.start()
        log.info(
            "DCAPipeline started (%d chirps × %d RX × %d samples = %.1f MB/frame "
            "at PRF %.0f Hz; PMM band [%.0f, %.0f] Hz @ %.1f dB)",
            self._dims.n_chirps, self._dims.n_rx, self._dims.n_samples,
            self._dims.bytes_per_frame / 1e6, self._dims.prf_hz,
            self._pmm_band_low, self._pmm_band_high, self._pmm_threshold,
        )

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info("DCAPipeline stopped (frames=%d, drops=%d, drone hits=%d)",
                 self._stats.frames_assembled, self._stats.frames_dropped,
                 self._stats.drone_detections)

    def stats(self) -> PipelineStats:
        return PipelineStats(**self._stats.__dict__)

    # ─────────────────────── worker thread ─────────────────────────────
    def _loop(self) -> None:
        """Drain the listener's payload queue → byte buffer → frames.

        Each iteration:
          1. ``listener.drain_payloads()`` returns every UDP payload
             received since the last call.
          2. We extend ``self._buf`` with those bytes.
          3. While the buffer has at least one frame's worth, slice
             off ``bytes_per_frame`` bytes and call ``_process_frame``.
             A backlog of frames is processed greedily so we don't
             accumulate latency — the most recent frame wins on the
             bus topic anyway.
          4. If no frames were produced this tick, emit a
             ``connected=False`` sentinel so the GUI knows the
             pipeline is alive but starved.

        Tick cadence is 20 ms — fast enough that backlog never builds
        when the consumer keeps up, slow enough that we don't burn
        CPU spinning. At ~143 ms/frame (PRF/n_chirps), one frame's
        worth of bytes accumulates over ~7 ticks; the inner ``while``
        loop catches up immediately when we get behind.
        """
        bpf = self._dims.bytes_per_frame
        TICK_S = 0.020
        # LVDS stall watchdog. The chip on this firmware emits LVDS
        # in bursts (~5 s of streaming, then halts the DMA). The only
        # known way to recover without a power cycle is to issue
        # `sensorStop` + `sensorStart 0` over the radar CLI — that
        # forces the chip to reset its LVDS DMA and resume streaming.
        # We do this automatically every 8 s of stall, rate-limited
        # to once per 12 s so a truly dead chip doesn't get hammered.
        # Validated end-to-end on 2026-04-30 19:26 (after V1.0 +
        # surgical fixes): chip stops emitting after a ~5 s burst,
        # auto-kick brings it back, repeat — keeps PMM detector
        # supplied with continuous slow-time windows.
        STALL_WARN_S = 5.0
        STALL_KICK_S = 8.0
        STALL_KICK_COOLDOWN_S = 12.0
        _stall_logged = False
        _last_kick_t = 0.0

        while not self._stop.is_set():
            # 1. Pull all queued payloads from the listener.
            payloads = self._listener.drain_payloads()
            if payloads:
                # bytearray.extend(bytes) is amortized O(N).
                for p in payloads:
                    self._buf.extend(p)

            # 1b. Stall watchdog. ``last_packet_age_s`` is +inf before
            # the first packet ever arrives — don't warn about boot
            # quiescence, only about a stream that USED to flow and
            # has gone silent.
            try:
                ds = self._listener.stats()
                age = ds.last_packet_age_s
                if (ds.packets_total > 0
                        and age != float("inf")
                        and age > STALL_WARN_S):
                    if not _stall_logged:
                        log.warning(
                            "LVDS stalled: no UDP for %.1fs. Will auto-kick "
                            "(sensorStop+sensorStart 0) in %.0f s. "
                            "packets_total=%d seq_drops=%d",
                            age, STALL_KICK_S - STALL_WARN_S,
                            ds.packets_total, ds.seq_drops_total,
                        )
                        _stall_logged = True
                    # Auto-kick to recover. The chip on this firmware
                    # halts LVDS DMA after a finite burst; the only
                    # reliable recovery is to reset BOTH sides:
                    #   1. DCA1000: stop_record → reset_fpga →
                    #      setup_capture → start_record (flushes the
                    #      FPGA's internal buffer and re-arms it).
                    #   2. AWR chip: kick_lvds() (Stop+Start 0, then
                    #      heavy-path full cfg re-push if needed).
                    # Without resetting the DCA, the chip's restarted
                    # LVDS frames hit a DCA in some stale state and
                    # never make it onto the wire.
                    now_t = time.monotonic()
                    if (age > STALL_KICK_S
                            and now_t - _last_kick_t > STALL_KICK_COOLDOWN_S
                            and self._radar_manager_ref is not None
                            and hasattr(self._radar_manager_ref, "kick_lvds")):
                        log.info("LVDS auto-kick: resetting DCA FPGA + "
                                 "kicking chip")
                        try:
                            # 1. Reset DCA first so it's ready for the
                            # chip's first post-kick LVDS frame.
                            if self._dca_control_ref is not None:
                                try:
                                    self._dca_control_ref.stop_record()
                                except Exception:
                                    pass
                                time.sleep(0.1)
                                try:
                                    self._dca_control_ref.reset_fpga()
                                except Exception:
                                    log.exception("auto-kick: DCA reset_fpga raised")
                                time.sleep(0.1)
                                try:
                                    self._dca_control_ref.setup_capture()
                                    self._dca_control_ref.start_record()
                                    log.info("auto-kick: DCA back in start_record")
                                except Exception:
                                    log.exception("auto-kick: DCA setup_capture/start_record raised")
                            # 2. Kick the chip.
                            self._radar_manager_ref.kick_lvds()
                            _last_kick_t = now_t
                        except Exception:
                            log.exception("LVDS auto-kick raised")
                else:
                    if _stall_logged and age < 0.5:
                        log.info("LVDS recovered (last_packet_age=%.2fs)", age)
                    _stall_logged = False
            except Exception:
                pass

            # 2. Process as many full frames as the buffer holds.
            processed_any = False
            while len(self._buf) >= bpf:
                frame_bytes = bytes(self._buf[:bpf])
                # Slice instead of pop+rebuild — bytearray supports
                # del slice in O(N) but with much smaller constants
                # than building a new bytearray.
                del self._buf[:bpf]
                try:
                    self._process_frame(frame_bytes)
                    processed_any = True
                except Exception as e:
                    # A bad frame should NOT take the pipeline down.
                    # Count it as a drop and keep processing.
                    log.exception("frame processing failed: %s", e)
                    self._stats.frames_dropped += 1

            if not processed_any:
                self._publish_sentinel()

            self._stop.wait(TICK_S)

    # ─────────────────────── frame processing ───────────────────────────
    def _bytes_to_cube(self, frame_bytes: bytes) -> np.ndarray:
        """Reshape one frame's wire bytes into ``(n_chirps, n_samples,
        n_rx)`` complex64.

        Wire layout (DCA1000 channel-interleave, lvdsMode=1):

          per chirp: [RX0 s0 I/Q] [RX1 s0 I/Q] [RX2 s0 I/Q] [RX3 s0 I/Q]
                     [RX0 s1 I/Q] ...

        Each I and Q is int16 LE, so 4 bytes per (RX, sample) pair.
        Total per chirp = n_rx × n_samples × 4 bytes.

        We reshape int16 → (n_chirps, n_samples, n_rx, 2) where the
        last dim is [I, Q], then collapse to complex64. This matches
        ``radar_dca.bin_parser._bytes_to_complex_chirp`` so the live
        path and the offline ``parse_bin_full`` path produce
        identical arrays from the same bytes.
        """
        d = self._dims
        # int16 view of the whole frame.
        raw = np.frombuffer(frame_bytes, dtype=np.int16)
        expected = d.n_chirps * d.n_samples * d.n_rx * 2
        if raw.size != expected:
            raise ValueError(
                f"frame int16 count {raw.size} != expected {expected} "
                f"({d.n_chirps}c × {d.n_samples}s × {d.n_rx}rx × 2)"
            )
        raw = raw.reshape(d.n_chirps, d.n_samples, d.n_rx, 2)
        return (raw[..., 0].astype(np.float32)
                + 1j * raw[..., 1].astype(np.float32)).astype(np.complex64)

    def _range_doppler(self, cube: np.ndarray) -> np.ndarray:
        """Run the range FFT (fast-time) on each chirp. Returns
        ``(n_chirps, n_range, n_rx)`` complex64. Hann window applied
        to suppress range sidelobes — same window as bin_parser.range_fft
        so live and offline give identical output."""
        n_chirps, n_samples, n_rx = cube.shape
        win = np.hanning(n_samples).astype(np.float32)
        windowed = cube * win[np.newaxis, :, np.newaxis]
        return np.fft.fft(windowed, axis=1).astype(np.complex64)

    def _integrate_rx(self, range_cube: np.ndarray) -> np.ndarray:
        """Coherent sum across the RX axis → ``(n_chirps, n_range)``.

        This is the cheapest "beamformer" — broadside look only, no
        steering. Fine for PMM (we want range×slow-time at boresight)
        but A/G eventually wants Capon for off-axis sources."""
        return range_cube.sum(axis=2)

    def _process_frame(self, frame_bytes: bytes) -> None:
        """Reshape one frame's bytes → range cube → slow-time grid →
        run mode-specific processing → emit RadarFrame on Topic.RADAR_AA.

        Active even when ``_publish_enabled`` is False (i.e. modes
        stock + ag) so the byte buffer always drains and we don't
        bleed memory. The publish step at the end is gated by mode."""
        d = self._dims
        cube = self._bytes_to_cube(frame_bytes)              # (C, S, R)
        range_cube = self._range_doppler(cube)               # (C, R, RX)
        # (n_chirps, n_range)
        chirp_x_range = self._integrate_rx(range_cube)
        # PMM detector wants (n_range, n_chirps).
        slow_time_grid = chirp_x_range.T  # (R, C)

        # ── PMM scan (A/A path) ───────────────────────────────────
        targets: List[RadarTarget] = []
        try:
            hits: List[Tuple[int, PMMResult]] = scan_range_bins(
                slow_time_grid,
                prf_hz=d.prf_hz,
                band_low_hz=self._pmm_band_low,
                band_high_hz=self._pmm_band_high,
                threshold_db=self._pmm_threshold,
            )
        except Exception:
            log.exception("scan_range_bins raised; treating as no hits")
            hits = []

        for range_bin, result in hits:
            range_m = float(range_bin) * d.range_resolution_m
            if range_m > self.max_range_m:
                continue   # outside our advertised range
            self._stats.drone_detections += 1
            self._stats.last_drone_range_m = range_m
            self._stats.last_drone_blade_freq_hz = result.blade_freq_hz
            # No AoA in the PMM path — emit at boresight.
            # x=right, y=forward, z=up (radar local frame).
            targets.append(RadarTarget(
                tid=range_bin,
                pos_x_m=0.0,
                pos_y_m=float(range_m),
                pos_z_m=0.0,
                vel_x_mps=0.0, vel_y_mps=0.0, vel_z_mps=0.0,
                size_x_m=0.5, size_y_m=0.5, size_z_m=0.5,
                confidence=float(result.confidence),
                source="pmm",
                num_points=1,
                coasting=False,
                hits=1, misses=0,
            ))

        # ── A/G long-range processor (range-Doppler + CFAR + AoA) ──
        # Runs every frame but only emits when relevant; the AoA is
        # done with a simple 4-RX FFT (delay-and-sum). Capon BF
        # requires the DDM-MIMO virtual-array geometry which mmw_demoDDM
        # encodes in TX cycling — until we decode that, the 4-RX path
        # gives ~30° angular resolution, which is enough for cueing
        # but not for fine tracking.
        ag_detections: List[RadarDetection] = self._ag_process(range_cube)

        self._frame_id += 1
        self._stats.frames_assembled += 1

        if self._publish_enabled:
            BUS.publish(Topic.RADAR_AA, RadarFrame(
                timestamp=time.time(),
                frame_id=self._frame_id,
                connected=True,
                profile=self.profile_name,
                max_range_m=self.max_range_m,
                fov_half_deg=self.az_half_deg,
                detections=ag_detections,
                targets=targets,
            ))

    # ─────────────────────── A/G long-range processor ──────────────────
    def _ag_process(self, range_cube: np.ndarray) -> List[RadarDetection]:
        """Coherent integration → range-Doppler map → CFAR → AoA → list.

        Steps:
          1. Coherent integrate ``self._ag_integrate_chirps`` chirps
             across slow-time. With N=512 vs the stock N=128, +6 dB
             SNR gain (10·log10(N)/sqrt(N)).
          2. Doppler FFT across the integrated chirps → range-Doppler
             magnitude map.
          3. Threshold via the selected CFAR algo (CA / OS / GO).
          4. For each detection, estimate AoA via 4-element FFT
             across the RX axis (delay-and-sum). Capon BF is a
             stretch goal once virtual-array calibration lands.
          5. Convert (range, az) to (x, y, z) in the radar frame.
        """
        d = self._dims
        n_chirps, n_range, n_rx = range_cube.shape

        # 1. Coherent integration — average groups of N chirps.
        N = max(8, min(int(self._ag_integrate_chirps), n_chirps))
        n_groups = n_chirps // N
        if n_groups < 2:
            # Not enough chirps for a meaningful Doppler FFT.
            return []
        # Trim to a multiple of N then reshape.
        trimmed = range_cube[: n_groups * N]
        # (n_groups, N, n_range, n_rx) → integrate within each group:
        integrated = trimmed.reshape(n_groups, N, n_range, n_rx).mean(axis=1)
        # 2. Doppler FFT across n_groups.
        win = np.hanning(n_groups).astype(np.float32)
        rd = np.fft.fftshift(
            np.fft.fft(integrated * win[:, None, None], axis=0),
            axes=0,
        )  # (n_doppler, n_range, n_rx)
        # Sum-over-RX magnitude map (cheap detection statistic).
        rd_mag = np.abs(rd.sum(axis=2))                       # (n_dop, n_range)

        # 3. CFAR threshold. Implementations are tight numpy loops;
        # ~10 ms per frame at our shapes.
        det_cells = self._cfar_2d(
            rd_mag,
            algo=str(self._ag_cfar_algo).lower(),
            threshold_db=float(self._ag_cfar_threshold_db),
        )
        if not det_cells:
            return []

        # 4. AoA per detection — 4-element zero-padded FFT across RX.
        # n_az=64 gives ~3° bin spacing; sin(angle) maps to FFT bin.
        n_az = 64
        # det_cells is list of (doppler_idx, range_idx)
        out: List[RadarDetection] = []
        for dop_idx, range_idx in det_cells:
            range_m = float(range_idx) * d.range_resolution_m
            if range_m > self.max_range_m:
                continue
            rx_slice = rd[dop_idx, range_idx, :]              # (n_rx,)
            az_spec = np.fft.fftshift(np.fft.fft(rx_slice, n=n_az))
            az_bin = int(np.argmax(np.abs(az_spec)))
            # Map FFT bin → angle: sin(theta) = (bin - n_az/2) / (n_az/2)
            sin_theta = (az_bin - n_az / 2) / (n_az / 2)
            sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
            az_deg = float(np.degrees(np.arcsin(sin_theta)))
            if abs(az_deg) > self.az_half_deg:
                continue
            # Doppler bin → radial velocity.
            # Doppler resolution = PRF / (N * n_groups). Center bin =
            # zero Doppler. λ at 77 GHz = 0.0039 m.
            lam_m = 3e8 / 77e9
            fd = (dop_idx - n_groups / 2) * (d.prf_hz / N / n_groups)
            vel_mps = -lam_m / 2.0 * fd  # negative because radial-towards-radar
            # Power → SNR estimate (relative to median noise floor).
            rd_db = 20.0 * np.log10(np.abs(rd_mag[dop_idx, range_idx]) + 1e-9)
            noise_db = 20.0 * np.log10(np.median(rd_mag) + 1e-9)
            snr_db = float(rd_db - noise_db)
            # Build a RadarDetection in sensor frame (+x right, +y
            # fwd, +z up). Range × azimuth → (x, y, z=0) since the
            # AWR2944P antenna is 1-D in azimuth (no elevation).
            x_m = range_m * float(np.sin(np.radians(az_deg)))
            y_m = range_m * float(np.cos(np.radians(az_deg)))
            out.append(RadarDetection(
                x_m=x_m, y_m=y_m, z_m=0.0,
                doppler_mps=float(vel_mps),
                snr_db=snr_db,
                noise_db=float(noise_db),
                range_m=range_m,
                az_deg=az_deg,
                el_deg=0.0,
                target_id=255,
            ))
        return out

    def _cfar_2d(
        self, mag: np.ndarray, *, algo: str, threshold_db: float,
    ) -> List[Tuple[int, int]]:
        """1-D CFAR along the range axis at every Doppler bin.

        We don't run a full 2-D CFAR (cost scales as guard²) — a
        per-Doppler 1-D scan catches what we care about (drones at
        non-zero Doppler, vehicles at radial velocity), and 2-D CFAR's
        marginal benefit isn't worth the CPU at our frame rate.

        Reference window: 16 cells either side of the test cell, with
        4 guard cells skipped on each side. Returns ``[(dop_idx,
        range_idx), …]`` sorted by descending magnitude.
        """
        n_dop, n_range = mag.shape
        N_REF = 16   # reference cells per side
        N_GUARD = 4  # guard cells per side
        thresh_lin = 10.0 ** (threshold_db / 20.0)
        out: List[Tuple[int, int, float]] = []

        # Reusable index slices for ref-window arithmetic.
        for d_idx in range(n_dop):
            row = mag[d_idx]
            for r in range(N_REF + N_GUARD, n_range - N_REF - N_GUARD):
                left = row[r - N_REF - N_GUARD : r - N_GUARD]
                right = row[r + N_GUARD + 1 : r + N_GUARD + 1 + N_REF]
                if algo == "ca":
                    noise_est = 0.5 * (left.mean() + right.mean())
                elif algo == "go":
                    noise_est = max(left.mean(), right.mean())
                elif algo == "os":
                    # Ordered-statistic: 75th percentile of combined window.
                    window = np.concatenate([left, right])
                    noise_est = float(np.partition(window, int(0.75 * len(window)))[int(0.75 * len(window))])
                else:
                    noise_est = 0.5 * (left.mean() + right.mean())  # fall back to CA
                if noise_est <= 0:
                    continue
                if row[r] > thresh_lin * noise_est:
                    out.append((d_idx, r, float(row[r])))

        # Cap at top-256 detections per frame so a noisy threshold
        # can't flood the bus.
        out.sort(key=lambda t: t[2], reverse=True)
        return [(d, r) for d, r, _ in out[:256]]

    def _publish_sentinel(self) -> None:
        """Emit a connected=False RadarFrame when no fresh ADC bytes
        are available. Only publishes when in A/A mode."""
        self._frame_id += 1
        if self._publish_enabled:
            BUS.publish(Topic.RADAR_AA, RadarFrame(
                timestamp=time.time(),
                frame_id=self._frame_id,
                connected=False,
                profile=self.profile_name,
                max_range_m=self.max_range_m,
                fov_half_deg=self.az_half_deg,
            ))

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
    n_chirps: int = 2304,    # 6 × 384 from awr2944P_aa.cfg (chirpCfg 0 5 + numLoops 384)
    n_rx: int = 4,
    n_samples: int = 384,
    chirp_period_s: float = 100e-6,
    range_resolution_m: float = 0.04,
) -> FrameDims:
    """Build FrameDims from explicit numbers.

    Once we wire a cfg parser, this becomes ``dims_from_cfg_file(path)``
    that reads the .cfg and computes everything. For now, defaults
    reflect ``radar/cfg/awr2944P_aa.cfg``.
    """
    return FrameDims(
        n_chirps=n_chirps,
        n_rx=n_rx,
        n_samples=n_samples,
        chirp_period_s=chirp_period_s,
        range_resolution_m=range_resolution_m,
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
        """Heartbeat. Drives the assemble→FFT→PMM→emit pipeline.

        For the scaffold we tick at the expected frame rate (PRF / n_chirps)
        and check whether the listener has accumulated enough bytes for
        a frame. If yes, process; if no, publish a sentinel RadarFrame
        with connected=False so the GUI shows DCA active but no data.
        """
        # Frame period in seconds = chirp_period × n_chirps.
        frame_period_s = self._dims.chirp_period_s * self._dims.n_chirps
        # Tick at half the frame rate to keep latency low.
        tick_s = max(0.05, frame_period_s / 2.0)

        last_byte_count = 0
        while not self._stop.is_set():
            # Pull listener stats — it tracks cumulative bytes.
            ds = self._listener.stats()
            new_bytes = max(0, ds.bytes_total - last_byte_count)
            last_byte_count = ds.bytes_total

            if new_bytes >= self._dims.bytes_per_frame:
                # Enough data for one or more frames. Process the most
                # recent (drop older frames if we fell behind — radar
                # latency matters more than completeness for a seeker).
                # For scaffold we log + emit a placeholder. M4.1 reads
                # from the listener's internal byte queue here.
                self._process_frame_stub(have_real_data=False)
            else:
                # Not enough new data; emit sentinel so GUI knows DCA
                # is alive but no raw samples reached us yet.
                self._publish_sentinel()

            self._stop.wait(tick_s)

    def _process_frame_stub(self, *, have_real_data: bool) -> None:
        """Placeholder for M4.2-M4.5. When real ADC bytes arrive,
        replace the zero-fill with the real reassembly + range FFT.

        Right now we:
          - synthesize a zero-filled (n_chirps, n_range) slow-time
            grid → scan_range_bins finds nothing → no DRONE detections.
          - emit a heartbeat RadarFrame with connected=True so the
            GUI knows the pipeline is *running* (vs. the bare DCAManager
            sentinel which is connected=False).
        """
        n_range = self._dims.n_samples  # range FFT len = n_samples (no zero-pad)
        slow_time_grid = np.zeros((n_range, self._dims.n_chirps),
                                  dtype=np.complex128)

        # M4.4 — PMM scan. Wired and ready; will return real hits once
        # the slow_time_grid has actual samples in it.
        hits: List[Tuple[int, PMMResult]] = scan_range_bins(
            slow_time_grid,
            prf_hz=self._dims.prf_hz,
            band_low_hz=self._pmm_band_low,
            band_high_hz=self._pmm_band_high,
            threshold_db=self._pmm_threshold,
        )

        # M4.5 — emit RadarTargets for hits. With zero input, hits is [].
        targets: List[RadarTarget] = []
        for range_bin, result in hits:
            range_m = range_bin * self._dims.range_resolution_m
            self._stats.drone_detections += 1
            self._stats.last_drone_range_m = range_m
            self._stats.last_drone_blade_freq_hz = result.blade_freq_hz
            # Boresight default: target straight ahead until M4.6 (Capon BF)
            # gives us real az/el. radar_dca emits in radar-frame coords:
            # x=right, y=forward, z=up.
            targets.append(RadarTarget(
                tid=range_bin,                    # range bin as a stable id
                pos_x_m=0.0,
                pos_y_m=float(range_m),
                pos_z_m=0.0,
                vel_x_mps=0.0,
                vel_y_mps=0.0,
                vel_z_mps=0.0,
                size_x_m=0.5, size_y_m=0.5, size_z_m=0.5,
                confidence=float(result.confidence),
                source="pmm",
                num_points=1,
                coasting=False,
                hits=1, misses=0,
            ))

        self._frame_id += 1
        self._stats.frames_assembled += 1
        # Only publish to Topic.RADAR_AA when we're actually in A/A mode
        # (composite flips _publish_enabled on set_mode). Suppresses bus
        # spam in stock + ag modes; raw-ADC bytes are still consumed.
        if self._publish_enabled:
            BUS.publish(Topic.RADAR_AA, RadarFrame(
                timestamp=time.time(),
                frame_id=self._frame_id,
                connected=True,
                profile=self.profile_name,
                max_range_m=self.max_range_m,
                fov_half_deg=self.az_half_deg,
                detections=[],
                targets=targets,
            ))

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

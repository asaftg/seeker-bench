"""DCA1000 raw-ADC → RadarFrame pipeline (clean rewrite, 2026-05-05).

Replaces ~1080 lines of layered patches with a single 5-stage pipeline:

    UDP payloads (DataPortListener)
        │
        ▼  Stage 1: reshape + range FFT (rfft → analytic, RX-major wire)
    range_cube (n_chirps, n_range, n_rx) complex64
        │
        ▼  Stage 2: slow-time MTI (mean-subtract across chirps)
              Kills walls, ground bounce, AND the AWR2944P chopper
              artifact at sample_rate/4 in one numpy op.
        │
        ▼  Stage 3: range-Doppler map (coherent integration + Doppler FFT)
    rd cube (n_doppler, n_range, n_rx)
        │
        ▼  Stage 4: vectorised CFAR + 4-RX zero-padded FFT for AoA
    cfar_detections : List[RadarDetection]
        │
        ▼  Stage 5: PMM scan on slow-time grid
    pmm_targets    + tracker.step(cfar_detections) → tracks
        │
        ▼  Mode-aware publish to Topic.RADAR_AA
            stock → tracks only
            ag    → tracks + cfar_detections
            aa    → tracks + cfar_detections + pmm_targets

Targets: 16+ fps sustained, no UDP drops, ~530 LOC. AoA via 5·(λ/2)
RX spacing → ±11.5° unambiguous; aliased outside but acceptable for cueing.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.fft as scipy_fft

from common.frame_bus import BUS
from common.frames import (
    RadarDetection,
    RadarFrame,
    RadarTarget,
    TargetClass,
    Topic,
)
from common.logging_setup import get_logger
from radar.clustering import ClusterParams, RadarClusterer
from radar_dca.data_port import DataPortListener
from radar_dca.pmm_detector import PMMResult, scan_range_bins

log = get_logger(__name__)


# ─────────────────────── frame dimensions (preserved from old file) ─────────


@dataclass
class FrameDims:
    """Shape of a raw-ADC frame as configured by the AWR cfg."""
    n_chirps: int
    n_rx: int
    n_samples: int
    bytes_per_sample: int = 2
    range_resolution_m: float = 0.04
    chirp_period_s: float = 100e-6

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
    chirp_period_s: float = 27.81e-6,
    range_resolution_m: float = 1.32,
) -> FrameDims:
    """Build FrameDims from explicit numbers (test entry point)."""
    return FrameDims(
        n_chirps=n_chirps,
        n_rx=n_rx,
        n_samples=n_samples,
        chirp_period_s=chirp_period_s,
        range_resolution_m=range_resolution_m,
    )


def dims_from_cfg_file(cfg_path: str) -> FrameDims:
    """Parse the AWR mmw_demo .cfg and compute FrameDims.

    Reads channelCfg / profileCfg / frameCfg; range resolution is computed
    from sampled bandwidth (slope × adc_capture_time), not swept BW.
    """
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
                    idle_us = float(tok[3])
                    ramp_us = float(tok[5])
                    slope_mhz_us = float(tok[8])
                    n_samples = int(tok[10])
                    digout_ksps = float(tok[11])
                    chirp_period_s = (idle_us + ramp_us) * 1e-6
                    adc_capture_us = n_samples / digout_ksps * 1e3
                    bw_hz = slope_mhz_us * 1e6 * adc_capture_us
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
            f"cfg {cfg_path} missing required keys: {missing}"
        )

    return FrameDims(
        n_chirps=int(chirp_indices) * int(n_loops),
        n_rx=int(n_rx),                              # type: ignore[arg-type]
        n_samples=int(n_samples),                    # type: ignore[arg-type]
        chirp_period_s=float(chirp_period_s),        # type: ignore[arg-type]
        range_resolution_m=float(range_res_m),       # type: ignore[arg-type]
    )


# ─────────────────────── pipeline params + presets ──────────────────────────


@dataclass
class PipelineParams:
    """All knobs in one place — set via mode preset or set_params()."""
    # AG / Range-Doppler
    integrate_chirps: int = 16
    cfar_algo: str = "ca"        # "ca" | "go" | "os" (os falls back to ca)
    cfar_threshold_db: float = 12.0
    # AoA + display
    az_half_deg: float = 60.0
    # Cap CFAR cells passed to AoA. The quarter-Nyquist chip artifact
    # can saturate hundreds of bins with identical magnitude; capping
    # at 64 keeps real-target peaks while starving the artifact pattern
    # of room to seed a tracker lock-on.
    aoa_max_detections: int = 64
    # Per-mode filtering — applied to detections + tracks before publish
    snr_min_db: float = 10.0
    range_min_m: float = 0.5
    range_max_m: float = 250.0
    speed_min_mps: float = 0.0
    # Cluster/track
    cluster_eps_pos_m: float = 8.0
    cluster_min_samples: int = 2
    confirm_min_hits: int = 2
    coast_max_frames: int = 30
    # PMM
    pmm_band_low_hz: float = 50.0
    pmm_band_high_hz: float = 500.0
    pmm_threshold_db: float = 24.0   # raised from 18; chopper artifacts fired at 18-22 dB


STOCK_PRESET: Dict[str, Any] = dict(
    # Tuned 2026-05-05: stock must mirror chip-TLV semantics —
    # silent scene = empty canvas, walker = 1-2 stable Kalman tracks.
    # Full radar horizon (1-250 m) — seeker mission is long-range
    # detection, range capping is not an option. CFAR=18dB + SNR>=22dB
    # starves the chip's quarter-Nyquist harmonic; the wider ±5 notch
    # in _notch_harmonic_artifact kills the artifact at source so the
    # surviving 60% of range bins spans 1-250 m.
    snr_min_db=22, range_min_m=1.0, speed_min_mps=0.0,
    cfar_threshold_db=18, cluster_min_samples=4,
    confirm_min_hits=3, aoa_max_detections=64,
)
AG_PRESET: Dict[str, Any] = dict(
    # AG mode: tracked centroids + raw CFAR cells. Slightly looser
    # than stock so operator sees clutter cells too.
    snr_min_db=18, range_min_m=0.5, speed_min_mps=0.0,
    cfar_threshold_db=18, cluster_min_samples=4,
    confirm_min_hits=3, aoa_max_detections=128,
)
AA_PRESET: Dict[str, Any] = dict(
    # AA mode: PMM is primary detector for drones; CFAR + tracks are
    # shown as backdrop. PMM threshold 28dB to reject chopper sidebands.
    snr_min_db=18, range_min_m=0.5, speed_min_mps=0.0,
    cfar_threshold_db=18, pmm_threshold_db=28,
    cluster_min_samples=4, confirm_min_hits=3,
    aoa_max_detections=128,
)
_PRESETS: Dict[str, Dict[str, Any]] = {
    "stock": STOCK_PRESET,
    "ag":    AG_PRESET,
    "aa":    AA_PRESET,
}


# ─────────────────────── stats ──────────────────────────────────────────────


@dataclass
class PipelineStats:
    """Counters surfaced to the GUI / diagnostics endpoint."""
    frames_assembled: int = 0
    frames_dropped: int = 0
    last_frame_t: float = 0.0
    detections_per_frame: float = 0.0
    targets_per_frame: float = 0.0


# ─────────────────────── pipeline ───────────────────────────────────────────


class DCAPipeline:
    """Worker thread: raw ADC bytes → RadarFrame on Topic.RADAR_AA.

    One instance per radar. Owns a RadarClusterer (DBSCAN + Kalman) for
    track stability so that "stock" mode produces zero output when the
    scene is silent and stable boxes when something is moving — same
    semantics as the on-chip TLV path, replicated host-side.
    """

    def __init__(
        self,
        *,
        listener: DataPortListener,
        dims: FrameDims,
        profile_name: str = "awr2944p_aa",
        max_range_m: float = 250.0,
        az_half_deg: float = 60.0,
        pmm_only: bool = False,
    ) -> None:
        self._listener = listener
        self._dims = dims
        self.profile_name = profile_name
        self.max_range_m = float(max_range_m)
        self.az_half_deg = float(az_half_deg)

        # In hybrid mode (chip TLV provides humans/vehicles via
        # RadarManager + Topic.RADAR), this pipeline only does PMM
        # for drones — Stage 3 (Doppler FFT) and Stage 4 (CFAR/AoA)
        # are skipped because the on-chip CFAR already did that work
        # better via DDMA. The host-side AoA from raw ADC would be
        # mis-angled anyway because the cfg enables DDMA on chip
        # (ddmPhaseShiftAntOrder) and this pipeline does not unfold
        # the per-TX phase modulation.
        self._pmm_only = bool(pmm_only)

        self.params = PipelineParams(
            az_half_deg=float(az_half_deg),
            range_max_m=float(max_range_m),
        )
        self._mode_name = "stock"
        self.set_mode("stock")

        # Tracker — one instance for the lifetime of the pipeline.
        self._tracker = RadarClusterer(self._cluster_params())

        # LVDS warn-once (auto-kick is broken without firmware fix; only
        # log the stall, don't try to recover).
        self._lvds_stalled_warned = False

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_id = 0
        self._buf = bytearray()
        self._stats = PipelineStats()

        # Pre-compute fast-time Hann window (constant across frames).
        self._hann_fast = np.hanning(dims.n_samples).astype(np.float32)

        # Range-gated zero-Doppler suppression. Static returns
        # (Doppler within ±halfwidth of centre) are dropped at ranges
        # beyond _cell_static_max_range_m — they are RF leakage / wall
        # bounces / sky returns by construction. Close-range static
        # cells survive (parked car / standing person at 1-10 m).
        # halfwidth=4 covers FFT spread of the chip artifact; a walker
        # at 0.5 m/s sits ~10 bins off zero and is unaffected.
        self._cell_zero_dop_halfwidth: int = 4
        self._cell_static_max_range_m: float = 12.0

    # ─────────────────────── public API ────────────────────────────────

    def start(self) -> None:
        """Spawn the worker thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="DCAPipeline", daemon=True,
        )
        self._thread.start()
        log.info(
            "DCAPipeline started (%d c × %d s × %d rx = %.1f MB/frame, "
            "PRF %.0f Hz, mode=%s)",
            self._dims.n_chirps, self._dims.n_samples, self._dims.n_rx,
            self._dims.bytes_per_frame / 1e6, self._dims.prf_hz, self._mode_name,
        )

    def stop(self) -> None:
        """Signal the worker thread to exit and join."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        log.info(
            "DCAPipeline stopped (frames=%d, drops=%d)",
            self._stats.frames_assembled, self._stats.frames_dropped,
        )

    def stats(self) -> PipelineStats:
        """Snapshot of pipeline counters."""
        return PipelineStats(**self._stats.__dict__)

    def diagnostics(self) -> dict:
        """Diagnostics dict for /api/radar/aa_diagnostics."""
        ds = self._listener.stats()
        return {
            "mode": self._mode_name,
            "pmm_only": self._pmm_only,
            "profile": self.profile_name,
            "frames_assembled": self._stats.frames_assembled,
            "frames_dropped": self._stats.frames_dropped,
            "last_frame_age_s": (
                time.time() - self._stats.last_frame_t
                if self._stats.last_frame_t > 0 else None
            ),
            "detections_per_frame": self._stats.detections_per_frame,
            "targets_per_frame": self._stats.targets_per_frame,
            "udp": {
                "listening": ds.listening,
                "packets_total": ds.packets_total,
                "bytes_per_s": ds.bytes_per_s,
                "seq_drops_total": ds.seq_drops_total,
                "queue_drops": self._listener.queue_drops(),
                "queue_depth": self._listener.queue_depth(),
                "last_packet_age_s": ds.last_packet_age_s,
            },
            "params": self.params.__dict__,
            "dims": {
                "n_chirps": self._dims.n_chirps,
                "n_samples": self._dims.n_samples,
                "n_rx": self._dims.n_rx,
                "range_res_m": self._dims.range_resolution_m,
                "prf_hz": self._dims.prf_hz,
            },
        }

    def set_mode(self, name: str) -> None:
        """Switch operating mode and apply its preset overrides.

        Modes:
            "stock" — empty detections, only Kalman-tracked centroids
            "ag"    — tracked centroids + raw CFAR detections
            "aa"    — tracked centroids + PMM "drone" hits + CFAR
        """
        key = name.lower().strip()
        if key not in _PRESETS:
            raise ValueError(f"unknown mode {name!r}; expected one of {list(_PRESETS)}")
        self._mode_name = key
        for k, v in _PRESETS[key].items():
            setattr(self.params, k, v)
        # Tracker config follows the active params.
        if hasattr(self, "_tracker"):
            self._tracker.params = self._cluster_params()
        log.info("DCAPipeline mode -> %s", key)

    def set_params(self, **kwargs: Any) -> None:
        """Override individual PipelineParams fields by keyword."""
        for k, v in kwargs.items():
            if not hasattr(self.params, k):
                # Soft-ignore unknown fields so legacy callers don't crash.
                log.warning("set_params: ignoring unknown field %r", k)
                continue
            setattr(self.params, k, v)
        if hasattr(self, "_tracker"):
            self._tracker.params = self._cluster_params()

    def set_pmm_only(self, v: bool) -> None:
        """Toggle PMM-only mode at runtime.

        When True, the pipeline skips Stage 3 (range-Doppler), Stage 4
        (CFAR + AoA), and the tracker step. Only Stage 5 (PMM scan on
        the MTI'd range cube) runs and is published. Used in hybrid
        deployments where the chip's TLV is the source of truth for
        humans/vehicles and DCAPipeline only needs to find drones.
        """
        self._pmm_only = bool(v)
        log.info("DCAPipeline pmm_only -> %s", self._pmm_only)

    # ─────────────────────── worker thread ─────────────────────────────

    def _loop(self) -> None:
        """Main loop: drain UDP → buffer → process whole frames."""
        bpf = self._dims.bytes_per_frame
        TICK_S = 0.020

        while not self._stop.is_set():
            for p in self._listener.drain_payloads():
                self._buf.extend(p)

            self._lvds_warn_if_stalled()

            while len(self._buf) >= bpf:
                frame_bytes = bytes(self._buf[:bpf])
                del self._buf[:bpf]
                try:
                    self._process_frame(frame_bytes)
                except Exception as e:
                    log.exception("frame processing failed: %s", e)
                    self._stats.frames_dropped += 1

            self._stop.wait(TICK_S)

    def _lvds_warn_if_stalled(self) -> None:
        """Log once when the UDP stream goes silent after starting."""
        try:
            ds = self._listener.stats()
        except Exception:
            return
        age = ds.last_packet_age_s
        if ds.packets_total > 0 and age != float("inf") and age > 5.0:
            if not self._lvds_stalled_warned:
                log.warning(
                    "LVDS stalled: no UDP for %.1fs (packets=%d drops=%d). "
                    "Auto-kick is disabled — power-cycle the AWR to recover.",
                    age, ds.packets_total, ds.seq_drops_total,
                )
                self._lvds_stalled_warned = True
        elif age < 0.5:
            self._lvds_stalled_warned = False

    # ─────────────────────── per-frame processing ──────────────────────

    def _process_frame(self, frame_bytes: bytes) -> None:
        """Run the 5-stage pipeline on one frame's worth of bytes.

        In pmm_only mode (hybrid deployment with chip TLV active) only
        Stages 1, 2, 5 run -- the chip's on-chip CFAR is the source of
        truth for humans/vehicles via Topic.RADAR, and host-side CFAR
        without DDMA-unfold would produce mis-angled detections anyway.
        """
        # Stage 1: reshape + range FFT.
        range_cube = self._stage1_range_fft(frame_bytes)
        # Stage 2: slow-time MTI + harmonic-artifact notch.
        range_cube -= range_cube.mean(axis=0, keepdims=True)
        self._notch_harmonic_artifact(range_cube)

        if self._pmm_only:
            # Skip Stage 3 (range-Doppler), Stage 4 (CFAR+AoA), tracker.
            # Only PMM runs on the MTI'd range cube.
            cfar_dets: List[RadarDetection] = []
            tracks: List[RadarTarget] = []
            pmm_targets = self._stage5_pmm(range_cube)
        else:
            # Full pipeline (legacy / standalone-AA mode).
            rd, rd_mag = self._stage3_range_doppler(range_cube)
            cfar_dets = self._stage4_cfar_aoa(rd, rd_mag)
            pmm_targets = self._stage5_pmm(range_cube)
            try:
                _, tracks = self._tracker.step(cfar_dets)
            except Exception:
                log.exception("tracker.step raised; treating as no tracks")
                tracks = []
        self._publish(cfar_dets, list(tracks), pmm_targets)

        self._frame_id += 1
        self._stats.frames_assembled += 1
        self._stats.last_frame_t = time.time()
        # EMA over per-frame counts (alpha=0.2 → ~5-frame memory).
        a = 0.2
        self._stats.detections_per_frame = (
            (1 - a) * self._stats.detections_per_frame + a * len(cfar_dets)
        )
        self._stats.targets_per_frame = (
            (1 - a) * self._stats.targets_per_frame + a * len(tracks)
        )
        if self._frame_id % 80 == 0:
            log.info(
                "DCAPipeline frame %d [%s]: cfar=%d tracks=%d pmm=%d",
                self._frame_id, self._mode_name,
                len(cfar_dets), len(tracks), len(pmm_targets),
            )

    # ─────────────────────── stages ────────────────────────────────────

    def _stage1_range_fft(self, frame_bytes: bytes) -> np.ndarray:
        """Bytes → range cube (n_chirps, n_range, n_rx) complex64.

        Wire layout is RX-major per chirp (non-interleaved):
        [RX0 s0..sN-1][RX1 s0..sN-1][RX2 s0..sN-1][RX3 s0..sN-1].
        Reshape (chirps, RX, samples), transpose → (chirps, samples, RX).
        Per-chirp-per-RX DC subtract, Hann window, rfft along samples.
        Scale POSITIVE-FREQUENCY bins by 2 (analytic-signal equivalence);
        DC and Nyquist stay as-is.
        """
        d = self._dims
        raw = np.frombuffer(frame_bytes, dtype=np.int16)
        expected = d.n_chirps * d.n_samples * d.n_rx
        if raw.size != expected:
            raise ValueError(
                f"frame int16 count {raw.size} != expected {expected}"
            )
        real_cube = (
            raw.reshape(d.n_chirps, d.n_rx, d.n_samples)
               .transpose(0, 2, 1)
               .astype(np.float32)
        )
        real_cube -= real_cube.mean(axis=1, keepdims=True)
        windowed = real_cube * self._hann_fast[np.newaxis, :, np.newaxis]
        rfft_out = scipy_fft.rfft(windowed, axis=1, workers=2)
        rfft_out[:, 1:-1, :] *= 2.0
        return rfft_out.astype(np.complex64)

    def _notch_harmonic_artifact(self, range_cube: np.ndarray) -> None:
        """Zero AWR2944P LO/ADC leakage harmonics in-place.

        The chip aliases internal mixer harmonics into the range axis
        every n_samples/16 bins (24 bins for n_samples=384). The exact
        peak position drifts ±3-4 bins between captures — the chip's
        sample clock isn't perfectly stable so the nominal grid is
        approximate. Empirical positions seen so far:

            08:09 capture: 24,  72, 120, 168     (on the grid)
            08:41 capture: 75, 116, 171, ...     (offset by +3, -4, +3)

        Notch radius ±5 catches both ±FFT-leakage and clock-drift
        wobble. ~38 % of range coverage is sacrificed to 7 dead-zone
        windows of ~13 m each, but the surviving bins span the full
        1-250 m radar horizon. The seeker mission is long-range drone
        detection — capping range to "fix" this would defeat the
        purpose. Killing the artifact at SOURCE here lets the rest of
        the pipeline do real work at 100-250 m.

        Operates on the complex range cube (n_chirps, n_range, n_rx).
        """
        n_range = range_cube.shape[1]
        period = self._dims.n_samples // 8
        if period <= 0:
            return
        step = max(period // 2, 1)
        radius = 5
        for b in range(step, n_range, step):
            lo = max(b - radius, 0)
            hi = min(b + radius + 1, n_range)
            range_cube[:, lo:hi, :] = 0

    def _stage3_range_doppler(
        self, range_cube: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Coherent integrate + Doppler FFT → (rd, rd_mag).

        rd has shape (n_doppler, n_range, n_rx) and is kept in memory
        for stage-4 AoA. rd_mag = |rd.sum(axis=2)| with shape
        (n_doppler, n_range) is the CFAR detection statistic.
        """
        n_chirps, n_range, n_rx = range_cube.shape
        N = max(8, min(int(self.params.integrate_chirps), n_chirps))
        n_groups = n_chirps // N
        if n_groups < 2:
            return (
                np.zeros((1, n_range, n_rx), dtype=np.complex64),
                np.zeros((1, n_range), dtype=np.float32),
            )
        trimmed = range_cube[: n_groups * N]
        integrated = trimmed.reshape(n_groups, N, n_range, n_rx).mean(axis=1)
        win = np.hanning(n_groups).astype(np.float32)
        rd = np.fft.fftshift(
            scipy_fft.fft(
                integrated * win[:, None, None], axis=0, workers=2,
            ),
            axes=0,
        )
        rd_mag = np.abs(rd.sum(axis=2)).astype(np.float32)
        return rd, rd_mag

    def _stage4_cfar_aoa(
        self, rd: np.ndarray, rd_mag: np.ndarray,
    ) -> List[RadarDetection]:
        """CFAR over rd_mag, then 4-RX AoA per surviving cell.

        Cap to top params.aoa_max_detections by magnitude. AoA is a
        64-pt zero-padded FFT across RX → sin θ = (bin - 32)/32. Range
        gates and SNR gate are applied after AoA so the operator's
        per-mode min-SNR / min-range knobs take effect here.
        """
        d = self._dims
        n_doppler, n_range = rd_mag.shape
        det_cells = self._cfar_2d(
            rd_mag,
            algo=str(self.params.cfar_algo).lower(),
            threshold_db=float(self.params.cfar_threshold_db),
        )
        if not det_cells:
            return []
        head = det_cells[: int(self.params.aoa_max_detections)]
        n_az = 64

        # Hoist constants out of the per-detection loop. Use the 25th
        # percentile of rd_mag (robust to strong signals/artifacts that
        # would otherwise inflate the median floor and compress SNR).
        noise_floor = float(np.percentile(rd_mag, 25) + 1e-9)
        noise_db_const = 20.0 * np.log10(noise_floor)
        lam_m = 3e8 / 77e9
        N = max(8, min(int(self.params.integrate_chirps), d.n_chirps))
        n_groups = d.n_chirps // N if d.n_chirps >= 16 else 1
        fd_scale = d.prf_hz / N / max(n_groups, 1)

        rng_min = float(self.params.range_min_m)
        rng_max = min(float(self.params.range_max_m), self.max_range_m)
        snr_min = float(self.params.snr_min_db)
        speed_min = float(self.params.speed_min_mps)
        az_lim = float(self.params.az_half_deg)

        out: List[RadarDetection] = []
        for dop_idx, range_idx in head:
            range_m = float(range_idx) * d.range_resolution_m
            if range_m < rng_min or range_m > rng_max:
                continue
            # AoA: 64-pt zero-padded FFT across RX, fftshift, argmax.
            rx_vec = rd[dop_idx, range_idx, :]
            az_spec = np.fft.fftshift(scipy_fft.fft(rx_vec, n=n_az))
            az_bin = int(np.argmax(np.abs(az_spec)))
            sin_theta = (az_bin - n_az / 2) / (n_az / 2)
            sin_theta = float(np.clip(sin_theta, -1.0, 1.0))
            az_deg = float(np.degrees(np.arcsin(sin_theta)))
            if az_deg > az_lim:
                az_deg = az_lim
            elif az_deg < -az_lim:
                az_deg = -az_lim
            # SNR.
            mag_db = 20.0 * np.log10(float(rd_mag[dop_idx, range_idx]) + 1e-9)
            snr_db = mag_db - noise_db_const
            if snr_db < snr_min:
                continue
            # Doppler bin → radial velocity.
            fd = (dop_idx - n_groups / 2.0) * fd_scale
            vel_mps = -lam_m / 2.0 * fd
            if abs(vel_mps) < speed_min:
                continue
            x_m = range_m * float(np.sin(np.radians(az_deg)))
            y_m = range_m * float(np.cos(np.radians(az_deg)))
            out.append(RadarDetection(
                x_m=x_m, y_m=y_m, z_m=0.0,
                doppler_mps=float(vel_mps),
                snr_db=float(snr_db),
                noise_db=float(noise_db_const),
                range_m=range_m,
                az_deg=az_deg,
                el_deg=0.0,
                target_id=255,
            ))
        return out

    def _stage5_pmm(self, range_cube: np.ndarray) -> List[RadarTarget]:
        """PMM scan over MTI'd slow-time grid → drone-class targets.

        chirp_x_range = range_cube.sum(axis=2); slow_time grid is its
        transpose. Stage-2 already MTI'd the range_cube so no extra
        clutter suppression is needed here.
        """
        d = self._dims
        chirp_x_range = range_cube.sum(axis=2)            # (n_chirps, n_range)
        slow_time_grid = chirp_x_range.T                  # (n_range, n_chirps)
        try:
            hits: List[Tuple[int, PMMResult]] = scan_range_bins(
                slow_time_grid,
                prf_hz=d.prf_hz,
                band_low_hz=float(self.params.pmm_band_low_hz),
                band_high_hz=float(self.params.pmm_band_high_hz),
                threshold_db=float(self.params.pmm_threshold_db),
            )
        except Exception:
            log.exception("scan_range_bins raised; treating as no hits")
            return []

        rng_min = float(self.params.range_min_m)
        rng_max = min(float(self.params.range_max_m), self.max_range_m)
        out: List[RadarTarget] = []
        for range_bin, result in hits:
            range_m = float(range_bin) * d.range_resolution_m
            if range_m < rng_min or range_m > rng_max:
                continue
            # PMM has no AoA on its own — place at boresight; fusion
            # against EO/thermal will refine angle.
            out.append(RadarTarget(
                tid=int(range_bin),
                pos_x_m=0.0,
                pos_y_m=range_m,
                pos_z_m=0.0,
                vel_x_mps=0.0, vel_y_mps=0.0, vel_z_mps=0.0,
                size_x_m=0.5, size_y_m=0.5, size_z_m=0.5,
                confidence=float(result.confidence),
                source="pmm",
                num_points=1,
                coasting=False,
                hits=1, misses=0,
            ))
        return out

    # ─────────────────────── CFAR ──────────────────────────────────────

    def _cfar_2d(
        self, mag: np.ndarray, *, algo: str, threshold_db: float,
    ) -> List[Tuple[int, int]]:
        """Vectorised cumsum CFAR along the range axis at every Doppler.

        Reference window: 16 cells either side, 4 guard. CA fallback for
        OS, GO supported. An early-bin one-sided right-only CFAR runs
        on bins 1..r_lo so close-range targets aren't excluded by the
        two-sided window.

        Returns (dop_idx, range_idx) pairs sorted by descending |mag|,
        capped to the top 256 cells.
        """
        n_dop, n_range = mag.shape
        N_REF = 16
        N_GUARD = 4
        thresh_lin = 10.0 ** (threshold_db / 20.0)

        cs = np.zeros((n_dop, n_range + 1), dtype=np.float64)
        cs[:, 1:] = np.cumsum(mag.astype(np.float64), axis=1)

        r_lo = N_REF + N_GUARD
        r_hi = n_range - N_REF - N_GUARD
        if r_hi <= r_lo:
            return []
        rs = np.arange(r_lo, r_hi)
        left_sum = cs[:, rs - N_GUARD] - cs[:, rs - N_REF - N_GUARD]
        right_sum = (
            cs[:, rs + N_GUARD + 1 + N_REF] - cs[:, rs + N_GUARD + 1]
        )
        left_mean = left_sum / float(N_REF)
        right_mean = right_sum / float(N_REF)

        if algo == "go":
            noise_est = np.maximum(left_mean, right_mean)
        else:
            noise_est = 0.5 * (left_mean + right_mean)

        test_mag = mag[:, r_lo:r_hi]
        det_mask = (noise_est > 0) & (test_mag > thresh_lin * noise_est)

        # Early-range one-sided right-only CFAR.
        early_lo = 1
        early_hi = r_lo
        if early_hi > early_lo:
            ers = np.arange(early_lo, early_hi)
            early_right = (
                cs[:, ers + N_GUARD + 1 + N_REF] - cs[:, ers + N_GUARD + 1]
            )
            early_noise = early_right / float(N_REF)
            early_test = mag[:, early_lo:early_hi]
            early_mask = (
                (early_noise > 0)
                & (early_test > thresh_lin * early_noise)
            )
            det_mask = np.concatenate([early_mask, det_mask], axis=1)
            test_mag = np.concatenate([early_test, test_mag], axis=1)
            r_lo = early_lo

        det_local = np.argwhere(det_mask)
        if det_local.size == 0:
            return []
        det_dops = det_local[:, 0]
        det_ranges = det_local[:, 1] + r_lo
        det_vals = test_mag[det_local[:, 0], det_local[:, 1]]

        # ── Range-gated zero-Doppler suppression ──
        # Static reflections at long range are ALWAYS clutter — RF
        # leakage / building walls / sky returns. Close-range static
        # returns (a person standing in a doorway, a parked car at 5m)
        # are the OPPOSITE: the only signal of interest in stock when
        # they're not moving. Split the difference: drop zero-Doppler
        # CFAR cells when range > _static_max_range_m, keep them at
        # close range. Real moving targets at any range pass through
        # untouched because they're outside the zero-Doppler band.
        n_dop, n_range = mag.shape
        n_dop_centre = n_dop // 2
        zhw = int(self._cell_zero_dop_halfwidth)
        max_static_bin = int(
            self._cell_static_max_range_m / max(self._dims.range_resolution_m, 1e-6)
        )

        is_zero_dop = np.abs(det_dops - n_dop_centre) <= zhw
        is_far = det_ranges > max_static_bin
        drop = is_zero_dop & is_far
        keep = ~drop
        if not keep.all():
            det_dops = det_dops[keep]
            det_ranges = det_ranges[keep]
            det_vals = det_vals[keep]
            if det_vals.size == 0:
                return []

        # Hard top-K cap: with the tighter CFAR threshold, real frames
        # rarely produce >128 cells. Anything more is the chip artifact
        # saturating the bin grid — capping starves the artifact pattern.
        TOPK = 128
        if det_vals.size > TOPK:
            top_idx = np.argpartition(det_vals, -TOPK)[-TOPK:]
            det_dops = det_dops[top_idx]
            det_ranges = det_ranges[top_idx]
            det_vals = det_vals[top_idx]

        order = np.argsort(-det_vals)
        return [(int(det_dops[i]), int(det_ranges[i])) for i in order]

    # ─────────────────────── publish + helpers ─────────────────────────

    def _publish(
        self,
        cfar_detections: List[RadarDetection],
        tracks: List[RadarTarget],
        pmm_targets: List[RadarTarget],
    ) -> None:
        """Publish a RadarFrame on Topic.RADAR_AA.

        pmm_only mode (hybrid with chip TLV):
            always → pmm_targets only (no cfar, no tracks; those come
                    from Topic.RADAR via RadarManager)

        Standalone (full-pipeline) mode:
            stock → tracks only
            ag    → tracks + cfar_detections
            aa    → tracks + pmm_targets + cfar_detections
        """
        if self._pmm_only:
            dets_out: List[RadarDetection] = []
            targets_out = list(pmm_targets)
        elif self._mode_name == "stock":
            dets_out = []
            targets_out = list(tracks)
        elif self._mode_name == "ag":
            dets_out = list(cfar_detections)
            targets_out = list(tracks)
        else:  # "aa"
            dets_out = list(cfar_detections)
            targets_out = list(tracks) + list(pmm_targets)

        BUS.publish(Topic.RADAR_AA, RadarFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=True,
            profile=self.profile_name,
            max_range_m=self.max_range_m,
            fov_half_deg=self.az_half_deg,
            detections=dets_out,
            targets=targets_out,
            num_points=len(dets_out),
            num_targets=len(targets_out),
        ))

    def _cluster_params(self) -> ClusterParams:
        """Build a ClusterParams from the active PipelineParams."""
        return ClusterParams(
            eps_pos_m=float(self.params.cluster_eps_pos_m),
            min_samples=int(self.params.cluster_min_samples),
            confirm_min_hits=int(self.params.confirm_min_hits),
            coast_max_frames=int(self.params.coast_max_frames),
        )

"""DCA1000 raw-ADC `.bin` file parser.

Reads the binary blob the DCA1000 dumps when capturing in raw-ADC LVDS
mode and produces real-valued chirp matrices (presented as complex64
with zero imaginary part for downstream-API compatibility).

Wire layout — empirically verified 2026-05-08 against `channelCfg 15 15`
recording (tools/diag_layout_decode.py, all four RX show uniform
energy ≈ 330 under this layout; any other reshape leaves two RX zero):

  - Per-sample format (cfg `adcCfg 2 0` = 16-bit REAL ADC; `lvdsStreamCfg
    -1 0 1 0` = HW-only ADC streaming, dataFmt=1):
      sample_lo  sample_hi
    Each ADC sample is a signed int16 little-endian. One sample =
    2 BYTES (NOT 4 — there is no IQ pair on this cfg). A previous
    revision of this file was written assuming complex-1x mode which
    halved the effective RX count; that bug is fixed here.

  - Per-chirp packing (RX-major, sample-minor):
      RX0_sample0  RX0_sample1  ...  RX0_sample(N-1)
      RX1_sample0  RX1_sample1  ...  RX1_sample(N-1)
      RX2_sample0  ...                          ...
      RX3_sample0  ...               RX3_sample(N-1)

  - Per-frame packing:
      chirp0  chirp1  chirp2  ...  chirp(numChirpsPerFrame - 1)

  - Frame boundary: implicit. The DCA doesn't insert per-frame
    delimiters; the file just contains numFrames worth of chirps
    back-to-back. Caller must know numChirpsPerFrame from the cfg.

Returned shape (preserves the prior public contract so downstream
replay.py / pmm_detector.py / herm_replay_v2.py keep working):
    (n_frames, n_chirps_per_frame, n_samples, n_rx)  complex64
The imaginary part is zero — downstream code that does an rfft (real
input) or treats the data as complex behaves identically; the
range-FFT + RX-AoA stages that previously read garbage out of the
"missing" RX channels now see real signal there.

Inputs:
  - ``bin_path``  : path to ``adc_data_Raw_0.bin`` (or post-processed
                    ``adc_data.bin``).
  - ``mmwave_cfg`` : path to the .mmwave.json that produced the
                    capture. We extract n_rx, n_samples, n_chirps_per_frame
                    from it so callers don't have to hand-thread these.

Returns:
  - 4-D complex64 numpy array, OR a streaming iterator of frames
    if the file is too large to hold in memory.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)


# ─────────────────────── cfg extraction ─────────────────────────────────


@dataclass
class CaptureDims:
    """Frame dimensions extracted from a Studio .mmwave.json."""
    n_rx: int                  # number of active RX antennas
    n_tx: int                  # number of active TX antennas (DDM divisor)
    n_samples: int             # numAdcSamples per chirp
    n_chirps_per_frame: int    # = numLoops × num_chirp_indices_per_loop
    chirp_period_s: float      # idleTime + rampEndTime, in seconds
    range_resolution_m: float  # c / (2 * BW) where BW = freqSlope × rampEnd
    framePeriodicity_s: float  # frame trigger period

    @property
    def prf_hz(self) -> float:
        """Pulse repetition frequency = 1 / chirp_period_s."""
        return 1.0 / self.chirp_period_s

    @property
    def bytes_per_sample(self) -> int:
        # Real int16 ADC sample. The cfg uses `adcCfg 2 0` = 16-bit real;
        # there is NO IQ pair on the wire. (Prior revision said 4 here,
        # which silently halved the effective RX count by causing the
        # reshape to skip every other RX as if it were a Q channel.)
        return 2

    @property
    def bytes_per_chirp(self) -> int:
        return self.n_samples * self.n_rx * self.bytes_per_sample

    @property
    def bytes_per_frame(self) -> int:
        return self.n_chirps_per_frame * self.bytes_per_chirp

    @property
    def max_range_m(self) -> float:
        return self.n_samples * self.range_resolution_m


def dims_from_mmwave_json(cfg_path: str | Path) -> CaptureDims:
    """Parse a Studio .mmwave.json into a ``CaptureDims``.

    Looks at ``mmWaveDevices[0].rfConfig`` for chirp + profile + frame
    parameters. Validates that the cfg uses a single profile and
    DDM-MIMO (which is what our fpv_long_range.mmwave.json gives).
    """
    cfg = json.loads(Path(cfg_path).read_text())
    rf = cfg["mmWaveDevices"][0]["rfConfig"]

    # Channel mask: count bits.
    rx_mask = int(rf["rlChanCfg_t"]["rxChannelEn"], 16)
    tx_mask = int(rf["rlChanCfg_t"]["txChannelEn"], 16)
    n_rx = bin(rx_mask).count("1")
    n_tx = bin(tx_mask).count("1")
    if n_rx == 0 or n_tx == 0:
        raise ValueError(f"empty rx/tx mask: rx=0x{rx_mask:x} tx=0x{tx_mask:x}")

    # Profile (assume single profile, profileId=0).
    profile = rf["rlProfiles"][0]["rlProfileCfg_t"]
    n_samples = int(profile["numAdcSamples"])
    idle_us = float(profile["idleTimeConst_usec"])
    ramp_us = float(profile["rampEndTime_usec"])
    chirp_period_s = (idle_us + ramp_us) * 1e-6

    # Bandwidth = slope (MHz/us) × ramp time (us) → MHz
    slope_mhz_per_us = float(profile["freqSlopeConst_MHz_usec"])
    bw_hz = slope_mhz_per_us * 1e6 * ramp_us
    c_m_per_s = 3e8
    range_resolution_m = c_m_per_s / (2.0 * bw_hz)

    # Frame: n_chirps_per_frame = numLoops × number_of_chirp_indices.
    # In DDM the "chirp indices" are the per-TX phase/freq variants —
    # one per active TX, all triggered together within a loop.
    frame = rf["rlFrameCfg_t"]
    n_loops = int(frame["numLoops"])
    chirp_start = int(frame["chirpStartIdx"])
    chirp_end = int(frame["chirpEndIdx"])
    n_chirp_indices = chirp_end - chirp_start + 1
    n_chirps_per_frame = n_loops * n_chirp_indices
    frame_period_s = float(frame["framePeriodicity_msec"]) * 1e-3

    return CaptureDims(
        n_rx=n_rx,
        n_tx=n_tx,
        n_samples=n_samples,
        n_chirps_per_frame=n_chirps_per_frame,
        chirp_period_s=chirp_period_s,
        range_resolution_m=range_resolution_m,
        framePeriodicity_s=frame_period_s,
    )


# ─────────────────────── parser ────────────────────────────────────────


def _bytes_to_complex_chirp(buf: bytes, n_samples: int, n_rx: int) -> np.ndarray:
    """Decode one chirp's bytes into a (n_samples, n_rx) complex64 array.

    Wire order (RX-major within chirp, real int16):
        RX0_s0  RX0_s1  ...  RX0_s(N-1)
        RX1_s0  RX1_s1  ...  RX1_s(N-1)
        RX2_s0  ...                  ...
        RX3_s0  ...          RX3_s(N-1)

    Each sample is a single signed int16 = 2 bytes (real ADC, no IQ).
    Total per chirp = n_samples * n_rx * 2 bytes.

    Returned shape is (n_samples, n_rx) complex64 with imag == 0 to
    keep the public contract identical to the prior buggy version.
    """
    if len(buf) != n_samples * n_rx * 2:
        raise ValueError(
            f"chirp buf length {len(buf)} != expected {n_samples * n_rx * 2}"
        )
    raw = np.frombuffer(buf, dtype=np.int16)
    # Wire is RX-major (n_rx rows, n_samples columns); transpose to the
    # canonical (n_samples, n_rx) shape downstream expects.
    real_part = raw.reshape(n_rx, n_samples).T.astype(np.float32)
    return real_part.astype(np.complex64)


def parse_bin_full(
    bin_path: str | Path,
    dims: CaptureDims,
) -> np.ndarray:
    """Load entire .bin into RAM and reshape into the canonical 4-D cube.

    Returns
    -------
    np.ndarray shape (n_frames, n_chirps_per_frame, n_samples, n_rx),
    dtype complex64.

    For a 2 GB recording at our cfg this is ~512 MB of complex64 in RAM.
    Use ``parse_bin_streaming`` if that's too much.
    """
    raw_bytes = Path(bin_path).read_bytes()
    bpf = dims.bytes_per_frame
    n_frames = len(raw_bytes) // bpf
    if n_frames == 0:
        raise ValueError(
            f"{bin_path} is too small ({len(raw_bytes)} B) for one frame "
            f"({bpf} B). Check the cfg matches the capture."
        )
    leftover = len(raw_bytes) - n_frames * bpf
    if leftover:
        log.warning("trailing %d bytes in %s (last frame partial); ignoring",
                    leftover, bin_path)

    # Reshape: int16 buffer → (n_frames, n_chirps, n_rx, n_samples), then
    # transpose to canonical (n_frames, n_chirps, n_samples, n_rx).
    n_int16 = n_frames * dims.n_chirps_per_frame * dims.n_rx * dims.n_samples
    raw = np.frombuffer(raw_bytes[: n_frames * bpf], dtype=np.int16)
    if raw.shape[0] != n_int16:
        raise RuntimeError(
            f"int16 count mismatch: got {raw.shape[0]}, expected {n_int16}"
        )
    raw = raw.reshape(n_frames, dims.n_chirps_per_frame, dims.n_rx,
                      dims.n_samples)
    real_cube = raw.transpose(0, 1, 3, 2).astype(np.float32)
    return real_cube.astype(np.complex64)


def parse_bin_streaming(
    bin_path: str | Path,
    dims: CaptureDims,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, frame_cube) pairs without loading the whole file.

    ``frame_cube`` is shape (n_chirps_per_frame, n_samples, n_rx),
    complex64. Use this for large recordings or when running real-time
    on a fixed memory budget.
    """
    bpf = dims.bytes_per_frame
    with open(bin_path, "rb") as f:
        frame_idx = 0
        while True:
            buf = f.read(bpf)
            if len(buf) == 0:
                return
            if len(buf) < bpf:
                log.warning("partial trailing frame (%d/%d bytes); dropping",
                            len(buf), bpf)
                return
            raw = np.frombuffer(buf, dtype=np.int16)
            raw = raw.reshape(dims.n_chirps_per_frame, dims.n_rx,
                              dims.n_samples)
            real_cube = raw.transpose(0, 2, 1).astype(np.float32)
            cube = real_cube.astype(np.complex64)
            yield frame_idx, cube
            frame_idx += 1


# ─────────────────────── range FFT helper ──────────────────────────────


def range_fft(frame_cube: np.ndarray) -> np.ndarray:
    """Range FFT across the n_samples (fast-time) axis.

    Input  shape (n_chirps, n_samples, n_rx) complex
    Output shape (n_chirps, n_range_bins, n_rx) complex,
    where n_range_bins = n_samples (no zero-pad).

    Hann window applied to suppress sidelobes.
    """
    n_chirps, n_samples, n_rx = frame_cube.shape
    win = np.hanning(n_samples).astype(np.float32)
    windowed = frame_cube * win[np.newaxis, :, np.newaxis]
    return np.fft.fft(windowed, axis=1)


def integrate_rx(frame_cube: np.ndarray) -> np.ndarray:
    """Coherent sum across the RX axis. Cheapest "beamformer" — broadside
    look only. Capon BF lands later.

    Input  shape (n_chirps, n_range_bins, n_rx)
    Output shape (n_chirps, n_range_bins) complex
    """
    return np.sum(frame_cube, axis=-1)

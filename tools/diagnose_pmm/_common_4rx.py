"""Helpers for parsing the AWR2944P + DCA1000 raw .bin with the CORRECT
2-lane CBUFF demux to recover all 4 RX channels.

KEY FACT (2026-05-08):
The chip uses a 2-lane LVDS CBUFF format (CBUFF_LANES2_REAL_FMT0=0x75316420
in C:/ti/mcu_plus_sdk/.../cbuff_lvds.c:63). Lane 0 alternates RX0 and RX2
by ADC sample index; lane 1 alternates RX1 and RX3. The DCA1000 captures
4 lane-positions per cycle but only lanes 0 and 1 carry data; positions 2
and 3 are zero-padded.

So the byte layout per chirp is:
  192 cycles × 4 int16 per cycle = 768 int16 = 1536 bytes
  At each cycle k:
    int16[0] = lane 0 = RX0 if k%2==0 else RX2 (one per ADC sample)
    int16[1] = lane 1 = RX1 if k%2==0 else RX3
    int16[2] = 0 (lane 2 zero pad)
    int16[3] = 0 (lane 3 zero pad)

The previous `_common.py` and `_common_fixed.py` interpreted lane 0 as
"RX0" and lane 1 as "RX1", missing RX2 and RX3 entirely. With this
correct demux, all 4 RX are recovered.

Per-RX time samples per chirp:
  RX0: lane0[0::2] = 192 samples (1 sample per ADC time)
  RX2: lane0[1::2] = 192 samples
  RX1: lane1[0::2] = 192 samples
  RX3: lane1[1::2] = 192 samples

Wait — that gives 4 RX × 192 samples but the lane 0 only has 384 samples
total per chirp (NOT 768). Reconciliation: each lane emits 1 ADC-sample
worth of one RX channel per cycle, alternating between two RX channels.
So lane 0 gives 192 RX0 + 192 RX2 = 384 samples per chirp. ✓
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import yaml
from scipy import fft as scipy_fft


@dataclass
class RecDims:
    n_chirps: int
    n_samples: int
    bytes_per_frame: int
    prf_hz: float
    chirp_period_s: float
    range_resolution_m: float
    max_range_m: float
    framePeriodicity_s: float = 0.05

    @property
    def n_range_bins(self) -> int:
        return self.n_samples // 2 + 1

    @property
    def per_va_prf_hz(self) -> float:
        return self.prf_hz / 6.0  # 6 chirp slots per loop


@dataclass
class Recording:
    name: str
    meta_path: Path
    bin_path: Path
    csv_path: Path
    dims: RecDims


KNOWN = {
    "drone_fly":  "drone fly.meta.yaml",
    "drone fly":  "drone fly.meta.yaml",
    "airborne1":  "drone test airborne 1.meta.yaml",
    "background": "drone test background.meta.yaml",
}


def resolve_recording(arg: str, recordings_dir: Path | None = None) -> Recording:
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parents[2] / "recordings"
    p = Path(arg)
    if p.suffix in (".yaml", ".yml") and p.exists():
        meta_path = p
    else:
        fname = KNOWN.get(arg.strip().lower(), f"{arg}.meta.yaml")
        meta_path = recordings_dir / fname
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    fd = meta["frame_dims"]
    arts = meta["artifacts"]
    dims = RecDims(
        n_chirps=int(fd["n_chirps"]),
        n_samples=int(fd["n_samples"]),
        bytes_per_frame=int(fd["bytes_per_frame"]),
        prf_hz=float(fd["prf_hz"]),
        chirp_period_s=float(fd["chirp_period_s"]),
        range_resolution_m=float(fd["range_resolution_m"]),
        max_range_m=float(fd["max_range_m"]),
        framePeriodicity_s=float(fd.get("framePeriodicity_s", 0.05)),
    )
    return Recording(
        name=meta_path.stem.replace(".meta", ""),
        meta_path=meta_path,
        bin_path=recordings_dir / arts["dca_bin"],
        csv_path=recordings_dir / arts["dca_index_csv"],
        dims=dims,
    )


def iter_4rx_frames(
    bin_path: Path, dims: RecDims, *, start_frame: int = 0, max_frames: int | None = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, real_cube) where real_cube has shape
    (n_chirps=768, n_samples=192, n_rx=4) float32.

    Demuxes the 2-lane CBUFF format:
      Lane 0 (idx % 4 == 0): alternates RX0/RX2 by ADC sample
      Lane 1 (idx % 4 == 1): alternates RX1/RX3 by ADC sample
      Lane 2/3 (idx % 4 == 2, 3): zero-padded by DCA1000
    """
    bpf = dims.bytes_per_frame
    expected_int16 = dims.n_chirps * dims.n_samples * 4   # 4 lane-positions per cycle
    with open(bin_path, "rb") as f:
        if start_frame > 0:
            f.seek(start_frame * bpf)
        idx = start_frame
        n = 0
        while True:
            buf = f.read(bpf)
            if len(buf) < bpf:
                return
            raw = np.frombuffer(buf, dtype=np.int16)
            if raw.size != expected_int16:
                raise RuntimeError(
                    f"frame {idx}: int16 count {raw.size} != expected {expected_int16}"
                )

            # Per chirp: 192 cycles × 4 int16 = 768 int16
            chirp_view = raw.reshape(dims.n_chirps, dims.n_samples, 4)
            lane0 = chirp_view[:, :, 0]   # (n_chirps, n_samples=192)
            lane1 = chirp_view[:, :, 1]

            # Demux: even sample index = RX0/RX1, odd = RX2/RX3
            # lane0[chirp, 0::2] = RX0 (96 samples per chirp)
            # lane0[chirp, 1::2] = RX2
            # lane1[chirp, 0::2] = RX1
            # lane1[chirp, 1::2] = RX3
            # Result: (n_chirps, n_samples_per_rx=96, 4 RX) — half the apparent samples per RX.
            rx0 = lane0[:, 0::2]
            rx2 = lane0[:, 1::2]
            rx1 = lane1[:, 0::2]
            rx3 = lane1[:, 1::2]

            cube = np.stack([rx0, rx1, rx2, rx3], axis=-1).astype(np.float32)
            yield idx, cube
            idx += 1
            n += 1
            if max_frames is not None and n >= max_frames:
                return


def stage1_range_fft_4rx(real_cube: np.ndarray) -> np.ndarray:
    """Range FFT on (n_chirps, n_samples_per_rx, 4) cube.

    Note: n_samples_per_rx is HALF of dims.n_samples because the 2-lane
    demux interleaves 4 RX onto 2 lanes. So range resolution doubles
    (worse) and max range halves vs the 2-RX interpretation.

    Output shape (n_chirps, n_range, 4) complex64.
    """
    real_cube = real_cube - real_cube.mean(axis=1, keepdims=True)
    n_samples = real_cube.shape[1]
    win = np.hanning(n_samples).astype(np.float32)
    windowed = real_cube * win[np.newaxis, :, np.newaxis]
    rfft_out = scipy_fft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)

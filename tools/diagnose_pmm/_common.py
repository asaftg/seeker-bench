"""Shared helpers for Phase-0 PMM diagnostics.

Knows how to:
  - Resolve a recording by friendly name (e.g. "drone fly") into the
    underlying _radar.bin / _radar.csv pair via the meta.yaml.
  - Iterate frames of a real-ADC capture (.bin) into (n_chirps,
    n_samples, n_rx) float32 arrays. The DCA1000 dumps real int16
    samples in RX-major order per chirp; we transpose to sample-major
    so the range FFT sees the layout it expects.
  - Run the same Stage-1 range FFT the live pipeline uses
    (radar_dca/dca_pipeline.py:_stage1_range_fft) so diagnostic plots
    match the live numbers.
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
    n_rx: int
    n_samples: int
    bytes_per_sample: int
    bytes_per_frame: int
    prf_hz: float
    chirp_period_s: float
    range_resolution_m: float
    max_range_m: float
    framePeriodicity_s: float = 0.05  # 20 fps default
    socket_recv_buffer_bytes: int = 0

    @property
    def n_range_bins(self) -> int:
        return self.n_samples // 2 + 1  # rfft positive bins incl. DC and Nyquist

    @property
    def per_va_prf_hz(self) -> float:
        # 4 TX DDMA → per-VA PRF after un-fold = total PRF / n_tx
        return self.prf_hz / 4.0

    @property
    def chirps_per_va(self) -> int:
        return self.n_chirps // 4

    @property
    def fps(self) -> float:
        return 1.0 / self.framePeriodicity_s


@dataclass
class Recording:
    name: str
    meta_path: Path
    bin_path: Path
    csv_path: Path
    dims: RecDims


# Friendly-name → meta filename map. The meta files use spaces in their
# friendly names (e.g. "drone fly.meta.yaml"); operator can pass either.
KNOWN_RECORDINGS = {
    "drone_fly": "drone fly.meta.yaml",
    "drone fly": "drone fly.meta.yaml",
    "airborne1": "drone test airborne 1.meta.yaml",
    "drone_test_airborne_1": "drone test airborne 1.meta.yaml",
    "background": "drone test background.meta.yaml",
    "drone_test_background": "drone test background.meta.yaml",
}


def _meta_to_dims(meta: dict) -> RecDims:
    fd = meta["frame_dims"]
    cap = meta.get("capture", {}) or {}
    return RecDims(
        n_chirps=int(fd["n_chirps"]),
        n_rx=int(fd["n_rx"]),
        n_samples=int(fd["n_samples"]),
        bytes_per_sample=int(fd.get("bytes_per_sample", 2)),
        bytes_per_frame=int(fd["bytes_per_frame"]),
        prf_hz=float(fd["prf_hz"]),
        chirp_period_s=float(fd["chirp_period_s"]),
        range_resolution_m=float(fd["range_resolution_m"]),
        max_range_m=float(fd["max_range_m"]),
        framePeriodicity_s=float(fd.get("framePeriodicity_s", 0.05)),
        socket_recv_buffer_bytes=int(cap.get("socket_recv_buffer_bytes") or 0),
    )


def resolve_recording(arg: str, recordings_dir: Path | None = None) -> Recording:
    """Accept a friendly name OR a path to a meta.yaml. Return Recording."""
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parents[2] / "recordings"

    p = Path(arg)
    if p.suffix in (".yaml", ".yml") and p.exists():
        meta_path = p
    else:
        fname = KNOWN_RECORDINGS.get(arg.strip().lower(), None)
        if fname is None:
            # Try literal "<arg>.meta.yaml"
            fname = f"{arg}.meta.yaml"
        meta_path = recordings_dir / fname
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Could not find meta for {arg!r}. Tried {meta_path}"
            )

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    arts = meta["artifacts"]
    bin_path = recordings_dir / arts["dca_bin"]
    csv_path = recordings_dir / arts["dca_index_csv"]
    dims = _meta_to_dims(meta)
    return Recording(
        name=meta_path.stem.replace(".meta", ""),
        meta_path=meta_path,
        bin_path=bin_path,
        csv_path=csv_path,
        dims=dims,
    )


def iter_real_frames(
    bin_path: Path, dims: RecDims, *, start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, real_cube) where real_cube has shape
    (n_chirps, n_samples, n_rx) float32, NOT yet DC-subtracted or windowed.

    Wire layout per Stage-1 of the live pipeline (RX-major per chirp,
    non-interleaved). Reshape (n_chirps, n_rx, n_samples) → transpose to
    (n_chirps, n_samples, n_rx).
    """
    bpf = dims.bytes_per_frame
    expected_int16 = dims.n_chirps * dims.n_rx * dims.n_samples
    with open(bin_path, "rb") as f:
        if start_frame > 0:
            f.seek(start_frame * bpf)
        idx = start_frame
        n = 0
        while True:
            buf = f.read(bpf)
            if len(buf) == 0 or len(buf) < bpf:
                return
            raw = np.frombuffer(buf, dtype=np.int16)
            if raw.size != expected_int16:
                raise RuntimeError(
                    f"frame {idx}: int16 count {raw.size} != expected {expected_int16}"
                )
            cube = (
                raw.reshape(dims.n_chirps, dims.n_rx, dims.n_samples)
                   .transpose(0, 2, 1)
                   .astype(np.float32)
            )
            yield idx, cube
            idx += 1
            n += 1
            if max_frames is not None and n >= max_frames:
                return


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    """Mirror radar_dca/dca_pipeline.py:_stage1_range_fft.

    Input  shape (n_chirps, n_samples, n_rx) float32
    Output shape (n_chirps, n_range, n_rx) complex64 with positive-freq
    bins scaled by 2 (analytic-signal equivalence). DC and Nyquist
    unscaled.
    """
    real_cube = real_cube - real_cube.mean(axis=1, keepdims=True)
    n_samples = real_cube.shape[1]
    win = np.hanning(n_samples).astype(np.float32)
    windowed = real_cube * win[np.newaxis, :, np.newaxis]
    rfft_out = scipy_fft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def notch_zones(n_samples: int, radius: int = 6) -> list[tuple[int, int]]:
    """Return the (lo, hi) range-bin zones the live pipeline zeros out.

    Mirrors dca_pipeline.py:_notch_harmonic_artifact at radius=6.
    """
    n_range = n_samples // 2 + 1
    step = n_samples // 8
    zones = []
    if step <= 0:
        return zones
    for b in range(step, n_range, step):
        lo = max(b - radius, 0)
        hi = min(b + radius + 1, n_range)
        zones.append((lo, hi))
    return zones


def in_notch_zone(range_bin: int, n_samples: int, radius: int = 6) -> bool:
    for lo, hi in notch_zones(n_samples, radius=radius):
        if lo <= range_bin < hi:
            return True
    return False


def frame_to_seconds(frame_idx: int, dims: RecDims) -> float:
    return frame_idx * dims.framePeriodicity_s

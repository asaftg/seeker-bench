"""Shared helpers for Phase-0/1 PMM diagnostics — TRULY CORRECTED LAYOUT.

This file's earlier "CORRECTED LAYOUT" header (claiming RX-interleaved
sample-major with RX2/RX3 zero) was wrong. The actual wire layout was
re-confirmed empirically on 2026-05-08 against the channelCfg=15
recording (tools/diag_layout_decode.py): a per-frame reshape of
(n_chirps, n_rx, n_samples) places UNIFORM signal energy on all four
RX channels, while the previous (n_chirps, n_samples, n_rx) reshape
(used here pre-fix) artefactually leaves RX2/RX3 at zero — that was
the "only 2 RX wired" diagnosis.

  Per-chirp wire layout (int16 indices, RX-major within chirp):
    [RX0_s0, RX0_s1, ..., RX0_s(N-1),
     RX1_s0, RX1_s1, ..., RX1_s(N-1),
     RX2_s0, RX2_s1, ..., RX2_s(N-1),
     RX3_s0, RX3_s1, ..., RX3_s(N-1)]

All four RX carry signal. The live pipeline
(`radar_dca/dca_pipeline.py:_stage1_range_fft`) was always correct;
it is `radar_dca/bin_parser.py` and this file that previously had
the wrong reshape and therefore the previous-agent's "RX2/RX3 = 0"
diagnosis. Both have been fixed 2026-05-08.

The 24-bin "chip artifact" notch in the live pipeline still applies
because that's a real chip-internal mixer harmonic seen on RX0/RX1
(and by extension RX2/RX3); it is independent of the reshape bug.
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
    n_rx_wire: int           # 4 — all RX wired and streaming on this EVM
    n_rx_active: int         # 4 — all four carry signal under the correct
                             #     reshape (was 2 pre-fix due to wrong stride)
    n_samples: int
    bytes_per_sample: int
    bytes_per_frame: int
    prf_hz: float
    chirp_period_s: float
    range_resolution_m: float
    max_range_m: float
    framePeriodicity_s: float = 0.05
    socket_recv_buffer_bytes: int = 0
    n_tx_ddma: int = 4       # 4 TX in DDMA (per cfg)
    n_chirps_per_va: int = 128   # 768 / 6 chirp slots = 128 — but DDMA un-mix
                                  # gives 128 only if we treat the 6-slot pattern
                                  # as a 4-TX × 128-loop schedule. See ddma_unfold.
    ddma_phase_ant_order: tuple[int, ...] = (0, 2, 3, 1)

    @property
    def n_range_bins(self) -> int:
        return self.n_samples // 2 + 1   # rfft positive bins

    @property
    def per_va_prf_hz(self) -> float:
        return self.prf_hz / 6.0   # 6 chirp slots per loop

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


KNOWN_RECORDINGS = {
    "drone_fly":   "drone fly.meta.yaml",
    "drone fly":   "drone fly.meta.yaml",
    "airborne1":   "drone test airborne 1.meta.yaml",
    "background":  "drone test background.meta.yaml",
}


def _meta_to_dims(meta: dict) -> RecDims:
    fd = meta["frame_dims"]
    cap = meta.get("capture", {}) or {}
    return RecDims(
        n_chirps=int(fd["n_chirps"]),
        n_rx_wire=int(fd["n_rx"]),
        n_rx_active=int(fd["n_rx"]),
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
    if recordings_dir is None:
        recordings_dir = Path(__file__).resolve().parents[2] / "recordings"
    p = Path(arg)
    if p.suffix in (".yaml", ".yml") and p.exists():
        meta_path = p
    else:
        fname = KNOWN_RECORDINGS.get(arg.strip().lower(), f"{arg}.meta.yaml")
        meta_path = recordings_dir / fname
    if not meta_path.exists():
        raise FileNotFoundError(f"meta not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f)
    arts = meta["artifacts"]
    return Recording(
        name=meta_path.stem.replace(".meta", ""),
        meta_path=meta_path,
        bin_path=recordings_dir / arts["dca_bin"],
        csv_path=recordings_dir / arts["dca_index_csv"],
        dims=_meta_to_dims(meta),
    )


def iter_real_frames(
    bin_path: Path, dims: RecDims, *, start_frame: int = 0,
    max_frames: int | None = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (frame_idx, real_cube) where real_cube has shape
    (n_chirps=768, n_samples=192, n_rx_active=4) float32.

    Parses int16 buffer as (n_chirps, n_rx, n_samples) — RX-major within
    each chirp, samples contiguous per RX — then transposes to the
    canonical (chirps, samples, rx) downstream shape.
    """
    bpf = dims.bytes_per_frame
    expected_int16 = dims.n_chirps * dims.n_rx_wire * dims.n_samples
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
            cube_full = raw.reshape(dims.n_chirps, dims.n_rx_wire, dims.n_samples)
            cube_active = cube_full.transpose(0, 2, 1).astype(np.float32)
            yield idx, cube_active
            idx += 1
            n += 1
            if max_frames is not None and n >= max_frames:
                return


def stage1_range_fft(real_cube: np.ndarray) -> np.ndarray:
    """Range FFT on real ADC samples.

    Input  shape (n_chirps, n_samples, n_rx) float32
    Output shape (n_chirps, n_range, n_rx) complex64
    """
    real_cube = real_cube - real_cube.mean(axis=1, keepdims=True)
    n_samples = real_cube.shape[1]
    win = np.hanning(n_samples).astype(np.float32)
    windowed = real_cube * win[np.newaxis, :, np.newaxis]
    rfft_out = scipy_fft.rfft(windowed, axis=1, workers=2)
    rfft_out[:, 1:-1, :] *= 2.0
    return rfft_out.astype(np.complex64)


def integrate_rx_active(range_cube: np.ndarray) -> np.ndarray:
    """Coherent sum across all active RX (broadside beam, ~6 dB SNR gain
    over RX0+RX1-only that the pre-fix code was effectively doing).

    Input  shape (n_chirps, n_range, n_rx_active=4) complex
    Output shape (n_chirps, n_range) complex
    """
    return range_cube.sum(axis=-1)


def frame_to_seconds(frame_idx: int, dims: RecDims) -> float:
    return frame_idx * dims.framePeriodicity_s

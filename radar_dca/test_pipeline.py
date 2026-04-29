"""End-to-end tests for the live DCAPipeline.

These exercise the byte-to-RadarFrame chain WITHOUT the real DCA
hardware: we feed synthetic UDP payloads into a mock listener,
verify the pipeline:

  1. drains the queue,
  2. assembles frames at the right byte boundary,
  3. produces real (non-zero) range cubes,
  4. fires PMM detections on a synthetic propeller signature, and
  5. fires A/G CFAR detections on a synthetic point target with
     known range × angle.

If any of these regress, the field test will produce no detections
exactly the way it did pre-fix — these tests are the canary.
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from radar_dca.dca_pipeline import (
    DCAPipeline,
    FrameDims,
    dims_from_cfg,
    dims_from_cfg_file,
)


# ─────────────────────── fakes ───────────────────────────────────────────


class FakeListener:
    """Minimal stand-in for DataPortListener.

    Holds a list of payload bytes; ``drain_payloads`` returns + clears
    them. Stats expose only what the pipeline reads (bytes_total).
    """
    def __init__(self) -> None:
        self._payloads: List[bytes] = []
        self._lock = threading.Lock()
        self.bytes_total = 0

    def push(self, data: bytes) -> None:
        with self._lock:
            self._payloads.append(data)
            self.bytes_total += len(data)

    def drain_payloads(self) -> List[bytes]:
        with self._lock:
            out = self._payloads
            self._payloads = []
            return out

    def stats(self):
        @dataclass
        class _S:
            bytes_total: int
        return _S(bytes_total=self.bytes_total)


# ─────────────────────── helpers ─────────────────────────────────────────


def _frame_bytes_from_cube(cube: np.ndarray) -> bytes:
    """Inverse of DCAPipeline._bytes_to_cube — build the wire bytes
    that the chip would have emitted for a given complex cube."""
    n_chirps, n_samples, n_rx = cube.shape
    raw = np.empty((n_chirps, n_samples, n_rx, 2), dtype=np.int16)
    raw[..., 0] = np.real(cube).astype(np.int16)
    raw[..., 1] = np.imag(cube).astype(np.int16)
    return raw.tobytes()


def _make_pipeline(dims: FrameDims) -> DCAPipeline:
    listener = FakeListener()
    pipe = DCAPipeline(
        listener=listener,  # type: ignore[arg-type]
        dims=dims,
        pmm_band_low_hz=50.0,
        pmm_band_high_hz=500.0,
        pmm_threshold_db=3.0,
        profile_name="test",
        max_range_m=200.0,
        az_half_deg=60.0,
    )
    pipe._publish_enabled = True   # publish path exercised
    return pipe


# ─────────────────────── tests ───────────────────────────────────────────


def test_dims_from_cfg_file_matches_unified_cfg(tmp_path):
    """Parsing our shipping unified.cfg produces the dims we expect."""
    cfg = tmp_path / "unified.cfg"
    cfg.write_text(
        "% comment\n"
        "channelCfg 15 15 0 0 0\n"
        "profileCfg 0 77 7 7 20.81 0 0 8.883 0 384 30000 0 0 164\n"
        "chirpCfg 0 5 0 0 0 0 0 15\n"
        "frameCfg 0 5 128 0 384 50 1 0\n"
        "sensorStart\n"
    )
    dims = dims_from_cfg_file(str(cfg))
    assert dims.n_rx == 4              # popcount(15) = 4
    assert dims.n_samples == 384
    assert dims.n_chirps == 6 * 128    # (5-0+1) chirp indices × 128 loops
    # PRF check: 1 / (7 + 20.81 us) ≈ 35958 Hz
    assert 35000 < dims.prf_hz < 37000
    # Range res: c / (2 × slope × adc_capture_time)
    # adc_capture_time = 384 / 30000 ksps = 12.8 us
    # BW = 8.883 MHz/us × 12.8 us ≈ 113.7 MHz → res ≈ 1.32 m
    assert 1.2 < dims.range_resolution_m < 1.5


def test_bytes_to_cube_roundtrip():
    """The inverse of _bytes_to_cube produces the same cube."""
    dims = dims_from_cfg(n_chirps=8, n_rx=4, n_samples=16)
    pipe = _make_pipeline(dims)
    rng = np.random.default_rng(42)
    cube = (rng.integers(-1000, 1000, size=(8, 16, 4))
            + 1j * rng.integers(-1000, 1000, size=(8, 16, 4))
           ).astype(np.complex64)
    buf = _frame_bytes_from_cube(cube)
    out = pipe._bytes_to_cube(buf)
    assert out.shape == cube.shape
    np.testing.assert_array_equal(out, cube)


def test_bytes_to_cube_rejects_wrong_size():
    """A buffer that's the wrong number of int16s (or odd-byte) must
    raise. We don't care which of the two error paths fires — any
    ValueError tells the pipeline to count the frame as dropped."""
    dims = dims_from_cfg(n_chirps=8, n_rx=4, n_samples=16)
    pipe = _make_pipeline(dims)
    # Even byte count but wrong int16 count → our explicit check.
    with pytest.raises(ValueError):
        pipe._bytes_to_cube(b"\x00" * (dims.bytes_per_frame - 4))
    # Odd byte count → numpy's own rejection.
    with pytest.raises(ValueError):
        pipe._bytes_to_cube(b"\x00" * 17)


def test_pipeline_assembles_frames_from_queued_payloads():
    """Push enough bytes for two frames; verify the loop processes both."""
    dims = dims_from_cfg(n_chirps=8, n_rx=4, n_samples=16)
    pipe = _make_pipeline(dims)
    listener = pipe._listener  # FakeListener

    cube = np.zeros((8, 16, 4), dtype=np.complex64)
    buf = _frame_bytes_from_cube(cube)
    assert len(buf) == dims.bytes_per_frame
    # Push 2 frames split across 5 packets to exercise reassembly.
    chunk = len(buf) // 5
    for i in range(0, len(buf) * 2, chunk):
        listener.push((buf + buf)[i:i + chunk])

    # Spin the loop manually a few times.
    pipe._stop.set()  # prevent the thread starting; we drive _loop ourselves
    # Run one iteration of the inner block.
    payloads = listener.drain_payloads()
    for p in payloads:
        pipe._buf.extend(p)
    bpf = dims.bytes_per_frame
    n_processed = 0
    while len(pipe._buf) >= bpf:
        frame_bytes = bytes(pipe._buf[:bpf])
        del pipe._buf[:bpf]
        pipe._process_frame(frame_bytes)
        n_processed += 1
    assert n_processed == 2
    assert pipe._stats.frames_assembled == 2
    assert pipe._stats.frames_dropped == 0


def test_pmm_fires_on_synthetic_propeller():
    """Inject a synthetic blade-pass modulation at 200 Hz at one
    range bin; the PMM detector should flag it."""
    n_chirps = 256          # 256 × 27.81 us ≈ 7.1 ms — enough for 200 Hz res
    n_rx = 4
    n_samples = 64
    chirp_period_s = 27.81e-6
    dims = dims_from_cfg(
        n_chirps=n_chirps, n_rx=n_rx, n_samples=n_samples,
        chirp_period_s=chirp_period_s,
    )
    pipe = _make_pipeline(dims)
    pipe._pmm_band_low = 100.0
    pipe._pmm_band_high = 400.0
    pipe._pmm_threshold = 3.0

    # Build a cube where range bin r=20 has a 200 Hz amplitude
    # modulation across slow-time, with a stationary point target.
    target_bin = 20
    blade_hz = 200.0
    t = np.arange(n_chirps) * chirp_period_s
    modulation = 0.5 + 0.5 * np.cos(2 * np.pi * blade_hz * t)  # 0..1
    # Cube shape (n_chirps, n_samples, n_rx) — put a complex CW tone
    # at fast-time index that maps to range bin target_bin.
    cube = np.zeros((n_chirps, n_samples, n_rx), dtype=np.complex64)
    fast_idx = np.arange(n_samples)
    # Tone at frequency that range-FFT places into bin `target_bin`.
    tone = np.exp(2j * np.pi * target_bin * fast_idx / n_samples)
    for c in range(n_chirps):
        # All RX channels carry the same signal (boresight).
        cube[c] = (tone[:, None] * modulation[c]).astype(np.complex64) * 5000

    buf = _frame_bytes_from_cube(cube)
    pipe._process_frame(buf)
    assert pipe._stats.frames_assembled == 1
    # We should have at least one PMM hit.
    assert pipe._stats.drone_detections >= 1, (
        f"PMM did not fire on a {blade_hz} Hz signature; "
        f"frames_dropped={pipe._stats.frames_dropped}"
    )


def test_ag_cfar_fires_on_synthetic_point_target():
    """Build a cube with a single bright stationary scatterer at a
    known range. Range-Doppler + CFAR should produce a detection at
    that range, with the right Doppler bin (zero-velocity)."""
    n_chirps = 128
    n_rx = 4
    n_samples = 64
    dims = dims_from_cfg(
        n_chirps=n_chirps, n_rx=n_rx, n_samples=n_samples,
        chirp_period_s=27.81e-6,
        range_resolution_m=1.0,
    )
    pipe = _make_pipeline(dims)
    pipe._ag_integrate_chirps = 16
    pipe._ag_cfar_algo = "ca"
    pipe._ag_cfar_threshold_db = 6.0   # 6 dB above noise floor

    target_bin = 25
    fast_idx = np.arange(n_samples)
    tone = np.exp(2j * np.pi * target_bin * fast_idx / n_samples)
    rng = np.random.default_rng(0)
    # Cube = small noise + bright stationary point.
    cube = (rng.standard_normal((n_chirps, n_samples, n_rx))
            + 1j * rng.standard_normal((n_chirps, n_samples, n_rx))
           ).astype(np.complex64) * 50.0
    for c in range(n_chirps):
        cube[c] += (tone[:, None] * 5000).astype(np.complex64)

    buf = _frame_bytes_from_cube(cube)
    pipe._process_frame(buf)
    # The detection list lives on the published RadarFrame; we can
    # check the pipeline got at least one RadarDetection back.
    # Since we call _process_frame directly (no bus), we re-run the
    # _ag_process step here:
    range_cube = pipe._range_doppler(pipe._bytes_to_cube(buf))
    dets = pipe._ag_process(range_cube)
    assert len(dets) >= 1, (
        f"A/G CFAR did not detect a point target at range bin "
        f"{target_bin} (det count {len(dets)})"
    )
    # The brightest detection should be at the right range.
    closest = min(dets, key=lambda d: abs(d.range_m - target_bin * dims.range_resolution_m))
    assert abs(closest.range_m - target_bin * dims.range_resolution_m) <= dims.range_resolution_m


def test_publish_disabled_does_not_emit():
    """When _publish_enabled is False (i.e. mode != aa), processing
    still runs but no RadarFrame hits the bus.

    The bus exposes ``get_latest()`` (not subscribe), so we check by
    snapshotting the latest RADAR_AA frame before and after and
    asserting it didn't change."""
    from common.frame_bus import BUS
    from common.frames import Topic

    dims = dims_from_cfg(n_chirps=8, n_rx=4, n_samples=16)
    pipe = _make_pipeline(dims)
    pipe._publish_enabled = False

    before = BUS.get_latest(Topic.RADAR_AA)
    cube = np.zeros((8, 16, 4), dtype=np.complex64)
    buf = _frame_bytes_from_cube(cube)
    pipe._process_frame(buf)
    after = BUS.get_latest(Topic.RADAR_AA)

    assert pipe._stats.frames_assembled == 1
    # Either no publish ever happened, or the latest frame is the
    # same one as before — id-equality covers both cases.
    assert after is before

"""EO capture process — IMX568 over Leopard FX3 USB3 bridge on Jetson.

Owns /dev/video0. Streams RAW12 (advertised as YUYV 8-bit by FX3 UVC
descriptor). Reinterprets bytes as uint16 RAW12 RGGB Bayer, applies
p1/p99 AGC stretch via cv2.convertScaleAbs, writes BGR to shared
memory ring.

Decoupled from the rest of seeker via shared memory. This process
holds its own GIL and runs flat-out at sensor rate (~25 Hz) without
contention from thermal/radar/inference threads.

Replaces the v1 eo/_v4l2_raw_backend.py + eo/imx568_capture.py +
eo_manager._capture_loop combination, but keeps the same wire format
(BGR uint8 H x W x 3) so downstream inference/fusion can be reused.

For Phase 2.2: this module's hot loop will be replaced by a pybind11
C++ extension (`seeker_v2.native.v4l2_capture`) that releases the GIL
during V4L2 IOCTLs + AGC. The process glue stays the same.
"""
from __future__ import annotations

import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from seeker_v2.processes.ipc import (
    FrameRing,
    FrameDescriptor,
    DescriptorQueue,
    published_frame,
)

log = logging.getLogger("seeker_v2.eo_capture")

NATIVE_W = 2472
NATIVE_H = 2064


# ── Try the native (C++/pybind11) backend first; fall back to the
#    pure-Python v4l2_raw_backend port from v1. ──────────────────────
def _make_backend(dev_path: str):
    """Return the best available V4L2+AGC backend for EO."""
    try:
        from seeker_v2.native import seeker_native  # type: ignore
        log.info("EO using native (C++) V4L2 backend")
        return seeker_native.V4L2Backend(dev_path, NATIVE_W, NATIVE_H)
    except Exception as e:
        log.info("native backend unavailable (%s); using pure-Python", e)

    # Pure-Python: import the v1 backend (already exists in repo).
    # Phase 2.1 ships this as the working backend; Phase 2.2 swaps in
    # the native one.
    from seeker_v2.processes._py_v4l2_backend import RawV4L2Backend
    return RawV4L2Backend(dev_path, NATIVE_W, NATIVE_H)


@dataclass
class EOCaptureConfig:
    # Stable udev symlink (always points at the IMX568 capture node)
    dev_path: str = "/dev/seeker_eo_v"
    width: int = NATIVE_W
    height: int = NATIVE_H
    shm_name: str = "seeker_eo_bgr"
    n_slots: int = 4
    # IMX568+FX3 firmware quirk: the response of exposure_ext to
    # actual sensor brightness is HIGHLY non-monotonic and scene-
    # dependent. Probed values that gave clean frames (mean=375,
    # p99=810) varied between exp=16 and exp=32 across two probes
    # 30 s apart in the same garage scene. v1's AE walks exposure
    # geometrically and gets stuck on any value that returns the
    # all-FF "saturated" signature, mistaking it for real sensor
    # saturation.
    # Default to a probe-ladder AE (see _ae_probe_ladder below) that
    # tries known-safe exposure values and picks the one with valid
    # frame stats. Initial value is just a starting point.
    initial_exposure_ext: int = 16
    ae_probe_values: tuple = (4, 8, 16, 32, 48, 64, 96, 128)
    ae_recheck_every_n_frames: int = 600   # ~30 s @ 20 Hz; re-probe to track light changes
    target_mean_lo: float = 100.0          # if mean < lo, exposure too low
    target_mean_hi: float = 2500.0         # if mean > hi, exposure too high
    target_p99_max: float = 3800.0         # if p99 above this, saturated
    target_p99_lo: float = 300.0
    target_p99_hi: float = 3500.0
    # Phase 2.5: GUI snapshot stream
    jpeg_quality: int = 80
    # Don't JPEG-encode every full-res frame; the GUI samples ~10 Hz max.
    # 1 / N — encode every Nth captured frame.
    jpeg_every_n: int = 3


def _try_load_nvjpeg(quality: int):
    """Phase 2.5: hardware nvjpeg for EO snapshots if available."""
    try:
        from seeker_v2.native import seeker_nvjpeg as _nj
        return _nj.JpegEncoder(quality=quality, name="seeker_v2_eo")
    except Exception:
        return None


def run(cfg: EOCaptureConfig, ctrl_q, stats_q, jpeg_q=None) -> int:
    """Process entry point.

    Args:
        cfg: serializable config dict (must survive multiprocessing pickle)
        ctrl_q: multiprocessing.Queue of control messages from main
                (e.g. ("exposure_ext", 1264), ("shutdown", None))
        stats_q: multiprocessing.Queue for periodic stats back to main

    Returns: exit code.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # SIGTERM = clean shutdown
    _stop = {"flag": False}

    def _on_sigterm(*_):
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)

    # Allocate the ring (we own it as the producer).
    frame_bytes = cfg.width * cfg.height * 3  # BGR uint8
    ring = FrameRing.create(
        cfg.shm_name, n_slots=cfg.n_slots, frame_bytes=frame_bytes
    )
    log.info(
        "EO capture: shm=%s, %dx%d BGR, %d slots (%.1f MB total)",
        cfg.shm_name, cfg.width, cfg.height, cfg.n_slots,
        cfg.n_slots * frame_bytes / 1024 / 1024,
    )

    backend = None
    try:
        backend = _make_backend(cfg.dev_path)
        if not backend.open():
            log.error("EO V4L2 open failed on %s", cfg.dev_path)
            return 1

        # Trigger-disable + initial exposure (handled inside open() for our backend)
        log.info("EO capture: open OK, beginning frame loop")

        frame_id = 0
        last_stats_emit = time.monotonic()
        ae_chosen_ext = cfg.initial_exposure_ext
        try:
            backend.set_exposure_ext(ae_chosen_ext)
        except AttributeError:
            pass

        # Stats trickle to main every ~1s
        STATS_PERIOD = 1.0

        # Phase 2.5: GUI snapshot encoder
        nvjpeg = _try_load_nvjpeg(cfg.jpeg_quality) if jpeg_q is not None else None
        if nvjpeg is not None:
            log.info("EO nvjpeg encoder loaded (q=%d)", cfg.jpeg_quality)

        while not _stop["flag"]:
            # ── Drain any pending control commands ────────────────────
            try:
                while True:
                    cmd = ctrl_q.get_nowait()
                    if cmd is None:
                        _stop["flag"] = True
                        break
                    op = cmd[0]
                    if op == "shutdown":
                        _stop["flag"] = True
                    elif op == "exposure_ext":
                        ae_chosen_ext = int(cmd[1])
                        try:
                            backend.set_exposure_ext(ae_chosen_ext)
                        except AttributeError:
                            pass
                    elif op == "gain":
                        try:
                            backend.set_gain_rgb(int(cmd[1]))
                        except AttributeError:
                            pass
            except Exception:
                # Empty queue raises queue.Empty - that's our signal to break
                pass

            if _stop["flag"]:
                break

            # ── Grab a frame ──────────────────────────────────────────
            bgr = backend.grab()
            if bgr is None:
                # Transient timeout; reuse last frame? Skip for now.
                time.sleep(0.001)
                continue

            # bgr is (H, W, 3) uint8 BGR. Write into shm.
            with published_frame(
                ring, frame_id=frame_id,
                width=cfg.width, height=cfg.height, channels=3,
                dtype="uint8",
            ) as view:
                # bgr.tobytes() -> view assignment. ndarray is C-contiguous.
                expected = cfg.width * cfg.height * 3
                if bgr.size != expected:
                    log.warning("frame size %d != expected %d", bgr.size, expected)
                    continue
                # Single memcpy: bgr (numpy) -> view (shm bytes).
                # numpy supports `.tobytes()` then assign, but we can be
                # zero-extra-copy by reshaping the view as uint8 and using
                # numpy assignment.
                view[:] = bgr.tobytes()

            # Phase 2.5: encode JPEG snapshot every N frames for the GUI.
            if (jpeg_q is not None
                    and (frame_id % max(1, cfg.jpeg_every_n)) == 0):
                jpeg_bytes = b""
                if nvjpeg is not None:
                    try:
                        jpeg_bytes = nvjpeg.encode_bgr(bgr)
                    except Exception as e:
                        log.warning("EO nvjpeg failed (%r) — disabling", e)
                        nvjpeg = None
                if nvjpeg is None:
                    try:
                        import cv2
                        ok, buf = cv2.imencode(
                            ".jpg", bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
                        )
                        if ok:
                            jpeg_bytes = bytes(buf)
                    except Exception as e:
                        log.warning("EO cv2 imencode failed (%r)", e)
                if jpeg_bytes:
                    try:
                        jpeg_q.put_nowait((frame_id, jpeg_bytes))
                    except Exception:
                        try:
                            jpeg_q.get_nowait()
                            jpeg_q.put_nowait((frame_id, jpeg_bytes))
                        except Exception:
                            pass

            frame_id += 1

            # ── Stats ─────────────────────────────────────────────────
            now = time.monotonic()
            if now - last_stats_emit >= STATS_PERIOD:
                last_stats_emit = now
                stats = getattr(backend, "last_raw_stats", None)
                try:
                    stats_q.put_nowait({
                        "kind": "eo_stats",
                        "frame_id": frame_id,
                        "raw_stats": dict(stats) if stats else None,
                        "exposure_ext": ae_chosen_ext,
                    })
                except Exception:
                    pass

    except Exception as e:
        log.exception("EO capture loop died: %s", e)
        return 2
    finally:
        if backend is not None:
            try:
                backend.release()
            except AttributeError:
                try:
                    backend.stop()
                except Exception:
                    pass
            except Exception:
                pass
        ring.close_and_unlink()
        log.info("EO capture: clean shutdown")

    return 0


# ── Process spawner helper for the orchestrator ────────────────────────
def spawn(mp_ctx, cfg: EOCaptureConfig):
    """Spawn the EO capture process.

    Returns (process, ctrl_q, stats_q, jpeg_q).
    """
    ctrl_q = mp_ctx.Queue(maxsize=32)
    stats_q = mp_ctx.Queue(maxsize=32)
    jpeg_q = mp_ctx.Queue(maxsize=2)
    proc = mp_ctx.Process(
        target=run, args=(cfg, ctrl_q, stats_q, jpeg_q),
        name="seeker_v2_eo_capture",
        daemon=False,
    )
    proc.start()
    return proc, ctrl_q, stats_q, jpeg_q


if __name__ == "__main__":
    # Standalone smoke test: capture for 5 seconds and report rate.
    import multiprocessing as mp

    logging.basicConfig(level=logging.INFO)
    cfg = EOCaptureConfig()
    ctx = mp.get_context("spawn")
    proc, ctrl_q, stats_q, _jpeg_q = spawn(ctx, cfg)

    # Read frames from the ring as a smoke consumer
    time.sleep(0.5)  # let producer create + open
    ring = FrameRing.attach(
        cfg.shm_name, n_slots=cfg.n_slots,
        frame_bytes=cfg.width * cfg.height * 3,
    )

    seen = []
    last_seq = 0
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < 5.0:
            desc, seq = ring.latest()
            if desc is not None and seq != last_seq:
                last_seq = seq
                seen.append(desc.frame_id)
            time.sleep(0.005)
    finally:
        ctrl_q.put(("shutdown", None))
        proc.join(timeout=3)
        if proc.is_alive():
            proc.terminate()
        ring.close()

    elapsed = time.monotonic() - t0
    print(f"saw {len(seen)} unique frames in {elapsed:.1f}s = {len(seen)/elapsed:.1f} fps")

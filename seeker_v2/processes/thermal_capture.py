"""Thermal capture process — FLIR Boson 640.

Owns /dev/video2 (or wherever Boson enumerates). Captures YUYV 8-bit
(Boson's onboard AGC) at 60 Hz native, applies our software AGC,
runs the tophat heat detector, JPEG-encodes once (cached so the WS
sender reuses), and publishes the BGR display + metadata to a
shared-memory ring.

The H/V YOLO classifier runs in the SEPARATE inference process. We
publish frames via shm; inference consumes and writes back HV
detections. MOSSE per-track update happens in the FUSION process so
this capture stays under 20 ms/frame even with many tracks.

Replaces v1 thermal/{thermal_manager,boson_capture,thermal_processor,
heat_detector}.py. Same algorithms, same parameters; just the threading
model is different.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from seeker_v2.processes.ipc import (
    DescriptorQueue,
    FrameDescriptor,
    FrameRing,
    published_frame,
)

log = logging.getLogger("seeker_v2.thermal_capture")


def _try_load_nvjpeg(quality: int):
    """Phase 2.3: prefer hardware nvjpeg encoder when present.

    On Jetson (JP5) the C++ extension is built and available as
    seeker_v2.native.seeker_nvjpeg. On Windows / x86 dev hosts the
    import fails and the caller falls back to cv2.imencode.
    """
    try:
        from seeker_v2.native import seeker_nvjpeg as _nj
        enc = _nj.JpegEncoder(quality=quality, name="seeker_v2_thermal")
        log.info("nvjpeg hw encoder loaded (q=%d)", quality)
        return enc
    except Exception as e:
        log.info("nvjpeg unavailable (%s) — using cv2.imencode", type(e).__name__)
        return None


@dataclass
class ThermalCaptureConfig:
    dev_path: str = "/dev/video2"  # may need auto-probe
    width: int = 640
    height: int = 512
    target_fps: int = 60
    shm_name: str = "seeker_thermal_bgr"
    n_slots: int = 4
    # AGC
    agc_low_pct: float = 2.0
    agc_high_pct: float = 98.0
    agc_stats_every: int = 5
    colormap: str = "WHITE_HOT"
    # Heat detector
    heat_tophat_kernel: int = 11   # Phase 1 fix #5
    heat_min_area: int = 8
    heat_max_detections: int = 20
    # JPEG
    jpeg_quality: int = 70  # Phase 1 fix #5: 75 -> 70


def _open_boson(cfg: ThermalCaptureConfig):
    import cv2
    cap = cv2.VideoCapture(cfg.dev_path, cv2.CAP_V4L2)
    if not cap.isOpened():
        log.error("Boson open failed at %s", cfg.dev_path)
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_FPS, cfg.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _apply_agc(gray: np.ndarray, low_pct: float, high_pct: float,
               cache: dict, stats_every: int) -> np.ndarray:
    """AGC stretch via cv2.convertScaleAbs with cached stats every Nth
    frame. Saves ~5 ms per cached frame on 640x512."""
    import cv2
    cache["frames_since"] = cache.get("frames_since", 0) + 1
    if cache["frames_since"] >= stats_every or "alpha" not in cache:
        cache["frames_since"] = 0
        sample = gray[::4, ::4]
        p_lo = float(np.percentile(sample, low_pct))
        p_hi = float(np.percentile(sample, high_pct))
        span = max(p_hi - p_lo, 4.0)
        cache["alpha"] = 255.0 / span
        cache["beta"] = -p_lo * cache["alpha"]
    return cv2.convertScaleAbs(gray, alpha=cache["alpha"], beta=cache["beta"])


def _apply_colormap(y8: np.ndarray, mapname: str) -> np.ndarray:
    import cv2
    if mapname.upper() == "INFERNO":
        return cv2.applyColorMap(y8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(y8, cv2.COLOR_GRAY2BGR)


def _tophat_heat_detect(y8: np.ndarray, kernel_size: int,
                        min_area: int, max_dets: int) -> list[dict]:
    """Tophat morphology heat detector. Same as v1 thermal/heat_detector.py."""
    import cv2
    if kernel_size <= 1:
        return []
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    th = cv2.morphologyEx(y8, cv2.MORPH_TOPHAT, kernel)
    sample = th[::4, ::4]
    med = float(np.median(sample))
    mad = float(np.median(np.abs(sample - med)) + 1.0)
    thresh = med + 6.0 * mad
    mask = (th > thresh).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(mask)
    dets = []
    for i in range(1, min(n, max_dets + 1)):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        cx, cy = cents[i]
        roi = th[y : y + h, x : x + w]
        peak = int(roi.max()) if roi.size else int(thresh)
        dets.append({
            "x": int(x), "y": int(y), "w": int(w), "h": int(h),
            "cx": float(cx), "cy": float(cy),
            "area": int(area), "peak": peak,
        })
    dets.sort(key=lambda d: d["peak"], reverse=True)
    return dets[:max_dets]


def run(cfg: ThermalCaptureConfig, ctrl_q, stats_q,
        det_q: Optional[DescriptorQueue] = None,
        jpeg_q=None) -> int:
    """Process entry point."""
    import cv2

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _stop = {"flag": False}

    def _on_sig(*_):
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    frame_bytes = cfg.width * cfg.height * 3
    ring = FrameRing.create(
        cfg.shm_name, n_slots=cfg.n_slots, frame_bytes=frame_bytes
    )
    log.info(
        "Thermal capture: shm=%s, %dx%d BGR, %d slots",
        cfg.shm_name, cfg.width, cfg.height, cfg.n_slots,
    )

    cap = _open_boson(cfg)
    if cap is None:
        ring.close_and_unlink()
        return 1

    nvjpeg = _try_load_nvjpeg(cfg.jpeg_quality)

    agc_cache: dict = {}
    frame_id = 0
    last_stats_emit = time.monotonic()
    fps_window: list = []

    try:
        while not _stop["flag"]:
            try:
                while True:
                    cmd = ctrl_q.get_nowait()
                    if cmd is None or cmd[0] == "shutdown":
                        _stop["flag"] = True
                        break
            except Exception:
                pass

            if _stop["flag"]:
                break

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.005)
                continue

            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            y8 = _apply_agc(
                gray, cfg.agc_low_pct, cfg.agc_high_pct,
                agc_cache, cfg.agc_stats_every,
            )

            heat_dets = _tophat_heat_detect(
                y8, cfg.heat_tophat_kernel,
                cfg.heat_min_area, cfg.heat_max_detections,
            )

            bgr = _apply_colormap(y8, cfg.colormap)

            # JPEG encode (cached so WS sender reuses these bytes).
            # Phase 2.3: hardware nvjpeg if available, else cv2.imencode.
            jpeg_bytes = b""
            if nvjpeg is not None:
                try:
                    jpeg_bytes = nvjpeg.encode_bgr(bgr)
                except Exception as e:
                    log.warning("nvjpeg encode failed (%r) — disabling", e)
                    nvjpeg = None
            if nvjpeg is None:
                ok, buf = cv2.imencode(
                    ".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality],
                )
                jpeg_bytes = bytes(buf) if ok else b""
            jpeg_size = len(jpeg_bytes)

            # Phase 2.5: hand the latest JPEG to the orchestrator.
            # Drop oldest on overflow — only the latest matters for the
            # GUI snapshot endpoint.
            if jpeg_q is not None and jpeg_size > 0:
                try:
                    jpeg_q.put_nowait((frame_id, jpeg_bytes,
                                       heat_dets))
                except Exception:
                    try:
                        jpeg_q.get_nowait()
                        jpeg_q.put_nowait((frame_id, jpeg_bytes,
                                           heat_dets))
                    except Exception:
                        pass

            with published_frame(
                ring, frame_id=frame_id,
                width=cfg.width, height=cfg.height, channels=3,
                dtype="uint8",
                meta={
                    "heat_dets": heat_dets,
                    "jpeg_size": jpeg_size,
                    "agc_alpha": agc_cache.get("alpha", 1.0),
                    "agc_beta": agc_cache.get("beta", 0.0),
                },
                extra_size=jpeg_size,
            ) as view:
                view[:] = bgr.tobytes()

            if det_q is not None:
                desc = FrameDescriptor(
                    frame_id=frame_id,
                    slot_idx=(ring._slot_idx if hasattr(ring, '_slot_idx') else 0),
                    mtime=time.monotonic(),
                    width=cfg.width, height=cfg.height, channels=3,
                    dtype="uint8",
                    meta={"heat_dets": heat_dets},
                )
                det_q.put(desc, drop_old=True)

            frame_id += 1
            fps_window.append(time.monotonic())
            cutoff = time.monotonic() - 5.0
            while fps_window and fps_window[0] < cutoff:
                fps_window.pop(0)

            now = time.monotonic()
            if now - last_stats_emit >= 1.0:
                last_stats_emit = now
                fps = len(fps_window) / 5.0 if fps_window else 0.0
                try:
                    stats_q.put_nowait({
                        "kind": "thermal_stats",
                        "frame_id": frame_id,
                        "fps_5s": fps,
                        "n_heat": len(heat_dets),
                    })
                except Exception:
                    pass

    except Exception:
        log.exception("Thermal capture loop died")
        return 2
    finally:
        try:
            cap.release()
        except Exception:
            pass
        ring.close_and_unlink()
        log.info("Thermal capture: clean shutdown")

    return 0


def spawn(mp_ctx, cfg: ThermalCaptureConfig, det_q=None):
    ctrl_q = mp_ctx.Queue(maxsize=32)
    stats_q = mp_ctx.Queue(maxsize=32)
    # Latest-JPEG queue: maxsize=2 so a stalled GUI doesn't backpressure
    # the capture loop. Phase 2.5.
    jpeg_q = mp_ctx.Queue(maxsize=2)
    proc = mp_ctx.Process(
        target=run, args=(cfg, ctrl_q, stats_q, det_q, jpeg_q),
        name="seeker_v2_thermal_capture", daemon=False,
    )
    proc.start()
    return proc, ctrl_q, stats_q, jpeg_q

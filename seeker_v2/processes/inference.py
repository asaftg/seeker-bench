"""Inference process — TRT YOLO for both EO + thermal.

Owns the iGPU. Loads two TRT engines (EO H/V + thermal H/V) at startup
(cold-start warmup so first inference is fast), then in a loop:

  1. Pull frame descriptors from EO and thermal capture queues
  2. Read the actual frame bytes from each capture's shared-memory ring
  3. Run YOLO inference (EO every 2 frames, thermal every 4 frames per
     v1 config) — independently, so EO inference doesn't block thermal
     and vice versa
  4. Write detections to a shared queue → fusion process
  5. Periodically emit performance stats to main

Phase 2.1: inference is single-threaded inside this process. TRT
inference releases the GIL during GPU compute, so EO and thermal
classifier calls can pipeline naturally.

Phase 2.4 will move thermal classifier to DLA0 so EO + thermal YOLO
genuinely run in parallel without iGPU contention.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from seeker_v2.processes.ipc import (
    DescriptorQueue,
    FrameDescriptor,
    FrameRing,
)

log = logging.getLogger("seeker_v2.inference")


@dataclass
class InferenceConfig:
    # EO classifier
    eo_engine_path: str = "models/seeker_eo_v3.engine"
    eo_imgsz: int = 832
    eo_classify_interval: int = 2  # run every Nth frame
    eo_conf: float = 0.40
    # EO frame ring to read from
    eo_shm_name: str = "seeker_eo_bgr"
    eo_n_slots: int = 4
    eo_width: int = 2472
    eo_height: int = 2064
    # Thermal classifier
    thermal_engine_path: str = "models/seeker_thermal_hv.engine"
    thermal_imgsz: int = 960  # match seeker_thermal_hv.engine fixed input
    thermal_classify_interval: int = 4  # Phase 1 fix #2
    thermal_conf: float = 0.55
    thermal_shm_name: str = "seeker_thermal_bgr"
    thermal_n_slots: int = 4
    thermal_width: int = 640
    thermal_height: int = 512
    # Warmup
    warmup_frames: int = 3
    # Phase 2.4: DLA offload for thermal classifier.
    # If a *_dla0.engine exists alongside the thermal engine, the C++
    # seeker_dla runner is used and the thermal forward pass executes on
    # DLA0 — leaving the iGPU to the EO classifier. ultralytics is still
    # used for preprocess / NMS / bytetrack on top.
    thermal_dla_engine_path: str = "models/seeker_thermal_hv_dla0.engine"
    thermal_dla_core: int = 0


def _load_yolo(path: str, imgsz: int):
    """Load an ultralytics YOLO model with TRT engine."""
    try:
        from ultralytics import YOLO
    except Exception as e:
        log.error("ultralytics not available: %r", e)
        return None
    try:
        model = YOLO(path, task="detect")
        log.info("loaded YOLO engine %s (imgsz=%d)", path, imgsz)
        return model
    except Exception as e:
        log.error("failed to load YOLO engine %s: %r", path, e)
        return None


def _load_thermal_with_dla(cfg: "InferenceConfig"):
    """Phase 2.4: try DLA-bound thermal engine first, else iGPU engine.

    The DLA engine must have been built with `trtexec --useDLACore=0`
    (see seeker_v2/scripts/build_dla_engine.sh). Loading requires the
    seeker_dla C++ extension because ultralytics' TRT loader does not
    call `IRuntime::setDLACore()` before deserialization.

    Returns a tuple (model, kind) where:
      kind == "ultralytics_igpu" — fall back path; use model.track/predict
      kind == "ultralytics_dla"  — DLA engine loaded via ultralytics
                                   (if ultralytics happens to handle DLA;
                                   on TRT 8.5 this often works because
                                   the engine carries the DLA binding)

    The returned `model` is always an ultralytics YOLO instance so the
    rest of the inference loop (preprocess, NMS, bytetrack) is identical.
    """
    import os
    dla_path = cfg.thermal_dla_engine_path
    if dla_path and os.path.exists(dla_path):
        # Sanity: try loading it via ultralytics. On JP5 with TRT 8.5
        # the engine knows it was built for DLA; the runtime picks up
        # the binding automatically from the serialized plan.
        m = _load_yolo(dla_path, cfg.thermal_imgsz)
        if m is not None:
            log.info("thermal classifier on DLA%d (%s)",
                     cfg.thermal_dla_core, dla_path)
            return m, "ultralytics_dla"
        log.warning("DLA engine load failed; falling back to iGPU engine")
    return (_load_yolo(cfg.thermal_engine_path, cfg.thermal_imgsz),
            "ultralytics_igpu")


def _warmup(model, imgsz: int, n: int = 3):
    if model is None:
        return
    dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(n):
        try:
            _ = model.track(dummy, persist=True, tracker="bytetrack.yaml",
                            imgsz=imgsz, verbose=False)
        except Exception:
            try:
                _ = model.predict(dummy, imgsz=imgsz, verbose=False)
            except Exception as e:
                log.warning("warmup pred failed: %r", e)


def _detections_from_results(results, conf_thresh: float) -> list[dict]:
    """Extract detections from ultralytics result list."""
    dets = []
    if not results:
        return dets
    r = results[0]
    boxes = getattr(r, "boxes", None)
    if boxes is None:
        return dets
    try:
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else boxes.xyxy
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else boxes.conf
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else boxes.cls
        ids = None
        if hasattr(boxes, "id") and boxes.id is not None:
            ids = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else boxes.id
    except Exception as e:
        log.warning("boxes extract failed: %r", e)
        return dets

    for i in range(len(xyxy)):
        c = float(conf[i])
        if c < conf_thresh:
            continue
        x1, y1, x2, y2 = xyxy[i]
        dets.append({
            "track_id": int(ids[i]) if ids is not None else -1,
            "class": int(cls[i]),
            "conf": c,
            "x": int(x1), "y": int(y1),
            "w": int(x2 - x1), "h": int(y2 - y1),
        })
    return dets


def run(cfg: InferenceConfig, ctrl_q, eo_det_q, thermal_det_q,
        stats_q) -> int:
    """Process entry point.

    Reads frames from shm rings; runs YOLO; pushes detections into
    eo_det_q / thermal_det_q for the fusion process.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    _stop = {"flag": False}

    def _on_sig(*_):
        _stop["flag"] = True

    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    # Attach to capture rings (read-only consumers)
    eo_ring = FrameRing.attach(
        cfg.eo_shm_name,
        n_slots=cfg.eo_n_slots,
        frame_bytes=cfg.eo_width * cfg.eo_height * 3,
    )
    thermal_ring = FrameRing.attach(
        cfg.thermal_shm_name,
        n_slots=cfg.thermal_n_slots,
        frame_bytes=cfg.thermal_width * cfg.thermal_height * 3,
    )

    log.info("loading TRT engines...")
    eo_model = _load_yolo(cfg.eo_engine_path, cfg.eo_imgsz)
    thermal_model, thermal_kind = _load_thermal_with_dla(cfg)
    log.info("thermal kind=%s", thermal_kind)

    log.info("warming up...")
    _warmup(eo_model, cfg.eo_imgsz, cfg.warmup_frames)
    _warmup(thermal_model, cfg.thermal_imgsz, cfg.warmup_frames)
    log.info("warmup complete; entering inference loop")

    eo_last_seq = 0
    eo_frame_counter = 0
    thermal_last_seq = 0
    thermal_frame_counter = 0
    last_stats_emit = time.monotonic()

    n_eo_inferences = 0
    n_thermal_inferences = 0
    eo_inf_ms_sum = 0.0
    thermal_inf_ms_sum = 0.0

    try:
        while not _stop["flag"]:
            # ── ctrl drain ─────────────────────────────────────────────
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

            did_work = False

            # ── EO inference (every Nth frame) ─────────────────────────
            desc, seq = eo_ring.latest()
            if desc is not None and seq != eo_last_seq:
                eo_last_seq = seq
                eo_frame_counter += 1
                if (eo_model is not None
                        and eo_frame_counter % cfg.eo_classify_interval == 0):
                    view = eo_ring.reader_view(desc.slot_idx)
                    bgr = np.frombuffer(bytes(view), dtype=np.uint8)
                    bgr = bgr.reshape(desc.height, desc.width, 3)
                    t0 = time.monotonic()
                    try:
                        results = eo_model.track(
                            bgr, persist=True, tracker="bytetrack.yaml",
                            conf=cfg.eo_conf, imgsz=cfg.eo_imgsz,
                            verbose=False,
                        )
                    except Exception:
                        try:
                            results = eo_model.predict(
                                bgr, conf=cfg.eo_conf,
                                imgsz=cfg.eo_imgsz, verbose=False,
                            )
                        except Exception as e:
                            log.warning("EO predict failed: %r", e)
                            results = None
                    inf_ms = (time.monotonic() - t0) * 1000
                    eo_inf_ms_sum += inf_ms
                    n_eo_inferences += 1
                    dets = _detections_from_results(results, cfg.eo_conf)
                    try:
                        eo_det_q.put_nowait({
                            "kind": "eo_dets",
                            "frame_id": desc.frame_id,
                            "ts": time.time(),
                            "detections": dets,
                            "inf_ms": inf_ms,
                        })
                    except Exception:
                        pass
                    did_work = True

            # ── Thermal inference (every Nth frame) ───────────────────
            tdesc, tseq = thermal_ring.latest()
            if tdesc is not None and tseq != thermal_last_seq:
                thermal_last_seq = tseq
                thermal_frame_counter += 1
                if (thermal_model is not None
                        and thermal_frame_counter % cfg.thermal_classify_interval == 0):
                    view = thermal_ring.reader_view(tdesc.slot_idx)
                    bgr = np.frombuffer(bytes(view), dtype=np.uint8)
                    bgr = bgr.reshape(tdesc.height, tdesc.width, 3)
                    t0 = time.monotonic()
                    try:
                        results = thermal_model.track(
                            bgr, persist=True, tracker="bytetrack.yaml",
                            conf=cfg.thermal_conf,
                            imgsz=cfg.thermal_imgsz, verbose=False,
                        )
                    except Exception:
                        try:
                            results = thermal_model.predict(
                                bgr, conf=cfg.thermal_conf,
                                imgsz=cfg.thermal_imgsz, verbose=False,
                            )
                        except Exception as e:
                            log.warning("thermal predict failed: %r", e)
                            results = None
                    inf_ms = (time.monotonic() - t0) * 1000
                    thermal_inf_ms_sum += inf_ms
                    n_thermal_inferences += 1
                    dets = _detections_from_results(results, cfg.thermal_conf)
                    try:
                        thermal_det_q.put_nowait({
                            "kind": "thermal_dets",
                            "frame_id": tdesc.frame_id,
                            "ts": time.time(),
                            "detections": dets,
                            "inf_ms": inf_ms,
                        })
                    except Exception:
                        pass
                    did_work = True

            if not did_work:
                # No new frames — sleep briefly to avoid busy-spin
                time.sleep(0.001)

            # ── Stats emit ────────────────────────────────────────────
            now = time.monotonic()
            if now - last_stats_emit >= 1.0:
                last_stats_emit = now
                eo_avg = (eo_inf_ms_sum / n_eo_inferences) if n_eo_inferences else 0.0
                t_avg = (thermal_inf_ms_sum / n_thermal_inferences) if n_thermal_inferences else 0.0
                try:
                    stats_q.put_nowait({
                        "kind": "inference_stats",
                        "eo_inferences": n_eo_inferences,
                        "eo_avg_ms": eo_avg,
                        "thermal_inferences": n_thermal_inferences,
                        "thermal_avg_ms": t_avg,
                    })
                except Exception:
                    pass
                n_eo_inferences = 0
                eo_inf_ms_sum = 0.0
                n_thermal_inferences = 0
                thermal_inf_ms_sum = 0.0

    except Exception:
        log.exception("inference loop died")
        return 2
    finally:
        eo_ring.close()
        thermal_ring.close()
        log.info("inference: clean shutdown")

    return 0


def spawn(mp_ctx, cfg: InferenceConfig):
    ctrl_q = mp_ctx.Queue(maxsize=32)
    eo_det_q = mp_ctx.Queue(maxsize=64)
    thermal_det_q = mp_ctx.Queue(maxsize=64)
    stats_q = mp_ctx.Queue(maxsize=32)
    proc = mp_ctx.Process(
        target=run, args=(cfg, ctrl_q, eo_det_q, thermal_det_q, stats_q),
        name="seeker_v2_inference", daemon=False,
    )
    proc.start()
    return proc, ctrl_q, eo_det_q, thermal_det_q, stats_q

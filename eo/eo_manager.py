"""EO pipeline orchestrator.

Mirrors ``thermal.thermal_manager.ThermalManager`` — two threads
(``EOCapture`` grabs frames, ``EOProcess`` runs YOLO) so slow inference
can't stall the camera path. Publishes ``EOFrame`` on ``Topic.EO``.

When EO is disabled in config or ``--no-eo`` is passed to ``main.py``,
EOManager is never instantiated and the thermal pipeline is byte-
identical to pre-Ticket-3 behaviour.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

import numpy as np

from common.config import load_config
from common.frame_bus import BUS
from common.frames import BBox, EODetection, EOFrame, TargetClass, Topic
from common.logging_setup import get_logger
from eo.eo_classifier import EOClassifier
from eo.fake_eo_source import FakeEOSource

log = get_logger(__name__)


class _CaptureLike(Protocol):
    raw16_available: bool
    device_index: Optional[int]
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(self) -> Optional[np.ndarray]: ...


# ── internal track struct for the persistence tracker ─────────────────
def _nms_same_class(dets: list[dict], iou_thresh: float = 0.45) -> list[dict]:
    """Greedy per-class NMS on raw YOLO detections.

    YOLO's own NMS sometimes leaves overlapping boxes when two anchors
    fire on the same object at different scales. One box per target
    matters here because downstream trackers key on IoU — a stray
    overlap becomes a duplicate track.
    """
    by_class: dict[str, list[dict]] = {}
    for d in dets:
        by_class.setdefault(d["class"], []).append(d)
    kept: list[dict] = []
    for cls, group in by_class.items():
        group.sort(key=lambda d: d["conf"], reverse=True)
        survivors: list[dict] = []
        for d in group:
            bx, by, bw, bh = d["bbox"]
            dup = False
            for s in survivors:
                sx, sy, sw, sh = s["bbox"]
                if _iou_xywh(sx, sy, sw, sh, bx, by, bw, bh) >= iou_thresh:
                    dup = True
                    break
            if not dup:
                survivors.append(d)
        kept.extend(survivors)
    return kept


def _iou_xywh(ax0, ay0, aw, ah, bx0, by0, bw, bh) -> float:
    ax1, ay1 = ax0 + aw, ay0 + ah
    bx1, by1 = bx0 + bw, by0 + bh
    ix0 = max(ax0, bx0); iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1); iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0); ih = max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(union) if union > 0 else 0.0


class EOManager:
    """Runs the whole EO pipeline behind two worker threads."""

    def __init__(
        self,
        use_fake: bool = False,
        device_index: int | str = "auto",
        enable_classifier: bool = True,
        reconnect_interval_s: float = 2.0,
        exclude_indices: Optional[list[int]] = None,
    ) -> None:
        self.use_fake = use_fake
        self.device_index = device_index
        self.enable_classifier = enable_classifier
        self.reconnect_interval_s = reconnect_interval_s
        self.exclude_indices = list(exclude_indices or [])

        cfg = load_config()
        ecfg = cfg.get("eo", {}) or {}
        ccfg = (ecfg.get("classifier") or {}) if isinstance(ecfg.get("classifier"), dict) else {}

        self._resolution = tuple(ecfg.get("resolution", [1280, 720]))
        self._target_fps = float(ecfg.get("target_fps", 30))
        self._hfov = float(ecfg.get("hfov_deg", 11.05))
        self._vfov = float(ecfg.get("vfov_deg", 9.23))

        # Test-webcam FOV override. The real IMX568 + 35mm is ~11° HFOV;
        # a generic webcam is ~60-70°. If fusion is told the wrong FOV,
        # pixel → angle math is wrong and cross-sensor association fails.
        _profile_name = str(ecfg.get("test_webcam", "none")).lower()
        _profiles = ecfg.get("test_webcam_profiles", {}) or {}
        if _profile_name and _profile_name != "none" and _profile_name in _profiles:
            p = _profiles[_profile_name] or {}
            ph = p.get("hfov_deg")
            pv = p.get("vfov_deg")
            if ph is not None and pv is not None:
                log.info(
                    "EO test_webcam='%s' overriding FOV %.2f°×%.2f° -> %.2f°×%.2f°",
                    _profile_name, self._hfov, self._vfov, float(ph), float(pv),
                )
                self._hfov = float(ph)
                self._vfov = float(pv)

        # Persistence tracker knobs — same pattern as the thermal HV tracker.
        self._conf_threshold = float(ccfg.get("conf_threshold", 0.40))
        self._min_bbox_px = int(ccfg.get("min_bbox_px", 900))  # ~30x30 px
        self._min_hits = int(ccfg.get("min_hits", 2))
        self._max_misses = int(ccfg.get("max_misses", 8))
        self._bbox_ema = float(ccfg.get("bbox_ema", 0.15))

        # classify_interval_frames: "auto" -> 1 on GPU, 4 on CPU. Webcams
        # typically run at 30 Hz (vs thermal's 60), so 4 still lands us
        # near 7-8 Hz inference which is enough for handheld targets.
        _raw_interval = ccfg.get("classify_interval_frames", "auto")
        if isinstance(_raw_interval, str) and _raw_interval.lower() == "auto":
            try:
                import torch
                _on_gpu = bool(torch.cuda.is_available())
            except Exception:
                _on_gpu = False
            # GPU: every-other-frame is fast enough AND keeps publish
            # rate close to webcam native. CPU: every 4th.
            self._classify_every = 2 if _on_gpu else 4
            log.info("eo classify_interval_frames=auto -> %d (%s)",
                     self._classify_every, "gpu" if _on_gpu else "cpu")
        else:
            self._classify_every = int(_raw_interval)

        self._classifier: Optional[EOClassifier] = None
        if enable_classifier and bool(ccfg.get("enabled", True)):
            # imgsz: "auto" -> 960 on GPU, 640 on CPU (parity with thermal HV)
            _raw_imgsz = ccfg.get("imgsz", "auto")
            if isinstance(_raw_imgsz, str) and _raw_imgsz.lower() == "auto":
                # EO frames are already rich RGB at 1920x1080 — 640 is
                # plenty for H/V detection at this FOV and keeps fps up.
                _imgsz = 640
            else:
                _imgsz = int(_raw_imgsz)
            try:
                self._classifier = EOClassifier(
                    fallback_model_path=str(ccfg.get("model", "models/yolov8n.pt")),
                    conf_threshold=self._conf_threshold,
                    imgsz=_imgsz,
                )
                log.info("EO classifier loaded (active=%s, imgsz=%d, conf=%.2f)",
                         self._classifier.active, _imgsz, self._conf_threshold)
            except Exception as e:
                log.warning("EO classifier init failed: %s", e)
                self._classifier = None

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._source: Optional[_CaptureLike] = None
        self._frame_id = 0
        self._tracks: list[dict] = []

        # Hand-off from capture to process thread (same pattern as thermal)
        self._latest_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq: int = 0

    # ───────────────────────── runtime device switch ─────────────────
    def set_device(self, new_index: int | str) -> None:
        """Reopen capture on a different cv2 device index at runtime.

        Used by the GUI's device-selector dropdown so the user can swap
        which physical camera drives the EO panel without restarting
        the whole process.
        """
        log.info("EO set_device -> %s (was %s)", new_index, self.device_index)
        self.device_index = new_index
        # Drop the current source; the capture loop will reopen on the
        # next iteration. This avoids cross-thread cv2 calls.
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass
            self._source = None

    # ───────────────────────── lifecycle ─────────────────────────────
    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="EOCapture", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, name="EOProcess", daemon=True
        )
        self._capture_thread.start()
        self._process_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._latest_cond:
            self._latest_cond.notify_all()
        for t in (self._process_thread, self._capture_thread):
            if t is not None:
                t.join(timeout=3.0)
        self._capture_thread = None
        self._process_thread = None
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass
            self._source = None

    # ───────────────────────── source wiring ─────────────────────────
    def _open_source(self) -> Optional[_CaptureLike]:
        if self.use_fake:
            src = FakeEOSource(
                width=int(self._resolution[0]),
                height=int(self._resolution[1]),
                fps=self._target_fps,
            )
            src.start()
            return src

        from eo.webcam_capture import WebcamCapture
        try:
            cap = WebcamCapture(
                device_index=self.device_index,
                width=int(self._resolution[0]),
                height=int(self._resolution[1]),
                exclude_indices=self.exclude_indices,
            )
            cap.start()
            return cap
        except RuntimeError as e:
            log.warning("EO source open failed: %s", e)
            return None

    # ───────────────────────── main loops ────────────────────────────
    def _capture_loop(self) -> None:
        log.info("EOManager capture thread starting (fake=%s)", self.use_fake)
        while not self._stop.is_set():
            if self._source is None:
                self._source = self._open_source()
                if self._source is None:
                    self._publish_disconnected()
                    self._stop.wait(self.reconnect_interval_s)
                    continue

            frame = self._source.grab()
            if frame is None:
                log.warning("EO grab returned None — treating as disconnect")
                self._publish_disconnected()
                try:
                    self._source.stop()
                except Exception:
                    pass
                self._source = None
                self._stop.wait(self.reconnect_interval_s)
                continue

            with self._latest_cond:
                self._latest_frame = frame
                self._latest_seq += 1
                self._latest_cond.notify_all()

        log.info("EOManager capture thread stopped")

    def _process_loop(self) -> None:
        last_seen_seq = 0
        while not self._stop.is_set():
            with self._latest_cond:
                while self._latest_seq == last_seen_seq and not self._stop.is_set():
                    self._latest_cond.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                frame = self._latest_frame
                last_seen_seq = self._latest_seq
            if frame is not None:
                try:
                    self._process_and_publish(frame)
                except Exception as e:
                    log.exception("EO process loop error: %s", e)
        log.info("EOManager process thread stopped")

    # ───────────────────────── pipeline ──────────────────────────────
    def _process_and_publish(self, frame: np.ndarray) -> None:
        self._frame_id += 1
        ts = time.time()

        # 1. YOLO (throttled). Tracks coast between runs so bboxes
        #    don't flicker off on non-classifier frames.
        run_classifier = (
            self._classifier is not None
            and (self._frame_id % self._classify_every == 0)
        )
        if run_classifier:
            try:
                raw = self._classifier.detect(frame)
            except Exception as e:
                log.warning("EO inference failed: %s", e)
                raw = []
            # Size gate drops tiny boxes (usually reflections / clutter).
            dets = [d for d in raw
                    if (d["bbox"][2] * d["bbox"][3]) >= self._min_bbox_px]
            # NMS within class — YOLO sometimes fires 2-3 overlapping
            # boxes on one vehicle. Without this, each overlap spawns a
            # duplicate track_id, which fusion then can't dedup.
            dets = _nms_same_class(dets, iou_thresh=0.45)
            self._update_tracks(dets)

        # 2. Build EODetection list from CONFIRMED tracks every frame
        #    so overlays hold steady between YOLO ticks.
        out_dets: list[EODetection] = []
        for trk in self._tracks:
            if trk["hits"] < self._min_hits:
                continue
            bx, by, bw, bh = trk["bbox"]
            try:
                tc = TargetClass(trk["class"])
            except ValueError:
                tc = TargetClass.UNKNOWN
            out_dets.append(EODetection(
                bbox=BBox(x=int(bx), y=int(by), w=int(bw), h=int(bh)),
                confidence=float(trk["conf"]),
                target_class=tc,
                track_id=int(trk["id"]),
            ))

        dev_idx = getattr(self._source, "device_index", None) if self._source else None
        ef = EOFrame(
            timestamp=ts,
            frame_id=self._frame_id,
            connected=True,
            bgr=frame,
            detections=out_dets,
            hfov_deg=self._hfov,
            vfov_deg=self._vfov,
            source_device=dev_idx,
        )
        BUS.publish(Topic.EO, ef)

    # ───────────────────────── persistence tracker ───────────────────
    def _update_tracks(self, dets: list[dict]) -> None:
        """Tiny IoU-matching tracker with hits/misses + bbox EMA.

        Mirrors the HV tracker in thermal_manager. Each detection either
        updates an existing track (IoU >= 0.3, same class) or spawns a
        new one. Unmatched tracks age out after max_misses.
        """
        IOU_MATCH = 0.30
        matched = [False] * len(self._tracks)
        for d in dets:
            bx, by, bw, bh = d["bbox"]
            best_t = -1
            best_iou = 0.0
            for ti, trk in enumerate(self._tracks):
                if matched[ti] or trk["class"] != d["class"]:
                    continue
                tb = trk["bbox"]
                iou = _iou_xywh(tb[0], tb[1], tb[2], tb[3], bx, by, bw, bh)
                if iou > best_iou:
                    best_iou = iou
                    best_t = ti
            if best_t >= 0 and best_iou >= IOU_MATCH:
                trk = self._tracks[best_t]
                a = self._bbox_ema
                trk["bbox"] = (
                    int(a * trk["bbox"][0] + (1 - a) * bx),
                    int(a * trk["bbox"][1] + (1 - a) * by),
                    int(a * trk["bbox"][2] + (1 - a) * bw),
                    int(a * trk["bbox"][3] + (1 - a) * bh),
                )
                trk["conf"] = float(d["conf"])
                trk["hits"] += 1
                trk["misses"] = 0
                matched[best_t] = True
            else:
                self._tracks.append({
                    "id": self._next_track_id(),
                    "bbox": (int(bx), int(by), int(bw), int(bh)),
                    "class": d["class"],
                    "conf": float(d["conf"]),
                    "hits": 1,
                    "misses": 0,
                })

        kept = []
        for ti, trk in enumerate(self._tracks):
            if ti < len(matched) and matched[ti]:
                kept.append(trk)
            else:
                trk["misses"] += 1
                if trk["misses"] <= self._max_misses:
                    kept.append(trk)
        self._tracks = kept

    _track_id_counter: int = 0
    def _next_track_id(self) -> int:
        self._track_id_counter += 1
        return self._track_id_counter

    # ───────────────────────── disconnected sentinel ─────────────────
    def _publish_disconnected(self) -> None:
        self._frame_id += 1
        ef = EOFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=False,
            hfov_deg=self._hfov,
            vfov_deg=self._vfov,
        )
        BUS.publish(Topic.EO, ef)

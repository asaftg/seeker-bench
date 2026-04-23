"""
Thermal pipeline orchestrator.

A single background thread that runs the full thermal pipeline:

    capture → AGC/colormap → heat detect → (optionally) classify
            → publish ThermalFrame on the FrameBus

The GUI backend subscribes to `Topic.THERMAL` and serves the
latest frame to the browser.

On capture failure (camera unplugged) the manager publishes a
`connected=False` sentinel frame and attempts to reconnect
every `reconnect_interval_s` seconds. Never crashes the process.
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Protocol

import cv2
import numpy as np

from common.config import load_config
from common.frame_bus import BUS
from common.frames import BBox, HeatTrackDebug, ThermalFrame, Topic
from common.logging_setup import get_logger
from thermal.classifier_hv import HumanVehicleClassifier
from thermal.detection_tracker import DetectionTracker, TrackerConfig
from thermal.digital_zoom import PRESETS as ZOOM_PRESETS, center_crop
from thermal.drone_classifier import Classifier
from thermal.fake_thermal_source import FakeThermalSource
from thermal.heat_detector import HeatDetector, HeatDetectorConfig
from thermal.thermal_processor import apply_agc, apply_colormap

log = get_logger(__name__)


class _CaptureLike(Protocol):
    raw16_available: bool
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def grab(self) -> Optional[np.ndarray]: ...


class ThermalManager:
    def __init__(
        self,
        use_fake: bool = False,
        device_index: int | str = "auto",
        enable_classifier: bool = True,
        reconnect_interval_s: float = 2.0,
    ) -> None:
        self.use_fake = use_fake
        self.device_index = device_index
        self.enable_classifier = enable_classifier
        self.reconnect_interval_s = reconnect_interval_s

        cfg = load_config()
        hdcfg = cfg.get("heat_detector", {})
        self._detector = HeatDetector(
            HeatDetectorConfig(
                threshold_k=float(hdcfg.get("threshold_k", 5.0)),
                background_kernel=int(hdcfg.get("background_kernel", 21)),
                min_blob_area_px=int(hdcfg.get("min_blob_area_px", 3)),
                max_blob_area_px=int(hdcfg.get("max_blob_area_px", 5000)),
                max_detections=int(hdcfg.get("max_detections_per_frame", 20)),
                algorithm=str(hdcfg.get("algorithm", "tophat")),
                tophat_kernel=int(hdcfg.get("tophat_kernel", 15)),
            )
        )

        trk_cfg = (hdcfg.get("tracker") or {})
        # OF bridge defaults to ON (see TrackerConfig docstring).
        # The bridge is gated on (a) a consecutive-bridge cap, (b) a
        # per-frame displacement cap, (c) a post-shift ROI warmth check,
        # and (d) the existing inlier-spread check. Previously-known
        # "ghost coasting purple box" behavior is fixed by those guards.
        _tcfg_defaults = TrackerConfig()
        self._tracker = DetectionTracker(
            TrackerConfig(
                enabled=bool(trk_cfg.get("enabled", True)),
                max_dist_px=float(trk_cfg.get("max_dist_px", 40.0)),
                min_hits=int(trk_cfg.get("min_hits", 5)),
                max_misses=int(trk_cfg.get("max_misses", 5)),
                ema=float(trk_cfg.get("ema", 0.5)),
                of_enabled=bool(trk_cfg.get("of_enabled", _tcfg_defaults.of_enabled)),
                max_of_bridges=int(trk_cfg.get("max_of_bridges", _tcfg_defaults.max_of_bridges)),
                of_min_warmth_contrast=float(trk_cfg.get("of_min_warmth_contrast", _tcfg_defaults.of_min_warmth_contrast)),
            )
        )

        self._classifier: Optional[Classifier] = None
        if enable_classifier:
            ccfg = cfg.get("classifier", {})
            try:
                self._classifier = Classifier(
                    enable_yolo=bool(ccfg.get("enabled", True)),
                    model_path=str(ccfg.get("model_path", "models/yolov8n.pt")),
                    trained_model_path=str(ccfg.get("trained_model_path", "models/seeker_thermal.pt")),
                    conf_threshold=float(ccfg.get("conf_threshold", 0.25)),
                    roi_padding_px=int(ccfg.get("roi_padding_px", 16)),
                    coco_to_target={int(k): v for k, v in (ccfg.get("class_to_target", ccfg.get("coco_to_target", {0: "drone"})) or {}).items()},
                )
            except Exception as e:  # defensive — classifier must never take down capture
                log.warning("Classifier init failed: %s", e)
                self._classifier = None

        # ── Phase B: human + vehicle classifier (Ticket 1) ────────────
        # Loaded only when classifier_hv_enabled: true in config.
        # When disabled the pipeline is byte-identical to Phase A.
        self._classifier_hv: Optional[HumanVehicleClassifier] = None
        ccfg = cfg.get("classifier", {})
        if enable_classifier and bool(ccfg.get("classifier_hv_enabled", False)):
            try:
                # imgsz: "auto" = 960 on GPU (better small-target range), 640 on CPU.
                _raw_imgsz = ccfg.get("classifier_hv_imgsz", "auto")
                if isinstance(_raw_imgsz, str) and _raw_imgsz.lower() == "auto":
                    try:
                        import torch
                        _hv_imgsz = 960 if torch.cuda.is_available() else 640
                    except Exception:
                        _hv_imgsz = 640
                else:
                    _hv_imgsz = int(_raw_imgsz)
                self._classifier_hv = HumanVehicleClassifier(
                    model_path=str(ccfg.get("classifier_hv_model", "models/seeker_thermal_hv.pt")),
                    conf_threshold=float(ccfg.get("classifier_hv_conf", 0.55)),
                    imgsz=_hv_imgsz,
                )
                self._hv_min_bbox_px = int(ccfg.get("classifier_hv_min_bbox_px", 2500))
                self._hv_min_hits = int(ccfg.get("classifier_hv_min_hits", 2))
                self._hv_max_misses = int(ccfg.get("classifier_hv_max_misses", 8))
                self._hv_bbox_ema = float(ccfg.get("classifier_hv_bbox_ema", 0.15))
                # Simple h/v persistence tracker: each entry is
                # {bbox, class, conf, hits, misses}. We IoU-match each new
                # YOLO det to the nearest cached track; a track must reach
                # `min_hits` before it's shown, and decays after `max_misses`
                # classifier runs without a matching detection.
                self._hv_tracks: list[dict] = []
                log.info(
                    "HV classifier loaded (active=%s, conf>=%s, min_bbox=%d, min_hits=%d)",
                    self._classifier_hv.active,
                    float(ccfg.get("classifier_hv_conf", 0.55)),
                    self._hv_min_bbox_px,
                    self._hv_min_hits,
                )
            except Exception as e:
                log.warning("HV classifier init failed: %s — h/v disabled", e)
                self._classifier_hv = None

        self._thcfg = cfg.get("thermal", {})
        # classify_interval_frames: "auto" adapts to hardware (1 on GPU, 6 on CPU).
        _raw_interval = (cfg.get("classifier", {}) or {}).get("classify_interval_frames", "auto")
        if isinstance(_raw_interval, str) and _raw_interval.lower() == "auto":
            try:
                import torch
                _on_gpu = bool(torch.cuda.is_available())
            except Exception:
                _on_gpu = False
            self._classify_every = 1 if _on_gpu else 6
            log.info("classify_interval_frames=auto -> %d (%s)",
                     self._classify_every, "gpu" if _on_gpu else "cpu")
        else:
            self._classify_every = int(_raw_interval)
        self._full_hfov = float(self._thcfg.get("hfov_deg", 75.0))
        self._full_vfov = float(self._thcfg.get("vfov_deg", 60.0))
        self._zoom_preset = str(self._thcfg.get("digital_zoom", {}).get("preset", "full"))

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._source: Optional[_CaptureLike] = None
        self._frame_id = 0
        self._last_classifications = None  # cache between classifier runs
        self._last_hv_dets: list = []      # cached h/v full-frame detections

        # Latest-frame handoff from capture thread to process thread.
        # The capture thread drains the camera as fast as it can and
        # overwrites `_latest_frame`; the process thread wakes on each
        # new sequence number and works on whatever is freshest. This
        # prevents slow YOLO/tophat work from stalling the camera.
        self._latest_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq: int = 0

    # ───────────────────────── runtime config ─────────────────────
    def set_zoom_preset(self, preset: str) -> bool:
        """Change the active digital-zoom preset at runtime.

        Returns True if accepted. Crops are applied on the next frame;
        detections produced after this point only see the cropped FOV,
        so blobs outside the narrowed FOV are not detected, counted,
        or rendered.

        Resets the temporal tracker because previous tracks are in a
        different coordinate space (different crop size) and would
        coast forever without matching anything new.
        """
        if preset not in ZOOM_PRESETS:
            return False
        self._zoom_preset = preset
        try:
            self._tracker.reset()
        except Exception:
            pass
        log.info("zoom preset -> %s (tracker reset)", preset)
        return True

    # ─────────────────── synthetic "Draw Target" ─────────────────
    def seed_synthetic_target(self, x: int, y: int, w: int, h: int) -> Optional[int]:
        """Seed a user-drawn bbox as a synthetic track in the tracker.

        Bbox is in display (post-zoom) coordinates — the same frame the
        GUI is drawing on. Returns the new track's ID or None if the
        bbox was rejected (too small / off-frame guarded at the GUI).
        """
        try:
            tid = self._tracker.seed_synthetic(
                BBox(x=int(x), y=int(y), w=int(w), h=int(h))
            )
            if tid is not None:
                log.info(
                    "ThermalManager seeded synthetic target id=%d bbox=(%d,%d,%d,%d)",
                    tid, x, y, w, h,
                )
            return tid
        except Exception as e:
            log.warning("seed_synthetic_target failed: %s", e)
            return None

    def clear_synthetic_target(self) -> int:
        """Remove any synthetic tracks. Returns count cleared."""
        try:
            return self._tracker.clear_synthetic()
        except Exception as e:
            log.warning("clear_synthetic_target failed: %s", e)
            return 0

    def set_device(self, new_index: int | str) -> None:
        """Reopen thermal capture on a different cv2 device index at runtime.

        Ticket 3: the GUI's camera-selector dropdown lets the user
        reassign which physical camera is Thermal vs EO without
        restarting the whole process. The capture loop will reopen on
        the next iteration (avoids cross-thread cv2 calls).
        """
        log.info("Thermal set_device -> %s (was %s)", new_index, self.device_index)
        self.device_index = new_index
        if self._source is not None:
            try:
                self._source.stop()
            except Exception:
                pass
            self._source = None
        try:
            self._tracker.reset()
        except Exception:
            pass

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        if self._capture_thread is not None:
            return
        self._stop.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="ThermalCapture", daemon=True
        )
        self._process_thread = threading.Thread(
            target=self._process_loop, name="ThermalProcess", daemon=True
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

    # ───────────────────────── main loop ─────────────────────────

    def _open_source(self) -> Optional[_CaptureLike]:
        if self.use_fake:
            src = FakeThermalSource(
                width=int(self._thcfg.get("resolution", [640, 512])[0]),
                height=int(self._thcfg.get("resolution", [640, 512])[1]),
                fps=float(self._thcfg.get("target_fps", 30)),
            )
            src.start()
            return src

        # Real hardware — import here so the rest of the manager
        # doesn't drag in cv2.VideoCapture at fake-mode import time
        from thermal.boson_capture import BosonCapture
        try:
            cap = BosonCapture(device_index=self.device_index)
            cap.start()
            return cap
        except RuntimeError as e:
            log.warning("Thermal source open failed: %s", e)
            return None

    def _capture_loop(self) -> None:
        """Tight grab loop. Nothing but reading frames from the camera.

        Any processing work (detect/classify/encode) happens on the
        process thread so it can NEVER stall the capture path. If the
        process thread falls behind, we simply overwrite `_latest_frame`
        and the next process tick sees the freshest frame — intermediate
        frames are dropped, which is the right behavior for real-time.
        """
        log.info("ThermalManager capture thread starting (fake=%s)", self.use_fake)
        while not self._stop.is_set():
            if self._source is None:
                self._source = self._open_source()
                if self._source is None:
                    self._publish_disconnected()
                    self._stop.wait(self.reconnect_interval_s)
                    continue

            frame = self._source.grab()
            if frame is None:
                log.warning("Thermal grab returned None — treating as disconnect")
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

        log.info("ThermalManager capture thread stopped")

    def _process_loop(self) -> None:
        """Consumes whatever the capture thread set as `_latest_frame`.

        Only runs when a NEW frame arrives (seq number changed). If
        processing is slower than capture, we silently drop intermediate
        frames — we only ever work on the freshest one.
        """
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
                    log.exception("process loop error: %s", e)
        log.info("ThermalManager process thread stopped")

    # ─────────────────────── pipeline stages ──────────────────────

    def _process_and_publish(self, frame: np.ndarray) -> None:
        self._frame_id += 1
        ts = time.time()
        orig_h, orig_w = frame.shape[:2]

        # ── 1. Derive raw16 + display from the FULL frame ──────────
        #    Detection always runs on the full sensor image so the
        #    adaptive threshold (MAD) sees the same noise statistics
        #    regardless of zoom level. This is the key insight: if we
        #    detect on a crop, the MAD changes with scene content and
        #    the slider feels different at every FOV.
        if frame.ndim == 2:
            raw16_full = frame.astype(np.uint16, copy=False)
            agc_lo = float(self._thcfg.get("agc", {}).get("low_percentile", 2))
            agc_hi = float(self._thcfg.get("agc", {}).get("high_percentile", 98))
            colormap_name = str(self._thcfg.get("agc", {}).get("colormap", "INFERNO"))
            agc = apply_agc(raw16_full, low_percentile=agc_lo, high_percentile=agc_hi)
            display_full = apply_colormap(agc, colormap_name)
        else:
            display_full = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            raw16_full = gray.astype(np.uint16, copy=False)

        # ── 2. Detect on the full frame ────────────────────────────
        detections = []
        try:
            detections = self._detector.detect(raw16_full)
        except Exception as e:
            log.warning("Heat detector failed: %s", e)

        # ── 3. Zoom: crop display + filter detections to crop ──────
        preset = self._zoom_preset
        if preset and preset != "full" and preset in ZOOM_PRESETS:
            from thermal.digital_zoom import crop_fraction as _cfrac
            target_hfov = ZOOM_PRESETS[preset].hfov_deg
            frac = _cfrac(self._full_hfov, target_hfov)
            cw = max(1, int(round(orig_w * frac)))
            ch = max(1, int(round(orig_h * frac)))
            cx0 = (orig_w - cw) // 2
            cy0 = (orig_h - ch) // 2
            hfov_cur = target_hfov
            vfov_cur = self._full_vfov * (target_hfov / self._full_hfov)

            # Filter: keep only detections whose center is inside crop
            cropped = []
            for d in detections:
                bcx = d.bbox.x + d.bbox.w * 0.5
                bcy = d.bbox.y + d.bbox.h * 0.5
                if cx0 <= bcx < cx0 + cw and cy0 <= bcy < cy0 + ch:
                    # Translate bbox to crop-relative coords
                    d.bbox = BBox(
                        x=max(0, d.bbox.x - cx0),
                        y=max(0, d.bbox.y - cy0),
                        w=d.bbox.w,
                        h=d.bbox.h,
                    )
                    cropped.append(d)
            detections = cropped

            # Crop + upscale display to original dims
            display = cv2.resize(
                display_full[cy0:cy0 + ch, cx0:cx0 + cw],
                (orig_w, orig_h),
                interpolation=cv2.INTER_LINEAR,
            )
            # Scale bbox coords from crop space → display space
            sx = orig_w / float(cw)
            sy = orig_h / float(ch)
            for d in detections:
                b = d.bbox
                d.bbox = BBox(
                    x=int(round(b.x * sx)),
                    y=int(round(b.y * sy)),
                    w=int(round(b.w * sx)),
                    h=int(round(b.h * sy)),
                )
            raw16 = raw16_full  # keep full for ThermalFrame
        else:
            display = display_full
            raw16 = raw16_full
            hfov_cur = self._full_hfov
            vfov_cur = self._full_vfov

        # ── 4. Temporal tracker ────────────────────────────────────
        # Tracker needs a grayscale view of the display image for its
        # Lucas-Kanade optical-flow bridge (works in bbox/display space,
        # so the crop-scaled `display` is the correct coordinate frame).
        try:
            display_gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY) \
                if display is not None and display.ndim == 3 else display
        except Exception:
            display_gray = None
        try:
            detections = self._tracker.update(detections, display_gray)
        except Exception as e:
            log.warning("DetectionTracker failed: %s", e)

        # Dev-mode snapshot of the tracker's internal state (all tracks,
        # including unconfirmed + coasting). Cheap enough to compute
        # every frame; the GUI only renders when developer mode is on.
        try:
            _snaps = self._tracker.snapshot()
            heat_tracks_debug = [
                HeatTrackDebug(
                    id=s.id, bbox=s.bbox,
                    hits=s.hits, misses=s.misses, age=s.age,
                    confirmed=s.confirmed, coasting=s.coasting,
                )
                for s in _snaps
            ]
        except Exception:
            heat_tracks_debug = []

        # ── 5. Classification (throttled) ──────────────────────────
        run_classifiers = (self._frame_id % self._classify_every == 0)

        # 5a. Drone classifier — ROI-based, only meaningful when heat
        # detector produced blobs (drones are tiny hot points at range).
        if run_classifiers and detections and self._classifier is not None:
            try:
                drone_results = self._classifier.classify(display, detections)
                for det, res in zip(detections, drone_results):
                    det.classification = res
                self._last_classifications = drone_results
            except Exception as e:
                log.warning("Drone classifier failed: %s", e)

        # 5b. H/V classifier (Phase B Ticket 1) — FULL-FRAME mode.
        # Heat-blob ROIs are unreliable for people (warm bodies fragment
        # into scattered spots) and useless for cars (often cold). So we
        # run YOLO on the whole display frame at `classify_interval_frames`
        # rate, feed results into a tiny persistence tracker, and merge
        # CONFIRMED tracks into the detections list every frame.
        if self._classifier_hv is not None:
            from common.frames import (
                BBox as _BBox,
                ClassificationResult,
                TargetClass,
                ThermalDetection,
            )

            def _iou_xywh(ax0, ay0, aw, ah, bx0, by0, bw, bh):
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

            # Only poll YOLO on classifier ticks. The hand-rolled
            # IoU+EMA tracker that used to live here was ripped out in
            # favour of ByteTrack (Ultralytics' model.track(persist=True)).
            # ByteTrack's Kalman filter coasts each object across missed
            # frames and associates low-confidence detections to existing
            # tracks, so fast pans / motion blur no longer respawn a new
            # ID every tick. Between classifier runs we republish the
            # last detection list; ByteTrack state lives inside the
            # ultralytics model and updates on the next call.
            if run_classifiers:
                try:
                    raw_hv = self._classifier_hv.track_full_frame(display)
                except Exception as e:
                    log.warning("HV full-frame step failed: %s", e)
                    raw_hv = []

                # Size gate: drop sub-threshold boxes (warm chips, hand patches).
                hv_dets = [
                    d for d in raw_hv
                    if (d["bbox"][2] * d["bbox"][3]) >= self._hv_min_bbox_px
                ]
                self._last_hv_dets = hv_dets

                # ByteTrack owns track IDs + persistence now. Rebuild
                # the `_hv_tracks` shape the downstream merge code
                # expects — one entry per ByteTrack ID present this
                # tick, hits=min_hits so confirmation logic below still
                # fires. Unconfirmed tracks (track_id=-1 on first frame)
                # are dropped so we don't flicker labels.
                self._hv_tracks = [
                    {
                        "id": int(d["track_id"]),
                        "bbox": tuple(int(v) for v in d["bbox"]),
                        "class": d["class"],
                        "conf": float(d["conf"]),
                        "hits": self._hv_min_hits,     # ByteTrack already confirmed
                        "misses": 0,
                    }
                    for d in hv_dets
                    if int(d.get("track_id", -1)) >= 0
                ]

            # Merge CONFIRMED h/v tracks into detections on EVERY frame
            # (not just classifier ticks) so overlays don't flicker.
            IOU_MERGE = 0.30
            for trk in self._hv_tracks:
                if trk["hits"] < self._hv_min_hits:
                    continue
                bx, by, bw, bh = trk["bbox"]
                try:
                    tc = TargetClass(trk["class"])
                except ValueError:
                    tc = TargetClass.UNKNOWN
                hv_conf = trk["conf"]

                best_i = -1
                best_iou = 0.0
                for i, det in enumerate(detections):
                    iou = _iou_xywh(
                        det.bbox.x, det.bbox.y, det.bbox.w, det.bbox.h,
                        bx, by, bw, bh,
                    )
                    if iou > best_iou:
                        best_iou = iou
                        best_i = i

                if best_i >= 0 and best_iou >= IOU_MERGE:
                    det = detections[best_i]
                    drone_conf = (
                        det.classification.confidence
                        if det.classification is not None
                        else 0.0
                    )
                    if hv_conf > drone_conf:
                        det.classification = ClassificationResult(
                            target_class=tc,
                            confidence=hv_conf,
                            classifier_used="yolo_hv",
                        )
                else:
                    detections.append(ThermalDetection(
                        bbox=_BBox(x=int(bx), y=int(by), w=int(bw), h=int(bh)),
                        area_px=int(bw * bh),
                        contrast=0.0,
                        classification=ClassificationResult(
                            target_class=tc,
                            confidence=hv_conf,
                            classifier_used="yolo_hv",
                        ),
                    ))

        # ── 6. Publish ─────────────────────────────────────────────
        tf = ThermalFrame(
            timestamp=ts,
            frame_id=self._frame_id,
            connected=True,
            raw16=raw16,
            agc8=display,
            detections=detections,
            hfov_deg=hfov_cur,
            vfov_deg=vfov_cur,
            zoom_preset=self._zoom_preset,
            heat_tracks=heat_tracks_debug,
        )
        BUS.publish(Topic.THERMAL, tf)

    def _publish_disconnected(self) -> None:
        self._frame_id += 1
        tf = ThermalFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=False,
        )
        BUS.publish(Topic.THERMAL, tf)

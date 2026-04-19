# seeker_bench — Repository Inventory

_Generated 2026-04-19_

---

## 1. File Tree (3 levels deep)

```
seeker_bench/
├── CLAUDE.md
├── README.md
├── START_SEEKER.bat
├── build_exe.bat
├── main.py
├── requirements.txt
├── seeker_bench.spec
├── yolo26n.pt
├── yolov8n.pt
│
├── build/
│   └── seeker_bench/
│       ├── Analysis-00.toc
│       ├── COLLECT-00.toc
│       ├── EXE-00.toc
│       ├── PKG-00.toc
│       ├── PYZ-00.pyz
│       ├── PYZ-00.toc
│       ├── base_library.zip
│       ├── localpycs/
│       ├── seeker_bench.exe
│       ├── seeker_bench.pkg
│       ├── warn-seeker_bench.txt
│       └── xref-seeker_bench.html
│
├── common/
│   ├── __init__.py
│   ├── config.py
│   ├── frame_bus.py
│   ├── frames.py
│   ├── logging_setup.py
│   └── tests/
│       ├── __init__.py
│       ├── test_frame_bus.py
│       └── test_frames.py
│
├── config/
│   └── app_config.yaml
│
├── datasets/
│   ├── .gitkeep
│   └── thermal_drone/
│       ├── drone_detection_example.png
│       ├── raw/
│       └── thermal_drone_raw.zip      ← (zip contents not shown)
│
├── dist/
│   └── seeker_bench/
│       ├── _internal/
│       └── seeker_bench.exe
│
├── gui/
│   ├── __init__.py
│   ├── app.py
│   ├── sensor_bridge.py
│   └── static/
│       ├── index.html
│       ├── css/
│       └── js/
│
├── logs/
│   ├── .gitkeep
│   ├── seeker.log
│   ├── seeker.log.1
│   ├── seeker.log.2
│   └── seeker.log.3
│
├── models/                            ← (model file contents not shown)
│   └── .gitkeep
│
├── radar/
│   └── __init__.py
│
├── runs/
│   └── detect/
│       └── runs/
│
├── scripts/
│   └── smoke_test.py
│
└── thermal/
    ├── __init__.py
    ├── __main__.py
    ├── boson_capture.py
    ├── detection_tracker.py
    ├── digital_zoom.py
    ├── drone_classifier.py
    ├── fake_thermal_source.py
    ├── heat_detector.py
    ├── thermal_manager.py
    ├── thermal_processor.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_classifier_fallback.py
    │   ├── test_digital_zoom.py
    │   ├── test_heat_detector.py
    │   └── test_thermal_processor.py
    └── training/
        ├── __init__.py
        ├── dataset.py
        ├── label_tool_readme.md
        ├── promote_model.py
        ├── record_for_training.py
        └── train.py
```

---

## 2. Python Modules — Public Classes and Top-Level Functions

> Test files and empty `__init__.py` files are skipped.

### `main.py`
| Kind | Signature |
|------|-----------|
| function | `parse_args() -> argparse.Namespace` |
| function | `main() -> int` |

---

### `common/config.py`
| Kind | Signature |
|------|-----------|
| function | `load_config(reload: bool = False) -> Dict[str, Any]` |

---

### `common/frame_bus.py`
| Kind | Signature |
|------|-----------|
| class | `FrameBus` |
| method | `FrameBus.publish(self, topic: str, frame: Any) -> None` |
| method | `FrameBus.get_latest(self, topic: str) -> Optional[Any]` |
| method | `FrameBus.wait_new(self, topic: str, timeout: Optional[float] = None) -> bool` |
| method | `FrameBus.clear(self, topic: str) -> None` |
| module-level | `BUS = FrameBus()` — singleton |

---

### `common/frames.py`
| Kind | Signature |
|------|-----------|
| class | `TargetClass(str, Enum)` — values: `UNKNOWN`, `HAND`, `DRONE`, `BIRD`, `NOISE` |
| dataclass | `BBox(x: int, y: int, w: int, h: int)` |
| method | `BBox.as_tuple(self) -> Tuple[int, int, int, int]` |
| dataclass | `ClassificationResult(target_class: TargetClass, confidence: float, classifier_used: str)` |
| dataclass | `ThermalDetection(bbox: BBox, area_px: int, contrast: float, classification: Optional[ClassificationResult] = None)` |
| dataclass | `ThermalFrame(timestamp: float, frame_id: int, connected: bool, raw16: Optional[np.ndarray] = None, agc8: Optional[np.ndarray] = None, detections: List[ThermalDetection] = ..., hfov_deg: float = 75.0, vfov_deg: float = 60.0, zoom_preset: str = "full")` |
| dataclass | `RadarFrame(timestamp: float, frame_id: int, connected: bool, detections: list = ...)` |
| class | `Topic` — constants: `THERMAL`, `RADAR`, `FUSED` |

---

### `common/logging_setup.py`
| Kind | Signature |
|------|-----------|
| function | `configure(level: str = "INFO", log_dir: Optional[str] = None, max_bytes: int = 5_000_000, backup_count: int = 3) -> None` |
| function | `get_logger(name: str) -> logging.Logger` |

---

### `gui/app.py`
| Kind | Signature |
|------|-----------|
| function | `create_app(thermal_manager=None) -> FastAPI` |

Routes registered inside `create_app`:
- `GET /` — serves `index.html`
- `GET /health` — returns `{"status": "ok"}`
- `GET /api/config/heat_detector` — returns current detector config
- `POST /api/config/thermal` — accepts `{"zoom_preset": str}`
- `POST /api/config/heat_detector` — accepts `{"threshold_k", "min_blob_area_px", "max_detections"}`
- `WS /ws/sensors` — streams sensor JSON frames

---

### `gui/sensor_bridge.py`
| Kind | Signature |
|------|-----------|
| function | `thermal_to_wire(tf: Optional[ThermalFrame], jpeg_quality: int = 80) -> Dict[str, Any]` |
| function | `radar_to_wire() -> Dict[str, Any]` |
| function | `fusion_to_wire() -> Dict[str, Any]` |

---

### `thermal/boson_capture.py`
| Kind | Signature |
|------|-----------|
| class | `BosonCapture(device_index: int \| str = "auto", width: int = 640, height: int = 512, prefer_raw16: bool = True)` |
| method | `BosonCapture.start(self) -> None` |
| method | `BosonCapture.stop(self) -> None` |
| method | `BosonCapture.grab(self) -> Optional[np.ndarray]` |
| method | `BosonCapture.is_open(self) -> bool` |
| function | `probe(duration_s: float = 2.0) -> Tuple[int, float]` |

---

### `thermal/detection_tracker.py`
| Kind | Signature |
|------|-----------|
| dataclass | `TrackerConfig(enabled: bool = True, max_dist_px: float = 40.0, min_hits: int = 5, max_misses: int = 5, ema: float = 0.5)` |
| class | `DetectionTracker(config: TrackerConfig \| None = None)` |
| method | `DetectionTracker.reset(self) -> None` |
| method | `DetectionTracker.update(self, detections: List[ThermalDetection]) -> List[ThermalDetection]` |

---

### `thermal/digital_zoom.py`
| Kind | Signature |
|------|-----------|
| dataclass | `ZoomPreset(name: str, hfov_deg: float)` — frozen |
| module-level | `PRESETS: Dict[str, ZoomPreset]` — keys: `full`, `wide`, `mid`, `narrow` |
| function | `crop_fraction(full_hfov_deg: float, target_hfov_deg: float) -> float` |
| function | `center_crop(frame: np.ndarray, full_hfov_deg: float, target_hfov_deg: float, upscale_to_original: bool = True) -> np.ndarray` |
| function | `apply_preset(frame: np.ndarray, full_hfov_deg: float, preset: str) -> np.ndarray` |

---

### `thermal/drone_classifier.py`
| Kind | Signature |
|------|-----------|
| function | `classify_by_shape(det: ThermalDetection) -> ClassificationResult` |
| class | `Classifier(enable_yolo: bool = True, model_path: str = "models/yolov8n.pt", trained_model_path: str = "models/seeker_thermal.pt", conf_threshold: float = 0.25, roi_padding_px: int = 16, coco_to_target: Optional[dict] = None)` |
| property | `Classifier.yolo_active -> bool` |
| method | `Classifier.classify(self, display_bgr: np.ndarray, detections: List[ThermalDetection]) -> List[Optional[ClassificationResult]]` |

---

### `thermal/fake_thermal_source.py`
| Kind | Signature |
|------|-----------|
| class | `FakeThermalSource(width: int = 640, height: int = 512, fps: float = 30.0, num_blobs: int = 1, seed: int = 42)` |
| method | `FakeThermalSource.start(self) -> None` |
| method | `FakeThermalSource.stop(self) -> None` |
| method | `FakeThermalSource.is_open(self) -> bool` |
| method | `FakeThermalSource.grab(self) -> Optional[np.ndarray]` |

---

### `thermal/heat_detector.py`
| Kind | Signature |
|------|-----------|
| dataclass | `HeatDetectorConfig(threshold_k: float = 5.0, background_kernel: int = 21, min_blob_area_px: int = 3, max_blob_area_px: int = 5000, max_detections: int = 20, algorithm: str = "tophat", tophat_kernel: int = 15)` |
| class | `HeatDetector(config: HeatDetectorConfig \| None = None)` |
| method | `HeatDetector.detect(self, frame_u16: np.ndarray) -> List[ThermalDetection]` |

---

### `thermal/thermal_manager.py`
| Kind | Signature |
|------|-----------|
| class | `ThermalManager(use_fake: bool = False, device_index: int \| str = "auto", enable_classifier: bool = True, reconnect_interval_s: float = 2.0)` |
| method | `ThermalManager.set_zoom_preset(self, preset: str) -> bool` |
| method | `ThermalManager.start(self) -> None` |
| method | `ThermalManager.stop(self) -> None` |

---

### `thermal/thermal_processor.py`
| Kind | Signature |
|------|-----------|
| function | `apply_agc(frame_u16: np.ndarray, low_percentile: float = 2.0, high_percentile: float = 98.0) -> np.ndarray` |
| function | `apply_colormap(frame_u8: np.ndarray, name: str = "INFERNO") -> np.ndarray` |
| function | `raw16_to_display(frame_u16: np.ndarray, colormap: str = "INFERNO", low_percentile: float = 2.0, high_percentile: float = 98.0) -> Tuple[np.ndarray, np.ndarray]` |

---

### `thermal/training/dataset.py`
| Kind | Signature |
|------|-----------|
| module-level | `TRAIN_CLASSES: List[TargetClass]` — `[DRONE, HAND, BIRD]` |
| function | `class_names() -> List[str]` |
| function | `class_index(cls: TargetClass) -> int` |
| dataclass | `DatasetLayout(root: Path, name: str = "seeker_thermal")` |
| function | `create(root: Path, name: str = "seeker_thermal") -> DatasetLayout` |
| function | `write_data_yaml(layout: DatasetLayout) -> None` |
| function | `split_unlabeled(layout: DatasetLayout, val_fraction: float = 0.2, seed: int = 0) -> Tuple[int, int]` |

---

### `thermal/training/train.py`
| Kind | Signature |
|------|-----------|
| function | `main() -> int` |

---

### `thermal/training/promote_model.py`
| Kind | Signature |
|------|-----------|
| function | `main() -> int` |

---

### `thermal/training/record_for_training.py`
| Kind | Signature |
|------|-----------|
| function | `main() -> int` |

---

### `scripts/smoke_test.py`
| Kind | Signature |
|------|-----------|
| function | `check(label: str, condition: bool, detail: str = "") -> None` |
| function | `main() -> int` |

---

## 3. Full Content of `config/app_config.yaml`

```yaml
# ───────────────────────────────────────────────────────────
# Seeker-01 Bench Test — Application Config
# ───────────────────────────────────────────────────────────
# Every tunable lives here. Code modules should never hardcode
# thresholds or paths; they read from this file via common.config.
# Changing a value here and restarting the app should be enough.

thermal:
  # Camera
  device_index: auto            # auto | 0 | 1 | 2 | 3
  resolution: [640, 512]        # FLIR Boson 640
  mode: RAW16                   # RAW16 (Y16) | AGC8 (YUY2)
  target_fps: 60

  # FOV (FLIR ADK 40640U075 is 75°H × ~60°V)
  hfov_deg: 75.0
  vfov_deg: 60.0

  # AGC
  agc:
    low_percentile: 2
    high_percentile: 98
    colormap: INFERNO           # INFERNO | IRONBOW | WHITE_HOT

  # Digital zoom presets (emulated narrow FOV)
  digital_zoom:
    enabled: true
    preset: full                # full | wide | mid | narrow | tight
    presets:
      full:   { hfov_deg: 75.0 }
      wide:   { hfov_deg: 37.5 }
      mid:    { hfov_deg: 18.75 }
      narrow: { hfov_deg: 12.5 }

heat_detector:
  # residual = frame - spatial_mean(frame)
  # threshold = median(residual) + k * 1.4826 * MAD(residual)
  # Defaults tuned for an INDOOR 8-bit AGC'd scene — raise threshold_k
  # or lower min_blob_area_px once you're pointing at sky backgrounds.
  threshold_k: 8.0              # lower = more sensitive (x MAD)
  background_kernel: 31         # spatial background kernel (odd)
  min_blob_area_px: 80          # ignores noise specks (< ~9x9 px)
  max_blob_area_px: 30000       # allow large targets (hot cup at narrow zoom)
  max_detections_per_frame: 6
  algorithm: tophat
  tophat_kernel: 31             # max target diameter in pixels (odd; bigger = catches larger objects)

  # Temporal tracker — suppresses flicker, smooths bboxes.
  # A blob must survive `min_hits` frames before a box is drawn;
  # a confirmed blob survives up to `max_misses` frames of missing
  # detections before being dropped.
  tracker:
    enabled: true
    max_dist_px: 40             # max centroid distance (px) to match across frames
    min_hits: 5                 # consecutive frames before a box is shown
    max_misses: 5               # grace frames after a track stops matching
    ema: 0.5                    # bbox smoothing: 0 = raw, 1 = frozen

classifier:
  enabled: true
  model_path: models/yolov8n.pt
  trained_model_path: models/seeker_thermal.pt   # preferred if exists
  conf_threshold: 0.40          # higher = fewer false DRONE labels; lower = catches faint drones
  classify_interval_frames: 15  # YOLO runs every Nth frame; tracker keeps labels sticky between runs
  roi_padding_px: 16
  # Model class-index → TargetClass mapping.
  # For the fine-tuned seeker_thermal.pt trained on thermal drone
  # data: class 0 = drone. Stock COCO weights are not loaded.
  class_to_target:
    0: drone

radar:
  enabled: false                # Phase A: radar disconnected placeholder
  # real radar params added in Phase B

fusion:
  enabled: false                # Phase A: no fusion

gui:
  host: 127.0.0.1
  port: 8080
  open_browser: true
  ws_fps: 60                    # GUI push rate; actual fps is min(ws_fps, camera_fps, encode_budget)
  thermal_jpeg_quality: 80

recording:
  output_dir: ./recordings
  format: placeholder           # Phase A: stub; HDF5 added later

logging:
  level: INFO
  log_dir: ./logs
  max_bytes: 5242880
  backup_count: 3
```

---

## 4. WebSocket Message Fields

All browser-bound messages are JSON objects sent on `WS /ws/sensors`. The top-level envelope contains three keys assembled in `gui/app.py` → `sensors()` handler and serialized by `gui/sensor_bridge.py`.

### Top-level envelope

```json
{
  "thermal": { ... },
  "radar":   { ... },
  "fusion":  { ... }
}
```

---

### `thermal` object — `thermal_to_wire()` in `gui/sensor_bridge.py`

**When camera is disconnected** (`tf is None` or `tf.connected == False`):

| Field | Type | Notes |
|-------|------|-------|
| `connected` | `false` | sentinel for GUI status pill |
| `frame_id` | `int` | `0` if `tf` is `None`, else `tf.frame_id` |
| `timestamp` | `float` | `0.0` if `tf` is `None`, else `tf.timestamp` |
| `jpeg_b64` | `null` | no image |
| `width` | `0` | |
| `height` | `0` | |
| `hfov_deg` | `75.0` | hardcoded fallback |
| `vfov_deg` | `60.0` | hardcoded fallback |
| `zoom_preset` | `"full"` | hardcoded fallback |
| `detections` | `[]` | empty list |

**When camera is connected**:

| Field | Type | Notes |
|-------|------|-------|
| `connected` | `true` | |
| `frame_id` | `int` | monotonically incrementing |
| `timestamp` | `float` | `time.time()` wall clock |
| `jpeg_b64` | `string \| null` | base64-encoded JPEG of the AGC display image; `null` if `agc8` is `None` |
| `width` | `int` | image width in pixels |
| `height` | `int` | image height in pixels |
| `hfov_deg` | `float` | current horizontal FOV (changes with zoom preset) |
| `vfov_deg` | `float` | current vertical FOV |
| `zoom_preset` | `string` | active zoom preset name (e.g. `"full"`, `"mid"`) |
| `detections` | `array` | list of detection objects (see below) |

**Each detection object** inside `detections[]`:

| Field | Type | Notes |
|-------|------|-------|
| `bbox.x` | `int` | top-left x, display-space pixels |
| `bbox.y` | `int` | top-left y |
| `bbox.w` | `int` | width |
| `bbox.h` | `int` | height |
| `area_px` | `int` | blob area in pixels |
| `contrast` | `float` | peak residual above background, 1 decimal place |
| `classification` | `object \| null` | `null` if unclassified |
| `classification.target_class` | `string` | `"unknown"`, `"hand"`, `"drone"`, `"bird"`, `"noise"` |
| `classification.confidence` | `float` | 0–1, 3 decimal places |
| `classification.classifier_used` | `string` | `"yolo"`, `"shape_heuristic"`, or `"none"` |

---

### `radar` object — `radar_to_wire()` in `gui/sensor_bridge.py`

Phase A stub; always the same shape:

| Field | Type | Value |
|-------|------|-------|
| `connected` | `false` | always disconnected in Phase A |
| `detections` | `[]` | always empty in Phase A |

---

### `fusion` object — `fusion_to_wire()` in `gui/sensor_bridge.py`

Phase A stub; always the same shape:

| Field | Type | Value |
|-------|------|-------|
| `active` | `false` | always inactive in Phase A |
| `tracks` | `[]` | always empty in Phase A |

---

## 5. Config Keys Read at Startup

All access is via `cfg.get(...)` on the dict returned by `common.config.load_config()`.

### Top-level section keys accessed

| Section key | Accessed in |
|-------------|-------------|
| `"logging"` | `main.py` |
| `"gui"` | `main.py`, `gui/app.py` |
| `"heat_detector"` | `thermal/thermal_manager.py` |
| `"classifier"` | `thermal/thermal_manager.py` |
| `"thermal"` | `thermal/thermal_manager.py` |

---

### `logging` sub-keys (via `log_cfg = cfg.get("logging", {})`)

| Key | Module | Default |
|-----|--------|---------|
| `logging.level` | `main.py` | `"INFO"` |
| `logging.log_dir` | `main.py` | `None` |
| `logging.max_bytes` | `main.py` | `5_000_000` |
| `logging.backup_count` | `main.py` | `3` |

---

### `gui` sub-keys (via `cfg.get("gui", {})`)

| Key | Module | Default |
|-----|--------|---------|
| `gui.host` | `main.py` | `"127.0.0.1"` |
| `gui.port` | `main.py` | `8080` |
| `gui.open_browser` | `main.py` | `True` |
| `gui.ws_fps` | `gui/app.py` | `20` |
| `gui.thermal_jpeg_quality` | `gui/app.py` | `80` |

---

### `heat_detector` sub-keys (via `hdcfg = cfg.get("heat_detector", {})`)

| Key | Module | Default |
|-----|--------|---------|
| `heat_detector.threshold_k` | `thermal/thermal_manager.py` | `5.0` |
| `heat_detector.background_kernel` | `thermal/thermal_manager.py` | `21` |
| `heat_detector.min_blob_area_px` | `thermal/thermal_manager.py` | `3` |
| `heat_detector.max_blob_area_px` | `thermal/thermal_manager.py` | `5000` |
| `heat_detector.max_detections_per_frame` | `thermal/thermal_manager.py` | `20` |
| `heat_detector.algorithm` | `thermal/thermal_manager.py` | `"tophat"` |
| `heat_detector.tophat_kernel` | `thermal/thermal_manager.py` | `15` |
| `heat_detector.tracker` | `thermal/thermal_manager.py` | `{}` |
| `heat_detector.tracker.enabled` | `thermal/thermal_manager.py` | `True` |
| `heat_detector.tracker.max_dist_px` | `thermal/thermal_manager.py` | `40.0` |
| `heat_detector.tracker.min_hits` | `thermal/thermal_manager.py` | `5` |
| `heat_detector.tracker.max_misses` | `thermal/thermal_manager.py` | `5` |
| `heat_detector.tracker.ema` | `thermal/thermal_manager.py` | `0.5` |

---

### `classifier` sub-keys (via `ccfg = cfg.get("classifier", {})`)

| Key | Module | Default |
|-----|--------|---------|
| `classifier.enabled` | `thermal/thermal_manager.py` | `True` |
| `classifier.model_path` | `thermal/thermal_manager.py` | `"models/yolov8n.pt"` |
| `classifier.trained_model_path` | `thermal/thermal_manager.py` | `"models/seeker_thermal.pt"` |
| `classifier.conf_threshold` | `thermal/thermal_manager.py` | `0.25` |
| `classifier.roi_padding_px` | `thermal/thermal_manager.py` | `16` |
| `classifier.class_to_target` | `thermal/thermal_manager.py` | `{0: "drone"}` |
| `classifier.coco_to_target` | `thermal/thermal_manager.py` | fallback alias for `class_to_target` |
| `classifier.classify_interval_frames` | `thermal/thermal_manager.py` | `5` |

---

### `thermal` sub-keys (via `self._thcfg = cfg.get("thermal", {})`)

| Key | Module | Default |
|-----|--------|---------|
| `thermal.resolution` | `thermal/thermal_manager.py` | `[640, 512]` |
| `thermal.target_fps` | `thermal/thermal_manager.py` | `30` |
| `thermal.hfov_deg` | `thermal/thermal_manager.py` | `75.0` |
| `thermal.vfov_deg` | `thermal/thermal_manager.py` | `60.0` |
| `thermal.digital_zoom.preset` | `thermal/thermal_manager.py` | `"full"` |
| `thermal.agc.low_percentile` | `thermal/thermal_manager.py` | `2` |
| `thermal.agc.high_percentile` | `thermal/thermal_manager.py` | `98` |
| `thermal.agc.colormap` | `thermal/thermal_manager.py` | `"INFERNO"` |

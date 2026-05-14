"""EO pipeline orchestrator.

Mirrors ``thermal.thermal_manager.ThermalManager`` — two threads
(``EOCapture`` grabs frames, ``EOProcess`` runs YOLO) so slow inference
can't stall the camera path. Publishes ``EOFrame`` on ``Topic.EO``.

When EO is disabled in config or ``--no-eo`` is passed to ``main.py``,
EOManager is never instantiated and the thermal pipeline is byte-
identical to pre-Ticket-3 behaviour.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np

from common.config import load_config
from common.frame_bus import BUS
from common.frames import BBox, EODetection, EOFrame, GimbalState, TargetClass, Topic
from common.logging_setup import get_logger
from eo.eo_classifier import EOClassifier
from eo.eo_processor import enhance, passthrough, scene_mean
from eo.eo_profiles import ProfileSelector, profiles_from_config
from eo.fake_eo_source import FakeEOSource

log = get_logger(__name__)


# Firmware-accepted ExposureExt range (verified empirically 2026-04-25).
# The FX3 bridge accepts ExposureExt as low as 1 (we don't really know
# below that); 50000 is well above any sane scene we've tested.
_AE_EXP_MIN = 1
_AE_EXP_MAX = 50000

# Where we cache the last successfully-converged ExposureExt across
# sessions. Same lens + similar lighting → next launch starts with the
# right value and skips the 30-55s SDK-helper-restart cascade entirely.
# Stored next to the config so it travels with the project; reset by
# deleting the file. Single-int JSON keeps the format trivially
# debuggable / hand-editable.
_AE_CACHE_PATH = Path(__file__).resolve().parents[1] / "logs" / "eo_ae_last.json"


def _load_cached_ae_exposure() -> Optional[int]:
    """Return last-converged ExposureExt from disk, or None if absent."""
    try:
        if not _AE_CACHE_PATH.exists():
            return None
        with open(_AE_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        v = int(data.get("exposure_ext"))
        if _AE_EXP_MIN <= v <= _AE_EXP_MAX:
            return v
        return None
    except Exception:
        return None


def _save_cached_ae_exposure(value: int) -> None:
    """Best-effort persist last-converged ExposureExt. Silent on failure
    — caching is an optimization, never a correctness requirement."""
    try:
        _AE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AE_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({"exposure_ext": int(value),
                       "saved_at": time.time()}, f)
    except Exception as e:
        log.debug("EO AE cache write failed: %r", e)


class _AEBracketController:
    """Bracket-based AE controller. Validated in
    ``scripts/sdk_ae_loop.py`` (converged 1264→9 in 8 iters on cloudy
    daylight saturated frame, 2026-04-25).

    Maintains two brackets — ``low_floor`` (largest exp known to be too
    dim) and ``high_brake`` (smallest exp known to clip or be too bright).
    Each step either tightens the bracket (binary search), or, when one
    bracket is missing, takes a coarse multiplicative move (×2 / ÷2)
    in the right direction.

    Use proportional control? No — empirically the IMX568 + 35 mm NIR
    has regimes where 4× exposure produces 12× p99. Proportional moves
    overshoot the analog cliff and oscillate between "all black" and
    "all white". Bracket binary-search converges deterministically even
    on extreme non-linearity.
    """

    __slots__ = ("target_lo", "target_hi", "low_floor", "high_brake")

    def __init__(self, target_lo: float, target_hi: float) -> None:
        self.target_lo = float(target_lo)
        self.target_hi = float(target_hi)
        self.low_floor: Optional[int] = None
        self.high_brake: Optional[int] = None

    def reset_brackets(self) -> None:
        """Forget what we've learned about this scene. Call when the
        scene clearly changed (camera moved, big light flip, manual
        override toggled off after a long pause)."""
        self.low_floor = None
        self.high_brake = None

    def in_band(self, p99: float, frac_clip: float) -> bool:
        return (self.target_lo <= p99 <= self.target_hi
                and frac_clip < 0.01)

    def step(self, exp: int, p99: float, mean: float,
             frac_clip: float) -> int:
        """Return the next ExposureExt to try given the latest stats."""
        is_sat = (frac_clip > 0.01) or (p99 >= 4090)
        is_dim = (p99 < self.target_lo) and not is_sat

        # Capture bracket state BEFORE we update it. The aggressive
        # first-step logic below needs to know "is this our very first
        # call?" — and that's only true when neither bracket has been
        # set yet on entry to this method.
        pre_step_no_brackets = (self.low_floor is None
                                and self.high_brake is None)

        if is_sat:
            if self.high_brake is None or exp < self.high_brake:
                self.high_brake = exp
        elif is_dim:
            if self.low_floor is None or exp > self.low_floor:
                self.low_floor = exp
        elif p99 > self.target_hi:
            if self.high_brake is None or exp < self.high_brake:
                self.high_brake = exp

        # Both brackets known → binary search inside them.
        if self.low_floor is not None and self.high_brake is not None:
            if self.high_brake - self.low_floor <= 1:
                # Bracket collapsed; accept low_floor (better dim than
                # saturated). The displayed image is AGC-stretched
                # downstream so dim-but-uncliped frames look fine.
                return self.low_floor
            new = (self.low_floor + self.high_brake) // 2
            if new == exp:
                new = exp - 1 if is_sat else exp + 1
            return max(_AE_EXP_MIN, min(_AE_EXP_MAX, new))

        # Coarse moves when only one bracket is known.
        if is_sat:
            # Aggressive ONE-SHOT first-step: when this is the literal
            # first call (no bracket info either side) AND every pixel
            # is pegged (frac_clip ≈ 1.0), the scene is at least 100x
            # too bright. Halving wastes 4-5 SDK helper restarts (~7s
            # each); a single /16 jumps right past the early descent.
            #
            # CRITICAL: this is gated on `pre_step_no_brackets` (state
            # captured at the very top of step() before we updated
            # high_brake), NOT just `low_floor is None`. After the
            # first jump high_brake is set but low_floor remains None,
            # and we must NOT /16 again — the second /16 would
            # overshoot into dim territory and force the controller
            # into an unnecessary binary search recovery (verified
            # in simulation: chained /16 turned an 8-step descent
            # into an 8-step descent, no win).
            if pre_step_no_brackets:
                if frac_clip >= 0.99:
                    return max(_AE_EXP_MIN, exp // 16)
                if frac_clip >= 0.50:
                    return max(_AE_EXP_MIN, exp // 4)
            return max(_AE_EXP_MIN, exp // 2)
        if p99 < 50:           # near noise floor
            return min(_AE_EXP_MAX, max(_AE_EXP_MIN, exp * 4))
        if is_dim:
            return min(_AE_EXP_MAX, max(_AE_EXP_MIN, exp * 2))
        if p99 > self.target_hi:
            return max(_AE_EXP_MIN, exp // 2)
        return exp


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
        # SDK stream frame delivery rate (imx568 backend only). Was hard-
        # coded to 10 fps — visibly choppy on moving vehicles. Default 20.
        self._stream_fps = float(ecfg.get("stream_fps", 20.0))
        self._hfov = float(ecfg.get("hfov_deg", 11.05))
        self._vfov = float(ecfg.get("vfov_deg", 9.23))

        # Backend selector: "imx568" → real hardware path (native 2472×2064,
        # mono, percentile AGC, profile-switched exposure). "webcam" → the
        # existing generic UVC path for dev laptops. Default to webcam for
        # back-compat with configs written before the IMX568 was online.
        self._sensor_backend = str(ecfg.get("sensor", "webcam")).lower()

        # Profile selector & AGC — only meaningful for the IMX568 backend.
        # Builds three profiles from config; scene-brightness-driven auto-
        # selection with hysteresis picks between them. Illuminator is
        # manually operator-controlled; we never toggle hardware.
        day, dusk, night = profiles_from_config(ecfg)
        self._profile_selector = ProfileSelector(
            day=day, dusk=dusk, night=night,
            min_dwell_s=float(ecfg.get("profile_min_dwell_s", 3.0)),
        )
        # AGC + enhancement chain config. Default = OFF (passthrough): the
        # FX3 bridge's own auto-exposure has already metered the scene;
        # piling our own gamma/CLAHE/unsharp on top of an 8-bit-from-12-bit
        # mono frame in low light produces nightmare artifacts. Any stage
        # you want, you opt into via YAML.
        agc_cfg = (ecfg.get("agc") or {}) if isinstance(ecfg.get("agc"), dict) else {}
        self._agc_enabled = bool(agc_cfg.get("enabled", False))
        self._agc_low_pct = float(agc_cfg.get("low_percentile", 0.5))
        self._agc_high_pct = float(agc_cfg.get("high_percentile", 99.5))

        ecfg_enh = (ecfg.get("enhance") or {}) if isinstance(ecfg.get("enhance"), dict) else {}
        denoise_cfg = (ecfg_enh.get("denoise") or {}) if isinstance(ecfg_enh.get("denoise"), dict) else {}
        self._enh_denoise_ksize = int(denoise_cfg.get("ksize", 3)) if bool(denoise_cfg.get("enabled", False)) else 0
        self._enh_gamma = float(ecfg_enh.get("gamma", 1.0))
        clahe_cfg = (ecfg_enh.get("clahe") or {}) if isinstance(ecfg_enh.get("clahe"), dict) else {}
        self._enh_clahe_clip = float(clahe_cfg.get("clip_limit", 1.5)) if bool(clahe_cfg.get("enabled", False)) else 0.0
        self._enh_clahe_grid = int(clahe_cfg.get("tile_grid", 8))
        unsh_cfg = (ecfg_enh.get("unsharp") or {}) if isinstance(ecfg_enh.get("unsharp"), dict) else {}
        self._enh_unsharp_amount = float(unsh_cfg.get("amount", 0.4)) if bool(unsh_cfg.get("enabled", False)) else 0.0
        self._enh_unsharp_radius = float(unsh_cfg.get("radius", 1.0))

        # Does the profile selector actually command the sensor's exposure?
        # On the LI-IMX568-GMSL2, UVC exposure writes are silently ignored
        # by the FX3 bridge firmware (verified: CameraTool's own exposure
        # slider has no effect, and our manual-exposure writes produced
        # frame-1-good-then-all-black). Default OFF: keep the bridge in
        # its own auto-exposure mode, let AGC handle display dynamic range,
        # and log the scene_mean / recommended-profile for engineering
        # visibility only. Flip to True once Leopard ships a firmware that
        # honors UVC exposure — we don't have to change any other code.
        self._profile_control_exposure = bool(
            ecfg.get("profile_control_exposure", False)
        )

        # Bridge AE lock — the user observed the static-scene image
        # "breathing" between frames when bridge AE is left to wander.
        # Set eo.manual_exposure_log2 to a UVC-style log2(seconds) value
        # (e.g. -6 ≈ 16 ms, -7 ≈ 8 ms, -5 ≈ 32 ms) and IMX568Capture
        # will write CAP_PROP_AUTO_EXPOSURE=manual + CAP_PROP_EXPOSURE
        # via cv2-DSHOW before PyAV opens the device. Set to null/None
        # to leave bridge AE running free.
        _mexp = ecfg.get("manual_exposure_log2", None)
        self._manual_exposure_log2 = (
            float(_mexp) if _mexp is not None else None
        )
        _mgain = ecfg.get("manual_gain", None)
        self._manual_gain = (
            float(_mgain) if _mgain is not None else None
        )
        # PRIMARY exposure-lock path on the IMX568/FX3 rig: Leopard SDK
        # ExposureExt vendor command. Set eo.manual_exposure_ext to an
        # int (e.g. 1000) and IMX568Capture will spawn the 32-bit Leopard
        # SDK helper subprocess to write ExposureExt + AE=off before
        # PyAV opens. Empirically this produces a rock-stable mean across
        # 30 s on a static scene; see scripts/eo_probe_breathing_at_fixed_expext.py.
        _mexp_ext = ecfg.get("manual_exposure_ext", None)
        self._manual_exposure_ext = (
            int(_mexp_ext) if _mexp_ext is not None else None
        )

        # Downscale native capture before AGC/YOLO/encode. 5 MP full-res
        # native is overkill for every downstream stage; cutting width in
        # half quarters the per-frame pixel work everywhere.
        self._display_max_width = int(ecfg.get("display_max_width", 1236))

        # Digital zoom (1 / 2 / 4 / 8). At 1×, the pipeline behaves
        # exactly as before: tiled SAHI on the full native frame.
        # At 2×, we center-crop native to 1/2 width and the classifier
        # still tiles that crop (wide-ish FOV + sharper pixels).
        # At 4× / 8×, the crop is small enough that tiling no longer
        # helps — we feed the crop straight to YOLO at imgsz=832
        # (single inference, narrow FOV, biggest model-px-on-target).
        # Display + recording always get the cropped+upscaled view so
        # operator and offline analysis see what the classifier saw.
        # Controlled via /api/config/eo POST {"zoom_level": N}.
        self._zoom_level: int = 1

        # JPEG-encode the final BGR frame ONCE on this thread, attach to
        # EOFrame.jpeg_bytes, so the asyncio WS sender doesn't pay the
        # cv2.imencode + base64 cost on every tick. Same quality knob
        # the GUI WS endpoint reads at gui/app.py — keep them in sync.
        gui_cfg = cfg.get("gui", {}) or {}
        self._eo_jpeg_quality = int(gui_cfg.get("eo_jpeg_quality", 82))

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
            # Per-class conf overrides — see app_config.yaml `classes_conf`.
            # When set, conf_threshold becomes the FLOOR (model receives
            # min(thresholds) at predict time) and per-class filtering
            # happens after. Lets us keep drone permissive (small hard
            # targets) while keeping person/vehicle strict (FP-prone on
            # streetlights/poles).
            _classes_conf = ccfg.get("classes_conf") or {}
            # Per-class HI override (new, optional). Default = same as
            # global conf_hi for all classes.
            _classes_conf_hi = ccfg.get("classes_conf_hi") or {}
            # Schmitt gate defaults (mutable at runtime via /api/config/eo_schmitt).
            _schmitt_cfg = ccfg.get("schmitt") or {}
            self._track_cfg = {
                "conf_hi":     float(_schmitt_cfg.get("conf_hi",     0.55)),
                "conf_lo":     float(_schmitt_cfg.get("conf_lo",     0.25)),
                "k_persist":   int(_schmitt_cfg.get("k_persist",   4)),
                "window":      int(_schmitt_cfg.get("window",      20)),
                "coast_ticks": int(_schmitt_cfg.get("coast_ticks", 3)),
            }
            # Per-class conf_hi map (HI is what triggers instant-pass).
            # Defaults to global conf_hi if not specified per-class.
            self._classes_conf_hi = {
                "person":  float(_classes_conf_hi.get("person",  self._track_cfg["conf_hi"])),
                "vehicle": float(_classes_conf_hi.get("vehicle", self._track_cfg["conf_hi"])),
                "drone":   float(_classes_conf_hi.get("drone",   self._track_cfg["conf_hi"])),
            }
            # Schmitt state (per-track accumulator). Mutable in-place
            # from the classifier worker — initialized here so the lazy
            # init in _classifier_loop is no longer needed.
            self._track_state: dict = {}
            self._track_tick: int = 0
            # Tiled (SAHI) inference config. See vision/sahi_inference.py.
            _tiling_cfg = ccfg.get("tiling") or {}
            self._tiling_enabled: bool = bool(_tiling_cfg.get("enabled", False))
            self._tiling_grid: tuple[int, int] = tuple(
                _tiling_cfg.get("grid", [2, 2])
            )
            self._tiling_overlap_frac: float = float(
                _tiling_cfg.get("overlap_frac", 0.25)
            )
            self._tiling_merge_iou: float = float(
                _tiling_cfg.get("merge_iou", 0.5)
            )
            try:
                self._classifier = EOClassifier(
                    fallback_model_path=str(ccfg.get("model", "models/yolov8n.pt")),
                    conf_threshold=self._conf_threshold,
                    imgsz=_imgsz,
                    per_class_conf=_classes_conf if _classes_conf else None,
                )
                log.info(
                    "EO classifier loaded (active=%s, imgsz=%d, conf=%.2f, "
                    "per_class_conf=%s)",
                    self._classifier.active, _imgsz, self._conf_threshold,
                    _classes_conf or "(none)",
                )
            except Exception as e:
                log.warning("EO classifier init failed: %s", e)
                self._classifier = None

        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._cls_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._source: Optional[_CaptureLike] = None
        self._frame_id = 0
        # Last ByteTrack detection list; republished every frame between
        # classifier ticks so overlays don't flicker.
        self._last_dets: list[dict] = []

        # MOSSE correlation tracker pool — runs at frame rate to keep
        # bboxes alive between YOLO ticks. ByteTrack's Kalman coast
        # only handles motion; MOSSE handles APPEARANCE-based tracking,
        # i.e. when YOLO's confidence dies due to motion blur during a
        # gimbal slew but the target is still visibly there. See
        # vision/mosse_tracker.py for algorithm. Disabled by default
        # while the integration matures; flip the YAML knob to enable.
        from vision.correlation_tracker_set import (
            CorrelationTrackerSet, CorrelationTrackerSetConfig)
        ct_cfg = (cfg.get("eo", {}).get("correlation_tracker") or {})
        self._mosse_pool = CorrelationTrackerSet(
            CorrelationTrackerSetConfig(
                enabled=bool(ct_cfg.get("enabled", False)),
                psr_lost=float(ct_cfg.get("psr_lost", 7.0)),
                lost_frames=int(ct_cfg.get("lost_frames", 5)),
                learning_rate=float(ct_cfg.get("learning_rate", 0.125)),
                sigma=float(ct_cfg.get("sigma", 2.0)),
                max_patch_dim=int(ct_cfg.get("max_patch_dim", 96)),
            )
        )

        # Optical-pose-feedback state. Used to detect when the published
        # gimbal pose advances WITHOUT the camera physically moving (lazy
        # servo deadband ate the PWM, or a stuck servo). Phase correlation
        # between consecutive downsampled grayscale EO frames measures
        # the actual image translation. When BUS-reported pose changed
        # by Δp_bus but the image only shifted Δp_optical, and the two
        # disagree by more than a small threshold, we trust the optics
        # and suppress the BUS update for the stamped pose-at-capture.
        # That keeps fusion's world-frame conversion stable through
        # phantom dpad clicks even when the maestro driver thinks it
        # successfully wrote a step.
        self._prev_frame_small: Optional[np.ndarray] = None
        self._prev_bus_pan: Optional[float] = None
        self._prev_bus_tilt: Optional[float] = None
        # Trusted pose: anchored to BUS but updated only when image
        # motion confirms the change. None until the first frame with
        # a valid GIMBAL bus message; falls back to BUS until anchored.
        self._trusted_pan: Optional[float] = None
        self._trusted_tilt: Optional[float] = None
        # Width of the downsampled phase-correlation window. Small for
        # speed; sub-pixel precision via Hanning window keeps it sensitive.
        self._of_target_w: int = 192  # 2026-05-14: 256->192 (44% fewer px, ~1.3ms saved per phaseCorrelate)
        # Phase-correlation cache for slew compensation. Single source of
        # truth: publish thread computes, classifier worker consumes
        # accumulator. State-gated (only runs when any track exists) so
        # cold-start empty-scene case pays zero compute. See plan doc
        # "EO bbox correctness + DEV-tab tuning UI".
        import threading as _threading
        self._pc_dx_disp_accum: float = 0.0  # writer: publish, reader+zero: classifier
        self._pc_dy_disp_accum: float = 0.0
        self._pc_last_dx_tick: float = 0.0   # writer+reader: publish thread only
        self._pc_last_dy_tick: float = 0.0
        self._pc_cache_lock = _threading.Lock()
        # 2026-05-14: separate raw-BUS tracker for bbox-shift derivation.
        # Independent of _prev_bus_pan (which gets suppressed by
        # _trusted_pan when phase-corr can't confirm motion — a feedback
        # loop that locks _prev_bus_pan when phase-corr undermeasures).
        self._pc_bus_prev_pan: Optional[float] = None
        self._pc_bus_prev_tilt: Optional[float] = None
        # Thresholds for "BUS reported motion not confirmed by optics".
        # If BUS says we moved >= 0.3° in either axis but optics say <0.1°,
        # we trust optics. 0.3° is the smallest BUS step under the gate
        # bump (~1.0°), with margin.
        self._bus_motion_min_deg: float = 0.3
        self._optical_confirm_deg: float = 0.1

        # Hand-off from capture to process thread (same pattern as thermal)
        self._latest_cond = threading.Condition()
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_seq: int = 0

        # Async classify hand-off. Process thread submits the latest
        # frame to the classifier worker thread (single-slot — newer
        # overwrites older if the worker is still busy on the previous
        # one), worker runs YOLO + ByteTrack and posts results back.
        # Decouples ~35 ms of YOLO inference from the publish path so
        # an occasional inference spike can't stall the EO frame rate.
        # ByteTrack `persist=True` is robust to skipped frames via its
        # Kalman predictor — verified by today's classify_every=2 path
        # which already feeds it every other frame.
        self._cls_in_lock = threading.Lock()
        self._cls_in_cond = threading.Condition(self._cls_in_lock)
        self._cls_in_pending: Optional[tuple[int, np.ndarray]] = None
        self._cls_out_lock = threading.Lock()
        self._cls_out_dets: list[dict] = []
        # One-shot signal: True the first publish-tick that reads a
        # fresh classifier result (worker just finished). Process
        # thread consumes/clears it. Lets the MOSSE pool distinguish
        # "reseed on YOLO ground truth" from "interpolate between
        # ticks" without burning a sequence number.
        self._cls_out_dets_fresh_flag: bool = False
        self._cls_out_fid: int = -1

        # Source lifecycle lock. Held during set_device / set_exposure_ext
        # so the capture thread can't reopen mid-switch and spawn a
        # second SDK helper that fights the first one for the camera.
        # See the bug report 2026-04-24: two parallel helpers locked the
        # device hard enough that only a USB replug recovered.
        self._source_lock = threading.Lock()
        # Set when an exposure change is in flight. Capture loop checks
        # this and pauses its reopen path until the change settles.
        self._switching_exposure = False

        # ── Software auto-exposure (Auto mode) ──
        #
        # The FX3 bridge's own AE (cam.AE = True) is broken on this rig
        # in bright daylight: it leaves the IMX568 fully analog-clipped
        # through the 35 mm NIR-pass lens and never recovers. Verified
        # 2026-04-25 with scripts/sdk_daylight_diagnostic.py — every
        # capture at default settings was raw u16 mean ≈ 4095 (full
        # analog ceiling). Even Leopard's own CameraTool produced a
        # pure-white frame on the same scene.
        #
        # So we run our own AE in software. When the user is in "Auto"
        # mode (manual_exposure_ext is None), this thread polls the
        # source's last_raw_stats and steps a bracket-based controller
        # toward p99 ∈ [target_lo, target_hi]. The chosen ExposureExt
        # lives in _ae_chosen_ext; the capture loop's source-open
        # routine reads (manual_exposure_ext or _ae_chosen_ext) when
        # spawning the helper. User-facing manual mode wins — the AE
        # thread no-ops while manual_exposure_ext is set.
        #
        # Convergence cost: each AE step requires restarting the SDK
        # helper subprocess (ExposureExt is committed at helper init).
        # Restart costs ~1-2 s of stream blackout, so we cap iterations
        # and accept we'll see a few flickers during initial AE on a
        # fresh scene. Once converged, the controller idles unless
        # stats drift outside the target band — same scene = no churn.
        ae_cfg = (ecfg.get("auto_exposure") or {}) if isinstance(ecfg.get("auto_exposure"), dict) else {}
        self._ae_enabled = bool(ae_cfg.get("enabled", True))
        # Acceptable raw u16 p99 band. Default [300, 3500] empirically
        # works for the daylight (cliff at p99≈4095, exp~5-9) AND
        # nighttime (with NIR torch, p99 climbs into hundreds at exp
        # ~1000-3000) regimes seen 2026-04-25.
        self._ae_target_lo = float(ae_cfg.get("target_p99_lo", 300.0))
        self._ae_target_hi = float(ae_cfg.get("target_p99_hi", 3500.0))
        # Initial seed when no prior AE state — what to try on first
        # frame after entering Auto mode. 1264 is the historical
        # reference exposure (well-lit indoor scene); the bracket
        # controller will rapidly cut from there if daylight, or hold
        # if scene is already in band, or grow if night.
        #
        # On startup we PREFER the last-converged value persisted from
        # a prior session (logs/eo_ae_last.json). Reasoning: the same
        # rig + same lens + similar lighting almost always converges to
        # within a 2x band of the previous session, so seeding with the
        # last-known-good value lets AE skip 4-7 SDK helper restarts
        # and reach in-band on the very first frame instead of after
        # ~30-55s of cascading restarts. If the cache is stale (lens
        # cap on, indoor → outdoor shift, etc.) the bracket controller
        # corrects in 1-2 normal steps anyway, so the worst-case is
        # parity with cold start, never worse.
        cfg_initial = int(ae_cfg.get("initial_exposure_ext", 1264))
        cached = _load_cached_ae_exposure()
        if cached is not None:
            log.info("EO AE: seeding from cached last-converged "
                     "exposure_ext=%d (cfg fallback was %d)",
                     cached, cfg_initial)
            self._ae_initial_ext = cached
        else:
            self._ae_initial_ext = cfg_initial
        self._ae_chosen_ext: Optional[int] = self._ae_initial_ext if self._ae_enabled else None
        self._ae_controller: Optional[_AEBracketController] = None
        self._ae_thread: Optional[threading.Thread] = None
        # Latched "AE has converged at least once this session" flag.
        # The controller's bracket state alone is not a good proxy for
        # convergence — a clean run from saturated → in-band can leave
        # low_floor=None forever (the AE never observed a "dim but not
        # saturated" reading because the binary search jumped past it).
        # We use this latch to drive the GUI's INITIALIZING scrim:
        # True the moment we first see in_band, and only reset on user-
        # initiated mode change (manual override toggled off after a
        # long pause, or set_device).
        self._ae_has_converged_once: bool = False
        # How long the AE thread sleeps between adjustments. Shorter =
        # more responsive but more stream blackouts during convergence.
        self._ae_step_period_s = float(ae_cfg.get("step_period_s", 1.5))
        # After we've converged once, only re-engage if frac_clip > this
        # (catastrophic over-exposure: scene got brighter) or p99 falls
        # below target_lo / 4 (catastrophic under-exposure: scene got
        # darker). Small drifts are tolerated to avoid jitter.
        self._ae_reengage_clip_frac = float(ae_cfg.get("reengage_frac_clip", 0.05))
        # Bracketed-state-firing-blank counter. If we've seen more than
        # this many consecutive grabs that returned None during AE
        # restarts, give up gracefully (stop trying, log loudly).
        self._ae_max_blank_iters = int(ae_cfg.get("max_blank_iters", 6))

    # ───────────────────────── runtime device switch ─────────────────
    def set_device(self, new_index: int | str) -> None:
        """Reopen capture on a different cv2 device index at runtime.

        Used by the GUI's device-selector dropdown so the user can swap
        which physical camera drives the EO panel without restarting
        the whole process.
        """
        log.info("EO set_device -> %s (was %s)", new_index, self.device_index)
        # Same race shape as set_exposure_ext: hold the lock + flag while
        # we tear down the old source so the capture thread can't spawn
        # a fresh helper on the OLD index in parallel.
        with self._source_lock:
            self._switching_exposure = True
            self.device_index = new_index
            old_source = self._source
            self._source = None

        if old_source is not None:
            try:
                old_source.stop()
            except Exception:
                pass

        time.sleep(0.6)
        with self._source_lock:
            self._switching_exposure = False

    # ───────────────────────── low-light boost ───────────────────────
    def set_lowlight_mode(self, enabled: bool) -> dict:
        """Toggle the AGC stretch + gamma midtone-lift display chain.

        At reference exposure, low-light scenes produce a raw frame
        that's correctly metered but visually almost black — the bridge
        AE saw the whole scene as dim and exposed for it, and now every
        pixel sits in the bottom 10 % of the 8-bit display range. AGC
        rescales [low_pct, high_pct] of pixel values back to full 0-255
        so what the camera saw is what the user sees on screen. Gamma >
        1 lifts midtones further at the cost of compressing highlights.

        OFF (default) — pure passthrough; what the bridge metered is
        what the panel shows. Best when the scene already fills the
        histogram (sunlit outdoor) because adding AGC then just amps
        sensor noise.

        ON — AGC stretch + gamma 1.6. Ugly in good light (you'll see
        the noise floor), essential in low light to see anything.

        Returns the resulting state. Live — no helper restart, the
        next published frame is already corrected.

        NOTE: an earlier version of this method ALSO enabled a 3-px
        median denoise + dropped YOLO conf to 0.20 + shrank the tracker
        min_bbox gate. Removed (2026-04-25) because the denoise visibly
        posterized low-light scenes (operator confirmed). If you need
        any of those individually, expose them as separate toggles —
        bundling them was a mistake.
        """
        new_val = bool(enabled)
        if new_val:
            self._agc_enabled = True
            self._enh_gamma = 1.6
            self._agc_low_pct = 0.5
            self._agc_high_pct = 99.5
        else:
            self._agc_enabled = False
            self._enh_gamma = 1.0
        log.info("EO low-light mode -> %s (agc=%s, gamma=%.2f)",
                 "ON" if new_val else "OFF",
                 self._agc_enabled, self._enh_gamma)
        return {
            "enabled": new_val,
            "agc_enabled": self._agc_enabled,
            "gamma": self._enh_gamma,
        }

    def get_lowlight_mode(self) -> dict:
        """Read current low-light boost state for GUI hydration."""
        return {
            "enabled": bool(self._agc_enabled and self._enh_gamma > 1.05),
            "agc_enabled": self._agc_enabled,
            "gamma": self._enh_gamma,
        }

    # ───────────────────────── runtime exposure switch ───────────────
    def set_exposure_ext(self, exposure_ext: Optional[int]) -> dict:
        """Flip the SDK stream backend between AE-on and a manual lock.

        ``exposure_ext=None`` → bridge auto-exposure on (safe outdoor
        default, what CameraTool does). ``exposure_ext=<int>`` → AE off
        + lock to that ExposureExt value.

        Implementation: the manager owns the source lifecycle. We take
        the source lock, stop the current SDK helper, update the value,
        then let the capture thread reopen on its next tick. The
        ``_switching_exposure`` flag tells the capture loop NOT to
        reopen until we've set the new value — without this, the
        capture thread races us, spawns a second helper, and we end
        up with two helpers fighting for the camera (which historically
        required a USB replug to recover from).
        """
        new_val = (None if exposure_ext is None else int(exposure_ext))
        with self._source_lock:
            already = (new_val == self._manual_exposure_ext
                       and self._source is not None)
            if already:
                return {
                    "exposure_ext": new_val,
                    "mode": "auto" if new_val is None else "manual",
                    "applied": True,
                    "noop": True,
                }
            # Block the capture loop from racing us.
            self._switching_exposure = True
            self._manual_exposure_ext = new_val
            # Flipping to Auto (new_val=None): wipe any AE bracket state
            # carried from a previous Auto session — the scene may have
            # changed completely since then. The AE thread will rebuild
            # brackets from the next stats reading. Also clear the
            # convergence latch so the GUI shows INITIALIZING during the
            # fresh search.
            if new_val is None and self._ae_controller is not None:
                self._ae_controller.reset_brackets()
            if new_val is None:
                self._ae_has_converged_once = False
            old_source = self._source
            self._source = None  # capture thread will see None but will
                                 # not call _open_source while
                                 # _switching_exposure is set.

        # Stop the old helper OUTSIDE the lock so a slow shutdown
        # (terminate → wait 3s → kill) doesn't block other API calls.
        if old_source is not None:
            try:
                old_source.stop()
            except Exception as e:
                log.warning("EO stop(old source) during exposure "
                            "switch threw %r", e)

        # Brief settle to let the FX3 bridge release the device. The
        # SDK helper subprocess sometimes takes ~0.5 s to fully release
        # the camera handle even after terminate(). Giving the bridge
        # this window before the capture loop reopens prevents the
        # "device already in use" error that caused the lockup bug.
        time.sleep(0.6)

        # Hand control back to the capture loop.
        with self._source_lock:
            self._switching_exposure = False

        log.info("EO exposure mode -> %s (capture loop will reopen)",
                 "AUTO" if new_val is None else f"MANUAL({new_val})")
        return {
            "exposure_ext": new_val,
            "mode": "auto" if new_val is None else "manual",
            "applied": True,
        }

    # ───────────────────────── AE state for GUI ──────────────────────
    def get_ae_state(self) -> dict:
        """Snapshot of software-AE state for the engineering tab.

        Returns ``mode`` ("manual" if user has overridden, else "auto"),
        the controller's current bracket, the most recent stats it acted
        on, and whether it's currently in-band. Cheap — no locking.
        """
        ctrl = self._ae_controller
        stats = None
        if self._source is not None:
            try:
                stats = getattr(self._source, "last_raw_stats", None)
            except Exception:
                stats = None
        return {
            "enabled": self._ae_enabled,
            "mode": "manual" if self._manual_exposure_ext is not None else "auto",
            "manual_exposure_ext": self._manual_exposure_ext,
            "ae_chosen_ext": self._ae_chosen_ext,
            "target_p99_lo": self._ae_target_lo,
            "target_p99_hi": self._ae_target_hi,
            "low_floor": getattr(ctrl, "low_floor", None) if ctrl else None,
            "high_brake": getattr(ctrl, "high_brake", None) if ctrl else None,
            "last_stats": stats,
        }

    # ───────────────────────── internal AE exposure apply ────────────
    def _apply_ae_exposure(self, new_ext: int) -> bool:
        """Commit a new ExposureExt picked by the software AE.

        Same shape as ``set_exposure_ext`` (lock + flag + tear down old
        helper + brief settle), but DOES NOT touch ``_manual_exposure_ext``
        — the user is still nominally in Auto mode; we're just updating
        the AE's chosen value. Returns True on success, False if a manual
        override or device switch raced us and won.
        """
        new_val = int(max(_AE_EXP_MIN, min(_AE_EXP_MAX, new_ext)))
        with self._source_lock:
            # If the user flipped to manual mid-step, don't fight them.
            if self._manual_exposure_ext is not None:
                return False
            if new_val == self._ae_chosen_ext and self._source is not None:
                return True  # noop
            self._switching_exposure = True
            self._ae_chosen_ext = new_val
            old_source = self._source
            self._source = None

        if old_source is not None:
            try:
                old_source.stop()
            except Exception as e:
                log.warning("EO stop(old source) during AE step threw %r", e)

        # Same settle window as set_exposure_ext — see that doc for why.
        time.sleep(0.6)

        with self._source_lock:
            self._switching_exposure = False
        return True

    # ───────────────────────── AE loop ───────────────────────────────
    def _ae_loop(self) -> None:
        """Software auto-exposure controller thread.

        Idle when the user has set a manual ExposureExt. Otherwise polls
        the source's last_raw_stats every ``_ae_step_period_s`` seconds
        and steps the bracket controller toward p99 ∈ [target_lo,
        target_hi]. Each step that actually changes the value triggers
        a source restart via ``_apply_ae_exposure`` (the SDK helper bakes
        ExposureExt at init, so we have no choice but to relaunch).

        Convergence is ~5-10 iterations from a worst-case initial seed.
        Once converged, we sit in the in-band branch and only re-engage
        if the scene shifts hard enough to clip or drop near the noise
        floor — small drifts are tolerated to avoid restart churn.
        """
        log.info("EO AE thread starting "
                 "(target p99 ∈ [%.0f, %.0f], step=%.1fs)",
                 self._ae_target_lo, self._ae_target_hi,
                 self._ae_step_period_s)
        blanks = 0
        while not self._stop.is_set():
            self._stop.wait(self._ae_step_period_s)
            if self._stop.is_set():
                break

            # User in manual mode → do nothing. Reset brackets so when
            # they flip back to Auto we don't carry stale state from a
            # different scene.
            if self._manual_exposure_ext is not None:
                if self._ae_controller is not None:
                    self._ae_controller.reset_brackets()
                blanks = 0
                continue

            if not self._ae_enabled:
                continue

            # Source not up yet (initial boot, between restarts) → wait.
            src = self._source
            if src is None:
                blanks += 1
                if blanks > self._ae_max_blank_iters:
                    log.warning(
                        "EO AE: source has been None for %d ticks — "
                        "either capture is failing to open or a long "
                        "switch is in flight. Will keep trying.", blanks
                    )
                    blanks = 0  # rate-limit the warning
                continue

            stats = None
            try:
                stats = src.last_raw_stats
            except Exception as e:
                log.debug("EO AE: last_raw_stats raised %r", e)
                stats = None

            if stats is None:
                # Non-SDK path (PyAV / YUY2 / fake / webcam) — software
                # AE is meaningless here, so go quiet. Bridge AE handles
                # those paths.
                blanks += 1
                if blanks > self._ae_max_blank_iters:
                    log.info("EO AE: source provides no raw stats — "
                             "AE disabled for this backend.")
                    return
                continue
            blanks = 0

            p99 = float(stats.get("p99", 0.0))
            mean = float(stats.get("mean", 0.0))
            frac_clip = float(stats.get("frac_clip", 0.0))

            # Lazy-init the controller. Done here (not in __init__) so
            # the very first step has fresh brackets per session.
            if self._ae_controller is None:
                self._ae_controller = _AEBracketController(
                    target_lo=self._ae_target_lo,
                    target_hi=self._ae_target_hi,
                )

            ctrl = self._ae_controller
            cur = self._ae_chosen_ext if self._ae_chosen_ext is not None \
                else self._ae_initial_ext

            # In-band? Just monitor for re-engage triggers.
            if ctrl.in_band(p99, frac_clip):
                # Latch convergence — drives the GUI scrim. See note in
                # _process_and_publish for why this is a separate flag
                # from the controller's bracket state.
                if not self._ae_has_converged_once:
                    self._ae_has_converged_once = True
                    # Persist the converged value so next session seeds
                    # from it — see _load_cached_ae_exposure docstring.
                    if self._ae_chosen_ext is not None:
                        _save_cached_ae_exposure(self._ae_chosen_ext)
                    log.info("EO AE converged (p99=%.0f, clip=%.3f, "
                             "exp=%s) — clearing INITIALIZING scrim",
                             p99, frac_clip, self._ae_chosen_ext)
                    try:
                        from common.events import emit as _emit
                        _emit("ae_converged", {
                            "p99": float(p99),
                            "frac_clip": float(frac_clip),
                            "exposure_ext": self._ae_chosen_ext,
                        })
                    except Exception:
                        pass
                # Re-engage thresholds: scene became MUCH brighter
                # (frac_clip blew through reengage gate) or MUCH darker
                # (p99 dropped below quarter of target_lo). Anything
                # in between is normal scene variation and tolerated.
                catastrophic_bright = frac_clip > self._ae_reengage_clip_frac
                catastrophic_dark = p99 < (self._ae_target_lo / 4.0)
                if catastrophic_bright or catastrophic_dark:
                    log.info("EO AE re-engaging (p99=%.0f, frac_clip=%.3f, "
                             "exp=%d) — scene changed", p99, frac_clip, cur)
                    ctrl.reset_brackets()
                    new_ext = ctrl.step(cur, p99, mean, frac_clip)
                    if new_ext != cur:
                        self._apply_ae_exposure(new_ext)
                continue

            # Out of band — step.
            new_ext = ctrl.step(cur, p99, mean, frac_clip)
            if new_ext == cur:
                # Bracket collapsed at current value.
                #   - If at AE_EXP_MAX or AE_EXP_MIN: scene is genuinely
                #     beyond the sensor envelope (too dim past max, too
                #     bright past min). Accept it and stop trying — log
                #     ONCE per stuck session, not every 1.5s. Otherwise
                #     each AE iteration triggers a source restart and
                #     the panel blinks black continuously.
                #   - If collapsed at intermediate value AND still out
                #     of band: brackets are stale (scene moved outside
                #     originally-tested range). Reset to let the next
                #     iteration search beyond the stuck region.
                if cur >= _AE_EXP_MAX or cur <= _AE_EXP_MIN:
                    if not getattr(self, "_ae_at_boundary_logged", False):
                        log.warning(
                            "EO AE: stuck at boundary exp=%d, p99=%.0f "
                            "out of band — sensor envelope exhausted, "
                            "holding", cur, p99,
                        )
                        self._ae_at_boundary_logged = True
                else:
                    self._ae_at_boundary_logged = False
                    if not ctrl.in_band(p99, frac_clip):
                        log.info("EO AE: bracket collapsed at exp=%d but "
                                 "p99=%.0f still out of band — resetting brackets",
                                 cur, p99)
                        ctrl.reset_brackets()
                continue
            log.info("EO AE step: exp %d -> %d (p99=%.0f mean=%.0f "
                     "clip=%.3f, lo_floor=%s, hi_brake=%s)",
                     cur, new_ext, p99, mean, frac_clip,
                     ctrl.low_floor, ctrl.high_brake)
            self._apply_ae_exposure(new_ext)

        log.info("EO AE thread stopped")

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
        # Async classify worker — only spun up if a classifier is
        # actually loaded. Without it, the publish path just always
        # sees an empty _cls_out_dets and the EO panel renders
        # without boxes (legitimate config: enable_classifier=False).
        if self._classifier is not None:
            self._cls_thread = threading.Thread(
                target=self._classifier_loop, name="EOClassify", daemon=True
            )
            self._cls_thread.start()
        # AE thread only runs when AE is enabled in config — saves a
        # daemon thread when someone explicitly disables software AE
        # (e.g. for debugging the bridge AE directly).
        if self._ae_enabled:
            self._ae_thread = threading.Thread(
                target=self._ae_loop, name="EOAutoExposure", daemon=True
            )
            self._ae_thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._latest_cond:
            self._latest_cond.notify_all()
        with self._cls_in_cond:
            self._cls_in_cond.notify_all()
        for t in (self._process_thread, self._capture_thread,
                  self._cls_thread, self._ae_thread):
            if t is not None:
                t.join(timeout=3.0)
        self._capture_thread = None
        self._process_thread = None
        self._cls_thread = None
        self._ae_thread = None
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

        # IMX568 backend — native 2472×2064, mono, profile-switched exposure.
        if self._sensor_backend == "imx568":
            from eo.imx568_capture import IMX568Capture
            # Effective ExposureExt: user manual override wins; else if our
            # software AE is enabled use its current best guess; else None
            # (which lets the bridge AE run — legacy behaviour for envs
            # where the bridge AE actually works).
            effective_ext: Optional[int]
            if self._manual_exposure_ext is not None:
                effective_ext = int(self._manual_exposure_ext)
            elif self._ae_enabled and self._ae_chosen_ext is not None:
                effective_ext = int(self._ae_chosen_ext)
            else:
                effective_ext = None
            try:
                cap = IMX568Capture(
                    device_index=self.device_index,
                    exclude_indices=self.exclude_indices,
                    manual_exposure_log2=self._manual_exposure_log2,
                    manual_gain=self._manual_gain,
                    manual_exposure_ext=effective_ext,
                    stream_fps=self._stream_fps,
                )
                cap.start()
                # Only command initial exposure/gain if profile control is
                # enabled. With control OFF (default on this FX3 bridge),
                # leave the camera in its own auto-exposure mode — that's
                # what CameraTool uses and it's the only setting that
                # actually produces a live image on this hardware.
                if self._profile_control_exposure:
                    p = self._profile_selector.current()
                    cap.set_exposure_ms(p.exposure_ms)
                    cap.set_gain(p.gain)
                    log.info("EO opened IMX568, profile_control ON, "
                             "initial profile=%s (exposure=%.1fms gain=%.1f)",
                             p.name.value, p.exposure_ms, p.gain)
                else:
                    log.info("EO opened IMX568, profile_control OFF — "
                             "bridge's auto-exposure remains active, "
                             "AGC handles display dynamic range")
                return cap
            except RuntimeError as e:
                log.warning("EO IMX568 open failed: %s", e)
                return None
            except Exception as e:
                log.warning("EO IMX568 open threw %r — will retry", e)
                return None

        # Webcam fallback — generic UVC path for dev without the kit.
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
        except Exception as e:
            # cv2.error is NOT a RuntimeError. Without this, an OpenCV
            # "Unknown C++ exception" from a flaky webcam bus kills the
            # EOCapture thread instead of retrying.
            log.warning("EO source open threw %r — will retry", e)
            return None

    # ───────────────────────── main loops ────────────────────────────
    def _capture_loop(self) -> None:
        log.info("EOManager capture thread starting (fake=%s)", self.use_fake)
        while not self._stop.is_set():
            # Take the source-lifecycle lock to read/open the source.
            # Without this, set_exposure_ext can null _source and the
            # capture loop spawns a NEW SDK helper while the API thread
            # is still tearing down the OLD one — two helpers fight for
            # the FX3 bridge and lock the device. The _switching_exposure
            # flag tells us "an exposure switch is in flight, don't open
            # anything yet, the API thread will hand control back."
            with self._source_lock:
                if self._switching_exposure:
                    need_reopen = False
                    src = None
                else:
                    src = self._source
                    need_reopen = (src is None)

            if self._switching_exposure:
                # Don't spam logs / publish disconnect every 50ms during
                # the ~0.6s switch window. Just idle.
                self._stop.wait(0.1)
                continue

            if need_reopen:
                opened = self._open_source()
                if opened is None:
                    self._publish_disconnected()
                    self._stop.wait(self.reconnect_interval_s)
                    continue
                # Re-check the flag under the lock before publishing the
                # new source — it's possible (rare) that an exposure
                # switch fired while _open_source was running. If so,
                # discard this source and let the switch handler win.
                with self._source_lock:
                    if self._switching_exposure:
                        try:
                            opened.stop()
                        except Exception:
                            pass
                        continue
                    self._source = opened
                    src = opened

            frame = src.grab()
            if frame is None:
                log.warning("EO grab returned None — treating as disconnect")
                self._publish_disconnected()
                # Drop the source under the lock so a concurrent
                # set_exposure_ext doesn't double-stop it.
                with self._source_lock:
                    if self._source is src:
                        self._source = None
                        drop = src
                    else:
                        drop = None
                if drop is not None:
                    try:
                        drop.stop()
                    except Exception:
                        pass
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
                    # Phase 1 fix: was 0.5. With 0.5 a missed notify
                    # (capture-vs-process race) cost 500ms = 10 dropped
                    # frames at 20fps. 0.05 caps the worst-case miss to
                    # ~1 frame.
                    self._latest_cond.wait(timeout=0.05)
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

    def _classifier_loop(self) -> None:
        """Async YOLO + ByteTrack worker.

        Pulls the latest frame submitted by the process thread (single-
        slot — older frames are discarded if a newer one arrives before
        we're done), runs the classifier, posts results to
        ``_cls_out_dets`` for the publish path to read.

        Frames may be skipped under load (the process thread always
        overwrites the pending slot with the freshest frame). ByteTrack
        ``persist=True`` handles missed frames via its Kalman predictor,
        same as the previous synchronous classify_every=2 path.
        """
        while not self._stop.is_set():
            with self._cls_in_cond:
                while self._cls_in_pending is None and not self._stop.is_set():
                    self._cls_in_cond.wait(timeout=0.5)
                if self._stop.is_set():
                    break
                fid, frame = self._cls_in_pending
                self._cls_in_pending = None
            if self._classifier is None:
                continue
            try:
                _zoom_lvl = int(getattr(self, "_zoom_level", 1) or 1)
                if self._tiling_enabled:
                    # 2026-05-14: SAHI on FULL NATIVE at ALL zoom levels.
                    # Cropping at high zoom collapses model conf.
                    from vision.sahi_inference import tiled_predict
                    raw = tiled_predict(
                        self._classifier._hv._model,
                        frame,
                        grid=self._tiling_grid,
                        overlap_frac=self._tiling_overlap_frac,
                        imgsz=self._classifier._hv.imgsz,
                        conf=self._classifier._hv.conf_threshold,
                        merge_iou=self._tiling_merge_iou,
                        per_class_conf=self._classifier._hv.per_class_conf
                            or None,
                    )
                elif False:
                    # 2026-05-14: at zoom 2x+, the cropped frame is
                    # already small (1236, 618, 309 wide for 2/4/8x).
                    # Tiling into 2x2 makes each tile ~155x130 at 8x —
                    # letterboxed to 832 = mostly padding -> conf collapses.
                    # Single-pass on the whole crop is far better.
                    # Engine is fixed batch=4 so we pad the batch with
                    # 3 frame copies and take only result[0].
                    _model = self._classifier._hv._model
                    _conf_floor = float(self._classifier._hv.conf_threshold)
                    _imgsz = int(self._classifier._hv.imgsz)
                    _pcc = self._classifier._hv.per_class_conf or None
                    _batch = [frame, frame, frame, frame]
                    _results = _model.predict(
                        _batch, conf=_conf_floor, imgsz=_imgsz,
                        verbose=False, batch=4,
                    )
                    raw = []
                    r0 = _results[0] if _results else None
                    if r0 is not None and r0.boxes is not None and len(r0.boxes) > 0:
                        _xyxy = r0.boxes.xyxy.cpu().numpy()
                        _confs = r0.boxes.conf.cpu().numpy()
                        _clss = r0.boxes.cls.cpu().numpy().astype(int)
                        _names = _model.names or {}
                        for (x1, y1, x2, y2), c, cid in zip(_xyxy, _confs, _clss):
                            cls_name = _names.get(int(cid), str(int(cid)))
                            if _pcc is not None:
                                floor = _pcc.get(cls_name)
                                if floor is not None and float(c) < floor:
                                    continue
                            raw.append({
                                "bbox": (int(max(0, x1)), int(max(0, y1)),
                                         int(max(1, x2 - x1)),
                                         int(max(1, y2 - y1))),
                                "class": cls_name,
                                "conf": float(c),
                            })
                    log.info("EO single-pass zoom=%dx frame=%s raw_dets=%d",
                              _zoom_lvl, frame.shape, len(raw))
                else:
                    raw = self._classifier.track(frame)

                if self._tiling_enabled:
                    # Tiled path doesn't carry ByteTrack IDs (the
                    # tracker can't reason across tile-frames sensibly).
                    log.debug("EO tiled_predict (fid=%d) frame=%s tiles=%s raw_dets=%d zoom=%dx hfov_eff=%.2f", fid, frame.shape, self._tiling_grid, len(raw), int(getattr(self, "_zoom_level", 1) or 1), self._hfov / float(max(1, getattr(self, "_zoom_level", 1))))
                    # MOSSE pool uses track_id as a dict key, so we
                    # synthesize unique monotonically-decreasing IDs.
                    # Each tile-tick spawns fresh MOSSE trackers; old
                    # ones decay via max_misses. Acceptable churn —
                    # the alternative (per-frame IoU-keyed re-ID) is a
                    # full mini-tracker that would replicate ByteTrack's
                    # job. If this churn shows up as MOSSE-pool memory
                    # growth, we add an LRU cap on the pool size.
                    if not hasattr(self, "_tile_id_seq"):
                        self._tile_id_seq = 1000000
                        # IoU tracker state: list of (track_id, cls, bbox_xywh)
                        # from the previous tick. We match current-tick dets
                        # against this so a vehicle detected on every tick
                        # keeps its stable ID (kills GUI flicker).
                        self._tile_prev_tracks: list = []
                    # 2026-05-14: rescale bboxes from FULL NATIVE -> DISPLAY,
                    # accounting for the zoom crop. Drop dets whose center
                    # is outside the visible crop region.
                    if frame is not None:
                        full_h, full_w = frame.shape[:2]
                    else:
                        full_h = full_w = 0
                    zl = max(1, _zoom_lvl)
                    crop_w = full_w // zl
                    crop_h = full_h // zl
                    crop_x0 = (full_w - crop_w) // 2
                    crop_y0 = (full_h - crop_h) // 2
                    if crop_w > 0 and self._display_max_width > 0:
                        scale = self._display_max_width / float(crop_w)
                    else:
                        scale = 1.0
                    _kept = []
                    for d in raw:
                        bx, by, bw, bh = d["bbox"]
                        cx, cy = bx + bw/2, by + bh/2
                        if not (crop_x0 <= cx <= crop_x0 + crop_w):
                            continue
                        if not (crop_y0 <= cy <= crop_y0 + crop_h):
                            continue
                        nx = int(round((bx - crop_x0) * scale))
                        ny = int(round((by - crop_y0) * scale))
                        nw = max(1, int(round(bw * scale)))
                        nh = max(1, int(round(bh * scale)))
                        d["bbox"] = (nx, ny, nw, nh)
                        _kept.append(d)
                    raw = _kept

                    # ── Phase-corr accumulator consume (slew compensation) ──
                    # Publish thread runs phaseCorrelate every tick when
                    # any track is in flight, and accumulates (dx, dy)
                    # in display pixel space into _pc_dx/dy_disp_accum.
                    # We read-and-zero the accumulator here under lock —
                    # whatever image translation happened across the N
                    # publish ticks since our last iteration. Apply it
                    # to _tile_prev_tracks BEFORE IoU matching so fresh
                    # YOLO dets (at the new panned position) still IoU-
                    # match the (shifted) prev tracks → same tid kept.
                    try:
                        with self._pc_cache_lock:
                            _dx_acc = self._pc_dx_disp_accum
                            _dy_acc = self._pc_dy_disp_accum
                            self._pc_dx_disp_accum = 0.0
                            self._pc_dy_disp_accum = 0.0
                        if (abs(_dx_acc) > 1.0 or abs(_dy_acc) > 1.0) and self._tile_prev_tracks:
                            self._tile_prev_tracks = [
                                (ptid, pcls,
                                 (pbb[0]+_dx_acc, pbb[1]+_dy_acc, pbb[2], pbb[3]))
                                for ptid, pcls, pbb in self._tile_prev_tracks
                            ]
                    except Exception as _e:
                        log.debug("prev-tracks accum-consume skip: %s", _e)


                    # IoU-based ID inheritance to kill flicker
                    def _bbox_iou(a, b):
                        ax1, ay1 = a[0], a[1]
                        ax2, ay2 = a[0]+a[2], a[1]+a[3]
                        bx1, by1 = b[0], b[1]
                        bx2, by2 = b[0]+b[2], b[1]+b[3]
                        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
                        iy = max(0, min(ay2, by2) - max(ay1, by1))
                        inter = ix * iy
                        if inter <= 0: return 0.0
                        return inter / (a[2]*a[3] + b[2]*b[3] - inter)
                    used_prev = set()
                    inherited_tids = set()
                    new_prev = []
                    for d in raw:
                        best_iou = 0.0
                        best_idx = -1
                        for i, (ptid, pcls, pbb) in enumerate(self._tile_prev_tracks):
                            if i in used_prev: continue
                            if pcls != d["class"]: continue
                            io = _bbox_iou(d["bbox"], pbb)
                            if io > best_iou and io >= 0.3:
                                best_iou = io
                                best_idx = i
                        if best_idx >= 0:
                            used_prev.add(best_idx)
                            tid = self._tile_prev_tracks[best_idx][0]
                            inherited_tids.add(int(tid))
                        else:
                            self._tile_id_seq += 1
                            tid = self._tile_id_seq
                        d.setdefault("track_id", tid)
                        new_prev.append((tid, d["class"], d["bbox"]))
                    self._tile_prev_tracks = new_prev
            except Exception as e:
                log.warning("EO async inference failed (fid=%d): %s", fid, e)
                continue

            # ── Two-tier Schmitt confirmation gate ─────────────────────
            # Per-track state accumulates evidence across YOLO ticks.
            # Confirmation rule (either path):
            #   HI: a single tick with conf >= conf_hi  → instant publish
            #   LO: K_PERSIST ticks with conf >= conf_lo within last WINDOW
            #       ticks → publish (slow path, kills isolated noise)
            # Once confirmed, the track stays sticky and is emitted with
            # last-known bbox for COAST_TICKS ticks of no fresh hit, so
            # MOSSE can coast it visually between YOLO ticks. After
            # COAST_TICKS ticks of no hit the track dies.
            #
            # This runs on the tiled-IoU-stable track_id path. For the
            # non-tiled fallback (self._classifier.track), 'raw' carries
            # ByteTrack IDs which behave identically in this gate.
            # _track_cfg/state initialized in __init__ (see ccfg.schmitt)
            cfg = self._track_cfg
            self._track_tick += 1
            tick = self._track_tick

            fresh_tids = set()
            for d in raw:
                tid = d.get("track_id")
                if tid is None: continue
                tid = int(tid)
                fresh_tids.add(tid)
                conf = float(d.get("conf", 0.0))
                st = self._track_state.get(tid)
                if st is None:
                    st = {"hits": [], "conf_max": 0.0, "confirmed": False,
                          "last_class": d.get("class","?"),
                          "last_conf":  conf,
                          "conf_ema":   conf,  # 2026-05-14: smoothed conf for stable display label
                          "last_bbox":  d.get("bbox"),
                          "last_seen":  tick}
                    self._track_state[tid] = st
                st["last_seen"]  = tick
                st["last_class"] = d.get("class", st["last_class"])
                st["last_conf"]  = conf
                # EMA smoothing — alpha=0.3 means new conf has 30% weight,
                # smoothed retains 70%. Slow enough to ride out single-frame
                # swings 0.30->0.90; fast enough to track real conf trends.
                st["conf_ema"] = 0.3 * conf + 0.7 * st.get("conf_ema", conf)
                st["last_bbox"]  = d.get("bbox", st["last_bbox"])
                if conf > st["conf_max"]: st["conf_max"] = conf
                if conf >= cfg["conf_lo"]: st["hits"].append(tick)
                # prune hits outside window
                st["hits"] = [t for t in st["hits"] if tick - t < cfg["window"]]
                if not st["confirmed"]:
                    # Per-class HI threshold (instant pass). Default = global cfg["conf_hi"].
                    _cls_hi = self._classes_conf_hi.get(
                        d.get("class","?"), cfg["conf_hi"]
                    )
                    if conf >= _cls_hi or len(st["hits"]) >= cfg["k_persist"]:
                        st["confirmed"] = True

            # Build out_dets = confirmed tracks (fresh this tick) +
            # confirmed-but-coasted tracks (last bbox, MOSSE will move
            # it between YOLO ticks). Garbage collect stale state.
            out_dets = []
            for tid, st in list(self._track_state.items()):
                if st["confirmed"]:
                    if tid in fresh_tids:
                        for d in raw:
                            if int(d.get("track_id", -1)) == tid:
                                # Override the displayed conf with the EMA-smoothed
                                # value to stop label-text flicker on YOLO conf
                                # swings (e.g., 0.30 <-> 0.90 frame-to-frame).
                                d["conf"] = float(st["conf_ema"])
                                out_dets.append(d)
                                break
                    elif (tick - st["last_seen"]) <= cfg["coast_ticks"]:
                        # Coast: synth det at last-known bbox. MOSSE
                        # in publish thread will template-match the bbox
                        # forward across the captured frames between
                        # YOLO ticks — handles moving targets correctly.
                        # 2026-05-14b: mark coasted=True so the MOSSE
                        # reseed in _process_and_publish does NOT yank
                        # the template back to this stale bbox — MOSSE
                        # is the source of truth for bbox position
                        # between YOLO ticks.
                        out_dets.append({
                            "bbox":     st["last_bbox"],
                            "class":    st["last_class"],
                            "conf":     float(st["conf_ema"]),  # smoothed
                            "track_id": tid,
                            "coasted":  True,
                        })
                    else:
                        del self._track_state[tid]
                else:
                    # Unconfirmed: drop state after WINDOW ticks of no hit
                    if (tick - st["last_seen"]) > cfg["window"]:
                        del self._track_state[tid]

            # Re-seed _tile_prev_tracks with confirmed coasted entries
            # so next-tick IoU matching can re-bind the same tid.
            if hasattr(self, "_tile_prev_tracks"):
                _seen_prev = {p[0] for p in self._tile_prev_tracks}
                for d in out_dets:
                    tid = int(d["track_id"])
                    if tid not in fresh_tids and tid not in _seen_prev:
                        self._tile_prev_tracks.append(
                            (tid, d["class"], d["bbox"]))

            # ── Containment-drop pass ─────────────────────────
            # SAHI cross-tile NMS uses IoU which fails when a small
            # bbox sits inside a larger one (IoU is small because the
            # bigger box has lots of area outside the small one). Drop
            # the smaller-area bbox when >=50% of its area is inside a
            # higher-conf bbox of the same class. Kills the twin-bbox
            # "vehicle detected in two tiles" artifact.
            def _contain_frac(a, b):
                # fraction of a area inside b
                ix = max(0, min(a[0]+a[2], b[0]+b[2]) - max(a[0], b[0]))
                iy = max(0, min(a[1]+a[3], b[1]+b[3]) - max(a[1], b[1]))
                inter = ix * iy
                aw = a[2] * a[3]
                return inter / aw if aw > 0 else 0.0
            # 2026-05-14: tid-aware sort. Stabilize containment-drop
            # across consecutive ticks by preferring candidates whose
            # tid was IoU-inherited from _tile_prev_tracks this tick.
            # Without this, the "big-box / small-box wins" alternation
            # between ticks causes track_id flips and GUI label flicker.
            _inherited = inherited_tids if 'inherited_tids' in dir() else set()
            def _tid_priority(d):
                # Lower sort key wins (Python sort is ascending).
                # 0 = inherited tid (stable identity from prev tick)
                # 1 = newly-minted tid
                # Tiebreaker: higher conf wins (negate for ascending sort)
                tid = int(d.get("track_id", -1))
                pri = 0 if tid in _inherited else 1
                return (pri, -float(d.get("conf", 0)))
            _sorted = sorted(out_dets, key=_tid_priority)
            _kept = []
            for d in _sorted:
                bb_d = d["bbox"]
                a_d  = (bb_d[2] * bb_d[3]) if isinstance(bb_d, tuple) else (bb_d.get("w",0) * bb_d.get("h",0))
                subsumed = False
                for k in _kept:
                    if k.get("class") != d.get("class"): continue
                    bb_k = k["bbox"]
                    a_k = (bb_k[2] * bb_k[3]) if isinstance(bb_k, tuple) else (bb_k.get("w",0) * bb_k.get("h",0))
                    # k is in _kept (came earlier in priority order).
                    # If they containment-overlap, drop d.
                    if a_k < a_d:
                        # k smaller than d — k is contained in d, not vice versa.
                        # But we still want to drop the LATER one (d) IF
                        # they overlap heavily, because k won priority.
                        if _contain_frac(bb_k, bb_d) >= 0.5:
                            subsumed = True; break
                    else:
                        if _contain_frac(bb_d, bb_k) >= 0.5:
                            subsumed = True; break
                if not subsumed:
                    _kept.append(d)
            out_dets = _kept

            # Periodic gate summary (every ~5s at 8Hz YOLO = 40 ticks).
            if (tick % 40) == 0:
                n_pending = sum(1 for st in self._track_state.values()
                                if not st["confirmed"])
                n_conf_fresh = sum(1 for d in out_dets
                                   if not d.get("coasted"))
                n_conf_coast = sum(1 for d in out_dets
                                   if d.get("coasted"))
                log.info("EO Schmitt gate (tick=%d): "
                         "conf_fresh=%d conf_coast=%d pending=%d "
                         "raw_in=%d -> out=%d",
                         tick, n_conf_fresh, n_conf_coast, n_pending,
                         len(fresh_tids), len(out_dets))

            raw = out_dets

            # Same size gate + per-class NMS as the previous synchronous
            # path. Runs on the worker thread, off the publish critical
            # path, so a 50-100 ms inference spike no longer stalls the
            # EO frame rate visible to the operator.
            dets = [d for d in raw
                    if (d["bbox"][2] * d["bbox"][3]) >= self._min_bbox_px]
            dets = _nms_same_class(dets, iou_thresh=0.45)
            with self._cls_out_lock:
                self._cls_out_dets = dets
                self._cls_out_fid = fid
                self._cls_out_dets_fresh_flag = True
        log.info("EOManager classify thread stopped")

    def get_schmitt_config(self) -> dict:
        """Snapshot of the Schmitt gate config (global + per-class HI + per-class LO).

        Used by gui/sensor_bridge to hydrate DEV-tab sliders on every WS tick.
        Safe to call from any thread — returns a dict copy.
        """
        out = dict(self._track_cfg)  # copy globals
        out["per_class"] = {
            cls: {
                "hi": float(self._classes_conf_hi.get(cls, self._track_cfg["conf_hi"])),
                "lo": float((self._classifier._hv.per_class_conf or {}).get(
                    cls, self._track_cfg["conf_lo"]
                )) if self._classifier and getattr(self._classifier, "_hv", None) else 0.0,
            }
            for cls in ("person", "vehicle", "drone")
        }
        return out

    def set_schmitt_config(self, **kw) -> None:
        """Mutate Schmitt config in place. Thread-safe via in-place dict
        item assignment (atomic in CPython).

        Accepts: conf_hi, conf_lo, k_persist, window, coast_ticks (globals);
        per_class={cls: {hi?, lo?}} (per-class overrides).

        INVARIANT: only mutates existing keys. Never rebinds the dict,
        never adds new keys. Worker thread reads keys directly without
        a lock, relying on this invariant.
        """
        for k in ("conf_hi", "conf_lo"):
            if k in kw:
                v = max(0.0, min(1.0, float(kw[k])))
                self._track_cfg[k] = v
        for k in ("k_persist", "window", "coast_ticks"):
            if k in kw:
                v = max(1, min(50, int(kw[k])))
                self._track_cfg[k] = v
        pc = kw.get("per_class") or {}
        for cls in ("person", "vehicle", "drone"):
            c = pc.get(cls) or {}
            if "hi" in c:
                self._classes_conf_hi[cls] = max(0.0, min(1.0, float(c["hi"])))
            if "lo" in c and self._classifier and getattr(self._classifier, "_hv", None):
                pcc = self._classifier._hv.per_class_conf
                if pcc is None:
                    self._classifier._hv.per_class_conf = pcc = {}
                pcc[cls] = max(0.0, min(1.0, float(c["lo"])))
        log.info("EO Schmitt cfg updated: %s", {**self._track_cfg, "per_class": self._classes_conf_hi})

    def set_zoom_level(self, level: int) -> None:
        """Set EO digital zoom. Valid: 1, 2, 4, 8.

        1× / 2× → classifier tiles the (possibly cropped) native frame.
        4× / 8× → classifier sees only the crop, single-tile at imgsz=832.
        Display + recording always show the cropped+upscaled view.
        """
        if level not in (1, 2, 4, 8):
            log.warning("EO set_zoom_level: invalid level %s (must be 1/2/4/8)", level)
            return
        prev = self._zoom_level
        self._zoom_level = int(level)
        if prev != self._zoom_level:
            # Reset tile-tracker state — bbox coords change reference
            # frame so previous-tick IoU matching is meaningless across
            # the zoom transition.
            try:
                self._tile_prev_tracks = []
                self._track_state = {}
                # 2026-05-14b: also kill the publish-thread
                # cache so stale bboxes from the old zoom don't
                # render briefly on the new cropped frame.
                with self._cls_out_lock:
                    self._cls_out_dets = []
                    self._cls_out_dets_fresh_flag = True
            except Exception:
                pass
            log.info("EO zoom: %dx -> %dx", prev, self._zoom_level)

        # ───────────────────────── pipeline ──────────────────────────────
    def _process_and_publish(self, frame: np.ndarray) -> None:
        self._frame_id += 1
        ts = time.time()
        # Snapshot the gimbal pose. See ThermalManager._process_and_publish
        # for the rationale — this binds pose to the frame at processing
        # start time, used by fusion to convert detections to world-frame
        # az/el using the actual pose at capture, not at fusion-tick time.
        gs_for_capture = BUS.get_latest(Topic.GIMBAL)
        if isinstance(gs_for_capture, GimbalState):
            bus_pan = float(gs_for_capture.pan_deg)
            bus_tilt = float(gs_for_capture.tilt_deg)
        else:
            bus_pan = None
            bus_tilt = None
        # Optical-feedback override: if BUS says we moved but the EO
        # pixels say we didn't, suppress the BUS update. See _trusted_pan
        # docstring on __init__ for rationale.
        # (Computation of phase correlation deferred until after we have
        # `frame` downsampled to grayscale — see below; we inject the
        # optical override into gimbal_pan_at_capture / gimbal_tilt_at_capture
        # before publishing the EOFrame.)
        gimbal_pan_at_capture = bus_pan
        gimbal_tilt_at_capture = bus_tilt

        # 0. IMX568 pipeline: downscale → AGC → profile switching.
        #
        # Downscale FIRST for display/JPEG-encode. Every subsequent
        # operation scales with pixel count; halving width quarters
        # the per-frame cost of AGC, JPEG encode, JSONL recording.
        #
        # 2026-05-08: when tiled inference is enabled, we keep the
        # NATIVE frame around to feed the tiled classifier — that's
        # the whole point of tiling (give YOLO native pixels for
        # long-distance targets). Display path still uses 1236-wide.
        # The native frame is NOT cached past _process_and_publish,
        # so memory cost is one extra 2472×2064×3 ≈ 15 MB per tick.
        # ALWAYS keep a reference to the native frame BEFORE the downscale —
        # used by (a) the tiled classifier path when tiling is enabled
        # and (b) the JSONL recorder to record native frames regardless
        # of the display_max_width setting (so on-disk recordings are
        # full fidelity for offline analysis). The downscale below
        # produces a NEW resize buffer so this reference stays valid
        # for the rest of the publish tick.
        native_frame: Optional[np.ndarray] = None
        if self._sensor_backend == "imx568":
            if self._display_max_width > 0 and frame.shape[1] > self._display_max_width:
                native_frame = frame
                scale = self._display_max_width / float(frame.shape[1])
                new_w = self._display_max_width
                new_h = int(round(frame.shape[0] * scale))
                # INTER_AREA is the right choice for downscaling — it
                # integrates over source pixels, which preserves small-
                # target detail better than INTER_LINEAR.
                import cv2 as _cv2
                frame = _cv2.resize(frame, (new_w, new_h),
                                    interpolation=_cv2.INTER_AREA)
            # ── Digital zoom (center crop) ──────────────────────────
            # Apply AFTER the display downscale so cross-sensor projection
            # math (which assumes display-coord bboxes scaled by the
            # capture's real HFOV/level) can read self._zoom_level. The
            # crop is taken from the native frame to preserve maximum
            # detail, then upscaled to display width (LINEAR for fewer
            # ringing artifacts on hard target edges).
            # 2026-05-14: preserve FULL native (pre-crop) for classifier.
            full_native_frame = native_frame
            zl = int(getattr(self, "_zoom_level", 1) or 1)
            if zl > 1 and native_frame is not None:
                import cv2 as _cv2
                nh, nw = native_frame.shape[:2]
                cw, ch = nw // zl, nh // zl
                x0 = (nw - cw) // 2
                y0 = (nh - ch) // 2
                # native_frame is replaced with the crop — classifier
                # below sees the cropped region as its "native".
                native_frame = native_frame[y0:y0+ch, x0:x0+cw].copy()
                # Display frame: crop+upscale to display_max_width.
                disp_w = self._display_max_width if self._display_max_width > 0 else cw
                disp_h = int(round(ch * (disp_w / cw)))
                frame = _cv2.resize(native_frame, (disp_w, disp_h),
                                    interpolation=_cv2.INTER_LINEAR)
            mean = scene_mean(frame)
            # Skip the selector entirely when WE are driving exposure.
            # Reason: scene_mean on the displayed frame is a function of
            # the exposure we just commanded, not of the scene alone — so
            # feeding it back into "should I switch profile?" creates an
            # oscillator (DAY 0.5ms → dim raw → "switch to NIGHT" → 80ms
            # → blown → "switch to DAY" → repeat every dwell_s). Lock to
            # the initial profile. Auto-switching is only sane when the
            # bridge's own AE runs and scene_mean genuinely reflects the
            # scene — i.e. when profile_control_exposure is OFF.
            if self._profile_control_exposure:
                switched = None
            else:
                switched = self._profile_selector.update(
                    mean, time.monotonic()
                )
            if switched is not None and self._source is not None:
                # Always log the recommended profile so the engineering
                # tab can show "selector recommends NIGHT" even when we
                # aren't driving the sensor — useful for validating that
                # the selector thresholds are tuned right before ever
                # flipping profile_control_exposure on.
                log.info("EO profile -> %s (scene mean=%.1f, "
                         "exposure=%.1fms gain=%.1f, control=%s)",
                         switched.name.value, mean,
                         switched.exposure_ms, switched.gain,
                         "ON" if self._profile_control_exposure else "OFF")
                if self._profile_control_exposure:
                    try:
                        self._source.set_exposure_ms(switched.exposure_ms)
                        self._source.set_gain(switched.gain)
                    except AttributeError:
                        # Webcam backend doesn't expose these; skip.
                        pass
            if self._agc_enabled:
                # Opt-in enhancement chain. Each stage is YAML-toggled —
                # see app_config.yaml `eo.enhance`. In low light, leave
                # CLAHE and unsharp DISABLED; they amplify grain into
                # ridge artifacts.
                frame = enhance(
                    frame,
                    denoise_ksize=self._enh_denoise_ksize,
                    low_pct=self._agc_low_pct,
                    high_pct=self._agc_high_pct,
                    gamma=self._enh_gamma,
                    clahe_clip=self._enh_clahe_clip,
                    clahe_grid=self._enh_clahe_grid,
                    unsharp_amount=self._enh_unsharp_amount,
                    unsharp_radius=self._enh_unsharp_radius,
                )
            else:
                # When the SDK RAW12 stream is the source, the frame has
                # already been Bayer-debayered to a real BGR with subtle
                # NIR color tint (the IMX568 IS a Bayer color sensor; the
                # 35 mm NIR-pass filter just makes R/G/B see near-identical
                # NIR levels, leaving faint pinkish/greenish color cast —
                # exactly what Leopard's CameraTool shows). Calling
                # passthrough() here would extract Y and re-expand as
                # GRAY2BGR, throwing the chroma away — that's why the EO
                # panel was rendering pure grayscale despite the sensor
                # delivering color. Detect SDK-stream mode via
                # last_raw_stats (only the SDK path populates it) and
                # skip passthrough so the genuine Bayer color reaches
                # the GUI. Legacy YUY2/PyAV paths still need passthrough's
                # luma-recovery — keep it on those.
                src = self._source
                sdk_mode = (src is not None
                            and getattr(src, "last_raw_stats", None) is not None)
                if not sdk_mode:
                    # Pure passthrough: collapse YUY2's pseudo-3-channel
                    # BGR back to one luma channel and re-expand for the
                    # JPEG encoder. No stretching, no gamma — what the
                    # bridge AE produced is what the GUI shows.
                    frame = passthrough(frame)
                # else: leave frame as-is. The SDK helper already AGC-
                # stretched u16→u8 before debayer, and any additional
                # tone mapping should go through enhance() opt-in.

        # 1. YOLO + ByteTrack (throttled).
        #
        # The hand-rolled IoU-matching tracker that used to live here was
        # ripped out in favour of ByteTrack (Ultralytics' built-in via
        # model.track(persist=True)). Kalman motion predictions + low-
        # confidence association mean IDs stay stable through fast pans,
        # brief occlusions, and motion blur — the exact failure mode
        # that was spamming the GUI with new #IDs every second.
        #
        # Tracks coast between YOLO ticks using the last-known ByteTrack
        # state (we just republish the last detection list until the
        # next classifier frame), so bboxes don't flicker off.
        # Async classify: hand the latest frame to the worker thread
        # if classifier is enabled and we're on a classify_every tick.
        # The worker runs YOLO + ByteTrack and posts results back to
        # _cls_out_dets; we read those below regardless of whether we
        # just submitted (results may be from a frame or two ago —
        # 50-100 ms detection lag, imperceptible to operators, and
        # the gimbal track predictor's Kalman smooths it out).
        run_classifier = (
            self._classifier is not None
            and (self._frame_id % self._classify_every == 0)
        )
        if run_classifier:
            with self._cls_in_cond:
                # Single-slot: if a frame is still pending, drop it.
                # The newer one is fresher and ByteTrack persist=True
                # handles the gap. Notify wakes the worker.
                #
                # When tiling is enabled, hand the classifier the
                # NATIVE 2472×2064 frame so per-tile inference gets
                # full sensor resolution. Otherwise hand it the
                # already-downscaled 1236-wide display frame.
                # 2026-05-14: classifier ALWAYS sees full native, never cropped.
                _full_native = locals().get("full_native_frame", native_frame)
                cls_input = _full_native if (self._tiling_enabled and
                                              _full_native is not None) else frame
                self._cls_in_pending = (self._frame_id, cls_input)
                self._cls_in_cond.notify()
        # Publish the worker's latest result (or empty list before any
        # result has come back). Coasting between worker updates is
        # the same coasting behaviour the synchronous path used to do.
        with self._cls_out_lock:
            self._last_dets = list(self._cls_out_dets)
            # Note whether THIS tick produced fresh classifier output
            # (so MOSSE can reseed on it instead of just running update).
            fresh_dets_this_tick = bool(self._cls_out_dets_fresh_flag)
            self._cls_out_dets_fresh_flag = False

        # ── Frame-rate MOSSE pool ────────────────────────────────
        # When YOLO is throttled (or its confidence drops during a
        # gimbal slew), `_last_dets` carries STALE bboxes from N frames
        # ago. The pool runs MOSSE on each ByteTrack ID every frame to
        # provide fresh, frame-rate per-target bboxes. On classifier
        # ticks: reseed each tracker on the new YOLO bbox (kills drift).
        # Between classifier ticks: tracker output replaces the stale
        # bbox in _last_dets. ByteTrack ID, conf, and class are
        # preserved — only the bbox is corrected by MOSSE.
        # ── Phase-correlation slew compensation (single source) ─────
        # Camera motion shifts the image globally; without compensation,
        # bboxes lag the actual target during gimbal pan. Measure the
        # true image-pixel shift via cv2.phaseCorrelate on a 192-wide
        # downsample (~1.7 ms on Xavier).
        #
        # GATING: state-based, NOT motion-based. We run phase-corr
        # every publish tick when there's anything to compensate (any
        # entry in _last_dets or any pending entry in _track_state).
        # We do NOT condition on "is the camera currently moving" —
        # by the time we observe motion, we've already published a
        # stale bbox for that frame. This is the architectural insight
        # from the post-mortem.
        #
        # Outputs:
        #   - _pc_last_dx_tick / _pc_last_dy_tick  (publish-thread only)
        #   - _pc_dx_disp_accum / _pc_dy_disp_accum (lock-protected,
        #     consumed and zeroed by the classifier worker before its
        #     next IoU match — preserves track IDs across pans).
        # Side effect: every entry in _last_dets has its bbox shifted
        # by (dx_tick, dy_tick) BEFORE MOSSE runs below — so MOSSE
        # reseed templates land on the actual target.
        _has_state = bool(self._last_dets) or bool(getattr(self, "_track_state", {}))
        _pc_dx = _pc_dy = 0.0
        try:
            if frame is not None and frame.size > 0:
                # Always compute small_gray when we have a frame, so
                # _prev_frame_small stays warm for the next tick when
                # state might appear. ~1 ms cost (resize+cvtColor).
                hh, ww = frame.shape[:2]
                target_w = self._of_target_w
                if ww > target_w:
                    scl = target_w / float(ww)
                    nw, nh = target_w, max(1, int(round(hh * scl)))
                    small = cv2.resize(frame, (nw, nh),
                                       interpolation=cv2.INTER_AREA)
                else:
                    small = frame
                    nw = ww
                if small.ndim == 3:
                    small_gray = cv2.cvtColor(
                        small, cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    small_gray = small.astype(np.float32)
                # phaseCorrelate only when we have state AND a prev
                # frame of matching shape. Skipped at cold-start
                # (next tick will have prev_frame_small populated).
                if (_has_state
                        and self._prev_frame_small is not None
                        and self._prev_frame_small.shape == small_gray.shape):
                    (dx_s, dy_s), _resp = cv2.phaseCorrelate(
                        self._prev_frame_small, small_gray)
                    scale_back = ww / float(nw)
                    _pc_dx_optical = float(dx_s) * scale_back
                    _pc_dy_optical = float(dy_s) * scale_back
                else:
                    _pc_dx_optical = _pc_dy_optical = 0.0
                # ALWAYS update prev (single source of truth).
                self._prev_frame_small = small_gray

                # 2026-05-14: BUS-derived shift is PRIMARY. Phase-corr
                # at fast slew undermeasures (AGC/blur degrades the
                # correlation peak). Independent _pc_bus_prev tracker
                # updated EVERY tick — NEVER suppressed by _trusted_pan.
                _pc_dx_bus = _pc_dy_bus = 0.0
                if (_has_state and bus_pan is not None and bus_tilt is not None
                        and self._pc_bus_prev_pan is not None
                        and self._pc_bus_prev_tilt is not None):
                    _bus_dpan  = float(bus_pan)  - float(self._pc_bus_prev_pan)
                    _bus_dtilt = float(bus_tilt) - float(self._pc_bus_prev_tilt)
                    zl = max(1, int(getattr(self, "_zoom_level", 1) or 1))
                    hfov_eff = float(self._hfov) / float(zl)
                    vfov_eff = float(self._vfov) / float(zl)
                    if hfov_eff > 0.01 and vfov_eff > 0.01:
                        _pc_dx_bus = -_bus_dpan  * (ww / hfov_eff)
                        _pc_dy_bus = -_bus_dtilt * (frame.shape[0] / vfov_eff)
                if bus_pan is not None and bus_tilt is not None:
                    self._pc_bus_prev_pan  = float(bus_pan)
                    self._pc_bus_prev_tilt = float(bus_tilt)

                # 2026-05-14: Cross-confirmation rule. Phase-corr at
                # 192-wide has ~5 px noise floor — phantom shifts at
                # rest cause overshoot. BUS over-reports motion (servo
                # commanded != actual). Use the smaller of the two
                # when they AGREE (same sign, magnitudes within 3x).
                # When they disagree → zero shift (defensive — bbox
                # stays put until both signals confirm motion).
                def _agree(a, b, max_ratio=3.0):
                    if a * b < 0: return False   # opposite signs
                    am, bm = abs(a), abs(b)
                    if am < 1.0 and bm < 1.0: return True  # both zero -> OK (no shift)
                    if am < 1.0 or bm < 1.0: return False  # one zero, other not
                    return max(am, bm) / max(min(am, bm), 0.001) <= max_ratio
                if _agree(_pc_dx_bus, _pc_dx_optical):
                    # Same sign, magnitudes consistent — trust the smaller.
                    _pc_dx = _pc_dx_optical if abs(_pc_dx_optical) <= abs(_pc_dx_bus) else _pc_dx_bus
                else:
                    _pc_dx = 0.0   # disagreement -> no shift
                if _agree(_pc_dy_bus, _pc_dy_optical):
                    _pc_dy = _pc_dy_optical if abs(_pc_dy_optical) <= abs(_pc_dy_bus) else _pc_dy_bus
                else:
                    _pc_dy = 0.0
            self._pc_last_dx_tick = _pc_dx
            self._pc_last_dy_tick = _pc_dy
            with self._pc_cache_lock:
                self._pc_dx_disp_accum += _pc_dx
                self._pc_dy_disp_accum += _pc_dy
            # Apply shift to every det in _last_dets BEFORE MOSSE.
            # MOSSE reseeds at d["bbox"]; if we don't shift first,
            # MOSSE re-anchors at the stale (pre-pan) position and
            # immediately drifts.
            if (abs(_pc_dx) > 1.0 or abs(_pc_dy) > 1.0) and self._last_dets:
                hh, ww = frame.shape[:2]
                kept = []
                _n_shifted = 0
                for d in self._last_dets:
                    bx, by, bw, bh = d["bbox"]
                    nx = int(round(bx + _pc_dx))
                    ny = int(round(by + _pc_dy))
                    # Drop tracks that panned off-screen.
                    if (nx + bw < 0 or nx > ww or
                            ny + bh < 0 or ny > hh):
                        continue
                    d["bbox"] = (nx, ny, bw, bh)
                    kept.append(d)
                    _n_shifted += 1
                self._last_dets = kept
                # Diagnostic: when shifted >5px, log both signals
                # and which path was chosen.
                if abs(_pc_dx) > 5.0 or abs(_pc_dy) > 5.0:
                    log.info("EO optflow: bus=(%.1f,%.1f) opt=(%.1f,%.1f) used=(%.1f,%.1f) n=%d fid=%d",
                              _pc_dx_bus, _pc_dy_bus,
                              _pc_dx_optical, _pc_dy_optical,
                              _pc_dx, _pc_dy, _n_shifted, self._frame_id)
        except Exception as _e:
            log.debug("phase-corr slew-comp skip: %s", _e)



        if self._mosse_pool.enabled:
            from vision.correlation_tracker_set import DetectorHit
            if fresh_dets_this_tick and self._last_dets:
                hits = []
                for d in self._last_dets:
                    raw_tid = d.get("track_id")
                    if raw_tid is None or int(raw_tid) < 0:
                        continue
                    # 2026-05-14b: Schmitt-emitted coast dets carry
                    # a stale YOLO bbox. Skipping them here keeps
                    # MOSSE template-matching from its last visual
                    # position, which is correct for moving targets.
                    if d.get("coasted"):
                        continue
                    bx, by, bw, bh = d["bbox"]
                    hits.append(DetectorHit(
                        track_id=int(raw_tid),
                        bbox_xywh=(int(bx), int(by), int(bw), int(bh)),
                    ))
                pool_out = self._mosse_pool.on_detector_tick(
                    frame, hits, self._frame_id)
            else:
                pool_out = self._mosse_pool.on_frame(frame, self._frame_id)
            # 2026-05-14: MOSSE bypass on big shifts. When the image
            # has shifted >30 px in this tick (rapid gimbal slew), the
            # MOSSE template search window walks off-target and PSR
            # collapses. The MOSSE-reported bbox is unreliable. Leave
            # _last_dets at the phase-corr-shifted position (from the
            # block above) — MOSSE re-spawns fresh trackers on the next
            # YOLO tick via on_detector_tick automatically.
            # 2026-05-14: bypass threshold raised 30 -> 200 px so MOSSE
            # gets to refine the OPT-based shift at moderate pans where
            # MOSSE template-match still works. MOSSE template walks off
            # only at very fast pans (>200 px shift between frames).
            _bypass_mosse = (abs(self._pc_last_dx_tick) > 200.0
                              or abs(self._pc_last_dy_tick) > 200.0)
            if not _bypass_mosse:
                # Override stale bboxes with MOSSE's frame-rate output.
                # IDs the pool dropped (lost streak) silently leave
                # _last_dets untouched — fusion's max_misses handles them.
                for d in self._last_dets:
                    tid = d.get("track_id")
                    if tid is None or int(tid) < 0:
                        continue
                    if int(tid) in pool_out.bboxes:
                        d["bbox"] = pool_out.bboxes[int(tid)]

        # 2. Republish last detection list every frame so overlays hold
        #    steady between YOLO ticks. ByteTrack's internal Kalman
        #    makes the coasting visually correct even at low classify
        #    rates because when a fresh tick lands, the new bbox lines
        #    up with where the track predicted it would be.
        out_dets: list[EODetection] = []
        for d in self._last_dets:
            bx, by, bw, bh = d["bbox"]
            try:
                tc = TargetClass(d["class"])
            except ValueError:
                tc = TargetClass.UNKNOWN
            # ByteTrack id sentinel cleanup (Phase A2): use None for
            # "no confirmed id yet" rather than the magic value -1.
            # Matches every other Optional[int] id field in the system
            # (FusedTrack.eo_track_id, etc.). Skip unconfirmed dets so
            # they don't flicker on the panel.
            raw_tid = d.get("track_id")
            if raw_tid is None or int(raw_tid) < 0:
                continue
            out_dets.append(EODetection(
                bbox=BBox(x=int(bx), y=int(by), w=int(bw), h=int(bh)),
                confidence=float(d["conf"]),
                target_class=tc,
                track_id=int(raw_tid),
            ))

        dev_idx = getattr(self._source, "device_index", None) if self._source else None
        # `initializing=True` means "AE is still searching for a usable
        # exposure — show the amber INITIALIZING scrim". We use the
        # latched `_ae_has_converged_once` flag rather than the bracket
        # state because a clean run (saturated → in-band) leaves
        # `low_floor` permanently None — the binary search never sampled
        # a "dim but not saturated" reading. An earlier version of this
        # check used `low_floor is None or high_brake is None` and got
        # stuck showing INITIALIZING forever on a perfectly converged AE.
        is_initializing = (
            self._ae_enabled
            and self._manual_exposure_ext is None
            and not self._ae_has_converged_once
        )
        # Latch the flag the first time we observe an in-band reading.
        if (self._ae_enabled and self._manual_exposure_ext is None
                and not self._ae_has_converged_once
                and self._ae_controller is not None
                and self._source is not None):
            try:
                stats = self._source.last_raw_stats
            except Exception:
                stats = None
            if stats is not None:
                p99 = float(stats.get("p99", 0.0))
                fc = float(stats.get("frac_clip", 0.0))
                if self._ae_controller.in_band(p99, fc):
                    first_time = not self._ae_has_converged_once
                    self._ae_has_converged_once = True
                    is_initializing = False
                    # Persist the converged value for next-session seed.
                    if self._ae_chosen_ext is not None:
                        _save_cached_ae_exposure(self._ae_chosen_ext)
                    log.info("EO AE converged (p99=%.0f, clip=%.3f, "
                             "exp=%s) — clearing INITIALIZING scrim",
                             p99, fc, self._ae_chosen_ext)
                    if first_time:
                        try:
                            from common.events import emit as _emit
                            _emit("ae_converged", {
                                "p99": float(p99),
                                "frac_clip": float(fc),
                                "exposure_ext": self._ae_chosen_ext,
                            })
                        except Exception:
                            pass
        # NOTE: EOFrame.initializing was rolled back from common/frames.py
        # by the user. We compute is_initializing above for AE-side state
        # tracking (latching _ae_has_converged_once) but no longer pass it
        # to EOFrame — the field doesn't exist on the wire schema right
        # now. If/when the INITIALIZING scrim is reintroduced, restore the
        # field on EOFrame and re-add the kwarg here.
        del is_initializing  # silences "unused" lint

        # ── Optical-pose feedback ────────────────────────────────
        # Compare consecutive grayscale frames via phase correlation.
        # When the BUS-reported pose moved by Δp_bus but the image only
        # shifted by Δp_optical, and the two disagree by more than a
        # threshold, trust the optics and suppress the BUS update for
        # the stamped pose-at-capture. Reason: lazy-servo deadband can
        # eat PWM commands so the maestro driver thinks it stepped (and
        # advances its _last_us, which feeds gimbal_state.pan/tilt) but
        # the camera physically didn't move. Without this override the
        # SAME static target gets a new fused-track ID every gated step
        # because fusion's world conversion uses the lying pose.
        # ('still not working.jsonl' showed 0→1.5° pan published with
        # zero pixel motion in the EO image content.)
        # ── BUS-trust override (uses phase-corr from earlier block) ─
        # Trust optical-flow over BUS pose: when BUS reports motion but
        # the image content didn't shift (lazy-servo deadband, stuck
        # servo), suppress the BUS update for the stamped pose-at-capture.
        # Reads _pc_last_dx_tick / _pc_last_dy_tick computed by the
        # single-source phase-corr block earlier in this method — no
        # re-compute, no second cv2.phaseCorrelate call.
        if (bus_pan is not None and bus_tilt is not None
                and self._prev_bus_pan is not None
                and self._prev_bus_tilt is not None
                and frame is not None and frame.size > 0):
            try:
                bus_dpan = bus_pan - self._prev_bus_pan
                bus_dtilt = bus_tilt - self._prev_bus_tilt
                if (abs(bus_dpan) > self._bus_motion_min_deg
                        or abs(bus_dtilt) > self._bus_motion_min_deg):
                    # _pc_last_d{x,y}_tick are in DISPLAY pixel space.
                    # Convert to degrees via display HFOV/VFOV.
                    fh, fw = frame.shape[:2]
                    zl = max(1, int(getattr(self, "_zoom_level", 1) or 1))
                    hfov_eff = float(self._hfov) / float(zl)
                    vfov_eff = float(self._vfov) / float(zl)
                    deg_per_px_h = hfov_eff / float(fw)
                    deg_per_px_v = vfov_eff / float(fh)
                    # Camera pan-right shifts content left, so dpan = -dx*deg/px.
                    dpan_optical = -self._pc_last_dx_tick * deg_per_px_h
                    dtilt_optical = self._pc_last_dy_tick * deg_per_px_v
                    if (abs(bus_dpan) > self._bus_motion_min_deg
                            and abs(dpan_optical) < self._optical_confirm_deg):
                        gimbal_pan_at_capture = self._prev_bus_pan
                    if (abs(bus_dtilt) > self._bus_motion_min_deg
                            and abs(dtilt_optical) < self._optical_confirm_deg):
                        gimbal_tilt_at_capture = self._prev_bus_tilt
                # Remember what we ACCEPTED as the bus pose so the next
                # comparison is against the override (not the raw BUS).
                self._prev_bus_pan = (gimbal_pan_at_capture
                                       if gimbal_pan_at_capture is not None
                                       else bus_pan)
                self._prev_bus_tilt = (gimbal_tilt_at_capture
                                        if gimbal_tilt_at_capture is not None
                                        else bus_tilt)
            except Exception as _opt_e:
                log.debug("EO optical pose feedback skipped: %r", _opt_e)
        elif bus_pan is not None and self._prev_bus_pan is None:
            self._prev_bus_pan = bus_pan
            self._prev_bus_tilt = bus_tilt

        # Encode the JPEG once, here, on the EO process thread. Removes
        # the dominant per-tick cost from the asyncio WS sender — see
        # gui/sensor_bridge.py:eo_to_wire and gui/app.py:_sender for the
        # consumers, which now reuse these bytes instead of re-encoding
        # the same frame on every WS tick. JSONL recorder also reuses
        # them via recording/encoders.encode_eo.
        jpeg_bytes: Optional[bytes] = None
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, int(self._eo_jpeg_quality)],
            )
            if ok:
                jpeg_bytes = bytes(buf)
        except Exception as e:
            log.warning("EO JPEG encode failed (frame_id=%d): %s",
                        self._frame_id, e)
            jpeg_bytes = None

        ef = EOFrame(
            timestamp=ts,
            frame_id=self._frame_id,
            connected=True,
            bgr=frame,
            detections=out_dets,
            hfov_deg=self._hfov / float(max(1, getattr(self, "_zoom_level", 1))),
            vfov_deg=self._vfov / float(max(1, getattr(self, "_zoom_level", 1))),
            source_device=dev_idx,
            gimbal_pan_at_capture=gimbal_pan_at_capture,
            gimbal_tilt_at_capture=gimbal_tilt_at_capture,
            jpeg_bytes=jpeg_bytes,
            jpeg_quality=int(self._eo_jpeg_quality),
            native_bgr=native_frame,
        )
        BUS.publish(Topic.EO, ef)

    # ───────────────────────── disconnected sentinel ─────────────────
    def _publish_disconnected(self) -> None:
        """Emit a disconnected/initializing EOFrame.

        We distinguish two no-frame states for the GUI:
        - `_switching_exposure` set → an AE step or user-initiated
          exposure change is restarting the SDK helper. The camera is
          fine; this is a planned blackout. Set `initializing=True` so
          the GUI shows "EO INITIALIZING…" instead of the alarming red
          DISCONNECTED pill.
        - software AE has not yet converged in this Auto session →
          same treatment. The first time the user points at a saturated
          daylight scene, the bracket controller does ~5-8 helper
          restarts to find a usable exposure (~30-55 s). During that
          window the frames are dark/black and the GUI should communicate
          "we're working on it" rather than "the camera is broken".
        - anything else → genuine disconnect (camera unplug, USB error).
          Leave `initializing=False`; the red pill is correct here.
        """
        self._frame_id += 1
        is_switching = bool(self._switching_exposure)
        ae_unconverged = (
            self._ae_enabled
            and self._manual_exposure_ext is None
            and (self._ae_controller is None
                 or self._ae_controller.low_floor is None
                 or self._ae_controller.high_brake is None)
        )
        # See note in _process_and_publish — initializing field rolled back
        # by user; computed here for future use but not passed to EOFrame.
        _ = is_switching or ae_unconverged
        ef = EOFrame(
            timestamp=time.time(),
            frame_id=self._frame_id,
            connected=False,
            hfov_deg=self._hfov,
            vfov_deg=self._vfov,
        )
        BUS.publish(Topic.EO, ef)

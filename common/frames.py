"""
Cross-module dataclasses.

Everything that crosses a module boundary is defined here. If a
field doesn't belong to an inter-module contract it should NOT be
here — keep module internals private to the module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np


# ───────────────────────────────────────────────────────────────
# Target classes
# ───────────────────────────────────────────────────────────────

class TargetClass(str, Enum):
    """High-level classification result for a detected blob.

    Inherits from `str` so it JSON-serializes cleanly into the
    WebSocket payload for the GUI.
    """
    UNKNOWN = "unknown"   # hot blob detected but unclassified
    HAND    = "hand"      # indoor test target (warm elongated blob)
    DRONE   = "drone"     # field target (warm compact blob)
    BIRD    = "bird"      # reserved for later phases
    NOISE   = "noise"     # classifier rejected as false alarm
    # Phase B — human+vehicle classifier (Ticket 1)
    PERSON  = "person"    # human detected by h/v classifier
    VEHICLE = "vehicle"   # vehicle detected by h/v classifier


# ───────────────────────────────────────────────────────────────
# Thermal frames
# ───────────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Axis-aligned pixel bounding box, inclusive top-left."""
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class ClassificationResult:
    target_class: TargetClass
    confidence: float               # 0..1
    classifier_used: str            # "yolo" | "shape_heuristic" | "none"


@dataclass
class ThermalDetection:
    """A single heat blob found in a thermal frame."""
    bbox: BBox
    area_px: int
    contrast: float                 # residual peak above background
    classification: Optional[ClassificationResult] = None


@dataclass
class ThermalFrame:
    """A fully-processed thermal frame published on the bus.

    raw16 is the 16-bit thermal counts image (may be None in fake
    sources). agc8 is the 8-bit AGC-normalized image used for
    display and classifier input. detections is the list of heat
    blobs found by the heat detector.

    `connected=False` is the sentinel "camera is gone" frame that
    the GUI uses to flip its status pill red without crashing.
    """
    timestamp: float                # time.time() wall clock
    frame_id: int
    connected: bool
    raw16: Optional[np.ndarray] = None   # uint16, shape (H, W)
    agc8: Optional[np.ndarray] = None    # uint8,  shape (H, W, 3) BGR
    detections: List[ThermalDetection] = field(default_factory=list)

    # Metadata about the FOV / crop the GUI needs to render overlays
    hfov_deg: float = 75.0
    vfov_deg: float = 60.0
    zoom_preset: str = "full"


# ───────────────────────────────────────────────────────────────
# EO (visible / NIR RGB) frames — Ticket 3
# ───────────────────────────────────────────────────────────────

@dataclass
class EODetection:
    """A single YOLO detection in an EO frame (person / vehicle)."""
    bbox: BBox
    confidence: float
    target_class: TargetClass   # PERSON | VEHICLE | UNKNOWN
    track_id: Optional[int] = None


@dataclass
class EOFrame:
    """A fully-processed EO (webcam) frame published on the bus.

    Mirrors ThermalFrame but carries a BGR uint8 display image
    instead of a raw16 thermal image.

    `connected=False` is the "camera is gone" sentinel — the GUI
    uses it to flip the EO status pill red without crashing.
    """
    timestamp: float
    frame_id: int
    connected: bool
    bgr: Optional[np.ndarray] = None   # uint8, shape (H, W, 3)
    detections: List[EODetection] = field(default_factory=list)

    # IMX568 (2472x2064 @ 2.74um -> 6.77x5.65mm active) + Commonlands
    # CIL350 (35mm EFL) — narrow telephoto.
    # HFOV = 2*atan(6.77/2/35) ≈ 11.05°
    # VFOV = 2*atan(5.65/2/35) ≈ 9.23°
    # DFOV ≈ 14.4° (inside the 11mm image-circle spec).
    # Per-pixel IFOV is ~7x finer than thermal's 75° — gimbal precision.
    hfov_deg: float = 11.05
    vfov_deg: float = 9.23
    source_device: Optional[int] = None  # cv2 device index in use


# ───────────────────────────────────────────────────────────────
# Fused tracks — Ticket 5 (cross-sensor fusion)
# ───────────────────────────────────────────────────────────────

@dataclass
class FusedTrack:
    """A single real-world target confirmed by one or more sensors.

    All positions are expressed in *angular space* relative to the
    boresight (az positive = right, el positive = up). We store
    angles because thermal and EO have very different pixel grids
    but a shared optical axis on the bench; angles are the sensor-
    neutral coordinate.

    The GUI layer re-projects (az, el, ang_w, ang_h) into each
    sensor's pixel grid to draw the "same" bbox on every panel.
    """
    id: int
    target_class: TargetClass
    confidence: float                   # best conf across contributing sensors
    sensors: List[str]                  # subset of ["eo", "thermal", "radar"]
    primary: str                        # whichever sensor supplied angles
    az_deg: float
    el_deg: float
    ang_w_deg: float
    ang_h_deg: float
    hits: int = 1
    misses: int = 0


# ───────────────────────────────────────────────────────────────
# Radar frames (Phase A: stub shape only, real fields added in Phase B)
# ───────────────────────────────────────────────────────────────

@dataclass
class RadarFrame:
    timestamp: float
    frame_id: int
    connected: bool
    # populated in Phase B:
    detections: list = field(default_factory=list)


# ───────────────────────────────────────────────────────────────
# Topic constants — use these instead of stringly-typed bus keys
# ───────────────────────────────────────────────────────────────

class Topic:
    THERMAL = "thermal"
    EO      = "eo"
    RADAR   = "radar"
    FUSED   = "fused"
    GIMBAL  = "gimbal"


# ───────────────────────────────────────────────────────────────
# Gimbal — Ticket 4 (pan/tilt servo controller)
# ───────────────────────────────────────────────────────────────

@dataclass
class GimbalState:
    """Latest state of the pan/tilt gimbal.

    Angles are in degrees in the sensor frame:
        pan_deg  — positive = right of boresight, 0 = centered
        tilt_deg — positive = up from horizon,    0 = horizontal
                   (mechanical range typically 0..22° on this rig)

    ``mode`` is "auto" when the gimbal is tracking a fused-track ID
    the user pressed TRACK on, "manual" otherwise. ``connected`` is
    False when the Maestro USB device isn't present — the rest of
    the app keeps running, the GUI just disables the dpad.
    """
    timestamp: float
    connected: bool
    pan_deg: float
    tilt_deg: float
    mode: str = "manual"          # "manual" | "auto"
    target_pan_deg: float = 0.0   # commanded setpoint (may lag actual)
    target_tilt_deg: float = 0.0
    tracked_target_id: Optional[int] = None
    error: Optional[str] = None   # last error string, or None

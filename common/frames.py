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
    RADAR   = "radar"
    FUSED   = "fused"

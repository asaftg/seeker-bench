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
    # True when the detection was seeded by a user "Draw Target" bbox
    # rather than by the heat detector. Synthetic detections propagate
    # through the tracker via optical flow only (no warmth requirement),
    # are skipped by classifiers, and are rendered as a magenta dashed
    # "USER TARGET" box in the GUI. Debug/demo handle for objects that
    # aren't hot (parked cars, trees, etc.).
    synthetic: bool = False


@dataclass
class HeatTrackDebug:
    """Debug-mode view of a heat-blob tracker entry.

    Only populated on ThermalFrame when the developer-mode toggle in
    the GUI is on (cheap enough to compute always, but we keep the wire
    field empty by default to avoid sending dozens of dicts every frame
    for users who don't care).
    """
    id: int
    bbox: BBox
    hits: int
    misses: int
    age: int
    confirmed: bool       # has reached min_hits — would be rendered as a production box
    coasting: bool        # missed this tick — coasting on last known position
    synthetic: bool = False  # user "Draw Target" seed (propagated via OF only)


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

    # Developer-mode only: raw state of the heat-blob tracker, including
    # unconfirmed and coasting tracks. Populated unconditionally by
    # ThermalManager; the GUI chooses whether to render based on its
    # own devMode flag.
    heat_tracks: List[HeatTrackDebug] = field(default_factory=list)


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
# Radar frames (AWR2944P mmw_demoDDM — Ticket 5a)
# ───────────────────────────────────────────────────────────────

@dataclass
class RadarDetection:
    """A single pre-detected point from the AWR2944P point cloud.

    Coordinates are in the sensor frame (metres):
        +x = right, +y = forward (boresight), +z = up.
    Doppler is signed: positive = approaching the sensor.

    range_m / az_deg / el_deg are filled in by the parser from (x,y,z)
    so the GUI / fusion never has to redo that trig.

    ``target_id`` is the DBSCAN cluster ID this point was assigned to
    by RadarManager (or 255 = unassigned / noise). Not a firmware
    Group-Tracker ID — the AWR2944P mmw_demoDDM build does not link
    gtrack.
    """
    x_m: float
    y_m: float
    z_m: float
    doppler_mps: float
    snr_db: float
    noise_db: float = 0.0
    range_m: float = 0.0
    az_deg: float = 0.0
    el_deg: float = 0.0
    target_id: int = 255


@dataclass
class RadarTarget:
    """A clustered radar target — rendered as a bounding box.

    All targets from this pipeline carry the single semantic class
    ``"radar_detection"`` (see project memory / Ticket 5a design).
    Vehicle / human classification is a late-fusion job against the
    EO + thermal panels, not radar-side.

    Size fields are bbox half-extents in metres (the full box spans
    pos ± size on each axis).

    ``source`` distinguishes where the box came from — currently
    always ``"dbscan"`` on this firmware build; reserved for
    ``"tracker"`` once/if a gtrack-linked firmware arrives.
    """
    tid: int
    pos_x_m: float
    pos_y_m: float
    pos_z_m: float
    vel_x_mps: float
    vel_y_mps: float
    vel_z_mps: float
    size_x_m: float = 0.5
    size_y_m: float = 0.5
    size_z_m: float = 0.5
    confidence: float = 1.0
    source: str = "dbscan"
    num_points: int = 0


@dataclass
class RadarFrame:
    """A parsed mmw_demo frame published on the bus.

    ``connected=False`` is the sentinel "radar is gone" frame used by
    the GUI to flip the radar panel to DISCONNECTED — matches the
    ThermalFrame / EOFrame convention.
    """
    timestamp: float
    frame_id: int
    connected: bool
    detections: List["RadarDetection"] = field(default_factory=list)
    targets: List["RadarTarget"] = field(default_factory=list)
    profile: str = ""
    num_points: int = 0
    num_targets: int = 0
    # Max range used by the firmware profile — the GUI scales the
    # polar canvas by this. Defaults to 50 m (stock DDM highRange
    # profile range limit).
    max_range_m: float = 50.0


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

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
    # Phase 2 fusion — radar contributor with no classification of its own.
    # A FusedTrack born from a radar-only observation carries this class
    # until EO/thermal confirms it (at which point the class is upgraded
    # to PERSON/VEHICLE/DRONE and locked).
    RADAR_TARGET = "radar_target"


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
    # Heat-track id from ``DetectionTracker`` (the per-sensor classical
    # CV tracker). Stamped onto the ThermalDetection at emit time so
    # downstream consumers — fusion, the GUI panel — can link a raw
    # det back to its persistent track id without bbox-IoU matching.
    # None on first-frame dets that haven't reached confirmation yet.
    track_id: Optional[int] = None


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

    # Gimbal pose at the time this frame entered the processing pipeline.
    # ThermalManager stamps these from BUS.get_latest(Topic.GIMBAL) at the
    # top of _process_and_publish, before detection runs. Fusion uses
    # these to convert each detection's az/el into world frame using the
    # pose that was actually true when the sensor saw the target —
    # NOT the pose at fusion-tick time, which can drift by 1-2° during a
    # fast slew and births phantom track IDs ('revert not helping ghosts'
    # cluster analysis showed the same physical target reborn 3-6 times
    # under different IDs because of this offset).
    # Optional[float]; None when the gimbal/state stream isn't yet
    # available (first frames after startup) — fusion falls back to its
    # legacy fusion-tick pose snapshot in that case.
    gimbal_pan_at_capture: Optional[float] = None
    gimbal_tilt_at_capture: Optional[float] = None


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

    # Software-AE / source-restart sentinel. Distinct from `connected`:
    # connected=False means "camera unplugged or hard failure"; while
    # initializing=True means "camera is fine, AE is bracketing toward a
    # usable exposure or the source is being restarted to commit a new
    # ExposureExt." The GUI shows a friendly "EO INITIALIZING…" badge
    # for this state instead of the alarming red DISCONNECTED pill.
    initializing: bool = False

    # IMX568 (2472x2064 @ 2.74um -> 6.77x5.65mm active) + Commonlands
    # CIL350 (35mm EFL) — narrow telephoto.
    # HFOV = 2*atan(6.77/2/35) ≈ 11.05°
    # VFOV = 2*atan(5.65/2/35) ≈ 9.23°
    # DFOV ≈ 14.4° (inside the 11mm image-circle spec).
    # Per-pixel IFOV is ~7x finer than thermal's 75° — gimbal precision.
    hfov_deg: float = 11.05
    vfov_deg: float = 9.23
    source_device: Optional[int] = None  # cv2 device index in use

    # See ThermalFrame.gimbal_pan_at_capture for the contract — same
    # purpose, populated by EOManager at process-start time.
    gimbal_pan_at_capture: Optional[float] = None
    gimbal_tilt_at_capture: Optional[float] = None

    # Pre-encoded JPEG bytes of `bgr` at `jpeg_quality`. Encoded ONCE on
    # the EO process thread so the asyncio WS sender doesn't pay the
    # cv2.imencode + base64 cost per tick (re-encoding the same frame
    # on every WS tick was the dominant per-tick cost — see
    # gui/sensor_bridge.py:eo_to_wire). Consumers that need the JPEG
    # (WS sender, JSONL recorder) reuse these bytes; consumers that
    # need raw pixels (fusion, gimbal, replay) keep using `bgr`.
    # None when no encode happened yet (disconnected sentinel frames).
    jpeg_bytes: Optional[bytes] = None
    jpeg_quality: int = 92


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
    # World-frame az/el of the track. Populated by fusion when running
    # in world-frame mode; None otherwise (legacy camera-frame fusion).
    # Why: az_deg/el_deg is camera-frame, computed in fusion as
    # (world − cur_pan_at_fusion_publish). A consumer that wants world
    # angles back has to add cur_pan again, but reading cur_pan at a
    # different moment (gimbal control loop runs faster than fusion's
    # 15Hz tick) leaks the gimbal-publish latency as PHANTOM VELOCITY in
    # the consumer's world-frame estimate. On `track test 6.jsonl` this
    # caused a static target's apparent world_az_dot ≈ -13 dps during
    # a slew, and the Phase-3 velocity feed-forward turned that into
    # runaway over-steering. Carrying world angles directly here removes
    # the round-trip.
    world_az_deg: Optional[float] = None
    world_el_deg: Optional[float] = None
    # Per-sensor tracker IDs that contributed to this fused track.
    # Lets the GUI label raw per-sensor detections with the same
    # fused id by direct id match (robust under EMA smoothing of the
    # fused track's stored angles, which IoU matching is not).
    # All three are stamped at observation-time and overwritten on
    # every fresh observation from that sensor; None when that sensor
    # has not contributed to this track recently.
    eo_track_id:      Optional[int] = None
    thermal_heat_id:  Optional[int] = None  # DetectionTracker heat-track id
    radar_tid:        Optional[int] = None  # RadarClusterer Kalman tid


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
    # Tracker state — populated by RadarClusterer's Kalman tracker.
    # ``coasting`` = True when this target was NOT matched to a fresh
    # DBSCAN cluster this frame; its position is Kalman-predicted from
    # the last hit. The GUI renders coasting boxes dashed/dim so the
    # operator can tell a dead-reckoned track from a measured one.
    coasting: bool = False
    hits: int = 0
    misses: int = 0


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
    # Half-angle (deg) of the azimuth gate applied in radar_manager.
    # The GUI uses this to draw the ±FOV sector lines so the operator
    # sees exactly which wedge is "in-gate".
    fov_half_deg: float = 60.0

    # See ThermalFrame.gimbal_pan_at_capture — same contract, populated
    # by RadarManager at frame ingest.
    gimbal_pan_at_capture: Optional[float] = None
    gimbal_tilt_at_capture: Optional[float] = None


# ───────────────────────────────────────────────────────────────
# Topic constants — use these instead of stringly-typed bus keys
# ───────────────────────────────────────────────────────────────

class Topic:
    THERMAL = "thermal"
    EO      = "eo"
    RADAR   = "radar"
    # Phase 3 raw-ADC path: PMM detector hits + A/G CFAR detections
    # from the host-side DCA1000 pipeline. Separate topic so stock
    # mode (pure TLV) and A/A overlay can both be read by the GUI
    # without the pipelines stepping on each other.
    RADAR_AA = "radar_aa"
    FUSED   = "fused"
    GIMBAL  = "gimbal"
    # Discrete event stream — user actions, system transitions, algo
    # diagnostics. Each publish is a free-form dict
    # {"type": str, "payload": {...}}; the recorder writes it to JSONL
    # and never inspects the payload. Don't bake schema into this topic.
    EVENTS  = "events"


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
    # Synthetic-target world-frame lock (None when no synth lock is
    # active). When set, the GUI/sensor_bridge can compute the
    # synthetic bbox image position directly from these world angles
    # plus the current gimbal pose, bypassing the OF tracker — which
    # is what the bbox-drift fix relies on. Captured once at synth
    # lock commit and held until track release.
    synth_world_az_deg: Optional[float] = None
    synth_world_el_deg: Optional[float] = None
    # Where the synth-locked world target *actually* appears in the
    # camera frame right now, in degrees off boresight. Computed from
    # the LK-measured camera motion since the synth anchor (so it
    # reflects PHYSICAL pose, not the controller's commanded pose
    # which can lie when the servo isn't following). sensor_bridge
    # uses these to position the synthetic bbox in the WS payload.
    # None when no LK measurement is available.
    target_resid_az_deg: Optional[float] = None
    target_resid_el_deg: Optional[float] = None
    # Lock-mode tracker output (gimbal.lock_mode.enabled).
    # When the operator presses TRACK on a fused track, gimbal_manager
    # spawns a per-sensor MOSSE lock tracker seeded from that track's
    # bbox content. The lock survives YOLO/heat/fusion dropouts —
    # bbox is published every gimbal tick regardless of classifier
    # state. None when lock mode is OFF (config disabled) or the
    # operator hasn't engaged a track. See vision/lock_tracker.py
    # for the state machine. lock_state is one of:
    #   "off" / "active" / "coasting" / "released"
    # GUI renders lock_bbox_eo / lock_bbox_thermal with priority over
    # the projected fused-track bbox; "coasting" gets an amber edge.
    lock_state: str = "off"
    lock_bbox_eo: Optional["BBox"] = None
    lock_bbox_thermal: Optional["BBox"] = None

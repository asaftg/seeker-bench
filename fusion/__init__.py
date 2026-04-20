"""Cross-sensor fusion.

FusionManager reads the latest ThermalFrame + EOFrame from the bus,
associates detections by angular position and class, and publishes a
list of FusedTrack objects (global IDs, angular position, contributing
sensor set) on Topic.FUSED.

Sensor priority (the "primary" sensor whose angles we trust):
    EO > Thermal > Radar

The GUI uses FusedTrack to draw a single green bbox on every panel
for targets confirmed by 2+ sensors, with the centroid of that bbox
being the world direction the drone should fly.
"""

"""Build per-second HOVER/MOVING/OFFSCREEN timeline for airborne1 drone recording.

Manually-curated visual labels from inspecting EO + thermal JPEGs at ~3s
intervals over the 152s recording. The drone is a DJI FPV quadcopter.

Phase definitions (per task spec):
- offscreen: drone not in frame
- moving: drone position changes meaningfully between consecutive sampled frames
         (or rapid blur indicates motion)
- hover: drone visible, position essentially stationary across consecutive frames
         (frame-to-frame translation < ~1/8 frame width). Slow apparent
         growth/shrink from drone moving along camera's line-of-sight (radial)
         or operator panning is still labeled hover - the drone itself is not
         translating laterally in 3D world frame as far as we can tell from
         EO/thermal alone.
- transition: ambiguous (drone entering/exiting frame, motion blur)

Camera EVENT at ~t=99s: operator zooms in on drone (trees disappear from FOV,
drone appears as bigger bright dot in narrower FOV). Hover analysis below
already accounts for this (drone position barely changes in zoomed FOV either).

The drone exits the (zoomed) FOV between t=112 and t=118, then never returns
through end of recording at t=152.

Output: ../recordings/airborne1_hover_timeline.csv
"""

from __future__ import annotations

import csv
import os

# (t_rel_s, phase, position, confidence, notes)
# Sample timestamps drawn from EO frame filenames (3s grid, ~51 samples).
# Each row represents the labeled state AT that timestamp (a 3s window
# centered on it is a reasonable interpretation but the CSV row is per-sample).
SAMPLES = [
    # 0-21s: drone not in frame (clear sky, no quadcopter visible in EO or thermal)
    (0.0,   "offscreen",  "",              "high", "empty sky, drone not yet in scene"),
    (3.1,   "offscreen",  "",              "high", "empty sky"),
    (6.1,   "offscreen",  "",              "high", "empty sky"),
    (9.1,   "offscreen",  "",              "high", "empty sky"),
    (12.2,  "offscreen",  "",              "high", "empty sky"),
    (15.2,  "offscreen",  "",              "high", "empty sky"),
    (18.2,  "offscreen",  "",              "high", "empty sky"),
    (21.3,  "offscreen",  "",              "high", "empty sky, drone enters between this sample and next"),
    # 24-30s: drone enters fast, big & close, motion-blurred (MOVING/transition)
    (24.3,  "transition", "lower_left",    "med",  "drone enters frame, big and close (lower-left in EO)"),
    (27.4,  "moving",     "lower_left",    "high", "drone clearly visible, X-shape, blurry (motion blur from rapid flight)"),
    (30.4,  "moving",     "lower_mid",     "high", "drone visible mid-frame, motion blur, transitioning to hover"),
    # 33-97s: classic hover - drone position barely changes, slowly recedes
    # (gets smaller as it moves further from operator), camera tracks loosely.
    # In thermal it's a stationary tiny dot at center for ~70 seconds.
    (33.4,  "hover",      "lower_mid",     "high", "drone holds position lower-mid, small quadcopter visible"),
    (36.5,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (39.5,  "hover",      "lower_mid",     "high", "stationary, lower-mid (slightly smaller -> further away)"),
    (42.6,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (45.6,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (48.6,  "hover",      "lower_mid",     "high", "stationary, lower-mid (drone is small dot)"),
    (51.7,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (54.7,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (57.7,  "hover",      "lower_mid",     "high", "stationary, lower-mid"),
    (60.8,  "hover",      "lower_mid",     "high", "stationary, lower-mid (small dot in EO and thermal)"),
    (63.8,  "hover",      "lower_mid",     "high", "stationary"),
    (66.9,  "hover",      "lower_mid",     "high", "stationary"),
    (69.9,  "hover",      "lower_mid",     "high", "stationary"),
    (72.9,  "hover",      "lower_mid",     "high", "stationary, small dot"),
    (76.0,  "hover",      "lower_mid",     "high", "stationary"),
    (79.0,  "hover",      "lower_mid",     "high", "stationary"),
    (82.0,  "hover",      "lower_mid",     "high", "stationary, drone dot starting to grow slightly (approaching radially)"),
    (85.1,  "hover",      "lower_mid",     "high", "stationary, drone slightly larger (approaching)"),
    (88.1,  "hover",      "lower_mid",     "high", "stationary, drone larger"),
    (91.2,  "hover",      "lower_mid",     "high", "stationary, drone larger"),
    (94.2,  "hover",      "lower_mid",     "high", "stationary, drone clearly larger now (closer)"),
    (97.2,  "hover",      "lower_mid",     "high", "stationary, drone visibly bigger (DJI FPV quadcopter shape clear)"),
    # 99-105s: camera zooms IN (trees disappear from thermal). Drone now appears
    # as bright larger dot in narrow FOV. Position still center -> still hover,
    # but frame-to-frame the drone shifts within the zoomed FOV (zoom amplifies
    # any motion). Labeling as hover with med confidence due to zoom transition.
    (100.3, "transition", "center",        "med",  "camera zoom event - thermal FOV narrows, drone visible center"),
    (103.3, "moving",     "mid_right",     "med",  "in zoomed view, drone has shifted to mid-right, slightly higher"),
    (106.3, "moving",     "center",        "med",  "drone shifts back toward center (still in zoomed view)"),
    (109.4, "hover",      "lower_mid",     "med",  "drone settled lower-mid in zoomed FOV"),
    (112.4, "hover",      "lower_mid",     "med",  "drone holding lower-mid in zoomed FOV (last clear sighting)"),
    # 115-152s: drone has flown out of (zoomed) FOV. Operator never re-acquires.
    (115.5, "transition", "",              "med",  "drone exits FOV between 112 and 118"),
    (118.5, "offscreen",  "",              "high", "empty (zoomed view, dark sky)"),
    (121.5, "offscreen",  "",              "high", "empty"),
    (124.6, "offscreen",  "",              "high", "empty"),
    (127.6, "offscreen",  "",              "high", "empty"),
    (130.6, "offscreen",  "",              "high", "empty (camera zoomed back out, normal sky FOV)"),
    (133.7, "offscreen",  "",              "high", "empty sky"),
    (136.7, "offscreen",  "",              "high", "empty sky"),
    (139.8, "offscreen",  "",              "high", "empty sky"),
    (142.8, "offscreen",  "",              "high", "empty sky"),
    (145.8, "offscreen",  "",              "high", "empty sky"),
    (148.9, "offscreen",  "",              "high", "empty sky"),
    (151.9, "offscreen",  "",              "high", "empty sky (end of recording)"),
]


def write_csv(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "t_rel_s",
            "phase",
            "drone_position_in_frame",
            "confidence",
            "notes",
        ])
        for t, phase, pos, conf, notes in SAMPLES:
            w.writerow([f"{t:.1f}", phase, pos, conf, notes])


def summarize() -> None:
    """Print summary using simple integration: each sample 'owns' the time
    until the next sample (so duration = next_t - this_t). Last sample has
    nominal 3s duration so we cover the full recording."""
    total_dur = 0.0
    by_phase = {"offscreen": 0.0, "hover": 0.0, "moving": 0.0, "transition": 0.0}

    times = [s[0] for s in SAMPLES]
    phases = [s[1] for s in SAMPLES]
    n = len(SAMPLES)
    for i in range(n):
        if i + 1 < n:
            d = times[i + 1] - times[i]
        else:
            d = 3.0  # last sample window
        by_phase[phases[i]] += d
        total_dur += d

    print(f"Total recording duration ~= {total_dur:.1f} s")
    print(f"  HOVER:      {by_phase['hover']:6.1f} s  ({by_phase['hover']/total_dur*100:.1f}%)")
    print(f"  MOVING:     {by_phase['moving']:6.1f} s  ({by_phase['moving']/total_dur*100:.1f}%)")
    print(f"  OFFSCREEN:  {by_phase['offscreen']:6.1f} s  ({by_phase['offscreen']/total_dur*100:.1f}%)")
    print(f"  TRANSITION: {by_phase['transition']:6.1f} s  ({by_phase['transition']/total_dur*100:.1f}%)")
    print()

    # Build segments by run-length encoding the phases.
    print("Segments:")
    seg_start = times[0]
    seg_phase = phases[0]
    segments: list[tuple[float, float, str]] = []
    for i in range(1, n):
        if phases[i] != seg_phase:
            seg_end = times[i]
            segments.append((seg_start, seg_end, seg_phase))
            seg_start = times[i]
            seg_phase = phases[i]
    seg_end = times[-1] + 3.0
    segments.append((seg_start, seg_end, seg_phase))

    print("\nClean HOVER segments (>= 5 s):")
    for s, e, p in segments:
        if p == "hover" and (e - s) >= 5.0:
            print(f"  hover     [{s:6.1f} -> {e:6.1f}]  duration {e - s:5.1f} s")

    print("\nMOVING segments:")
    for s, e, p in segments:
        if p == "moving":
            print(f"  moving    [{s:6.1f} -> {e:6.1f}]  duration {e - s:5.1f} s")

    print("\nAll segments (RLE):")
    for s, e, p in segments:
        print(f"  {p:11s} [{s:6.1f} -> {e:6.1f}]  duration {e - s:5.1f} s")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.normpath(os.path.join(here, "..", "recordings", "airborne1_hover_timeline.csv"))
    write_csv(out)
    print(f"Wrote {out}")
    print()
    summarize()

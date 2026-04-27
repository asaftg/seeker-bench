"""Probe which UVC properties the FX3 bridge ACTUALLY honors.

The bridge has a track record of silently ignoring property writes
(CAP_PROP_CONVERT_RGB on this rig, CAP_PROP_EXPOSURE earlier in this
session — got readback=-13.0 when we asked for -6.0). So before we
bake any "manual exposure" / "fixed gain" code into the capture path,
prove which properties actually change the device state.

Method: open the camera with cv2.CAP_DSHOW, read every UVC property,
write a non-default value to each, read it back, log the deltas.
Anything where the readback CHANGED from the initial read is a knob
we can use; anything that didn't change is a knob the bridge ignores.

Run with seeker stopped so we have exclusive access to the device.
"""
from __future__ import annotations
import sys

import cv2

PROPS = [
    ("BRIGHTNESS",      cv2.CAP_PROP_BRIGHTNESS),
    ("CONTRAST",        cv2.CAP_PROP_CONTRAST),
    ("SATURATION",      cv2.CAP_PROP_SATURATION),
    ("HUE",             cv2.CAP_PROP_HUE),
    ("GAIN",            cv2.CAP_PROP_GAIN),
    ("EXPOSURE",        cv2.CAP_PROP_EXPOSURE),
    ("AUTO_EXPOSURE",   cv2.CAP_PROP_AUTO_EXPOSURE),
    ("GAMMA",           cv2.CAP_PROP_GAMMA),
    ("SHARPNESS",       cv2.CAP_PROP_SHARPNESS),
    ("BACKLIGHT",       cv2.CAP_PROP_BACKLIGHT),
    ("AUTO_WB",         cv2.CAP_PROP_AUTO_WB),
    ("WB_TEMPERATURE",  cv2.CAP_PROP_WB_TEMPERATURE),
    ("ISO_SPEED",       cv2.CAP_PROP_ISO_SPEED),
    ("CONVERT_RGB",     cv2.CAP_PROP_CONVERT_RGB),
    ("FPS",             cv2.CAP_PROP_FPS),
    ("BUFFERSIZE",      cv2.CAP_PROP_BUFFERSIZE),
]

# Test write values per property — these are "obviously different from
# default" so a successful write produces a measurable readback delta.
TEST_WRITES = {
    "BRIGHTNESS":   [0.0, 0.5, 128.0, -64.0, 64.0],
    "CONTRAST":     [0.0, 32.0, 64.0, 1.0],
    "GAIN":         [0.0, 1.0, 8.0, 16.0, 64.0],
    "EXPOSURE":     [-13.0, -10.0, -8.0, -6.0, -4.0, -2.0],
    "AUTO_EXPOSURE":[0.0, 0.25, 0.5, 0.75, 1.0, 3.0],
    "GAMMA":        [50.0, 100.0, 150.0, 220.0],
    "SHARPNESS":    [0.0, 1.0, 2.0, 4.0],
    "BACKLIGHT":    [0.0, 1.0, 2.0],
    "ISO_SPEED":    [100.0, 400.0, 800.0],
    "CONVERT_RGB":  [0.0, 1.0],
    "BUFFERSIZE":   [1.0, 2.0, 4.0],
}


def main() -> int:
    print("Opening device 0 with CAP_DSHOW...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("FAIL: could not open device 0")
        return 1

    print("\n=== INITIAL PROPERTY VALUES ===")
    initial = {}
    for name, prop in PROPS:
        v = cap.get(prop)
        initial[name] = v
        print(f"  {name:<18s} = {v}")

    print("\n=== WRITE TESTS — looking for properties that actually move ===")
    accepted: list[str] = []
    rejected: list[str] = []
    for name, prop in PROPS:
        writes = TEST_WRITES.get(name)
        if not writes:
            continue
        before = cap.get(prop)
        for v in writes:
            cap.set(prop, v)
            after = cap.get(prop)
            mark = "[CHANGED]" if abs(after - before) > 1e-6 else "[no-op]"
            print(f"  {name:<18s} set {v!r:>8s} -> readback {after!r:<10s} {mark}")
            if abs(after - before) > 1e-6:
                if name not in accepted:
                    accepted.append(name)
                before = after
        # restore initial after testing this property
        cap.set(prop, initial[name])
        # if nothing moved, mark as rejected
        if name not in accepted and name not in rejected:
            rejected.append(name)

    print("\n=== SUMMARY ===")
    print(f"Properties the bridge ACCEPTS (readback changed): {accepted}")
    print(f"Properties the bridge IGNORES (readback static):  {rejected}")

    cap.release()
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

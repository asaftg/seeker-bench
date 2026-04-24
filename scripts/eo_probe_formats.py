"""Enumerate EVERY video format the LI-IMX568 UVC descriptor advertises.

Goes deeper than cv2 -- uses pygrabber (DirectShow wrapper) to read the
device's raw media type list. If this shows only a few degraded YUY2
modes, the sensor is genuinely not initialized (Leopard driver needed).
If it shows many modes (MJPG, raw Bayer, high-res YUY2), then the
driver is fine and we had a cv2 format-negotiation problem.

    pip install pygrabber
    python -m scripts.eo_probe_formats
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        from pygrabber.dshow_graph import FilterGraph
    except Exception as e:
        print(f"pygrabber missing: {e}")
        print("Install it with:  pip install pygrabber")
        return 1

    g = FilterGraph()
    devs = g.get_input_devices()
    print("== DirectShow video capture devices ==")
    for i, name in enumerate(devs):
        print(f"  [{i}] {name}")
    print()

    if not devs:
        print("No DSHOW devices found.")
        return 1

    for i, name in enumerate(devs):
        # We only care about the LI kit; thermal is already handled.
        if "IMX" not in name and "Leopard" not in name and "LI-" not in name:
            continue
        print(f"== Format list for [{i}] {name} ==")
        try:
            g.add_video_input_device(i)
            caps = g.get_input_device().get_formats()
        except Exception as e:
            print(f"  failed to enumerate formats: {e}")
            continue
        if not caps:
            print("  (empty format list)")
        else:
            for row in caps:
                print(f"  {row}")
        print()

    print("Interpretation:")
    print("  Many entries with MJPG/YUY2/Bayer at 1920x1080 or 2464x2064")
    print("    -> sensor is initialized; driver is NOT the blocker.")
    print("  Only a handful of YUY2 modes at <=1280x960")
    print("    -> sensor is in a crippled fallback mode; Leopard driver")
    print("       IS the blocker and that email was the right call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

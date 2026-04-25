"""Grab a frame from the IMX568 using PyAV (ffmpeg) instead of OpenCV.

Why: OpenCV's DirectShow backend on this FX3 bridge hands back a
per-channel-constant flat buffer (every pixel equal to the scene's mean
brightness) — CameraTool shows a real scene but cv2.VideoCapture gives
us a grey rectangle whose value tracks the scene average. PyAV talks
to DirectShow through ffmpeg, which uses a different filter graph and
decoder pipeline; if the raw USB stream is fine, PyAV should see it.

Usage::

    pip install av
    python scripts/eo_pyav_probe.py

Saves ``scripts/eo_snapshots/imx568_pyav.png`` + per-channel stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    import av
except ImportError:
    print("PyAV not installed. Run: pip install av")
    sys.exit(1)

import cv2
import numpy as np

OUT_DIR = Path(__file__).resolve().parent / "eo_snapshots"
OUT_DIR.mkdir(exist_ok=True)

# Friendly name as Windows Device Manager shows it. If this doesn't match
# what ffmpeg sees, run `ffmpeg -list_devices true -f dshow -i dummy` to
# get the actual string.
DEVICE_NAME = "LI-IMX568"


def list_devices() -> None:
    """Print the DirectShow device names ffmpeg can see.

    PyAV bundles ffmpeg as a DLL, not a separate .exe, so we can't shell
    out. Instead we force an open of ``video=dummy`` which ffmpeg always
    rejects — but before rejecting, its verbose logging prints the full
    device list to the ffmpeg log callback. We enable av's verbose log
    level, then catch the expected failure.
    """
    print("DirectShow devices (forcing ffmpeg to enumerate):")
    av.logging.set_level(av.logging.VERBOSE)
    try:
        av.open("video=__list_devices__",
                format="dshow",
                options={"list_devices": "true"})
    except Exception as e:
        # Expected: ffmpeg writes the list to its log, then raises.
        # The list goes to stderr-ish output during the call; nothing
        # we can capture programmatically without a log callback.
        print(f"  ({type(e).__name__}: {e})")
    av.logging.set_level(av.logging.WARNING)
    print("  (If you saw device names printed above, use the EXACT "
          "friendly name — e.g. python scripts/eo_pyav_probe.py "
          "\"LI-USB30-IMX568 M\")")
    print("  (If nothing printed, PyAV's log callback is off; "
          "check Device Manager > Cameras for the exact name.)")


def try_open(device: str) -> bool:
    """Try both "just open" (let ffmpeg pick defaults) and "forced native"
    modes. PyAV 17 removed ``av.AVError`` — we catch broad Exception so
    the script survives whatever PyAV decides to raise.
    """
    print(f"\n== Trying DirectShow device: {device!r} ==")
    # First pass: no format forcing — lets ffmpeg negotiate whatever the
    # bridge offers (often a smaller resolution, but we just want SOME
    # real frame to verify the decode path).
    attempts = [
        ("defaults", {}),
        ("forced 2472x2064 yuyv422",
         {"video_size": "2472x2064",
          "pixel_format": "yuyv422",
          "framerate": "20"}),
        ("forced 2472x2064 mjpeg",
         {"video_size": "2472x2064",
          "pixel_format": "mjpeg",
          "framerate": "20"}),
    ]
    for label, opts in attempts:
        print(f"  -- {label} opts={opts}")
        try:
            container = av.open(
                f"video={device}",
                format="dshow",
                options=opts,
            )
        except Exception as e:
            print(f"     av.open failed: {type(e).__name__}: {e}")
            continue
        try:
            if _read_and_report(container):
                container.close()
                return True
        finally:
            try:
                container.close()
            except Exception:
                pass
    return False


def _read_and_report(container) -> bool:
    stream = container.streams.video[0]
    print(f"     stream: codec={stream.codec_context.name} "
          f"{stream.width}x{stream.height} pix_fmt={stream.pix_fmt}")

    frame_count = 0
    for packet in container.demux(stream):
        for frame in packet.decode():
            arr = frame.to_ndarray(format="bgr24")
            frame_count += 1
            if frame_count >= 5:
                # Sample a few frames so we're past any decoder warmup.
                h, w = arr.shape[:2]
                stats = []
                for ch, name in enumerate(("B", "G", "R")):
                    c = arr[:, :, ch]
                    stats.append(
                        f"{name}: min={c.min():3d} max={c.max():3d} "
                        f"mean={c.mean():6.1f} std={c.std():6.1f}"
                    )
                print(f"  frame {frame_count}: shape=({h},{w},3)")
                for s in stats:
                    print(f"    {s}")

                luma = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                per_ch_std = [float(arr[:, :, c].std()) for c in range(3)]
                print(f"    Y: min={luma.min()} max={luma.max()} "
                      f"mean={luma.mean():.1f} std={luma.std():.1f}")
                print(f"    per-channel std = {per_ch_std}")
                if max(per_ch_std) > 2.0:
                    print("  >>> REAL SCENE CONTENT — PyAV works <<<")
                else:
                    print("  >>> Frame is flat. PyAV sees same failure "
                          "as OpenCV; sensor is producing a constant "
                          "buffer at the USB layer. <<<")

                out = OUT_DIR / "imx568_pyav.png"
                cv2.imwrite(str(out), arr)
                print(f"     saved -> {out}")
                return True
        if frame_count >= 5:
            break

    return False


if __name__ == "__main__":
    list_devices()

    # Try common names the bridge might expose itself as.
    candidates = [
        "LI-IMX568",
        "USB Camera",
        "USB Video Device",
        "LI-USB30-IMX568",
        "IMX568",
    ]
    if len(sys.argv) > 1:
        candidates.insert(0, sys.argv[1])

    for dev in candidates:
        try:
            if try_open(dev):
                break
        except Exception as e:
            print(f"  exception for {dev!r}: {e}")

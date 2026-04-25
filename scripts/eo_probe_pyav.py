"""Probe whether PyAV can open the FX3 bridge in raw YUY2 (yuyv422)
without DirectShow's broken auto-decode in OpenCV's path.

PyAV bypasses OpenCV entirely — it goes ffmpeg → libavdevice (dshow) →
libavformat → us. ffmpeg's dshow demuxer is a different implementation
than OpenCV's, so it might honor yuyv422 where OpenCV silently swaps
in BGR.

Strategy:
1. Use ffmpeg to list dshow video devices, pick the IMX/Leopard one.
2. Open it with av.open(..., format='dshow', options={pixel_format,
   video_size}) and grab one frame.
3. If the frame format is 'yuyv422', extract Y plane and report stats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    import av
except Exception as e:
    print(f"PyAV import failed: {e!r}")
    sys.exit(2)


def _find_device_name() -> str | None:
    """Walk dshow's device list via ffmpeg and pick the IMX/Leopard one."""
    # av's "list devices" emits to stderr — capture it.
    import subprocess, imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        cp = subprocess.run(
            [exe, "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        print(f"ffmpeg device-list call failed: {e!r}")
        return None
    out = (cp.stderr or "") + "\n" + (cp.stdout or "")
    print("---- ffmpeg dshow device list ----")
    print(out)
    print("---- /list ----\n")
    # Lines look like:  [dshow @ ...] "Device Name" (video)
    candidates = []
    for line in out.splitlines():
        if "(video)" not in line.lower():
            continue
        if '"' not in line:
            continue
        name = line.split('"', 2)[1]
        candidates.append(name)
    print(f"video device candidates: {candidates}")
    for name in candidates:
        n = name.lower()
        if "imx" in n or "leopard" in n or "li-" in n or "fx3" in n:
            return name
    if candidates:
        return candidates[0]
    return None


def _probe(device_name: str) -> int:
    print(f"\nProbing dshow device: {device_name!r}")
    options = {
        # ask for the raw 4:2:2 layout the bridge actually delivers.
        "pixel_format": "yuyv422",
        "video_size": "2472x2064",
        "rtbufsize": "256M",
    }
    try:
        container = av.open(
            f"video={device_name}", format="dshow", options=options,
        )
    except Exception as e:
        print(f"av.open failed: {e!r}")
        return 1

    streams = [s for s in container.streams if s.type == "video"]
    if not streams:
        print("no video streams")
        container.close()
        return 1

    vs = streams[0]
    print(f"stream codec: {vs.codec_context.name}, "
          f"pix_fmt: {vs.codec_context.pix_fmt}, "
          f"size: {vs.codec_context.width}x{vs.codec_context.height}")

    # Pull a few frames so the bridge's AE settles.
    frame = None
    for i, packet in enumerate(container.demux(vs)):
        for f in packet.decode():
            frame = f
        if frame is not None and i >= 4:
            break

    if frame is None:
        print("no frames decoded")
        container.close()
        return 1

    print(f"\nframe format: {frame.format.name}, "
          f"size: {frame.width}x{frame.height}, "
          f"planes: {len(frame.planes)}")
    for i, p in enumerate(frame.planes):
        print(f"  plane[{i}]: line_size={p.line_size}, "
              f"buffer_size={p.buffer_size}")

    if frame.format.name in ("yuyv422", "yuv422p"):
        # YUY2 packed = Y0 U0 Y1 V0 Y2 U1 Y3 V1 ...
        # plane 0 is the packed buffer for yuyv422.
        if frame.format.name == "yuyv422":
            buf = np.frombuffer(bytes(frame.planes[0]), dtype=np.uint8)
            row = frame.planes[0].line_size
            packed = buf[: row * frame.height].reshape(frame.height, row)
            # take the Y bytes (every other byte starting at 0, up to 2*W)
            y = packed[:, : 2 * frame.width : 2].copy()
        else:  # yuv422p — Y plane is plane 0 directly
            buf = np.frombuffer(bytes(frame.planes[0]), dtype=np.uint8)
            row = frame.planes[0].line_size
            y = buf[: row * frame.height].reshape(frame.height, row)[:, : frame.width].copy()
        print(f"\nRAW Y plane: shape={y.shape} dtype={y.dtype} "
              f"min={y.min()} max={y.max()} mean={y.mean():.1f} "
              f"std={y.std():.1f}")
        out_dir = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "pyav_raw_y.png"
        try:
            import cv2
            cv2.imwrite(str(out_path), y)
            print(f"saved {out_path}")
        except Exception as e:
            print(f"could not write png: {e!r}")
        container.close()
        return 0
    else:
        print(f"unexpected frame format {frame.format.name} — not YUY2")
        container.close()
        return 1


def main() -> int:
    name = _find_device_name()
    if not name:
        print("no dshow video device found — is the camera plugged in / "
              "is CameraTool closed?")
        return 2
    return _probe(name)


if __name__ == "__main__":
    sys.exit(main())

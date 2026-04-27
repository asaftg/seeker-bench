"""LeopardSDKStreamCapture — 64-bit consumer of the 32-bit Leopard
SDK streaming helper.

Why
---
The FX3 USB3 bridge exposes two output paths:

  1. UVC preview pin: 2592×1944 YUV with internal AGC + debayer baked
     in. PyAV/ffmpeg-dshow streams this. Visible breathing, noisy
     YUY2 reconstruction. This was our previous path.

  2. Vendor SDK pin (LeopardCamera.dll, 32-bit only): 2472×2064 RAW12
     direct from the sensor, exactly the data CameraTool saves to
     .raw / .bmp. No internal AGC, no debayer, no breathing.

This class drives path #2. The SDK is 32-bit and the seeker is 64-bit,
so we spawn ``leopard_sdk_helper.py --stream`` under
``tools/python311-x86/python.exe`` as a long-lived subprocess and
read framed binary payloads off its stdout.

Frame format on the wire
------------------------
    [4-byte little-endian uint32 = payload length N]
    [N bytes raw payload]

Payload is RAW12 packed as uint16 LE (raw_value << 4). To recover the
12-bit pixel value: ``raw12 = (uint16 >> 4)``. Values are Bayer mosaic
in BG pattern (cv2.COLOR_BAYER_BG2BGR) — empirically determined by
matching channel ratios to a Leopard CameraTool reference BMP captured
of the identical scene (verified 2026-04-24).

Header (on stderr)
------------------
The helper writes one line ``STREAM_HDR {json}`` to stderr after init
and before the first frame. The parent reads it to learn W, H, etc.

Public surface
--------------
    cap = LeopardSDKStreamCapture(exposure_ext=2000)
    cap.start()
    while True:
        frame = cap.grab()              # BGR uint8 (H, W, 3)
        if frame is None: break
        ...
    cap.stop()

The object is thread-safe for ``stop()``; ``grab()`` is single-consumer.

Failure modes
-------------
    - Helper subprocess dies → grab() returns None and a flag is set.
    - PyAV / CameraTool / VLC / Windows Camera holding the device →
      helper's SDK Open() fails. We surface a clear error.
"""
from __future__ import annotations

import io
import json
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
PY32 = REPO / "tools" / "python311-x86" / "python.exe"


class LeopardSDKStreamCapture:
    """Spawn helper, consume framed RAW12 stream, hand back BGR frames."""

    def __init__(
        self,
        exposure_ext: Optional[int] = None,
        width: int = 2472,
        height: int = 2064,
        bayer_pattern: Optional[int] = None,
        ae: str = "on",
        stream_fps: float = 10.0,
        warmup: int = 12,
    ) -> None:
        # bayer_pattern=None means "this is a monochrome sensor — DO NOT
        # debayer; broadcast the single-channel u8 to BGR instead." The
        # default used to be cv2.COLOR_BAYER_BG2BGR which made sense
        # only on the IMX568 *color* variant (LI-IMX568-GMSL2-C). Our
        # rig has the *monochrome* variant (LI-IMX568-GMSL2-M, no Bayer
        # filter array) — running a 2x2 demosaic on uniform mono pixels
        # produces tiny interpolation artifacts and a half-resolution
        # luma plane. Default flipped to None on 2026-04-25. Pass an
        # explicit cv2.COLOR_BAYER_xx2BGR if you ever attach a color
        # variant of the same sensor.
        # exposure_ext=None + ae="on" → bridge auto-exposure, the safe
        # default for "I'm going to point this at an arbitrary scene
        # tomorrow morning" use. Set exposure_ext to a specific int +
        # ae="off" to lock the camera to a calibrated manual value
        # (used by sdk_capture_and_compare.py to reproduce the
        # leopard_reference.bmp at sub-decimal-point precision).
        self.exposure_ext = (None if exposure_ext is None
                             else int(exposure_ext))
        self.ae = str(ae)
        self.width = int(width)
        self.height = int(height)
        self.bayer_pattern = bayer_pattern
        self.stream_fps = float(stream_fps)
        self.warmup = int(warmup)
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._header: Optional[dict] = None
        self._stop_flag = threading.Event()
        self._first_frame_seen = False
        self._frames_grabbed = 0
        # Per-frame raw u16 statistics, refreshed by every successful
        # grab(). Read by EOManager's software AE thread, which uses
        # raw u16 p99 to drive ExposureExt — a much more stable signal
        # than the post-AGC 8-bit mean. Kept as a plain dict (no lock)
        # because it's read-only from outside and Python attribute
        # writes are atomic enough for monitoring; the AE polls every
        # ~1 s, not every frame. None until the first frame arrives.
        # Schema:
        #   {"mean": float, "std": float, "p99": float, "max": int,
        #    "frac_clip": float (0..1), "seq": int}
        self.last_raw_stats: Optional[dict] = None

    # ───────────── lifecycle ─────────────

    def start(self, header_timeout_s: float = 8.0) -> None:
        if self._proc is not None:
            return
        cmd = [
            str(PY32), str(HELPER),
            "--ae", self.ae,
            "--data-mode", "RAW12",
            "--capture-width", str(self.width),
            "--capture-height", str(self.height),
            "--capture-warmup", str(self.warmup),
            "--stream-fps", str(self.stream_fps),
            "--stream",
        ]
        # Only pass --exposure-ext when we have a concrete value; with
        # ae="on" + no exposure_ext the bridge runs its own auto-
        # exposure loop, which is what we want for an arbitrary scene
        # (sunlit outdoor → dim indoor) without recompiling.
        if self.exposure_ext is not None:
            # Insert AFTER the `--ae <value>` pair (positions 2 and 3)
            # so we don't break the argparse pairing of `--ae` with its
            # choice argument.
            cmd[4:4] = ["--exposure-ext", str(self.exposure_ext)]
        log.info("LeopardSDKStreamCapture: spawn %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            bufsize=0,                    # unbuffered binary
        )
        # stderr drain thread: parses STREAM_HDR + logs everything else
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        # wait for header
        t0 = time.time()
        while self._header is None:
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"helper exited before sending header "
                    f"(rc={self._proc.returncode})")
            if time.time() - t0 > header_timeout_s:
                raise RuntimeError(
                    f"timed out ({header_timeout_s}s) waiting for "
                    f"STREAM_HDR from helper")
            time.sleep(0.05)
        log.info("LeopardSDKStreamCapture: header=%s", self._header)

    def stop(self) -> None:
        self._stop_flag.set()
        if self._proc is None:
            return
        try:
            self._proc.terminate()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=3.0)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        self._proc = None

    # ───────────── stderr drain ─────────────

    def _drain_stderr(self) -> None:
        try:
            for raw in iter(self._proc.stderr.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                if line.startswith("STREAM_HDR "):
                    try:
                        self._header = json.loads(line[len("STREAM_HDR "):])
                    except Exception:
                        log.warning("bad STREAM_HDR: %s", line)
                elif line.startswith("STREAM_END "):
                    log.info("helper end: %s", line[len("STREAM_END "):])
                elif line.startswith("STREAM_ERR "):
                    log.warning("helper: %s", line)
                else:
                    # General log line from helper
                    log.debug("helper: %s", line)
        except Exception as e:
            log.warning("stderr drain ended: %r", e)

    # ───────────── frame I/O ─────────────

    def _read_exact(self, n: int) -> Optional[bytes]:
        out = bytearray()
        while len(out) < n:
            if self._stop_flag.is_set():
                return None
            chunk = self._proc.stdout.read(n - len(out))
            if not chunk:
                return None
            out.extend(chunk)
        return bytes(out)

    def grab(self) -> Optional[np.ndarray]:
        """Return the next frame as BGR uint8 (H, W, 3). None on EOF.

        Per-frame side effects:
          - ``self.last_raw_stats`` updated with raw u16 statistics
            (mean / p99 / max / frac_clip) computed on a strided
            sample of the full 5 MP frame. This is what the software
            AE in EOManager reads. Stride = 8 (~78 k samples) keeps
            the cost <1 ms per frame.
          - ``self._frames_grabbed`` incremented.

        Decode pipeline (rewritten 2026-04-25 to handle daylight)
        ---------------------------------------------------------
        Earlier history: ``raw8 = np.clip(raw12, 0, 255)`` worked only
        because ``raw12 = (u16 >> 4)`` and the bridge was running its
        own AE that pinned scenes near a low mean — the 8-bit clip
        was effectively a no-op. In bright daylight through the 35 mm
        NIR-pass lens, the bridge AE is unable to drop exposure low
        enough; raw u16 lands at the analog ceiling (4095) and >>4
        gives 255 everywhere — pure white screen. (See
        ``scripts/sdk_daylight_diagnostic.py`` 2026-04-25.)

        New decode: AGC-stretch directly from raw u16. p1→0, p99→255,
        clipped. This produces a usable display image at *any* non-
        analog-clipped exposure (raw u16 p99 anywhere in [50, 4090]),
        which is exactly the regime our software AE drives the sensor
        into. Saturated frames still display white because p99 is at
        the ceiling; that's the correct visual feedback to the operator
        that AE hasn't converged yet.
        """
        if self._proc is None:
            return None
        # Frame length
        hdr = self._read_exact(4)
        if hdr is None or len(hdr) < 4:
            return None
        n = struct.unpack("<I", hdr)[0]
        expected = self.width * self.height * 2
        if n != expected:
            log.warning("frame length %d != expected %d", n, expected)
        payload = self._read_exact(n)
        if payload is None or len(payload) < n:
            return None
        # u16 LE buffer. Empirically (2026-04-25) the SDK packs raw12
        # in the LOW 12 bits of each uint16: u16 ∈ [0, 4095]. The low
        # nibble is always 0xF (the SDK's padding choice), which we
        # don't care about — AGC normalization absorbs that constant.
        u16 = np.frombuffer(payload, dtype="<u2").reshape(
            self.height, self.width)

        # ── raw stats on a strided sample (cheap, runs every frame) ──
        sample = u16[::8, ::8]
        s_p1 = float(np.percentile(sample, 1))
        s_p99 = float(np.percentile(sample, 99))
        s_mean = float(sample.mean())
        s_max = int(sample.max())
        # frac_clip: pixels at/near the analog ceiling. 4080 chosen
        # over 4095 because the AGC stretch can saturate quantization
        # the last 1-2 LSB; treat anything >= 4080 as analog-saturated.
        s_frac_clip = float((sample >= 4080).mean())
        self.last_raw_stats = {
            "mean": s_mean, "p1": s_p1, "p99": s_p99,
            "max": s_max, "frac_clip": s_frac_clip,
            "seq": self._frames_grabbed,
        }

        # ── AGC stretch u16 → u8 (p1, p99) → (0, 255). ──
        # Guard against degenerate range (uniform frame): force a
        # minimum span so we never divide by zero. With a 4-LSB span
        # the result is essentially black or white anyway, which is
        # correct visual feedback for "frame is saturated" or "frame
        # is at noise floor".
        span = max(s_p99 - s_p1, 4.0)
        # Keep math in float32 then cast — np.clip on uint16 with a
        # fractional offset does odd integer rounding.
        scaled = (u16.astype(np.float32) - s_p1) * (255.0 / span)
        raw8 = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
        # Mono → BGR. Two cases:
        #   bayer_pattern is None (default, mono sensor) → broadcast the
        #     single-channel intensity to all 3 BGR channels. No demosaic,
        #     full sensor resolution preserved, true grayscale output.
        #   bayer_pattern is a cv2.COLOR_BAYER_* constant → run that
        #     demosaic. Only use when an actual color-variant sensor is
        #     attached; on the LI-IMX568-GMSL2-M (mono) this would
        #     spread uniform pixels over a 2×2 grid via interpolation
        #     and produce phantom color noise + half resolution.
        if self.bayer_pattern is None:
            bgr = cv2.cvtColor(raw8, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(raw8, self.bayer_pattern)
        self._frames_grabbed += 1
        if not self._first_frame_seen:
            self._first_frame_seen = True
            log.info("first SDK stream frame: shape=%s mean=%.1f "
                     "raw u16 stats: mean=%.0f p99=%.0f max=%d clip=%.1f%%",
                     bgr.shape, float(bgr.mean()),
                     s_mean, s_p99, s_max, s_frac_clip * 100)
        return bgr

    # convenience
    def grab_gray(self) -> Optional[np.ndarray]:
        f = self.grab()
        if f is None:
            return None
        return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None


# ─────────────────────── standalone smoke test ──────────────────────

def _main():
    """python -m eo.leopard_stream_capture — opens the stream, displays
    live frames in an OpenCV window. Ctrl-C / 'q' to exit.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--exposure-ext", type=int, default=2000)
    ap.add_argument("--max-frames", type=int, default=120)
    ap.add_argument("--save-frame", default=None,
                    help="Save the Nth frame to this BMP path then exit.")
    ap.add_argument("--save-frame-n", type=int, default=15)
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    cap = LeopardSDKStreamCapture(exposure_ext=args.exposure_ext)
    cap.start()
    t0 = time.time()
    n = 0
    try:
        while n < args.max_frames:
            f = cap.grab()
            if f is None:
                print("EOF or helper died")
                break
            n += 1
            fps = n / max(time.time() - t0, 0.001)
            if args.save_frame and n == args.save_frame_n:
                cv2.imwrite(args.save_frame, f)
                print(f"saved frame #{n} -> {args.save_frame}")
                break
            if not args.no_display:
                # Resize for display
                h, w = f.shape[:2]
                disp = cv2.resize(f, (w // 2, h // 2))
                cv2.imshow("Leopard SDK stream", disp)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if n % 5 == 0:
                print(f"  frame {n:>4d}  {fps:.2f} fps  "
                      f"mean={f.mean():.1f}")
    finally:
        cap.stop()
        if not args.no_display:
            cv2.destroyAllWindows()
    print(f"done: {n} frames in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    _main()

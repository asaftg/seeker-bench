"""LI-IMX568-GMSL2 USB3 capture driver.

Native 2472x2064 global-shutter mono sensor via Leopard Imaging's
FX3-based EVA bridge. The bridge advertises a YUY2-labeled UVC stream
on DirectShow; Windows auto-decodes the payload to BGR uint8, which is
what OpenCV hands us. We treat the camera as mono and use the Y
channel (via cvtColor) for AGC / detection input.

API mirrors ``eo.webcam_capture.WebcamCapture`` so ``EOManager`` can
consume either source interchangeably:

    start() / stop() / grab() -> BGR uint8 (H, W, 3)

Extensions beyond WebcamCapture:
  * set_exposure_ms(x) / set_gain(x) — runtime profile control
  * get_scene_mean() — cheap luminance probe the profile-selector uses

Standalone::

    python -m eo.imx568_capture             # auto-detect index, 5s probe
    python -m eo.imx568_capture --device 0  # force index
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import cv2
import numpy as np

from common.logging_setup import get_logger

log = get_logger(__name__)


# Native resolution confirmed via eo_imx568_probe.py — anything lower
# and the FX3 bridge falls back to a cropped/binned mode we don't want.
NATIVE_W = 2472
NATIVE_H = 2064

# Pixel formats to try, in order of preference. YUY2 first because
# ``scripts/eo_list_formats.py`` showed it's the only carrier the FX3
# bridge actually delivers real pixels on — every other FOURCC we request
# gets silently remapped to YUY2 by DirectShow anyway, but asking for MJPG
# explicitly has been observed to hand back one good frame and then a
# stream of black frames (seen 2026-04-24: "scene mean=147.9" on frame 1,
# "scene mean=0.0" four seconds later). ``None`` = "don't force a
# format, let DirectShow negotiate whatever the driver's default is" —
# last-ditch fallback for a future bridge firmware.
# Keep this list in sync with scripts/eo_list_formats.py.
_FOURCC_CANDIDATES: list[Optional[str]] = ["YUY2", None]

# Capture backends to try, in order. DSHOW is the only backend that
# actually opens the FX3 bridge on this laptop — MSMF returns
# isOpened()==False every time (tested 2026-04-24 across five device
# indices), and CAP_ANY just delegates back to DSHOW anyway. Keeping
# DSHOW alone avoids a per-reconnect 100 ms × N-indices stall we'd
# otherwise eat probing dead backends every few seconds.
_BACKEND_CANDIDATES: list[tuple[int, str]] = [
    (cv2.CAP_DSHOW, "DSHOW"),
]


def _fourcc_to_str(fourcc_int: int) -> str:
    """Decode a 4-byte FOURCC integer (as returned by CAP_PROP_FOURCC)."""
    try:
        return "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    except Exception:
        return "?"


def _measure_frame_brightness(frame: np.ndarray) -> tuple[float, float]:
    """Return (recovered Y mean, G-saturation fraction) on a 16× subsample.

    Mirrors the trust-weighted recovery in eo_processor._to_luma so the
    AE loop sees the same "true Y" the user sees on screen, not the raw
    BGR mean (which is ~150 for a saturated green frame whose true Y is
    actually 240+).
    """
    if frame is None or frame.size == 0:
        return 0.0, 0.0
    slab = frame[::16, ::16]
    if slab.ndim == 3 and slab.shape[2] == 3:
        g = slab[..., 1].astype(np.float32)
        r = slab[..., 2].astype(np.float32)
        gw = np.clip((255.0 - g) / 15.0, 0.0, 1.0)
        rw = np.clip(r / 15.0, 0.0, 1.0)
        yg = np.clip(g - 135.0, 0.0, 255.0)
        yr = np.clip(r + 179.0, 0.0, 255.0)
        tw = np.maximum(gw + rw, 1e-3)
        y = (yg * gw + yr * rw) / tw
        sat = float((g >= 254.0).mean())
        return float(y.mean()), sat
    # 2-D fallback (raw YUY2 mode shouldn't reach here, but be safe).
    return float(slab.astype(np.float32).mean()), 0.0


def _probe_software_ae_direction(
    cap: "cv2.VideoCapture",
) -> tuple[int, float]:
    """One-shot startup probe: does CAP_PROP_EXPOSURE respond, and which
    direction is brighter?

    Strategy: flip to manual AE, write two test exposures separated by
    several stops, and watch the recovered-Y mean move. Returns:
        (sign, baseline_value) where
            sign = +1  if higher CAP_PROP_EXPOSURE → brighter (UVC convention)
            sign = -1  if higher CAP_PROP_EXPOSURE → dimmer (some bridges)
            sign =  0  if neither test value moved the mean — bridge
                       ignores the writes; software AE is hopeless on
                       this hardware and we restore bridge AE.

    Total cost ~1 s. Logs are explicit so the developer can SEE which
    branch fired without re-running with a debugger.
    """
    def _settle_and_sample(value: float) -> Optional[tuple[float, float]]:
        try:
            cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        except Exception:
            return None
        time.sleep(0.25)
        last_frame = None
        for _ in range(5):
            ok, fr = cap.read()
            if ok and fr is not None:
                last_frame = fr
        if last_frame is None:
            return None
        return _measure_frame_brightness(last_frame)

    # Flip to manual AE first — without this, the bridge's internal AE
    # may overwrite our exposure writes inside the 250 ms settle window.
    try:
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception as e:
        log.info("software AE probe: AE=manual write threw %r", e)
        return 0, -7.0

    # Two probe values, ~3 stops apart. -7 ≈ 7.8 ms (typical indoor),
    # -10 ≈ 1 ms (much darker). On a working bridge with conventional
    # UVC sign, -7 should be brighter than -10.
    sample_low = _settle_and_sample(-10.0)
    sample_high = _settle_and_sample(-7.0)

    if sample_low is None or sample_high is None:
        log.info("software AE probe: no frames after exposure write — disabling")
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass
        return 0, -7.0

    y_low, sat_low = sample_low
    y_high, sat_high = sample_high
    delta = y_high - y_low

    if abs(delta) < 5.0:
        log.warning(
            "software AE probe: bridge does not respond to CAP_PROP_EXPOSURE "
            "(Y at -10s=%.1f, Y at -7s=%.1f, delta=%.1f) — leaving bridge AE on",
            y_low, y_high, delta,
        )
        try:
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        except Exception:
            pass
        return 0, -7.0

    sign = 1 if delta > 0 else -1
    # Pick the start exposure on the DIM side. The AE loop will brighten
    # if we overshoot, but starting bright + walking down means we spend
    # the first few frames in saturation — visible as a momentary white
    # flash on the GUI. Start at the darker end and let the loop find
    # the target mean from below.
    if sign > 0:
        baseline = -12.0  # conventional bridge: darker = more negative
    else:
        baseline = -3.0   # reversed bridge: darker = more positive
    try:
        cap.set(cv2.CAP_PROP_EXPOSURE, baseline)
    except Exception:
        pass
    log.info(
        "software AE probe: sign=%+d (delta=%.1f), baseline exposure=%.1f log₂s",
        sign, delta, baseline,
    )
    return sign, baseline


# ───────────────────── PyAV / ffmpeg-dshow backend ────────────────────────
#
# OpenCV's DirectShow path on this OpenCV build silently ignores
# CAP_PROP_CONVERT_RGB=0 on this bridge's UVC stream, so it ALWAYS hands
# us the destructive YUY2->BGR auto-decode (B=Y-227, G=Y+135, R=Y-179
# clipped) which permanently destroys the Y∈[120,179] band. Probed
# 2026-04-24 across {DSHOW, MSMF} × {flag-before-open, after-format,
# after-grab} × {refourcc on/off}: every combo gave broken BGR.
#
# PyAV bypasses OpenCV entirely — it goes through ffmpeg's libavdevice
# dshow input, which honors `pixel_format=yuyv422` and hands back the
# raw YUY2 bytes. That gives us the same clean Y plane Leopard's
# CameraTool sees (verified by scripts/eo_probe_pyav.py: min=0 max=255
# mean=79 std=71 on a real workshop scene where the OpenCV path
# delivered min=0 max=255 mean=202 with G saturated everywhere).
#
# This wrapper exposes the cv2.VideoCapture-shaped subset that grab()
# relies on (read / release / isOpened) so the rest of IMX568Capture
# is unchanged. read() returns the raw (H, 2*W) YUY2 packed buffer;
# the existing _raw_yuy2_mode path slices Y from it.

try:
    import av as _av  # PyAV — pip-installable, bundles libavdevice
    _PYAV_AVAILABLE = True
except Exception:  # pragma: no cover — env-specific
    _av = None
    _PYAV_AVAILABLE = False


class _PyAVDshowBackend:
    """cv2.VideoCapture-lookalike that pipes raw YUY2 from ffmpeg-dshow.

    The interface is intentionally narrow — only what IMX568Capture.grab()
    and stop() touch — so swapping it in for ``cv2.VideoCapture`` is
    transparent to the rest of the class. set()/get() are no-ops because
    DirectShow exposure control lives in IAMCameraControl, which ffmpeg's
    dshow demuxer does not expose; software AE on the PyAV path is left
    for a follow-up (the bridge's hardware AE already produces a usable
    image, the dead-zone problem is what we came here to fix).
    """

    def __init__(self, device_name: str, width: int, height: int) -> None:
        self._device_name = device_name
        self._w = width
        self._h = height
        self._container: Optional["_av.container.InputContainer"] = None
        self._stream = None
        self._demux_iter = None
        # Buffer one decoded packet's worth of frames so we can return
        # them on subsequent read() calls without re-demuxing.
        self._frame_queue: list[np.ndarray] = []

    def open(self) -> bool:
        if not _PYAV_AVAILABLE:
            return False
        try:
            self._container = _av.open(
                f"video={self._device_name}",
                format="dshow",
                options={
                    "pixel_format": "yuyv422",
                    "video_size": f"{self._w}x{self._h}",
                    "rtbufsize": "256M",
                },
            )
        except Exception as e:
            log.info("PyAV dshow open failed for %r: %r", self._device_name, e)
            self._container = None
            return False
        streams = [s for s in self._container.streams if s.type == "video"]
        if not streams:
            self._container.close()
            self._container = None
            return False
        self._stream = streams[0]
        # Confirm the demuxer actually negotiated YUY2 — if it fell back
        # to something else, abort so we use the OpenCV fallback path
        # instead of silently producing garbage.
        if self._stream.codec_context.pix_fmt != "yuyv422":
            log.info(
                "PyAV dshow opened but pix_fmt=%s (expected yuyv422)",
                self._stream.codec_context.pix_fmt,
            )
            self._container.close()
            self._container = None
            self._stream = None
            return False
        self._demux_iter = self._container.demux(self._stream)
        return True

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Return (ok, raw_yuy2_buf) where the buffer is a contiguous
        ``(H, 2*W)`` uint8 array — the same shape the existing
        ``_raw_yuy2_mode`` slicing in grab() expects."""
        if self._container is None or self._demux_iter is None:
            return False, None
        if self._frame_queue:
            return True, self._frame_queue.pop(0)
        try:
            for packet in self._demux_iter:
                # Drain decoded frames into our queue; usually exactly one
                # per packet for raw video.
                for frame in packet.decode():
                    if frame.format.name != "yuyv422":
                        continue
                    plane = frame.planes[0]
                    line_size = plane.line_size
                    raw = np.frombuffer(bytes(plane), dtype=np.uint8)
                    raw = raw[: line_size * frame.height].reshape(
                        frame.height, line_size,
                    )
                    # If line_size has padding past 2*W, drop it.
                    if line_size > 2 * frame.width:
                        raw = raw[:, : 2 * frame.width]
                    self._frame_queue.append(np.ascontiguousarray(raw))
                if self._frame_queue:
                    return True, self._frame_queue.pop(0)
        except StopIteration:
            return False, None
        except Exception as e:
            log.debug("PyAV demux/decode threw %r", e)
            return False, None
        return False, None

    def release(self) -> None:
        try:
            if self._container is not None:
                self._container.close()
        except Exception:
            pass
        self._container = None
        self._stream = None
        self._demux_iter = None
        self._frame_queue.clear()

    def isOpened(self) -> bool:  # noqa: N802 (cv2 spelling)
        return self._container is not None

    # cv2.VideoCapture API stubs — IMX568Capture occasionally calls .set()
    # for exposure / gain on the OpenCV path. On PyAV path these are no-ops
    # (return False) because ffmpeg's dshow input doesn't expose camera
    # control and we'd need a separate IAMCameraControl COM call to drive
    # exposure. Returning False is the documented "the property could not
    # be set" signal, which is exactly what software AE wants to hear.

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        return False

    def get(self, prop: int) -> float:
        return 0.0


def _find_dshow_video_device(name_hint: str = "imx") -> Optional[str]:
    """Locate an FX3 / IMX568 dshow device name via ffmpeg's device list.

    Returns the first device whose name contains *any* of the hints
    'imx', 'leopard', 'li-', 'fx3' (case-insensitive). Returns None if
    no matching device is found, in which case the caller falls back
    to the OpenCV path.
    """
    if not _PYAV_AVAILABLE:
        return None
    try:
        import subprocess
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        cp = subprocess.run(
            [exe, "-hide_banner", "-list_devices", "true",
             "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        log.debug("ffmpeg device-list call failed: %r", e)
        return None
    text = (cp.stderr or "") + "\n" + (cp.stdout or "")
    candidates: list[str] = []
    for line in text.splitlines():
        if "(video)" not in line.lower() or '"' not in line:
            continue
        candidates.append(line.split('"', 2)[1])
    hints = ("imx", "leopard", "li-", "fx3")
    for c in candidates:
        cl = c.lower()
        if any(h in cl for h in hints):
            return c
    return None


def _set_uvc_manual_exposure(
    device_index: int,
    exposure_log2: float = -6.0,
    gain: Optional[float] = None,
) -> dict:
    """Briefly open the camera with cv2-DSHOW to write manual exposure.

    On Windows DirectShow, UVC camera-control properties (auto-exposure
    flag, exposure time, gain, brightness) are persistent in the device
    across a close/reopen so long as the device is not power-cycled or
    re-enumerated. So the trick is:

        1. cv2.VideoCapture(idx, CAP_DSHOW)          ← brief open
        2. cap.set(CAP_PROP_AUTO_EXPOSURE, 0.25)     ← manual mode
        3. cap.set(CAP_PROP_EXPOSURE, exposure_log2) ← lock exposure
        4. cap.release()                             ← properties stay
        5. PyAV opens & grabs                        ← stable frames

    The user observed visible "breathing" on a static scene — bridge AE
    was hunting frame-to-frame. With AE off and a fixed exposure, the
    image stops drifting and we can tune one knob to match Leopard.

    Returns a dict of {prop_name: (set_value, readback)} for the log so
    we can tell whether the bridge accepted the writes (some FX3 firmware
    silently ignores property writes — hence the log).

    Why log2 for exposure_log2: cv2's DSHOW backend reports exposure in
    "log2 seconds" (UVC EXPOSURE_TIME_ABSOLUTE convention). -6 ≈ 1/64s
    ≈ 16 ms, -7 ≈ 8 ms, -5 ≈ 32 ms. Adjust to taste.
    """
    result: dict = {}
    try:
        cap = cv2.VideoCapture(int(device_index), cv2.CAP_DSHOW)
    except Exception as e:
        result["open_error"] = repr(e)
        return result
    if not cap.isOpened():
        result["open_error"] = "isOpened()==False"
        try:
            cap.release()
        except Exception:
            pass
        return result
    try:
        # 0.25 = manual (V4L2 convention DSHOW sometimes inherits).
        # 1 = manual / 3 = auto on some DirectShow drivers.
        # Try 0.25 first, then 1, log both readbacks.
        for ae_value in (0.25, 1.0):
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, ae_value)
            rb = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)
            result[f"AUTO_EXPOSURE={ae_value}"] = rb
        cap.set(cv2.CAP_PROP_EXPOSURE, float(exposure_log2))
        result["EXPOSURE_set"] = float(exposure_log2)
        result["EXPOSURE_readback"] = cap.get(cv2.CAP_PROP_EXPOSURE)
        if gain is not None:
            cap.set(cv2.CAP_PROP_GAIN, float(gain))
            result["GAIN_set"] = float(gain)
            result["GAIN_readback"] = cap.get(cv2.CAP_PROP_GAIN)
    finally:
        try:
            cap.release()
        except Exception:
            pass
    return result


def _set_leopard_exposure_ext(
    exposure_ext: int,
    ae_off: bool = True,
    gain: Optional[int] = None,
) -> dict:
    """Spawn the 32-bit helper to write LPCamera.ExposureExt via the
    Leopard SDK's vendor-extension I2C path.

    This is the path that ACTUALLY works on this bridge. The standard
    UVC properties (CAP_PROP_AUTO_EXPOSURE, CAP_PROP_EXPOSURE, etc) are
    silently ignored by the FX3 firmware — proven exhaustively in
    scripts/eo_probe_uvc_props.py. ExposureExt is a vendor-extension
    register that the FX3 actually honors; setting it disables bridge
    AE and pins the sensor exposure to a stable value.

    Empirically (scripts/eo_probe_breathing_at_fixed_expext.py): with
    ExposureExt=1000 the recovered Y mean is rock-stable across 30 s
    (span=1.7, std=0.6) — was wildly drifting before. The "static
    scene keeps changing" complaint is solved by this single write.

    The 32-bit subprocess split is forced because LeopardCamera.dll is
    PE machine 0x14c (x86) and our seeker is 64-bit. tools/python311-x86
    is an embeddable Python with pythonnet pre-installed for this.

    Returns the helper's JSON result dict for logging. The helper closes
    the SDK session before returning; the FX3 retains the manual-exposure
    state across that close, and PyAV can immediately reopen the device.
    """
    import json as _json
    import os as _os
    import subprocess as _subprocess

    # Resolve helper paths relative to this file so it works from any cwd.
    here = _os.path.dirname(_os.path.abspath(__file__))
    repo = _os.path.dirname(here)
    py32 = _os.path.join(repo, "tools", "python311-x86", "python.exe")
    helper = _os.path.join(here, "leopard_sdk_helper.py")

    if not _os.path.exists(py32):
        return {"helper_unavailable": f"32-bit Python missing at {py32}"}
    if not _os.path.exists(helper):
        return {"helper_unavailable": f"helper script missing at {helper}"}

    cmd = [py32, helper,
           "--exposure-ext", str(int(exposure_ext)),
           "--ae", "off" if ae_off else "on",
           "--json"]
    if gain is not None:
        cmd.extend(["--gain", str(int(gain))])

    try:
        cp = _subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return {"helper_threw": repr(e)}

    if cp.returncode != 0 and not cp.stdout.strip():
        return {"helper_failed": cp.returncode, "stderr": cp.stderr[-500:]}

    try:
        return _json.loads(cp.stdout)
    except Exception:
        return {"helper_unparsable_stdout": cp.stdout[-500:],
                "stderr": cp.stderr[-500:]}


def _frame_has_spatial_content(frame: np.ndarray) -> bool:
    """True iff the frame carries real scene data (not a flat fake buffer).

    The FX3-over-YUY2 failure mode produces a frame where each of the B,
    G, R channels is a single constant value, so the *per-channel* std is
    zero even though the whole-frame std can look non-zero (because the
    three channels are different constants). We test per-channel std on a
    downsampled slice: ~1 ms, unambiguous.
    """
    if frame is None or frame.size == 0:
        return False
    if frame.ndim == 2:
        return float(frame[::8, ::8].std()) > 1.0
    slab = frame[::8, ::8]
    return max(float(slab[:, :, c].std()) for c in range(slab.shape[2])) > 1.0


class IMX568Capture:
    """LI-IMX568-GMSL2 USB3 wrapper, BGR uint8 output.

    The sensor is *mono*. BGR comes out of DirectShow because its YUY2
    decode path always produces 3-channel output. All three channels
    carry the same luma; we keep them as BGR so downstream YOLO
    (trained on color imagery) doesn't have to special-case.
    """

    def __init__(
        self,
        device_index: int | str = "auto",
        exclude_indices: Optional[list[int]] = None,
        manual_exposure_log2: Optional[float] = None,
        manual_gain: Optional[float] = None,
        manual_exposure_ext: Optional[int] = None,
        use_sdk_stream: bool = True,
        stream_fps: float = 20.0,
    ) -> None:
        self.requested_index = device_index
        self.exclude_indices = list(exclude_indices or [])
        # PRIMARY path: 32-bit Leopard SDK helper streaming RAW12 over a
        # subprocess pipe. This is the only path that delivers true
        # Leopard-CameraTool image quality on this FX3 bridge — verified
        # 2026-04-24 with frame-to-frame B/G/R diff < 0.03 vs the
        # reference BMP. PyAV/OpenCV YUY2 paths produce visible breathing
        # and YUY2-decode artifacts and are now fallback-only.
        self._use_sdk_stream = bool(use_sdk_stream)
        self._sdk_stream_mode: bool = False  # set True if SDK path opens
        # Cap to 25 fps — beyond that the FX3 USB3 bus saturates under
        # RAW12 (10 MB/frame * 25 = 250 MB/s, near the link's practical
        # ceiling). 10 was the historical default and bottlenecks moving
        # targets visibly.
        self._stream_fps = float(max(1.0, min(25.0, stream_fps)))
        # When set, we lock the bridge to manual exposure with this value
        # (UVC log2 seconds, e.g. -6 ≈ 16ms) before PyAV opens the device.
        # Killing AE hunting was the single biggest "image stops breathing
        # frame to frame" win — see _set_uvc_manual_exposure docstring.
        # NOTE: this is the LEGACY UVC path. On this FX3 bridge it does
        # nothing — bridge ignores all UVC writes. Kept as a fallback for
        # other hardware. Prefer manual_exposure_ext on the IMX568 rig.
        self._manual_exposure_log2 = manual_exposure_log2
        self._manual_gain = manual_gain
        # PRIMARY exposure-lock path: Leopard SDK ExposureExt. When set,
        # we spawn the 32-bit helper subprocess before PyAV opens, write
        # ExposureExt via the vendor I2C extension, and the FX3 retains
        # the manual-exposure lock across the helper close + PyAV open.
        # See _set_leopard_exposure_ext docstring + the empirical proof
        # in scripts/eo_probe_breathing_at_fixed_expext.py.
        self._manual_exposure_ext = manual_exposure_ext

        self.device_index: Optional[int] = None
        self.actual_width: int = 0
        self.actual_height: int = 0
        # API-compat shim for EOManager's _CaptureLike protocol — IMX568
        # does not expose a 16-bit raw path through UVC, so raw16 is
        # unavailable here. Classifier/detector read the BGR field.
        self.raw16_available: bool = False
        self._cap: Optional[cv2.VideoCapture] = None
        self._fourcc: str = "?"  # which pixel format won negotiation
        # True iff CAP_PROP_CONVERT_RGB=0 took effect AND the resulting
        # buffer is the YUY2 byte stream we expect (HxWx2 or HxW*2).
        # When True, grab() slices the raw bytes for a clean mono Y plane;
        # when False, grab() falls back to BGR-with-broken-color-decode and
        # eo_processor's _to_luma has to recover from the green tint.
        self._raw_yuy2_mode: bool = False

        # ── Software AE state ─────────────────────────────────────────────
        #
        # The bridge's internal AE is unreliable on this rig — it routinely
        # drives the sensor far enough that DirectShow's YUY2→BGR decode
        # clips G across most of the frame, which the YUY2 dead-zone
        # recovery cannot undo. So after open we do a one-shot probe to
        # learn whether CAP_PROP_EXPOSURE actually responds and in which
        # direction (the convention varies by bridge), then run a closed-
        # loop tick from grab() that nudges exposure to keep:
        #     * G-saturation fraction below ~5%
        #     * recovered Y mean roughly in [80, 160]
        #
        # If the probe finds the bridge ignores writes, we re-enable
        # bridge AE and disable software AE — never end up worse than
        # before this code existed.
        self._sw_ae_enabled: bool = False    # set True after successful probe
        self._sw_ae_sign: int = 0            # +1 conventional, -1 reversed, 0 broken
        self._sw_ae_value: float = -7.0      # current exposure (UVC log₂-seconds)
        self._sw_ae_min: float = -13.0       # ~0.12 ms
        self._sw_ae_max: float = -3.0        # ~125 ms
        self._sw_ae_frame_counter: int = 0
        # Target band: keep recovered Y mean roughly mid-frame so the
        # operator sees a normal-brightness image, and dim the bridge
        # the moment G starts clipping at 255 (G saturates well before Y
        # reaches the top of its band, see _measure_frame_brightness).
        # An earlier guess pushed the band down to 25..55 on the theory
        # that Leopard CameraTool runs the sensor very dim — that was
        # based on a misread of Leopard's R/G/B histogram (those three
        # numbers being equal just means Leopard's output is mono Y
        # replicated to BGR, NOT that the sensor is at low exposure).
        # Reverted to a normal mid-band target.
        self._sw_ae_target_low: float = 80.0
        self._sw_ae_target_high: float = 160.0
        self._sw_ae_sat_limit: float = 0.05  # 5% G-clipped pixels triggers dim
        self._sw_ae_tick_period: int = 10    # adjust at ~3 Hz on a 30 fps stream

    # ───────────────────────── lifecycle ─────────────────────────

    def start(self) -> None:
        """Open the IMX568 at native resolution and commit to the first
        combo that delivers a native-res frame — flat or not.

        Why NOT a full matrix search for "spatial content":
          * Windows UVC is exclusive-open. Once we hold a cap on idx 0,
            every subsequent `cv2.VideoCapture(0, …)` on MSMF or DSHOW
            returns `isOpened()==False`. The search looks like
            "nothing else works" but really it's self-blocked.
          * Repeated open/close cycles against the FX3 bridge push it
            into a bad state (seen 2026-04-24: after a 10s probe, the
            very same cap that produced a flat-but-valid test frame
            returned None from its first real grab).
          * If the first combo hands back a flat buffer, the sensor is
            stuck at the USB layer and NO OpenCV combo will fix it —
            the fix is a USB replug, not another backend try.

        Strategy: walk (backend × FOURCC × index), pick the first combo
        that opens at NATIVE resolution, log a warning if the test frame
        is flat, and return. The user either sees real content or sees
        the flat-frame warning plus "replug the cable". No more silent
        black panel, no more 10-second stall.
        """
        if self._cap is not None:
            return

        # ── Leopard SDK RAW12 stream FIRST (preferred path) ───────────
        #
        # Spawns a 32-bit subprocess running leopard_sdk_helper.py
        # --stream which drives LeopardCamera.dll directly, captures
        # RAW12 (2472×2064 packed Bayer mosaic), and pipes [length][bytes]
        # frames back over stdout. We decode (uint16>>4) → debayer
        # cv2.COLOR_BAYER_BG2BGR → 8-bit BGR. Empirically matches
        # Leopard CameraTool's BMP output to sub-decimal-point per
        # channel (B/G/R diff < 0.03 vs reference, 2026-04-24).
        #
        # If anything goes wrong (32-bit Python missing, SDK DLL
        # missing, helper crashes, header timeout), we fall through
        # to the legacy PyAV / OpenCV YUY2 paths.
        if self._use_sdk_stream:
            try:
                from eo.leopard_stream_capture import LeopardSDKStreamCapture
                # AE policy:
                #   manual_exposure_ext=None  →  bridge auto-exposure ON
                #     (safe default — handles dim-indoor → sunlit-outdoor
                #      without blowing the frame to white).
                #   manual_exposure_ext=<int> →  bridge AE OFF + manual lock
                #     (used to reproduce the calibration reference exactly,
                #      or to freeze exposure for engineering tests).
                if self._manual_exposure_ext is None:
                    exp_ext = None
                    ae_mode = "on"
                else:
                    exp_ext = int(self._manual_exposure_ext)
                    ae_mode = "off"
                sdk = LeopardSDKStreamCapture(
                    exposure_ext=exp_ext,
                    width=NATIVE_W,
                    height=NATIVE_H,
                    # bayer_pattern=None → mono sensor (LI-IMX568-GMSL2-M).
                    # The IMX568 monochrome variant has no Bayer filter
                    # array — every pixel is uniform luma. Running a 2×2
                    # demosaic on uniform mono pixels is wrong (phantom
                    # color + half resolution). The previous default
                    # cv2.COLOR_BAYER_BG2BGR was a leftover from when the
                    # rig was tested against a color variant.
                    bayer_pattern=None,
                    ae=ae_mode,
                    stream_fps=self._stream_fps,
                    warmup=12,
                )
                # Header timeout extended to 20s (was 12s) — observed
                # 2026-04-25 that when the FX3 bridge has been left in a
                # different resolution mode by a prior CameraTool session,
                # the first stream renegotiation can take ~15s.
                sdk.start(header_timeout_s=20.0)
                # Verify by grabbing one settle frame.
                test = sdk.grab()
                if test is None:
                    log.warning("SDK stream: header arrived but first "
                                "grab returned None — falling back")
                    sdk.stop()
                else:
                    self._cap = sdk  # type: ignore[assignment]
                    self.device_index = 0  # SDK opens by handle, not index
                    self.actual_width = NATIVE_W
                    self.actual_height = NATIVE_H
                    self._fourcc = "RAW12"
                    self._sdk_stream_mode = True
                    self._raw_yuy2_mode = False
                    self._sw_ae_enabled = False  # SDK pins exposure
                    log.info(
                        "IMX568Capture mode: SDK_RAW12_STREAM "
                        "(exposure_ext=%s, ae=%s, %dx%d, Bayer BG, "
                        "first frame mean=%.1f)",
                        ("auto" if exp_ext is None else str(exp_ext)),
                        ae_mode, NATIVE_W, NATIVE_H, float(test.mean()),
                    )
                    return
            except Exception as e:
                log.warning("SDK stream backend unavailable (%r) — "
                            "falling back to PyAV/OpenCV", e)
                # CRITICAL: kill the orphaned helper subprocess. Without
                # this, a STREAM_HDR timeout leaves the helper alive
                # holding the camera, and the PyAV/DSHOW fallback below
                # then fails with "device already in use by other
                # application". Verified 2026-04-25 against a stuck-in-
                # 2592x1944 bridge state from a prior CameraTool session.
                try:
                    sdk_local = locals().get("sdk")
                    if sdk_local is not None:
                        sdk_local.stop()
                except Exception as cleanup_e:
                    log.warning("SDK stream cleanup also threw %r "
                                "— camera may be stuck; USB replug "
                                "may be required.", cleanup_e)
                # Brief settle so the bridge releases the device handle
                # before PyAV grabs it.
                time.sleep(0.5)

        # ── PyAV / ffmpeg-dshow first ─────────────────────────────────
        #
        # Try to open the bridge through ffmpeg's dshow demuxer with an
        # explicit yuyv422 request before falling back to OpenCV. This
        # is the ONLY path on this rig that delivers a clean Y plane —
        # OpenCV's DirectShow always swaps in a destructive YUY2->BGR
        # auto-decode regardless of CAP_PROP_CONVERT_RGB. See the
        # _PyAVDshowBackend docstring above for the probe trail.
        if _PYAV_AVAILABLE:
            device_name = _find_dshow_video_device()
            if device_name:
                # Lock bridge AE BEFORE PyAV opens the device. PyAV holds
                # the device exclusive once open() succeeds, so we have
                # exactly one window to write the manual-exposure command.
                #
                # PRIMARY: Leopard SDK ExposureExt via 32-bit subprocess
                # helper. This is the only path that actually works on
                # this FX3 bridge — proven in eo_probe_uvc_props.py (UVC
                # writes ignored) and eo_probe_breathing_at_fixed_expext.py
                # (ExposureExt → mean stable across 30s).
                if self._manual_exposure_ext is not None:
                    rep = _set_leopard_exposure_ext(
                        exposure_ext=int(self._manual_exposure_ext),
                        ae_off=True,
                        gain=int(self._manual_gain) if self._manual_gain is not None else None,
                    )
                    log.info("Leopard SDK exposure prelude: %s", rep)
                # FALLBACK: legacy UVC log2 path. Kept for non-IMX568
                # hardware; on the FX3 bridge it's a known no-op.
                elif self._manual_exposure_log2 is not None:
                    locked = False
                    for probe_idx in range(4):
                        rep = _set_uvc_manual_exposure(
                            probe_idx,
                            exposure_log2=float(self._manual_exposure_log2),
                            gain=self._manual_gain,
                        )
                        if "open_error" in rep:
                            continue
                        log.info(
                            "UVC manual-exposure prelude on idx %d: %s",
                            probe_idx, rep,
                        )
                        locked = True
                        break
                    if not locked:
                        log.warning(
                            "UVC manual-exposure prelude could not open "
                            "any cv2 dshow index — bridge AE will run free"
                        )
                pyav_cap = _PyAVDshowBackend(device_name, NATIVE_W, NATIVE_H)
                if pyav_cap.open():
                    # Read one settle frame to confirm the stream is alive.
                    ok, test = pyav_cap.read()
                    if ok and test is not None:
                        self._cap = pyav_cap  # type: ignore[assignment]
                        # Best-effort device index for telemetry; PyAV
                        # opens by name, so we don't really know the index.
                        self.device_index = 0
                        self.actual_width = NATIVE_W
                        self.actual_height = NATIVE_H
                        self._fourcc = "YUY2"
                        self._raw_yuy2_mode = True   # grab() will slice Y
                        # Software AE on the PyAV path needs a separate
                        # IAMCameraControl COM hook (PyAV's dshow demuxer
                        # doesn't expose UVC camera-control). Disabled
                        # for now — bridge hardware AE is producing a
                        # clean image without our intervention.
                        self._sw_ae_enabled = False
                        log.info(
                            "IMX568Capture mode: PYAV_RAW_YUY2 (device "
                            "%r, raw Y plane via ffmpeg-dshow yuyv422)",
                            device_name,
                        )
                        return
                    else:
                        log.info(
                            "PyAV opened %r but first read returned no "
                            "frame — falling back to OpenCV",
                            device_name,
                        )
                        pyav_cap.release()
                else:
                    log.info(
                        "PyAV could not open dshow device %r — falling "
                        "back to OpenCV", device_name,
                    )
            else:
                log.info(
                    "PyAV: no IMX568/Leopard/FX3 device found in dshow "
                    "device list — falling back to OpenCV",
                )

        candidates = self._candidate_indices()
        last_err: Optional[str] = None

        for idx in candidates:
            if idx in self.exclude_indices:
                continue

            wrong_device_on_index = False

            for backend_id, backend_name in _BACKEND_CANDIDATES:
                if wrong_device_on_index:
                    break
                for fourcc_tag in _FOURCC_CANDIDATES:
                    label = f"idx {idx} {backend_name}/{fourcc_tag or 'DEFAULT'}"

                    try:
                        cap = cv2.VideoCapture(idx, backend_id)
                    except Exception as e:
                        last_err = f"{label}: ctor threw {e!r}"
                        log.info("%s", last_err)
                        continue
                    if not cap.isOpened():
                        cap.release()
                        last_err = f"{label}: isOpened()==False"
                        log.info("%s", last_err)
                        continue

                    try:
                        if fourcc_tag is not None:
                            cap.set(cv2.CAP_PROP_FOURCC,
                                    cv2.VideoWriter_fourcc(*fourcc_tag))
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        # Force AE ON. Without this, a previous run that
                        # set MANUAL mode (CAP_PROP_AUTO_EXPOSURE=0.25)
                        # can leave the bridge stuck at a fixed exposure
                        # across reboots — image looks dark or blown
                        # depending on what value it last latched. The
                        # 0.75 = AUTO convention is DSHOW-specific.
                        # Bridge may ignore the write; harmless either way.
                        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
                    except Exception as e:
                        cap.release()
                        last_err = f"{label}: set-props threw {e!r}"
                        continue

                    ok, test = cap.read()
                    if not ok or test is None:
                        cap.release()
                        last_err = f"{label}: opened but no frames"
                        continue

                    h, w = test.shape[:2]
                    if (w, h) != (NATIVE_W, NATIVE_H):
                        cap.release()
                        last_err = (f"{label}: returned {w}x{h}, not "
                                    f"native {NATIVE_W}x{NATIVE_H}")
                        log.info("%s — skipping, not IMX568", last_err)
                        # Not IMX568 on this index — don't bother
                        # trying other backends / formats here, move on.
                        wrong_device_on_index = True
                        break

                    negotiated = _fourcc_to_str(
                        int(cap.get(cv2.CAP_PROP_FOURCC))
                    )
                    has_content = _frame_has_spatial_content(test)

                    # ── Software AE bring-up ─────────────────────────
                    #
                    # Probe whether CAP_PROP_EXPOSURE actually responds
                    # and in which direction (UVC convention varies by
                    # bridge), pick a sane starting exposure if so, and
                    # leave software AE armed so grab() can keep the
                    # frame out of G-saturation. If the probe finds the
                    # bridge ignores writes, this falls back to bridge
                    # AE and disables software AE so we never end up
                    # worse than before this code existed.
                    sw_ae_sign, sw_ae_baseline = _probe_software_ae_direction(cap)
                    self._sw_ae_sign = sw_ae_sign
                    self._sw_ae_value = sw_ae_baseline
                    self._sw_ae_enabled = (sw_ae_sign != 0)

                    # ── Raw YUY2 attempt ───────────────────────────────
                    #
                    # The FX3 bridge ships a MONO sensor through YUY2 with
                    # U=V=0. DirectShow's auto-decode of that into BGR
                    # produces a frame where:
                    #     B = Y - 227 (clipped to 0)
                    #     G = Y + 135 (clipped to 255 above Y=120)
                    #     R = Y - 179 (clipped to 0 below Y=179)
                    # which permanently destroys ~60 codes of dynamic range
                    # in the Y∈[120,179] band — the visible "oil-painting
                    # mid-tone patches" everyone has been blaming on the
                    # enhancement chain.
                    #
                    # CAP_PROP_CONVERT_RGB=0 disables that decode and hands
                    # us the raw YUY2 byte stream. Slicing every-other
                    # byte gives us the FULL 0..255 mono Y plane with no
                    # dead zone. This is the only configuration that
                    # produces an image that looks like a real day sensor.
                    #
                    # If the bridge or this opencv build refuses raw mode,
                    # we leave it on the BGR path and let eo_processor
                    # recover what it can from the broken color decode.
                    raw_ok = False
                    try:
                        if cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                            ok2, raw_test = cap.read()
                            if ok2 and raw_test is not None:
                                rh, rw = raw_test.shape[:2]
                                # YUY2 is 2 bytes per pixel. Acceptable
                                # shapes: (H, W, 2) packed-channel, or
                                # (H, 2W) flat-byte. Anything else means
                                # the bridge ignored the flag.
                                if (raw_test.ndim == 3 and rw == NATIVE_W
                                        and rh == NATIVE_H
                                        and raw_test.shape[2] == 2):
                                    raw_ok = True
                                elif (raw_test.ndim == 2
                                      and rh == NATIVE_H
                                      and rw == 2 * NATIVE_W):
                                    raw_ok = True
                                else:
                                    # Bridge accepted the flag but format
                                    # is something we don't expect — turn
                                    # the auto-decode back on.
                                    cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                            else:
                                cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                    except Exception as e:
                        log.debug("raw YUY2 attempt threw %r — staying BGR", e)
                        try:
                            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
                        except Exception:
                            pass

                    # COMMIT — regardless of flatness. See docstring.
                    self._cap = cap
                    self.device_index = idx
                    self.actual_width = w
                    self.actual_height = h
                    self._fourcc = negotiated
                    self._raw_yuy2_mode = raw_ok
                    log.info(
                        "IMX568Capture mode: %s (raw YUY2 %s)",
                        "RAW_YUY2_MONO" if raw_ok else "BGR_FALLBACK",
                        "enabled — clean mono Y plane"
                            if raw_ok else
                            "rejected — falling back to broken-color BGR + "
                            "_to_luma green-tint recovery",
                    )
                    if has_content:
                        log.info(
                            "IMX568Capture opened on %s at %dx%d "
                            "(requested=%s, negotiated=%s) — "
                            "frame has real content",
                            label, w, h,
                            fourcc_tag or "DEFAULT", negotiated,
                        )
                    else:
                        log.warning(
                            "IMX568Capture opened on %s at %dx%d "
                            "(requested=%s, negotiated=%s) but test "
                            "frame is FLAT (per-ch std ~0). Sensor is "
                            "streaming a degenerate buffer at the USB "
                            "layer — unplug/replug the USB3 cable to "
                            "reset the FX3 bridge. Proceeding anyway so "
                            "the pipeline stays alive.",
                            label, w, h,
                            fourcc_tag or "DEFAULT", negotiated,
                        )
                    return

        raise RuntimeError(
            f"Could not open IMX568 (tried {candidates}, "
            f"excluded {self.exclude_indices}). Last error: {last_err}. "
            f"Make sure CameraTool / other viewers are closed."
        )

    def stop(self) -> None:
        if self._cap is not None:
            try:
                # SDK stream uses .stop(); cv2/PyAV use .release().
                if self._sdk_stream_mode:
                    self._cap.stop()
                else:
                    self._cap.release()
            except Exception:
                pass
            self._cap = None
            self._sdk_stream_mode = False

    def __enter__(self) -> "IMX568Capture":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    # ───────────────────────── capture ───────────────────────────

    def grab(self) -> Optional[np.ndarray]:
        """Return the next BGR uint8 frame (H, W, 3), or None on failure.

        In RAW_YUY2_MONO mode (preferred): pulls the raw YUY2 byte stream,
        slices the Y plane (every other byte), and re-expands to BGR for
        downstream consumers. Skips DirectShow's broken color decode
        entirely so the full 0..255 sensor range survives to the GUI.

        In BGR_FALLBACK mode: returns whatever DirectShow's YUY2->BGR
        gave us, green tint and all. eo_processor._to_luma is responsible
        for recovering luma in that case.
        """
        if self._cap is None:
            return None

        # SDK RAW12 stream path: the consumer already delivers BGR uint8
        # from a Bayer-BG debayer; no YUY2 slicing, no software AE
        # needed (the helper pinned ExposureExt at start).
        if self._sdk_stream_mode:
            try:
                return self._cap.grab()  # LeopardSDKStreamCapture.grab()
            except Exception as e:
                log.warning("SDK stream grab threw %r", e)
                return None

        try:
            ok, frame = self._cap.read()
        except Exception:
            return None
        if not ok or frame is None:
            return None

        if not self._raw_yuy2_mode:
            # Run the software-AE feedback loop on the BGR fallback path.
            # No-op if the startup probe disabled it.
            self._software_ae_tick(frame)
            return frame

        # Raw YUY2 → mono Y plane → BGR (replicated luma).
        # YUY2 byte order is [Y0 U Y1 V] per 2 pixels: bytes 0,2,4,... = Y.
        if frame.ndim == 3 and frame.shape[2] == 2:
            # OpenCV packed (H, W, 2): channel 0 holds Y for both pixels.
            y = frame[:, :, 0]
        elif frame.ndim == 2:
            # Flat byte stream (H, 2W): take every other byte.
            y = frame[:, 0::2]
        else:
            # Shouldn't happen if start()'s shape check passed, but be safe.
            return frame
        # Ensure contiguous uint8 — downstream uses np operations that
        # don't tolerate strided views silently.
        y = np.ascontiguousarray(y, dtype=np.uint8)
        return cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

    def is_open(self) -> bool:
        if self._cap is None:
            return False
        if self._sdk_stream_mode:
            return bool(getattr(self._cap, "is_alive", False))
        return self._cap.isOpened()

    @property
    def last_raw_stats(self) -> Optional[dict]:
        """Forward LeopardSDKStreamCapture's per-frame raw u16 stats
        when we're in SDK-stream mode. Returns None on the legacy
        PyAV/YUY2 paths — those don't have access to the underlying
        12-bit data because the FX3 bridge has already done its own
        AGC and debayer in those modes. The software AE in EOManager
        polls this; if it's None, AE simply doesn't engage (which is
        the right behaviour — non-SDK paths use the bridge's AE)."""
        if self._sdk_stream_mode and self._cap is not None:
            return getattr(self._cap, "last_raw_stats", None)
        return None

    # ───────────────────────── software AE ───────────────────────

    def _software_ae_tick(self, frame: np.ndarray) -> None:
        """One feedback step of the closed-loop software AE.

        Called once per ``grab()`` on the BGR fallback path. Throttles
        itself to ``_sw_ae_tick_period`` frames so we adjust at ~3 Hz,
        not 30 Hz (gives the sensor time to commit the new exposure
        between pokes — UVC commits asynchronously).

        Logic:
          * G-clip > 5%  OR  recovered Y mean > 160 → step DIM
          * recovered Y mean < 80                   → step BRIGHT
          * otherwise                                → in-band, no change

        Step size is 1 UVC stop (×2 / ÷2 in seconds). Direction is
        determined at startup by ``_probe_software_ae_direction``; we
        multiply by ``_sw_ae_sign`` so a reversed-convention bridge
        gets the same effective behaviour.
        """
        if not self._sw_ae_enabled or self._cap is None:
            return
        self._sw_ae_frame_counter += 1
        if self._sw_ae_frame_counter < self._sw_ae_tick_period:
            return
        self._sw_ae_frame_counter = 0

        y_mean, sat = _measure_frame_brightness(frame)

        # Dim if either: too bright, OR G channel is clipping (which
        # destroys recoverable dynamic range regardless of mean).
        too_bright = (y_mean > self._sw_ae_target_high) or (sat > self._sw_ae_sat_limit)
        too_dark = y_mean < self._sw_ae_target_low

        if not (too_bright or too_dark):
            return

        # 1 stop = ±1.0 in log₂-seconds. Sign is the bridge's convention.
        # If conventional (sign=+1): brighter means LARGER exposure value;
        # so to dim we subtract 1, to brighten we add 1.
        # If reversed (sign=-1): we flip the math.
        step = -1.0 if too_bright else 1.0
        new_value = self._sw_ae_value + step * float(self._sw_ae_sign)
        new_value = max(self._sw_ae_min, min(self._sw_ae_max, new_value))
        if new_value == self._sw_ae_value:
            return  # already railed against the limit
        try:
            self._cap.set(cv2.CAP_PROP_EXPOSURE, float(new_value))
        except Exception as e:
            log.debug("software AE write threw %r — disabling", e)
            self._sw_ae_enabled = False
            return
        log.debug(
            "AE %s: Y=%.1f sat=%.1f%% exposure %.1f→%.1f log₂s",
            "dim" if too_bright else "bright",
            y_mean, sat * 100.0, self._sw_ae_value, new_value,
        )
        self._sw_ae_value = new_value

    # ───────────────────────── controls ──────────────────────────
    #
    # UVC exposure is in log-2 seconds: CAP_PROP_EXPOSURE=-6 means
    # 2^-6 = 1/64 s ≈ 15.6 ms. Some bridges respect it, some don't —
    # we set it and don't error if the write is ignored (the scene's
    # measured brightness tells the profile selector whether it took).

    def set_exposure_ms(self, exposure_ms: float) -> bool:
        """Request a manual exposure in milliseconds. Returns True on success."""
        if self._cap is None or exposure_ms <= 0:
            return False
        try:
            # Flip to manual exposure mode (0.25 = manual on DSHOW).
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            # UVC: exposure in log-2 seconds.
            log2_s = np.log2(exposure_ms / 1000.0)
            return bool(self._cap.set(cv2.CAP_PROP_EXPOSURE, float(log2_s)))
        except Exception as e:
            log.debug("set_exposure_ms(%.2f) threw %r", exposure_ms, e)
            return False

    def set_auto_exposure(self, enabled: bool) -> bool:
        if self._cap is None:
            return False
        try:
            # 0.75 = auto, 0.25 = manual (DSHOW convention).
            return bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,
                                      0.75 if enabled else 0.25))
        except Exception:
            return False

    def set_exposure_ext(self, exposure_ext: Optional[int]) -> bool:
        """Deprecated. Do NOT call from API/GUI threads.

        This method used to do its own stop()+start() of the SDK helper.
        That ran on the FastAPI request thread while the EO capture
        thread was independently observing the gap and ALSO calling
        ``_open_source()`` — two SDK helpers ended up fighting for the
        FX3 bridge and one orphan locked the camera hard enough to
        require a USB replug.

        Source-lifecycle changes are now owned exclusively by
        ``EOManager.set_exposure_ext()`` which serializes them under
        ``_source_lock`` + the ``_switching_exposure`` flag so the
        capture loop can't race. This method intentionally does
        nothing so any old caller is a silent no-op rather than a
        device lockup.
        """
        log.warning("IMX568Capture.set_exposure_ext is a no-op; route "
                    "through EOManager.set_exposure_ext instead.")
        return False

    def set_gain(self, gain: float) -> bool:
        """Request analog gain. Range depends on bridge; typically 0..100."""
        if self._cap is None or gain < 0:
            return False
        try:
            return bool(self._cap.set(cv2.CAP_PROP_GAIN, float(gain)))
        except Exception:
            return False

    # ───────────────────────── helpers ───────────────────────────

    def _candidate_indices(self) -> list[int]:
        if isinstance(self.requested_index, int):
            return [self.requested_index]
        if self.requested_index == "auto" or self.requested_index is None:
            # IMX568 showed up on index 0 on this rig after the USB-port
            # swap, but don't hardcode — probe 0..4 for other hosts.
            return [0, 1, 2, 3, 4]
        try:
            return [int(self.requested_index)]
        except (TypeError, ValueError):
            return [0, 1, 2, 3, 4]


# ───────────────────────────────────────────────────────────────
# Self-test
# ───────────────────────────────────────────────────────────────
def probe(duration_s: float = 5.0,
          device_index: int | str = "auto") -> Tuple[int, float, float]:
    """Open, stream for N seconds, return (frames, fps, mean_brightness)."""
    cap = IMX568Capture(device_index=device_index)
    cap.start()
    frames = 0
    mean_acc = 0.0
    t0 = time.time()
    try:
        while time.time() - t0 < duration_s:
            f = cap.grab()
            if f is not None:
                frames += 1
                # Sample the mean on every 5th frame — cheap enough to
                # log without flooding stdout for a 5s probe.
                if frames % 5 == 0:
                    mean_acc += float(f.mean())
    finally:
        cap.stop()
    elapsed = time.time() - t0
    fps = frames / elapsed if elapsed > 0 else 0.0
    mean = mean_acc / max(1, frames // 5)
    return frames, fps, mean


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--duration", type=float, default=5.0)
    args = ap.parse_args()

    f, fps, mean = probe(args.duration, device_index=args.device)
    print(f"IMX568 probe: {f} frames in {args.duration:.1f}s = {fps:.1f} fps, "
          f"scene mean brightness {mean:.1f}/255")

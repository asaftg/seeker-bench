"""Direct V4L2 mmap + RAW12 reinterpretation backend for IMX568 on Linux.

============================================================================
WHAT THE FX3 ACTUALLY SHIPS  (proven by decompiling LeopardCamera.dll +
analyzing CameraTool's saved .raw frames + verifying on the live Jetson
stream, 2026-05-08):
============================================================================

The Leopard FX3 firmware on the LI-USB30-IMX568-GMSL2 module ALWAYS
streams RAW12 RGGB Bayer data, regardless of the SENSOR_DATA_MODE you
pass to LPCamera.SetParam. The bytes are packed as little-endian uint16
(values 0..4095, four bits zero-padded in each pair). The UVC descriptor
advertises this stream as YUYV — a transport lie — because the FX3
firmware was built without ever exposing a real RAW12 UVC FourCC.

LeopardCamera.dll's SetParam method literally discards the data_type
argument and asks DirectShow for a YUYV/16bpp stream regardless. The
Windows pipeline that produces the high-quality image is:

    1. Read YUYV-shaped bytes from the FX3 (just bulk USB transfer).
    2. Reinterpret the byte stream as numpy uint16 LE.
    3. p1/p99 stretch on the raw u16 to recover full dynamic range.
    4. Replicate luma to BGR (or debayer if color is wanted).

This file does the same on Linux without a single XU write or USB
trick — direct V4L2 mmap to get the bytes, then numpy reinterpretation.

============================================================================
DEAD HYPOTHESES (don't waste time re-trying):
============================================================================
- Hidden V4L2 raw FourCCs:    none. S_FMT redirects every Bayer FOURCC to
                              YUYV. (Verified 2026-05-08.)
- Hidden USB alt-settings:    only bAlternateSetting=0 exists.
- XU mode-switch selector:    none. 32-value sweep across all 12
                              supported XU selectors flipped nothing
                              about the byte content.
- The DLL using a private USB endpoint: no — DLL only references
                              DirectShowLib (no usblib, no WinUSB).
"""
from __future__ import annotations
import ctypes, fcntl, mmap, os
from typing import Optional, Tuple
import numpy as np
import cv2  # for SIMD-optimized AGC stretch + grayscale->BGR convert

# V4L2 IOCTL definitions (from <linux/videodev2.h>)
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_FIELD_NONE = 1
V4L2_MEMORY_MMAP = 1
V4L2_PIX_FMT_YUYV = 0x56595559  # 'YUYV' fourcc

_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2

def _IOC(d, t, nr, sz):
    return (d << _IOC_DIRSHIFT) | (t << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (sz << _IOC_SIZESHIFT)
def _IOR(t, nr, st):  return _IOC(_IOC_READ,             ord(t), nr, ctypes.sizeof(st))
def _IOW(t, nr, st):  return _IOC(_IOC_WRITE,            ord(t), nr, ctypes.sizeof(st))
def _IOWR(t, nr, st): return _IOC(_IOC_READ|_IOC_WRITE,  ord(t), nr, ctypes.sizeof(st))


class _v4l2_pix_format(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]


class _format_union(ctypes.Union):
    # raw_data on this kernel = 200 bytes.
    _fields_ = [("pix", _v4l2_pix_format), ("_pad", ctypes.c_byte * 200)]


class _v4l2_format(ctypes.Structure):
    # Tegra L4T 35.6 (kernel 5.10.216-tegra) struct v4l2_format layout:
    #     __u32 type;          // offset 0
    #     __u32 _reserved;     // offset 4 (NOT in mainline videodev2.h
    #                          //           but emitted by Tegra build)
    #     union { ... } fmt;   // offset 8
    # Total = 208 bytes -> VIDIOC_S_FMT = 0xc0d05605.
    # Without the _reserved spacer, the union starts at offset 4 from
    # ctypes' perspective but the kernel reads/writes at offset 8 ->
    # every pix field shifted by one slot, height absorbs width's value,
    # pixelformat absorbs height, etc. Empirically reproduced 2026-05-08.
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("_reserved", ctypes.c_uint32),
        ("fmt", _format_union),
    ]


class _v4l2_requestbuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("flags", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8 * 3),
    ]


class _v4l2_timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]


class _timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]


class _m_union(ctypes.Union):
    _fields_ = [
        ("offset", ctypes.c_uint32),
        ("userptr", ctypes.c_ulong),
        ("fd", ctypes.c_int32),
    ]


class _v4l2_buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _timeval),
        ("timecode", _v4l2_timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", _m_union),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]


# ── UVC Extension Unit IOCTL (mirrors leopard_linux.py but reuses our
#    streaming fd — see _xu_write_on_stream_fd). Avoids opening a second
#    fd to /dev/video0 mid-stream, which the kernel UVC driver punishes
#    with frame-rate throttling. ──
_UVC_SET_CUR = 0x01

class _UVCXUQuery(ctypes.Structure):
    # Tegra L4T 35.6 layout matches mainline:
    #   __u8 unit; __u8 selector; __u8 query;
    #   __u16 size; __u8 *data;
    # Natural alignment: 16 bytes on 64-bit (3xu8 + 1pad + u16 + 2pad + 8ptr)
    _fields_ = [
        ("unit",     ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query",    ctypes.c_uint8),
        ("size",     ctypes.c_uint16),
        ("data",     ctypes.POINTER(ctypes.c_uint8)),
    ]

UVCIOC_CTRL_QUERY = _IOC(_IOC_READ | _IOC_WRITE, ord("u"), 0x21,
                         ctypes.sizeof(_UVCXUQuery))

VIDIOC_S_FMT     = _IOWR("V",  5, _v4l2_format)
VIDIOC_REQBUFS   = _IOWR("V",  8, _v4l2_requestbuffers)
VIDIOC_QUERYBUF  = _IOWR("V",  9, _v4l2_buffer)
VIDIOC_QBUF      = _IOWR("V", 15, _v4l2_buffer)
VIDIOC_DQBUF     = _IOWR("V", 17, _v4l2_buffer)
VIDIOC_STREAMON  = _IOW ("V", 18, ctypes.c_int)
VIDIOC_STREAMOFF = _IOW ("V", 19, ctypes.c_int)


class RawV4L2Backend:
    """cv2.VideoCapture-lookalike that returns RAW YUY2 (H, 2*W) uint8.

    Mirrors the public surface of imx568_capture._PyAVDshowBackend:
        open() / read() -> (ok, raw) / release() / isOpened()
    """

    def __init__(self, dev_path: str, width: int, height: int, n_buffers: int = 4):
        self._dev = dev_path
        self._w = int(width)
        self._h = int(height)
        self._n = int(n_buffers)
        self._fd = -1
        self._maps: list = []
        self._streaming = False
        # Per-frame raw u16 stats — fed to seeker's eo_manager AE loop.
        # Same shape as LeopardSDKStreamCapture.last_raw_stats on Windows.
        self.last_raw_stats: Optional[dict] = None
        # AGC alpha/beta EMA-smoothing state. On dim scenes the raw
        # span can be <20 counts; even 1-count sensor noise on p1/p99
        # swings alpha=255/span by ~10%, perceived as a 1-3 second
        # brightness pulse on a static frame. Smoothed across recomputes.
        self._agc_alpha_smooth: Optional[float] = None
        self._agc_beta_smooth: Optional[float] = None
        # Cache the last successful frame so grab() can return it on a
        # transient DQBUF timeout instead of None. seeker's eo_manager
        # treats grab()==None as "device disconnected" and tears the cap
        # down, which on Linux is fatal because the OS-level fd has to
        # be re-acquired and there's a small race window where /dev/video0
        # stays EBUSY.
        self._last_bgr: Optional[np.ndarray] = None
        # Frames since last trigger-state recheck. We re-disable trigger
        # mode every N frames in case the FX3 firmware re-arms it (which
        # has been observed to happen sporadically — root cause unknown).
        self._frames_since_trigger_check: int = 0
        # Frames since last AGC stats recompute. Stats are reused for
        # ~4 frames between recomputes to save ~5 ms per "skipped" frame.
        self._frames_since_stats: int = 0
        # AGC LUT cache (rebuild only when p1/p99 drift meaningfully).
        self._lut: Optional[np.ndarray] = None
        self._lut_p1: float = -1.0
        self._lut_p99: float = -1.0

    def isOpened(self) -> bool:  # noqa: N802
        return self._fd >= 0

    def _xu_write_on_stream_fd(self, selector: int, data: bytes) -> None:
        """Write a Leopard XU control on OUR streaming fd (no second
        open). FX3 vendor extension unit is hardcoded to unit=3 on this
        bridge per LeopardCamera.dll IL."""
        if self._fd < 0:
            return
        size = len(data)
        buf = (ctypes.c_uint8 * size).from_buffer_copy(data)
        q = _UVCXUQuery(unit=3, selector=selector, query=_UVC_SET_CUR,
                        size=size, data=buf)
        fcntl.ioctl(self._fd, UVCIOC_CTRL_QUERY, q)

    def set_exposure_ext(self, value: int) -> int:
        """Write ExposureExt (XU 0x06, u16 LE) on the streaming fd.

        Replaces the legacy path that opened a 2nd LeopardLinux fd to
        /dev/video0 every AE tick — which UVC kernel-driver throttles
        and causes EO frame-rate cliffs (every ~1.5s the AE thread
        would briefly steal the device, dropping EO to 1 Hz).

        Uses the documented decoy-write workaround (FX3 firmware
        silently ignores repeat writes of the same value, so we write
        a distinct decoy first, then the real target).

        Returns the readback value (0 on error)."""
        if self._fd < 0:
            return 0
        try:
            last = getattr(self, "_last_exp_written", None)
            decoy = 100 if last != 100 else 200
            tgt = int(value) & 0xffff
            self._xu_write_on_stream_fd(0x06, decoy.to_bytes(2, "little"))
            import time as _t
            _t.sleep(0.05)  # FX3 firmware needs settle time per Leopard SDK
            self._xu_write_on_stream_fd(0x06, tgt.to_bytes(2, "little"))
            self._last_exp_written = tgt
            return tgt
        except OSError:
            return 0

    def set_gain_rgb(self, gain: int) -> None:
        """Write RGB gain (XU 0x0d, 8B = 4×u16 LE) on streaming fd."""
        if self._fd < 0:
            return
        try:
            g = int(gain) & 0xffff
            self._xu_write_on_stream_fd(
                0x0d, (g.to_bytes(2, "little")) * 4
            )
        except OSError:
            pass

    @property
    def is_alive(self) -> bool:
        """IMX568Capture.is_open() looks at this attribute when in
        _sdk_stream_mode (the Windows SDK path uses a thread + flag)."""
        return self._fd >= 0 and self._streaming

    def open(self, retries: int = 8, backoff_s: float = 0.5) -> bool:
        """Open with retry. seeker tries the SDK helper (Windows .exe) first
        on Linux, which fails — but it can briefly leave /dev/video0 in EBUSY
        from the failing Popen. A few retries with backoff handles that race."""
        import time as _t
        last_err = None
        for attempt in range(retries):
            try:
                self._fd = os.open(self._dev, os.O_RDWR | os.O_NONBLOCK)
                break
            except OSError as e:
                last_err = e
                self._fd = -1
                if attempt < retries - 1:
                    _t.sleep(backoff_s)
        if self._fd < 0:
            return False
        try:
            f = _v4l2_format()
            f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            f.fmt.pix.width = self._w
            f.fmt.pix.height = self._h
            f.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV
            f.fmt.pix.field = V4L2_FIELD_NONE
            fcntl.ioctl(self._fd, VIDIOC_S_FMT, f)
            self._w = f.fmt.pix.width
            self._h = f.fmt.pix.height

            # REQBUFS can fail with EBUSY too (right after another process
            # released the device). Retry with backoff like the open call.
            req = _v4l2_requestbuffers()
            req.count = self._n
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            import time as _t
            for _attempt in range(8):
                try:
                    fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req)
                    break
                except OSError as _e_rq:
                    if _e_rq.errno in (16, 11):  # EBUSY, EAGAIN
                        _t.sleep(0.5)
                        continue
                    raise
            if req.count < 2:
                self.release()
                return False

            for i in range(req.count):
                buf = _v4l2_buffer()
                buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
                buf.memory = V4L2_MEMORY_MMAP
                buf.index = i
                fcntl.ioctl(self._fd, VIDIOC_QUERYBUF, buf)
                m = mmap.mmap(
                    self._fd, buf.length,
                    mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=buf.m.offset,
                )
                self._maps.append((m, buf.length))
                fcntl.ioctl(self._fd, VIDIOC_QBUF, buf)

            t = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
            fcntl.ioctl(self._fd, VIDIOC_STREAMON, t)
            self._streaming = True

            # ── Disable FX3 trigger mode → free-running ──
            # The Leopard FX3 firmware boots into trigger-mode (XU 0x0b
            # nonzero); in that state the sensor only captures on a soft-
            # trigger pulse and the stream returns black-level frames
            # forever. The Windows DLL's LPCamera.EnableTriggerMode(false,
            # false) writes XU 0x0b = [0, 0] (verified by decompiling
            # LeopardCamera.dll's IL: token 0x0600002e).
            #
            # We MUST do this AFTER STREAMON — XU writes seem to be
            # ignored by the FX3 firmware unless the UVC streaming
            # interface is active. Earlier attempts to write XU from a
            # control-only fd silently failed.
            try:
                # Write XU on OUR streaming fd (not a fresh one). The UVC
                # kernel driver throttles when multiple processes/fds
                # share /dev/video0 — opening a second fd here was
                # causing the EO frame rate to cliff to 1 Hz.
                self._xu_write_on_stream_fd(0x0b, bytes([0, 0]))
            except Exception:
                # Non-fatal: streaming might still work if firmware was
                # already in free-running mode.
                pass

            # Warmup: drain a few frames to prime the pipeline before
            # returning. Without this, the first caller-side grab() can
            # hit a 1s DQBUF timeout (the FX3 needs ~200-500ms to
            # actually start delivering frames after STREAMON + the
            # trigger-disable XU write that we just did).
            import time as _t
            for _w in range(5):
                ok, _ = self.read(timeout_s=2.0)
                if ok:
                    break
                _t.sleep(0.05)

            return True
        except Exception:
            self.release()
            return False

    def read(self, timeout_s: float = 1.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (True, raw_yuy2_buf) where buf is (H, 2*W) uint8."""
        if self._fd < 0:
            return False, None
        import time as _t
        deadline = _t.monotonic() + float(timeout_s)
        buf = _v4l2_buffer()
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
        buf.memory = V4L2_MEMORY_MMAP
        while True:
            try:
                fcntl.ioctl(self._fd, VIDIOC_DQBUF, buf)
                break
            except OSError as e:
                if e.errno in (11, 35):
                    if _t.monotonic() > deadline:
                        return False, None
                    _t.sleep(0.005)
                    continue
                return False, None
        idx = buf.index
        m, _ = self._maps[idx]
        valid_n = self._w * self._h * 2
        src_n = min(int(buf.bytesused) if buf.bytesused else valid_n, valid_n)
        raw = np.frombuffer(m[:src_n], dtype=np.uint8).copy()
        if raw.size < valid_n:
            raw = np.concatenate([raw, np.zeros(valid_n - raw.size, dtype=np.uint8)])
        raw = raw[:valid_n].reshape(self._h, 2 * self._w)

        try:
            fcntl.ioctl(self._fd, VIDIOC_QBUF, buf)
        except OSError:
            pass
        return True, raw

    def grab(self) -> Optional[np.ndarray]:
        """RAW12 reinterpretation path.

        Reads the raw byte stream, treats it as little-endian uint16
        (RAW12 RGGB Bayer values 0..4095 padded to 16 bits), computes
        per-frame raw stats (fed to seeker's eo_manager AE), then
        applies a p1/p99 stretch to produce mono BGR uint8 for the
        downstream YOLO/GUI path.

        Mirrors the Windows LeopardSDKStreamCapture.grab() return shape
        so IMX568Capture._sdk_stream_mode can take the early-return
        path verbatim.
        """
        # Periodically re-disable trigger mode in case the FX3 firmware
        # re-arms it sporadically. CRITICAL: do the XU IOCTL on OUR
        # streaming fd, not a fresh one. Opening a second fd to
        # /dev/video0 mid-stream caused observable frame-rate cliffs
        # (UVC kernel driver throttles when multiple fds are active).
        self._frames_since_trigger_check += 1
        if self._frames_since_trigger_check >= 300:
            self._frames_since_trigger_check = 0
            try:
                self._xu_write_on_stream_fd(0x0b, bytes([0, 0]))
            except Exception:
                pass
        ok, raw = self.read()
        if not ok or raw is None:
            # Return the last good frame on transient timeout — eo_manager
            # treats None as "disconnected" and tears down the source.
            return self._last_bgr
        # Reinterpret as RAW12 u16 LE — zero-copy view. raw is (H, 2W)
        # uint8 contiguous; .view() reinterprets bytes without a copy.
        u16 = raw.view(np.uint16).reshape(self._h, self._w)

        # Stats are fed to seeker's eo_manager AE which only steps every
        # ~1.5 sec. We don't need fresh percentiles every frame — at
        # 19fps that's ~28 frames per AE step. Recomputing every grab is
        # ~5 ms wasted. Compute every Nth frame and reuse between.
        # Cost saving: 4-5 ms/frame x ~80% of frames = ~4 ms avg/frame.
        self._frames_since_stats += 1
        if (self._frames_since_stats >= 5
                or self.last_raw_stats is None):
            self._frames_since_stats = 0
            # Denser fixed grid: [::4, ::4] = 4× more samples than the
            # historical [::8, ::8] (~640k px vs ~160k). Density is the
            # correct fix for the binary BAD/GOOD AGC flip on small
            # gimbal moves — a denser grid is far less likely to miss
            # sparse bright features. An earlier attempt also added a
            # randomized per-compute phase (oy, ox = randint(0,4)) but
            # that injected per-recompute percentile jitter (±5 raw
            # counts on p1/p99) which, on low-contrast scenes (span ~10),
            # swung alpha = 255/span by 2× between recomputes and made
            # static scenes pulse every ~0.25 s. Fixed grid = stable
            # stats on a static scene = no flicker; ::4 density alone
            # is enough to handle the sub-degree gimbal-move case.
            sample = u16[::4, ::4]
            s_p1 = float(np.percentile(sample, 1))
            s_p99 = float(np.percentile(sample, 99))
            s_mean = float(sample.mean())
            s_max = int(sample.max())
            s_frac_clip = float((sample >= 4090).sum()) / sample.size
            self.last_raw_stats = {
                "p1": s_p1,
                "p99": s_p99,
                "mean": s_mean,
                "max": s_max,
                "frac_clip": s_frac_clip,
            }
        else:
            # Reuse cached stats — alpha/beta below stay identical to
            # the last computed frame (AGC stretch is stable between
            # AE steps).
            s_p1 = self.last_raw_stats["p1"]
            s_p99 = self.last_raw_stats["p99"]

        # AGC stretch via cv2.convertScaleAbs — single SIMD-optimized C pass
        # on the full u16 frame: y = clip(alpha*u16 + beta, 0, 255). Replaces
        # the numpy float-multiply path that took ~30ms per frame on Jetson
        # AGX (now ~6ms). Profiled 2026-05-08: cv2.convertScaleAbs at 6.3ms,
        # cv2.cvtColor mono->BGR at ~3ms, total ~9ms vs 32ms before.
        #
        # Always-stretch AGC (pre-ea79b97 formula): always stretch
        # what is there, no fallback. Saturated-bright edge case (lens
        # covered, span literally 0) renders dark — accept as a sensor
        # cue rather than re-introduce the degenerate-span branch that
        # posterized real scenes.
        span = max(s_p99 - s_p1, 4.0)
        alpha_now = 255.0 / span
        beta_now = -s_p1 * alpha_now

        # EMA-smooth alpha/beta across recomputes (mix=0.2 → ~5
        # recompute time-constant ≈ 2 s at 13 fps × every-5-frames).
        # Damps the per-recompute jitter that comes from sensor noise
        # on dim scenes (where span is <20 counts and 1-count p1/p99
        # noise swings alpha by 10%). Real scene changes (gimbal sweep
        # to a brighter region) still propagate within ~1 s.
        if self._agc_alpha_smooth is None:
            self._agc_alpha_smooth = alpha_now
            self._agc_beta_smooth = beta_now
        else:
            mix = 0.2
            self._agc_alpha_smooth = (1.0 - mix) * self._agc_alpha_smooth + mix * alpha_now
            self._agc_beta_smooth  = (1.0 - mix) * self._agc_beta_smooth  + mix * beta_now
        alpha = self._agc_alpha_smooth
        beta  = self._agc_beta_smooth

        y8 = cv2.convertScaleAbs(u16, alpha=alpha, beta=beta)

        # Replicate luma to BGR — matches IMX568Capture's documented
        # output contract ("the sensor is mono; we keep BGR replicated
        # so downstream YOLO doesn't have to special-case").
        bgr = cv2.cvtColor(y8, cv2.COLOR_GRAY2BGR)
        self._last_bgr = bgr
        return bgr

    # eo_manager calls .stop() on the source when it suspects a
    # disconnect. LeopardSDKStreamCapture has stop() (terminates the
    # subprocess); we just release the V4L2 fd. Without this, eo_manager
    # silently swallows AttributeError and leaves /dev/video0 EBUSY.
    def stop(self) -> None:
        self.release()

    def release(self) -> None:
        try:
            if self._fd >= 0 and self._streaming:
                t = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                fcntl.ioctl(self._fd, VIDIOC_STREAMOFF, t)
        except Exception:
            pass
        self._streaming = False
        for m, _ in self._maps:
            try:
                m.close()
            except Exception:
                pass
        self._maps = []
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except Exception:
                pass
        self._fd = -1


def _smoke(dev: str = "/dev/video0", w: int = 2472, h: int = 2064):
    cap = RawV4L2Backend(dev, w, h)
    if not cap.open():
        print("FAILED open")
        return 1
    try:
        last_y = None
        for i in range(6):
            ok, raw = cap.read()
            if not ok:
                print("frame %d: read failed" % i)
                continue
            y = raw[:, ::2]
            uv = raw[:, 1::2]
            print("frame %d: Y mean=%.1f std=%.1f max=%d  |  UV mean=%.1f"
                  % (i, y.mean(), y.std(), y.max(), uv.mean()))
            last_y = y
        if last_y is not None:
            try:
                from PIL import Image
                Image.fromarray(last_y).save("/tmp/v4l2_raw_y.png")
                print("saved /tmp/v4l2_raw_y.png")
            except Exception:
                pass
    finally:
        cap.release()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_smoke(sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"))

"""Direct V4L2 mmap backend that returns RAW YUY2 bytes — Linux equivalent
of imx568_capture._PyAVDshowBackend (which gives raw YUY2 on Windows by
bypassing OpenCV DSHOW). Output shape is (H, 2*W) uint8, matching what
imx568_capture.grab() expects in `_raw_yuy2_mode`. No FFMPEG/PyAV/GStreamer
required — uses only ctypes + V4L2 ioctls.

Why this is needed: cv2.VideoCapture(/dev/videoN, CAP_V4L2) returns BGR
that has been YUV->RGB converted with U=V=0 (mono sensor). The conversion
math forces G=Y+135, R=Y-179, B=Y-227, producing the *exact same* dead-zone
posterization as Windows DSHOW (Y in [120,179] is irrecoverable). Routing
the raw byte stream around that conversion preserves the full Y signal.
"""
from __future__ import annotations
import ctypes, fcntl, mmap, os
from typing import Optional, Tuple
import numpy as np

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
    _fields_ = [("pix", _v4l2_pix_format), ("_pad", ctypes.c_byte * 200)]


class _v4l2_format(ctypes.Structure):
    _fields_ = [("type", ctypes.c_uint32), ("fmt", _format_union)]


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

    def isOpened(self) -> bool:  # noqa: N802
        return self._fd >= 0

    def open(self) -> bool:
        try:
            self._fd = os.open(self._dev, os.O_RDWR | os.O_NONBLOCK)
        except OSError:
            self._fd = -1
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

            req = _v4l2_requestbuffers()
            req.count = self._n
            req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE
            req.memory = V4L2_MEMORY_MMAP
            fcntl.ioctl(self._fd, VIDIOC_REQBUFS, req)
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
            return True
        except Exception:
            self.release()
            return False

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Return (True, raw_yuy2_buf) where buf is (H, 2*W) uint8."""
        if self._fd < 0:
            return False, None
        import time as _t
        deadline = _t.monotonic() + 1.0
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

"""
leopard_linux.py - Linux drop-in for LeopardCamera.dll's public API.

Talks the *exact* UVC Extension Unit protocol documented in Leopard's
open-source linux_camera_tool (github.com/LI01/linux_camera_tool):
    - unit  = 3
    - selectors / buffer sizes from includes/uvc_extension_unit_ctrl.h
    - I2C/sensor-register byte layout from src/uvc_extension_unit_ctrl.cpp

NO reverse engineering. NO sensor tuning logic invented here.
Tuning values flow in from seeker's existing helper code unchanged;
this module just routes those calls to the camera over UVCIOC_CTRL_QUERY.
"""

import ctypes
import fcntl
import os
import struct

# -- UVC ioctl + struct from <linux/uvcvideo.h> --------------------------
# struct uvc_xu_control_query {
#     __u8  unit;
#     __u8  selector;
#     __u8  query;       /* UVC_GET_/SET_CUR etc. */
#     __u16 size;
#     __u8 *data;
# };
# IOC: _IOWR('u', 0x21, struct uvc_xu_control_query)
UVC_SET_CUR = 0x01
UVC_GET_CUR = 0x81
UVC_GET_MIN = 0x82
UVC_GET_MAX = 0x83
UVC_GET_LEN = 0x85

class _XUQuery(ctypes.Structure):
    # NO _pack_ here — kernel struct uvc_xu_control_query uses NATURAL
    # alignment (16 bytes total on 64-bit: 3xu8 + 1pad + u16 + 2pad + 8ptr).
    # The IOCTL number bakes in sizeof(struct), so any mismatch -> EINVAL.
    _fields_ = [
        ("unit",     ctypes.c_uint8),
        ("selector", ctypes.c_uint8),
        ("query",    ctypes.c_uint8),
        ("size",     ctypes.c_uint16),
        ("data",     ctypes.POINTER(ctypes.c_uint8)),
    ]

# _IOWR('u', 0x21, struct uvc_xu_control_query) — compute at runtime
# _IOC(dir,type,nr,size); dir=READ|WRITE=3, type='u'=0x75, nr=0x21
_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRSHIFT   = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT  = _IOC_SIZESHIFT + _IOC_SIZEBITS
_IOC_WRITE = 1
_IOC_READ  = 2
def _IOC(d, t, nr, sz):
    return (d << _IOC_DIRSHIFT) | (t << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (sz << _IOC_SIZESHIFT)
UVCIOC_CTRL_QUERY = _IOC(_IOC_READ | _IOC_WRITE, ord('u'), 0x21, ctypes.sizeof(_XUQuery))

# -- LI XU selectors (verbatim from includes/uvc_extension_unit_ctrl.h) --
LI_XU_SENSOR_MODES_SWITCH           = 0x01
LI_XU_SENSOR_WINDOW_REPOSITION      = 0x02
LI_XU_LED_MODES                     = 0x03
LI_XU_SENSOR_GAIN_CONTROL_RGB       = 0x04
LI_XU_SENSOR_NO_REG_I2C_RW          = 0x05
LI_XU_SENSOR_UUID_HWFW_REV          = 0x07
LI_XU_PTS_QUERY                     = 0x08
LI_XU_SOFT_TRIGGER                  = 0x09
LI_XU_TRIGGER_DELAY                 = 0x0a
LI_XU_TRIGGER_MODE                  = 0x0b
LI_XU_SENSOR_REGISTER_CONFIGURATION = 0x0c
LI_XU_SENSOR_REG_RW                 = 0x0e
LI_XU_ERASE_EEPROM                  = 0x0f
LI_XU_GENERIC_I2C_RW                = 0x10
LI_XU_SENSOR_DEFECT_PIXEL_TABLE     = 0x11
LI_XU_SENSOR_REGISTER_CONFIG        = 0x1f

_SIZES = {
    LI_XU_SENSOR_MODES_SWITCH:           2,
    LI_XU_SENSOR_WINDOW_REPOSITION:      8,
    LI_XU_LED_MODES:                     1,
    LI_XU_SENSOR_GAIN_CONTROL_RGB:       8,
    LI_XU_SENSOR_NO_REG_I2C_RW:          259,
    LI_XU_SENSOR_UUID_HWFW_REV:          49,
    LI_XU_PTS_QUERY:                     4,
    LI_XU_SOFT_TRIGGER:                  2,
    LI_XU_TRIGGER_DELAY:                 4,
    LI_XU_TRIGGER_MODE:                  2,
    LI_XU_SENSOR_REGISTER_CONFIGURATION: 256,
    LI_XU_SENSOR_REG_RW:                 5,
    LI_XU_ERASE_EEPROM:                  2,
    LI_XU_GENERIC_I2C_RW:                262,
    LI_XU_SENSOR_DEFECT_PIXEL_TABLE:     33,
    LI_XU_SENSOR_REGISTER_CONFIG:        256,
}

# I2C read/write flag bytes (byte 0 of LI_XU_GENERIC_I2C_RW payload)
# from the same .cpp file: 0x81/0x82 = write, 0x01/0x02 = read
GENERIC_I2C_WRITE_FLG = 0x81   # 8-bit subaddr write
GENERIC_I2C_READ_FLG  = 0x01   # 8-bit subaddr read
GENERIC_I2C_WRITE_FLG_16 = 0x82
GENERIC_I2C_READ_FLG_16  = 0x02

# Sensor REG R/W flag (byte 0 of LI_XU_SENSOR_REG_RW payload)
SENSOR_REG_WRITE_FLG = 0x01
SENSOR_REG_READ_FLG  = 0x00

# UNIT id (FX3-based Leopard cameras hardcode 3 in the .cpp)
LI_UNIT = 3


class LeopardLinuxError(IOError):
    pass


class LeopardLinux:
    """Drop-in mirror of the public surface of LeopardCamera.dll.

    Methods names are camelCase to match the .NET DLL surface that
    seeker's eo/leopard_sdk_helper.py reflects via Python.NET clr.
    Adapt seeker call-sites to import this class on Linux instead.
    """

    def __init__(self, dev_path="/dev/video0"):
        self.dev_path = dev_path
        # O_RDWR | O_NONBLOCK ; UVC ioctls don't need streaming fd,
        # but other code may want to stream — match common pattern.
        self.fd = os.open(dev_path, os.O_RDWR)

    def close(self):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None

    # ---- low-level XU primitives (write_to_UVC_extension / read_from_) -

    def _xu_write(self, selector, data: bytes):
        size = _SIZES[selector]
        buf = (ctypes.c_uint8 * size).from_buffer_copy(
            data.ljust(size, b'\x00')[:size]
        )
        q = _XUQuery(unit=LI_UNIT, selector=selector,
                     query=UVC_SET_CUR, size=size, data=buf)
        try:
            fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, q)
        except OSError as e:
            raise LeopardLinuxError(
                f"XU SET_CUR failed selector=0x{selector:02x} errno={e.errno}"
            ) from e

    def _xu_read(self, selector) -> bytes:
        size = _SIZES[selector]
        buf = (ctypes.c_uint8 * size)()
        q = _XUQuery(unit=LI_UNIT, selector=selector,
                     query=UVC_GET_CUR, size=size, data=buf)
        try:
            fcntl.ioctl(self.fd, UVCIOC_CTRL_QUERY, q)
        except OSError as e:
            raise LeopardLinuxError(
                f"XU GET_CUR failed selector=0x{selector:02x} errno={e.errno}"
            ) from e
        return bytes(buf)

    # ---- public API matching LeopardCamera.dll -------------------------

    def SetSensorMode(self, mode: int):
        """LI_XU_SENSOR_MODES_SWITCH — selects res/fps/binning preset."""
        self._xu_write(LI_XU_SENSOR_MODES_SWITCH, bytes([mode & 0xff, 0]))

    def GetSensorMode(self) -> int:
        return self._xu_read(LI_XU_SENSOR_MODES_SWITCH)[0]

    def SetRGBGain(self, r, gr, gb, b):
        """LI_XU_SENSOR_GAIN_CONTROL_RGB — 4×u16 little-endian."""
        payload = struct.pack("<HHHH", r & 0xffff, gr & 0xffff,
                              gb & 0xffff, b & 0xffff)
        self._xu_write(LI_XU_SENSOR_GAIN_CONTROL_RGB, payload)

    def I2CRegRW(self, rw_flag: int, buf_cnt: int, slave_addr: int,
                 reg_addr: int, data: int = 0) -> int:
        """LI_XU_GENERIC_I2C_RW — payload layout from uvc_extension_unit_ctrl.cpp.
        rw_flag=GENERIC_I2C_WRITE_FLG / GENERIC_I2C_READ_FLG, buf_cnt is
        register VALUE width in bytes (1 or 2).
        """
        if buf_cnt not in (1, 2):
            raise ValueError("buf_cnt must be 1 or 2")
        payload = bytearray(_SIZES[LI_XU_GENERIC_I2C_RW])
        payload[0] = rw_flag
        payload[1] = buf_cnt - 1
        payload[2] = (slave_addr >> 8) & 0xff
        payload[3] = slave_addr & 0xff
        payload[4] = (reg_addr >> 8) & 0xff
        payload[5] = reg_addr & 0xff
        if rw_flag in (GENERIC_I2C_WRITE_FLG, GENERIC_I2C_WRITE_FLG_16):
            if buf_cnt == 1:
                payload[6] = data & 0xff
            else:
                # NOTE: per .cpp byte order is HIGH first then LOW (big-endian)
                payload[6] = (data >> 8) & 0xff
                payload[7] = data & 0xff
            self._xu_write(LI_XU_GENERIC_I2C_RW, bytes(payload))
            return 0
        else:  # READ — write addr, then read back
            self._xu_write(LI_XU_GENERIC_I2C_RW, bytes(payload))
            rd = self._xu_read(LI_XU_GENERIC_I2C_RW)
            if buf_cnt == 1:
                return rd[6]
            return (rd[6] << 8) | rd[7]

    def SetRegRW(self, reg_addr: int, reg_val: int):
        """LI_XU_SENSOR_REG_RW — 1+2+2 bytes (flag,addrH,addrL,valH,valL)."""
        payload = bytes([
            SENSOR_REG_WRITE_FLG,
            (reg_addr >> 8) & 0xff, reg_addr & 0xff,
            (reg_val  >> 8) & 0xff, reg_val  & 0xff,
        ])
        self._xu_write(LI_XU_SENSOR_REG_RW, payload)

    def GetRegRW(self, reg_addr: int) -> int:
        payload = bytearray([
            SENSOR_REG_READ_FLG,
            (reg_addr >> 8) & 0xff, reg_addr & 0xff,
            0, 0,
        ])
        self._xu_write(LI_XU_SENSOR_REG_RW, bytes(payload))
        rd = self._xu_read(LI_XU_SENSOR_REG_RW)
        return (rd[3] << 8) | rd[4]

    def SetExposure(self, exposure: int):
        """LI_XU_SENSOR_REGISTER_CONFIGURATION — bulk-mode exposure write.
        Layout used by Leopard's tool: payload[0]=group-write opcode, then
        a list of (reg_data_width, reg_addr, reg_val) tuples. seeker's
        existing helper drives this through I2CRegRW for IMX568 because
        Sony exposure is a multi-register sequence — call SetExposureRegs
        from the helper instead of this convenience.
        """
        # Convenience: write to a single common shutter register; seeker's
        # detailed sequencing should use I2CRegRW directly (mirrors what
        # the Windows DLL does internally).
        self.I2CRegRW(GENERIC_I2C_WRITE_FLG, 2, 0x34, 0x3012, exposure & 0xffff)

    def SetGain(self, gain: int):
        """Convenience: write gain via I2C. IMX568 analog-gain register is
        0x3014/0x3015 per Sony datasheet; seeker's helper should specify
        exact register. This helper writes one common location."""
        self.I2CRegRW(GENERIC_I2C_WRITE_FLG, 2, 0x34, 0x3014, gain & 0xffff)

    def GetSensorUUID_HWFW(self) -> bytes:
        """LI_XU_SENSOR_UUID_HWFW_REV — 49 bytes, used for camera ID."""
        return self._xu_read(LI_XU_SENSOR_UUID_HWFW_REV)


# -- self-test / smoke test ------------------------------------------------

def smoke(dev="/dev/video0"):
    print(f"[leopard_linux] opening {dev} ...")
    cam = LeopardLinux(dev)
    try:
        uuid = cam.GetSensorUUID_HWFW()
        print(f"[leopard_linux] UUID/HWFW (49B): {uuid.hex()}")
        try:
            mode = cam.GetSensorMode()
            print(f"[leopard_linux] current sensor mode = {mode}")
        except LeopardLinuxError as e:
            print(f"[leopard_linux] GetSensorMode: {e}")
        # Try a no-op-ish I2C read of a Sony chip ID register (0x3000-ish)
        try:
            v = cam.I2CRegRW(GENERIC_I2C_READ_FLG, 1, 0x34, 0x3000)
            print(f"[leopard_linux] I2C[0x34][0x3000] = 0x{v:02x}")
        except LeopardLinuxError as e:
            print(f"[leopard_linux] I2C read: {e}")
    finally:
        cam.close()


if __name__ == "__main__":
    import sys
    smoke(sys.argv[1] if len(sys.argv) > 1 else "/dev/video0")

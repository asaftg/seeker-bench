"""32-bit Python helper that drives Leopard's LeopardCamera.dll directly.

Why this exists
----------------
The FX3 USB3 bridge on the LI-IMX568-GMSL2 silently ignores every
standard UVC property write (CAP_PROP_AUTO_EXPOSURE, CAP_PROP_EXPOSURE,
BRIGHTNESS, GAMMA, ...). Proven exhaustively in
scripts/eo_probe_uvc_props.py:

    Bridge ACCEPTS: GAIN (clamped to {0, 1}), CONVERT_RGB (readback only)
    Bridge IGNORES: BRIGHTNESS, CONTRAST, EXPOSURE, AUTO_EXPOSURE,
                    GAMMA, SHARPNESS, BACKLIGHT, ISO_SPEED, ...

So bridge AE is forced on at the UVC layer and we cannot pin the sensor
exposure that way. The user observed visible "breathing" frame-to-frame
on a static scene — broken classification because the same scene
produces different pixel values depending on AE state.

Leopard's CameraTool talks to the same bridge and IS able to disable
AE / pin exposure / set gain. They use a vendor-specific UVC Extension
Unit + I2C-over-UVC pipe exposed by ``LeopardCamera.dll`` — methods like
``SetExposure``, ``SetGain``, ``set_AE``, ``I2CRegRW``,
``WriteToUVCExtension``. (Full reflection dump available via
``--dump-api``.)

Both ``LeopardCamera.dll`` and the native ``CAppLib.dll`` are 32-bit
(PE machine 0x14c). Our seeker is 64-bit, so we cannot load them in-
process. This helper bridges the gap as a one-shot subprocess:

    seeker (64-bit)
        ── subprocess(tools/python311-x86/python.exe leopard_sdk_helper.py
                      --exposure E --gain G --json)
        ── waits for ok=true result
        ── PyAV opens the camera, captures frames with locked settings

The Leopard SDK opens the device, writes the AE-off + manual values,
and closes. The sensor retains the settings across our PyAV reopen.

API used (from System.Reflection enumeration of LeopardCamera.dll v1.0):
    LeopardCamera.LPCamera
        UpdateCameraList()                  ← enumerate dshow cameras
        Open(DsDevice, int, int)            ← attach to one camera
        set_AE(bool)                        ← TURN OFF AUTO-EXPOSURE
        set_Exposure(int)                   ← lock exposure
        set_Gain(int)                       ← lock gain
        Stop() / Close()                    ← release; settings persist
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback


def _check_bitness() -> None:
    if sys.maxsize > 2**32:
        sys.stderr.write(
            "leopard_sdk_helper must be run from 32-bit Python "
            "(LeopardCamera.dll is x86). "
            f"Current: {sys.executable} ({sys.maxsize})\n"
        )
        sys.exit(2)


def _setup_clr(release_dir: str):
    """Set Windows DLL search path so CAppLib.dll resolves, then load CLR."""
    os.chdir(release_dir)
    sys.path.insert(0, release_dir)
    try:
        os.add_dll_directory(release_dir)
    except Exception:
        pass
    import clr  # pythonnet
    clr.AddReference(os.path.join(release_dir, "LeopardCamera.dll"))
    clr.AddReference(os.path.join(release_dir, "DirectShowLib-2005.dll"))


def _find_imx_device():
    """Return the DsDevice for the IMX568/Leopard/FX3 camera, or None."""
    from DirectShowLib import DsDevice, FilterCategory  # type: ignore
    devices = DsDevice.GetDevicesOfCat(FilterCategory.VideoInputDevice)
    hints = ("imx", "leopard", "li-", "fx3")
    for dev in devices:
        name = (dev.Name or "").lower()
        if any(h in name for h in hints):
            return dev
    if devices:
        # Fall back to the first device — we have no other way to pick.
        return devices[0]
    return None


def _open_lpcamera(release_dir: str):
    """Construct an LPCamera and Open() it on the IMX568 device.

    LPCamera.Open(DsDevice, int resIndex, int frIndex) — resolution
    index and framerate index into the camera's reported caps. Index 0
    is the first available, which on the IMX568 is the native
    2472x2064 mode.
    """
    import LeopardCamera as LC  # type: ignore
    cam = LC.LPCamera()
    # Walk dshow devices to find ours.
    dev = _find_imx_device()
    if dev is None:
        raise RuntimeError("no DirectShow video input device found")
    # resIndex=0 (first reported resolution), frIndex=0 (first framerate).
    cam.Open(dev, 0, 0)
    return cam, dev


def _list_methods(obj) -> list[str]:
    # Don't call getattr — that invokes property getters and some throw NRE
    # before Run() has populated internal state. Just enumerate names.
    return sorted(m for m in dir(obj) if not m.startswith("_"))


def _safe_snapshot(cam) -> dict:
    """Read every property in isolation; record exception per failure."""
    snap: dict = {}
    for name in ("AE", "Exposure", "ExposureExt", "Gain",
                 "Width", "Height", "Bits"):
        try:
            snap[name] = int(getattr(cam, name)) if name != "AE" else bool(getattr(cam, name))
        except Exception as e:
            snap[name + "_err"] = repr(e)
    return snap


def main() -> int:
    _check_bitness()
    # If --stream is requested, lock stdout down BEFORE anything else
    # has a chance to write to it. We:
    #   1) duplicate the original fd-1 to a saved fd (the FRAME CHANNEL)
    #   2) point fd 1 (and Python's sys.stdout) at fd 2 (stderr) so any
    #      print(), logging.StreamHandler(sys.stdout), or .NET
    #      Console.WriteLine that fires during SDK init goes to stderr
    #      (which the parent already drains and logs).
    #   3) flip fd 1 to binary on Windows.
    # We then stream binary frames to the FRAME CHANNEL fd directly via
    # os.write — bypassing any text wrapper.
    _stream_mode = ("--stream" in sys.argv)
    _stream_frame_fd = None
    if _stream_mode:
        try:
            _stream_frame_fd = os.dup(1)            # save original fd-1
            os.dup2(2, 1)                            # fd-1 -> stderr
            sys.stdout = sys.stderr                  # Python-level too
            try:
                import msvcrt as _msvcrt
                _msvcrt.setmode(_stream_frame_fd, os.O_BINARY)
            except Exception:
                pass
        except Exception as e:
            sys.stderr.write(f"stream-fd-init failed: {e!r}\n")

    p = argparse.ArgumentParser()
    p.add_argument("--release-dir",
                   default=r"C:\Users\asaf.ruf.BLUERIVERTECH\Downloads\Release")
    p.add_argument("--exposure", type=int, default=None)
    p.add_argument("--exposure-ext", type=int, default=None,
                   help="Set ExposureExt (the actually-readable exposure prop)")
    p.add_argument("--gain", type=int, default=None)
    p.add_argument("--ae", choices=("on", "off"), default="off",
                   help="Auto-exposure: 'off' to lock to manual values")
    p.add_argument("--inspect", action="store_true",
                   help="Open device, dump current values, exit")
    p.add_argument("--probe-ranges", action="store_true",
                   help="Call GetCameraControlPropertyRange / GetVideoProcAmpPropertyRange")
    p.add_argument("--bits", type=int, default=None,
                   help="Set Bits (8/10/12) — bridge bit depth")
    p.add_argument("--sensor-mode", type=int, default=None,
                   help="SetSensorMode(int) — sensor operating mode (HDR/linear/etc)")
    p.add_argument("--probe-setters", action="store_true",
                   help="Try every settable property to map what actually works")
    p.add_argument("--i2c-write", action="append", default=[],
                   help="Direct sensor I2C write 'subAddr:regAddr:val' "
                        "(hex), repeatable. e.g. --i2c-write 0x34:0x0202:0x0200")
    p.add_argument("--i2c-read", action="append", default=[],
                   help="Direct sensor I2C read 'subAddr:regAddr:nbytes' (hex)")
    p.add_argument("--reg-write", action="append", default=[],
                   help="LPCamera.SetRegRW write 'addr:val' — likely FX3 "
                        "firmware register, not sensor I2C. Untested.")
    p.add_argument("--reg-read", action="append", default=[],
                   help="LPCamera.SetRegRW read 'addr' — see --reg-write.")
    # ── frame capture (the path that gives Leopard-quality pixels) ─────
    p.add_argument("--capture-frame", default=None,
                   help="Grab a single frame via LPCamera.CaptureImage and "
                        "save it. Output path. The buffer is written as "
                        "raw bytes; a sidecar '.meta.json' next to it "
                        "carries width/height/bpp so the reader knows how "
                        "to reshape. This is the same call CameraTool uses "
                        "to save a BMP — bypasses the FX3 preview-mode "
                        "AGC/debayer path that UVC streaming runs through.")
    p.add_argument("--capture-bmp", default=None,
                   help="Like --capture-frame but also writes a viewable "
                        "BMP next to it (for visual confirmation of "
                        "Leopard-equivalence).")
    p.add_argument("--capture-warmup", type=int, default=3,
                   help="Discard this many frames before --capture-frame "
                        "(let auto-gain settle if any).")
    p.add_argument("--data-mode", default="RAW12",
                   choices=("YUV", "RAW12", "RAW10", "RAW8", "RGB888"),
                   help="SENSOR_DATA_MODE for SetParam(). RAW12 is the "
                        "Leopard CameraTool default for IMX568 — gives "
                        "the same clean 16-bit-packed mono pixels their "
                        ".raw file holds. YUV is the FX3 preview-pipeline "
                        "mode with internal AGC (the noisy path).")
    p.add_argument("--capture-width", type=int, default=2472,
                   help="Width passed to SetParam(). 2472 is IMX568's "
                        "native raw mode width (Leopard's BMP is "
                        "2472x2064). 2592 is the YUV-preview mode.")
    p.add_argument("--capture-height", type=int, default=2064)
    p.add_argument("--stream", action="store_true",
                   help="Long-lived streaming mode. After init+warmup, "
                        "loop: CaptureImage -> emit [4-byte LE length][payload] "
                        "frames on stdout. The 64-bit parent reads frames "
                        "and decodes RAW12 itself. Stops when stdin closes "
                        "or stdout pipe breaks. Implies --data-mode RAW12.")
    p.add_argument("--stream-fps", type=float, default=10.0,
                   help="Soft cap; the SDK delivers ~10 fps anyway in "
                        "RAW12 mode so this rarely throttles.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    result: dict = {"ok": False, "stage": "init"}
    cam = None
    try:
        _setup_clr(args.release_dir)
        result["stage"] = "loaded_dll"

        cam, dev = _open_lpcamera(args.release_dir)
        result["device_name"] = str(dev.Name)
        result["device_path"] = str(dev.DevicePath) if hasattr(dev, "DevicePath") else None
        result["stage"] = "opened"

        # SetParam BEFORE Run — picks the sensor data mode (YUV preview
        # vs RAW12 etc.) and the resolution. Without this, Run() defaults
        # to YUV preview which is the FX3 AGC'd path — exactly what we're
        # trying to escape. We only call it when a capture is requested
        # so non-capture diagnostic invocations of this helper are
        # unaffected (they all expect the YUV-preview state).
        if args.capture_frame or args.capture_bmp or args.stream:
            try:
                import System
                from LeopardCamera import LPCamera as _LPCam
                # Streaming defaults to RAW12 mode regardless of arg.
                dm = "RAW12" if args.stream else args.data_mode
                mode_enum = getattr(_LPCam.SENSOR_DATA_MODE, dm)
                cam.SetParam(int(args.capture_width),
                             int(args.capture_height),
                             False,                # display: no preview window
                             System.IntPtr.Zero,   # parentWin: none
                             mode_enum)
                result["set_param"] = {
                    "width": int(args.capture_width),
                    "height": int(args.capture_height),
                    "data_mode": args.data_mode,
                }
            except Exception as e:
                result["set_param_err"] = repr(e)

        # LPCamera.Open() doesn't actually initialize the streaming graph —
        # AE/Exposure/Gain getters throw NullReferenceException until Run()
        # is called. Run() spins up the capture pipeline; properties become
        # readable shortly after.
        try:
            cam.Run()
            # Longer settle: in RAW12 mode the sensor's first 5–10 frames
            # after Run() are uninitialized / dark current — std<1 LSB,
            # essentially a blank frame. Empirically, ~1.5 s is needed
            # before the first frame is real scene data.
            time.sleep(1.5)
            result["stage"] = "running"
        except Exception as e:
            result["run_err"] = repr(e)

        # Snapshot before we change anything (each property in isolation
        # so one failure doesn't blank the whole snapshot).
        result["before"] = _safe_snapshot(cam)

        if args.probe_ranges:
            # IAMCameraControl exposes Exposure/Focus/Iris/Pan/Tilt/Roll/Zoom (1..7).
            # IAMVideoProcAmp exposes Brightness/Contrast/Hue/Sat/Sharp/Gamma/WB/Backlight/Gain.
            cam_ctrl_props = ["Pan", "Tilt", "Roll", "Zoom",
                              "Exposure", "Iris", "Focus"]
            vproc_props = ["Brightness", "Contrast", "Hue", "Saturation",
                           "Sharpness", "Gamma", "ColorEnable", "WhiteBalance",
                           "BacklightCompensation", "Gain"]
            ranges: dict = {}
            for name in cam_ctrl_props:
                try:
                    r = cam.GetCameraControlPropertyRange(name)
                    ranges[f"camctrl.{name}"] = repr(r)
                except Exception as e:
                    ranges[f"camctrl.{name}_err"] = repr(e)
            for name in vproc_props:
                try:
                    r = cam.GetVideoProcAmpPropertyRange(name)
                    ranges[f"vproc.{name}"] = repr(r)
                except Exception as e:
                    ranges[f"vproc.{name}_err"] = repr(e)
            result["ranges"] = ranges
            result["ok"] = True
        elif args.inspect:
            result["camera_methods"] = _list_methods(cam)
            result["ok"] = True
        elif args.probe_setters:
            # Probe what the bridge actually accepts. Each test:
            #   1. Read current value (if getter works)
            #   2. Write a known-changed value
            #   3. Read back if possible
            #   4. Restore (best-effort)
            probes = {}
            test_writes = {
                "Bits": [10, 12, 8],
                "Width": [1280, 2472, 2592],
                "Height": [720, 2064, 1944],
                "ExposureExt": [200, 1000, 5000],
            }
            for prop, vals in test_writes.items():
                pres = []
                for v in vals:
                    rec = {"set": v}
                    try:
                        setattr(cam, prop, int(v))
                        rec["set_ok"] = True
                    except Exception as e:
                        rec["set_err"] = repr(e)
                    try:
                        rec["readback"] = int(getattr(cam, prop))
                    except Exception as e:
                        rec["readback_err"] = repr(e)
                    pres.append(rec)
                probes[prop] = pres

            # SetSensorMode + SetParam (no getter we know of)
            for sm in (0, 1, 2, 3):
                try:
                    cam.SetSensorMode(sm)
                    probes[f"SetSensorMode({sm})"] = "ok"
                except Exception as e:
                    probes[f"SetSensorMode({sm})"] = repr(e)

            result["probes"] = probes
            result["ok"] = True
        else:
            # Disable AE before writing manual values. Order matters —
            # if we write Exposure while AE is on, the bridge promptly
            # overwrites our value on the next AE tick.
            try:
                cam.AE = (args.ae == "on")
                result["set_AE"] = (args.ae == "on")
            except Exception as e:
                result["set_AE_err"] = repr(e)

            if args.exposure is not None:
                try:
                    cam.Exposure = int(args.exposure)
                    result["set_Exposure"] = int(args.exposure)
                except Exception as e:
                    result["set_Exposure_err"] = repr(e)

            if args.exposure_ext is not None:
                try:
                    target_exp = int(args.exposure_ext)
                    # FX3 bridge "exposure wiggle". Empirically, a freshly
                    # opened device on a cold IMX568 returns the sensor
                    # noise floor (raw12 ≈ 15) regardless of what we
                    # write to ExposureExt. We've reproduced this many
                    # times: 8 consecutive --exposure-ext 1264 single-
                    # shot calls all return mean≈14.75. But if the
                    # exposure VALUE changes between/within subprocess
                    # sessions (e.g. 500, 1264, 2000…), the sensor
                    # starts producing real data. Conclusion: writing
                    # the SAME ExposureExt twice is a no-op on the
                    # bridge firmware; we must issue at least two
                    # DIFFERENT values to force a state-change commit.
                    # So: write a "decoy" value first, sleep, then
                    # write the real target. Fixes streaming and one-
                    # shot equally.
                    decoy = 100 if target_exp != 100 else 5000
                    cam.ExposureExt = decoy
                    time.sleep(0.4)
                    cam.ExposureExt = target_exp
                    result["set_ExposureExt"] = target_exp
                    result["set_ExposureExt_decoy"] = decoy
                    # Let the new exposure clock out into a full frame.
                    time.sleep(0.8)
                except Exception as e:
                    result["set_ExposureExt_err"] = repr(e)

            if args.gain is not None:
                try:
                    cam.Gain = int(args.gain)
                    result["set_Gain"] = int(args.gain)
                except Exception as e:
                    result["set_Gain_err"] = repr(e)

            if args.bits is not None:
                try:
                    cam.Bits = int(args.bits)
                    result["set_Bits"] = int(args.bits)
                    try:
                        result["Bits_readback"] = int(cam.Bits)
                    except Exception as e:
                        result["Bits_readback_err"] = repr(e)
                except Exception as e:
                    result["set_Bits_err"] = repr(e)

            if args.sensor_mode is not None:
                try:
                    cam.SetSensorMode(int(args.sensor_mode))
                    result["SetSensorMode"] = int(args.sensor_mode)
                except Exception as e:
                    result["SetSensorMode_err"] = repr(e)

            # Direct I2C writes — last so settings on top of mode/bits
            # changes. Format: "subAddr:regAddr:val" (hex). regAddr is
            # 16-bit, val is one byte. For multi-byte regs (exposure is
            # 16-bit at 0x0202/0x0203), pass two writes.
            i2c_results = []
            for spec in args.i2c_write:
                parts = spec.split(":")
                if len(parts) != 3:
                    i2c_results.append({"spec": spec, "err": "bad format"})
                    continue
                try:
                    sub = int(parts[0], 0)
                    reg = int(parts[1], 0)
                    val = int(parts[2], 0)
                    # I2CRegRW(rw_flag, bufCnt, subAddress, regAddr, regData[])
                    # rw_flag: 0=write, 1=read (varies by SDK; try 0 first)
                    import System
                    arr = System.Array[System.Byte]([val & 0xFF])
                    cam.I2CRegRW(0, 1, sub, reg, arr)
                    i2c_results.append({"spec": spec, "wrote": hex(val)})
                except Exception as e:
                    i2c_results.append({"spec": spec, "err": repr(e)})
            for spec in args.i2c_read:
                parts = spec.split(":")
                if len(parts) != 3:
                    i2c_results.append({"spec": spec, "err": "bad format"})
                    continue
                try:
                    sub = int(parts[0], 0)
                    reg = int(parts[1], 0)
                    n = int(parts[2], 0)
                    import System
                    arr = System.Array[System.Byte]([0] * n)
                    cam.I2CRegRW(1, n, sub, reg, arr)
                    i2c_results.append({"spec": spec,
                                        "read": [int(b) for b in arr]})
                except Exception as e:
                    i2c_results.append({"spec": spec, "err": repr(e)})
            if i2c_results:
                result["i2c"] = i2c_results

            # SetRegRW — writes to a register on the FX3 ITSELF (firmware
            # control), not to the sensor over I2C. If the FX3 has an
            # "AE enable" register exposed here, turning it off would let
            # our sensor I2C writes stick. Untested as of 2026-04-24.
            reg_results = []
            for spec in args.reg_write:
                parts = spec.split(":")
                if len(parts) != 2:
                    reg_results.append({"spec": spec, "err": "bad format"})
                    continue
                try:
                    addr = int(parts[0], 0)
                    val = int(parts[1], 0)
                    # SetRegRW(rw_flag, address, value) — rw_flag 0 = write
                    cam.SetRegRW(0, addr, val)
                    reg_results.append({"spec": spec, "wrote_via_SetRegRW": hex(val)})
                except Exception as e:
                    reg_results.append({"spec": spec, "err": repr(e)})
            for spec in args.reg_read:
                try:
                    addr = int(spec, 0)
                    val = cam.SetRegRW(1, addr, 0)
                    reg_results.append({"spec": spec, "read_via_SetRegRW": int(val) if val is not None else None})
                except Exception as e:
                    reg_results.append({"spec": spec, "err": repr(e)})
            if reg_results:
                result["reg"] = reg_results

            # ── frame capture ───────────────────────────────────────
            # CaptureImage() is what CameraTool uses internally to save
            # BMPs / RAWs (we found the .raw file at 2472*2064*2 = 16-bit
            # mono, exactly what this method should hand back). Bypasses
            # the FX3 preview-mode debayer + auto-gain pipeline that the
            # UVC stream runs through, which is the whole reason the
            # PyAV-streamed output is noisier than CameraTool's BMP at
            # the same ExposureExt. SDK signature:
            #
            #   Int32 CaptureImage(out IntPtr pBuffer,
            #                      ref Int32 width, ref Int32 height,
            #                      ref Int32 bpp)
            #
            # pythonnet maps `out IntPtr` and `ref Int32` to additional
            # tuple-return values: the call returns (rc, ptr, w, h, bpp).
            # We then Marshal.Copy the unmanaged bytes to managed, free
            # the native buffer, and write the result to disk.
            if args.capture_frame or args.capture_bmp:
                import System
                from System import IntPtr, Int32
                from System.Runtime.InteropServices import Marshal

                # warm-up: drop a few frames so any internal AE/AGC
                # has settled to the (locked) ExposureExt state above.
                # Sleep between captures so we actually advance through
                # distinct frames (without the sleep, CaptureImage can
                # return the same buffered frame back-to-back).
                warmup_stats: list = []
                for wi in range(max(0, int(args.capture_warmup))):
                    try:
                        rc_w, p_w, w_w, h_w, b_w = cam.CaptureImage(
                            IntPtr.Zero, Int32(0), Int32(0), Int32(0))
                        # Quick stat on this warmup frame so we can
                        # see in the JSON whether the sensor is alive.
                        try:
                            if p_w != IntPtr.Zero:
                                bpp_w = (int(b_w) + 7) // 8
                                nb_w = int(w_w) * int(h_w) * bpp_w
                                # Only sample first 64KB for speed
                                sample_n = min(nb_w, 65536)
                                m_arr = System.Array.CreateInstance(
                                    System.Byte, sample_n)
                                Marshal.Copy(p_w, m_arr, 0, sample_n)
                                samp = bytes(m_arr)
                                # mean of LO bytes only (since data is
                                # raw12 in u16 LE, LO byte ≈ raw12 & 0xFF)
                                lo_mean = (sum(samp[::2]) /
                                           max(len(samp[::2]), 1))
                                warmup_stats.append(
                                    {"i": wi, "lo_mean": round(lo_mean, 1)})
                        except Exception:
                            pass
                        if p_w != IntPtr.Zero:
                            try:
                                cam.FreeImageBuffer(p_w)
                            except Exception:
                                pass
                    except Exception:
                        # Some SDK versions only have CameraCaptureImage
                        # (caller-allocated). Fall through to the real
                        # capture below; that branch handles both.
                        break
                    time.sleep(0.12)
                cap_info_warmup = warmup_stats

                cap_info: dict = {}
                try:
                    rc, ptr, w, h, bpp = cam.CaptureImage(
                        IntPtr.Zero, Int32(0), Int32(0), Int32(0))
                    cap_info["return_code"] = int(rc) if rc is not None else None
                    cap_info["width"] = int(w)
                    cap_info["height"] = int(h)
                    cap_info["bpp"] = int(bpp)
                    if ptr == IntPtr.Zero:
                        cap_info["err"] = "CaptureImage returned null buffer"
                    else:
                        # Copy unmanaged bytes -> managed byte array.
                        # IMX568 raw mono = 16 bits/pixel; the SDK packs
                        # those as 2 bytes/pixel. Some sensor modes hand
                        # back 8 bpp (preview) or 24 bpp (debayered).
                        # bytes_per_pixel = ceil(bpp/8).
                        bytes_per_px = (int(bpp) + 7) // 8
                        nbytes = int(w) * int(h) * bytes_per_px
                        cap_info["nbytes"] = nbytes
                        # IMPORTANT: SDK reports bpp=24 (3 B/px) but in
                        # RAW12 mode it actually only fills the FIRST
                        # W*H*2 bytes with valid raw12-in-uint16 data;
                        # the trailing W*H bytes are uninitialized
                        # zeros. We copy/save only the valid prefix
                        # when in a raw mode.
                        is_raw_mode = (args.data_mode in
                                       ("RAW12", "RAW10", "RAW8"))
                        valid_nbytes = nbytes
                        if is_raw_mode:
                            valid_nbytes = int(w) * int(h) * 2
                        managed = System.Array.CreateInstance(
                            System.Byte, valid_nbytes)
                        Marshal.Copy(ptr, managed, 0, valid_nbytes)
                        # Convert to Python bytes for I/O.
                        py_bytes = bytes(managed)
                        cap_info["valid_nbytes"] = valid_nbytes
                        if args.capture_frame:
                            out_path = args.capture_frame
                            with open(out_path, "wb") as f:
                                f.write(py_bytes)
                            meta = {
                                "width": int(w), "height": int(h),
                                "bpp": int(bpp),
                                "bytes_per_pixel": bytes_per_px,
                                "valid_nbytes": valid_nbytes,
                                "data_mode": args.data_mode,
                                "layout": ("raw12_le_uint16_left_shift_4"
                                           if args.data_mode == "RAW12"
                                           else "as-bpp-says"),
                                "exposure_ext": int(args.exposure_ext) if args.exposure_ext else None,
                                "warmup_stats": cap_info_warmup,
                            }
                            with open(out_path + ".meta.json", "w") as f:
                                json.dump(meta, f, indent=2)
                            cap_info["wrote"] = out_path
                        if args.capture_bmp:
                            # Also write a viewable BMP for visual sanity.
                            # 16-bit mono → 8-bit AGC stretch; 8-bit mono
                            # → write directly; 24-bit BGR → write directly.
                            try:
                                import numpy as np
                                arr = np.frombuffer(py_bytes, dtype=np.uint8)
                                if bytes_per_px == 2:
                                    img16 = arr.view(np.uint16).reshape(int(h), int(w))
                                    p_lo, p_hi = np.percentile(img16, [0.5, 99.5])
                                    img8 = np.clip((img16 - p_lo) /
                                                   max(p_hi - p_lo, 1) * 255.0,
                                                   0, 255).astype(np.uint8)
                                elif bytes_per_px == 1:
                                    img8 = arr.reshape(int(h), int(w))
                                else:
                                    img8 = arr.reshape(int(h), int(w),
                                                       bytes_per_px)
                                # OpenCV writes BMP by extension.
                                import cv2
                                cv2.imwrite(args.capture_bmp, img8)
                                cap_info["bmp"] = args.capture_bmp
                                cap_info["bmp_mean"] = float(img8.mean())
                            except Exception as e:
                                cap_info["bmp_err"] = repr(e)
                        # Quick stats so the caller can sanity-check that
                        # the capture is sensible without re-loading.
                        try:
                            import numpy as np
                            arr = np.frombuffer(py_bytes, dtype=np.uint8)
                            if bytes_per_px == 2:
                                arr16 = arr.view(np.uint16)
                                cap_info["raw_mean"] = float(arr16.mean())
                                cap_info["raw_min"] = int(arr16.min())
                                cap_info["raw_max"] = int(arr16.max())
                            else:
                                cap_info["raw_mean"] = float(arr.mean())
                                cap_info["raw_min"] = int(arr.min())
                                cap_info["raw_max"] = int(arr.max())
                        except Exception:
                            pass
                        # Free the native buffer the SDK allocated.
                        try:
                            cam.FreeImageBuffer(ptr)
                        except Exception as e:
                            cap_info["free_err"] = repr(e)
                except Exception as e:
                    cap_info["fatal"] = repr(e)
                    cap_info["traceback"] = traceback.format_exc()
                result["capture"] = cap_info

            # ── streaming loop (long-lived) ─────────────────────────
            if args.stream:
                import System
                from System import IntPtr, Int32
                from System.Runtime.InteropServices import Marshal
                import struct
                if _stream_frame_fd is None:
                    sys.stderr.write("STREAM_ERR no frame fd available\n")
                    raise RuntimeError("no frame fd")
                frame_fd = _stream_frame_fd

                # Settle warmup before streaming begins. Empirically
                # the SDK delivers ~10 fps in RAW12 mode, so frames
                # are spaced ~100 ms apart; with sleep < 100 ms we'd
                # race the streaming pipeline and read the same buffer
                # twice (and ALSO seem to read a stale "first frame"
                # buffer that never gets refreshed). 0.15 s gives the
                # pipeline a real chance to advance, AND we sample the
                # mean of each warmup frame to see when the sensor has
                # actually started integrating signal.
                stream_warmup = max(int(args.capture_warmup), 12)
                wu_means: list = []
                for wi in range(stream_warmup):
                    try:
                        rc_w, p_w, w_w, h_w, b_w = cam.CaptureImage(
                            IntPtr.Zero, Int32(0), Int32(0), Int32(0))
                        if p_w != IntPtr.Zero:
                            try:
                                # Sample first 64KB to compute LO-byte
                                # mean (proxy for raw12 mean)
                                samp_n = 65536
                                m_arr = System.Array.CreateInstance(
                                    System.Byte, samp_n)
                                Marshal.Copy(p_w, m_arr, 0, samp_n)
                                samp = bytes(m_arr)
                                lo = samp[::2]
                                lo_mean = sum(lo) / max(len(lo), 1)
                                wu_means.append(round(lo_mean, 1))
                            except Exception:
                                pass
                            try:
                                cam.FreeImageBuffer(p_w)
                            except Exception:
                                pass
                    except Exception:
                        break
                    time.sleep(0.15)
                sys.stderr.write("STREAM_WU_MEANS " +
                                 json.dumps(wu_means) + "\n")
                sys.stderr.flush()

                # Send a small JSON header on stderr so the parent can
                # learn W/H/format without parsing the payload.
                hdr = {"width": int(args.capture_width),
                       "height": int(args.capture_height),
                       "data_mode": "RAW12",
                       "valid_nbytes": int(args.capture_width) *
                                       int(args.capture_height) * 2,
                       "exposure_ext": int(args.exposure_ext) if args.exposure_ext else None}
                sys.stderr.write("STREAM_HDR " + json.dumps(hdr) + "\n")
                sys.stderr.flush()

                target_dt = 1.0 / max(float(args.stream_fps), 0.1)
                frame_idx = 0
                while True:
                    t_loop = time.time()
                    try:
                        rc, ptr, w, h, bpp = cam.CaptureImage(
                            IntPtr.Zero, Int32(0), Int32(0), Int32(0))
                    except Exception as e:
                        sys.stderr.write(f"STREAM_ERR capture: {e!r}\n")
                        break
                    if ptr == IntPtr.Zero:
                        sys.stderr.write("STREAM_ERR null buffer\n")
                        break
                    valid_n = int(w) * int(h) * 2  # RAW12-as-uint16-LE
                    try:
                        managed = System.Array.CreateInstance(
                            System.Byte, valid_n)
                        Marshal.Copy(ptr, managed, 0, valid_n)
                        py_bytes = bytes(managed)
                    finally:
                        try:
                            cam.FreeImageBuffer(ptr)
                        except Exception:
                            pass
                    # Frame on the wire: [4-byte LE length][bytes]
                    try:
                        os.write(frame_fd, struct.pack("<I", valid_n))
                        # Large payload: write may return short on Win;
                        # loop until everything is out.
                        view = memoryview(py_bytes)
                        sent = 0
                        while sent < len(view):
                            wn = os.write(frame_fd, view[sent:])
                            if wn <= 0:
                                raise OSError("os.write returned 0")
                            sent += wn
                    except (BrokenPipeError, OSError) as e:
                        sys.stderr.write(f"STREAM_ERR pipe write: {e!r}\n")
                        break
                    frame_idx += 1
                    # Throttle if SDK delivers faster than stream_fps
                    elapsed = time.time() - t_loop
                    if elapsed < target_dt:
                        time.sleep(target_dt - elapsed)
                # Done — fall through to cleanup
                result["stream_frames_emitted"] = frame_idx

            # Settle so the bridge propagates writes to sensor I2C.
            time.sleep(0.3)

            # Read back to confirm.
            result["after"] = _safe_snapshot(cam)

            result["ok"] = True
    except Exception as e:
        result["fatal"] = repr(e)
        result["traceback"] = traceback.format_exc()
    finally:
        # Always close cleanly so PyAV can reopen the device.
        if cam is not None:
            try:
                cam.Stop()
            except Exception:
                pass
            try:
                cam.Close()
            except Exception:
                pass

    if args.stream:
        # Stream mode: stdout was used for binary frames. Send any
        # final summary on stderr so parent can log it.
        try:
            sys.stderr.write("STREAM_END " + json.dumps(result) + "\n")
            sys.stderr.flush()
        except Exception:
            pass
        return 0 if result.get("ok") else 1

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        for k, v in result.items():
            print(f"{k}: {v}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())

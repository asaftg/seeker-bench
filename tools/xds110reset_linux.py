#!/usr/bin/env python3
"""Linux equivalent of TI's xds110reset.exe — pulses nSRST on an
AWR2944P chip via its onboard XDS110 debug bridge using libusb.

Why this exists: when the AWR's CLI / mmw_demoDDM firmware wedges
(typical after a Ctrl+C of the host or a stuck calibration), the
Windows code path calls
    C:\\ti\\ccs\\ccs\\ccs_base\\common\\uscif\\xds110\\xds110reset.exe
to recover the chip without a physical 12V barrel cycle. CCS ships
the same binary for x86_64 Linux but NOT for aarch64 — and the
AWR2944P's debug bridge is fully software-resettable from Linux via
libusb, so we don't need TI's binary at all.

Protocol reference: OpenOCD's xds110.c (canonical Linux-side source
for the XDS110 vendor protocol). Bulk endpoints OUT=0x02 IN=0x83.
Frame: '*' (0x2A) | size_u16_LE | payload. Opcodes XDS_CONNECT=0x01,
XDS_DISCONNECT=0x02, XDS_SET_SRST=0x0e (payload byte 0=assert,
1=deassert).

Usage (after `pip install pyusb` + udev rule for 0451:bef3):
    python3 tools/xds110reset_linux.py            # 50ms pulse
    python3 tools/xds110reset_linux.py --hold 100 # 100ms pulse

udev rule (so non-root works):
    SUBSYSTEM=="usb", ATTRS{idVendor}=="0451",
        ATTRS{idProduct}=="bef3", MODE="0666"
Save to /etc/udev/rules.d/71-ti-xds110.rules then:
    sudo udevadm control --reload && sudo udevadm trigger
"""
from __future__ import annotations

import argparse
import struct
import sys
import time

try:
    import usb.core
    import usb.util
except ImportError:
    sys.stderr.write(
        "ERROR: pyusb not installed. Install with:\n"
        "    .venv/bin/pip install pyusb\n"
    )
    sys.exit(2)


VID, PID = 0x0451, 0xBEF3
EP_OUT, EP_IN = 0x02, 0x83
XDS_CONNECT, XDS_DISCONNECT, XDS_SET_SRST = 0x01, 0x02, 0x0E

# JTAG vendor interface number on XDS110 (per OpenOCD xds110.c).
JTAG_INTERFACE = 2


def _frame(payload: bytes) -> bytes:
    return b"*" + struct.pack("<H", len(payload)) + payload


def _txn(dev, payload: bytes, in_len_payload: int) -> bytes:
    dev.write(EP_OUT, _frame(payload), timeout=4000)
    resp = bytes(dev.read(EP_IN, 3 + in_len_payload, timeout=4000))
    if not resp.startswith(b"*"):
        raise RuntimeError(f"XDS110 bad framing: {resp!r}")
    size = struct.unpack("<H", resp[1:3])[0]
    err = struct.unpack("<I", resp[3:7])[0]
    if err != 0:
        raise RuntimeError(f"XDS110 cmd 0x{payload[0]:02x} returned err 0x{err:08x}")
    return resp[3 : 3 + size]


def xds110_pulse_nrst(hold_ms: int = 50) -> None:
    """Assert nSRST low, hold for `hold_ms`, deassert. Returns when
    deassertion is acknowledged. After this returns, the AWR2944P
    re-runs its boot ROM and starts mmw_demoDDM fresh.

    IMPORTANT: only touch interface ``JTAG_INTERFACE`` (the XDS110's
    vendor JTAG class). The XDS110 also exposes two cdc_acm interfaces
    (the radar CLI + data UARTs at /dev/ttyACM0 and /dev/ttyACM1) and
    a usbhid interface — if we detach those kernel drivers, they
    never rebind on their own and /dev/seeker_radar_cli vanishes
    until the host re-enumerates the USB device.
    """
    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if dev is None:
        raise RuntimeError(
            f"XDS110 not found (USB VID:PID {VID:04x}:{PID:04x}). "
            f"Is the AWR EVM's J8 USB cable plugged in?"
        )
    # Do NOT call dev.set_configuration() — the device is already
    # configured by the kernel and re-setting forces a USB-level reset
    # that drops cdc_acm bindings.
    detached_jtag = False
    try:
        if dev.is_kernel_driver_active(JTAG_INTERFACE):
            dev.detach_kernel_driver(JTAG_INTERFACE)
            detached_jtag = True
    except (usb.core.USBError, NotImplementedError):
        pass
    usb.util.claim_interface(dev, JTAG_INTERFACE)
    try:
        _txn(dev, bytes([XDS_CONNECT]), 4)
        _txn(dev, bytes([XDS_SET_SRST, 0]), 4)  # assert (drive nSRST low)
        time.sleep(hold_ms / 1000.0)
        _txn(dev, bytes([XDS_SET_SRST, 1]), 4)  # deassert
        _txn(dev, bytes([XDS_DISCONNECT]), 4)
    finally:
        usb.util.release_interface(dev, JTAG_INTERFACE)
        if detached_jtag:
            try:
                dev.attach_kernel_driver(JTAG_INTERFACE)
            except (usb.core.USBError, NotImplementedError):
                pass
        usb.util.dispose_resources(dev)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hold", type=int, default=50,
                    help="nSRST hold time in milliseconds (default 50)")
    args = ap.parse_args()
    print(f"Pulsing nSRST on XDS110 (VID:PID {VID:04x}:{PID:04x}) for {args.hold} ms ...")
    xds110_pulse_nrst(args.hold)
    print("OK — chip should now be re-booting mmw_demoDDM. Allow ~1 s before sending CLI commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

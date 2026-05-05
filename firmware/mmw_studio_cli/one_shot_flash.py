"""one_shot_flash.py - End-to-end AWR2944P firmware flash for mmw_studio_cli.

Flashes the freshly-built awr2x44P_mmw_studio_cli.appimage onto the chip via
TI's uart_uniflash.py over the XDS110 virtual COM (J8). Uses the FT4232H on
J10 (Port C bit 6 = nRESET, Port D bits 2/3/4 = SOP[2:0]) to drive the chip
into UART boot mode (SOP[2:0] = 101 = "QSPI flash programming") and back to
functional mode (001 = QSPI boot) without operator jumper changes.

Reference:
    AWR2944PEVM schematic SPRUJ22C §2.10 (SOP table)
    radar_setup.md       (J17/J18/J20 jumper map; flash transport = J8 XDS110)
    radar_dca/ftdi_pin_holder.py (FT4232H pin map for nRESET + SOP)
    scripts/reflash_chip.py      (proven manual workflow this script automates)

What "one shot" means: the user runs `one_shot_flash.bat`, the script
automates SOP set, nRESET pulse, flash, SOP clear, nRESET pulse, then
verifies the chip is alive in functional mode.

Falls back to operator instructions if FT4232H is unavailable (cable not
plugged into J10, Zadig drivers not installed, another process holding it).

Usage (from this directory):
    one_shot_flash.bat              # full auto if FT4232H ready, otherwise
                                    # prints jumper instructions and waits
    one_shot_flash.bat --manual     # always use manual jumper workflow
    one_shot_flash.bat --port COM11 # override XDS110 SBL-UART port
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import serial
import serial.tools.list_ports

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
APPIMAGE = HERE / "awr2x44P_mmw_studio_cli.appimage"
FLASH_CFG = HERE / "flash.cfg"

UART_UNIFLASH = Path(
    "C:/ti/mcu_plus_sdk_awr2x44p_10_02_00_04/tools/boot/uart_uniflash.py"
)

# AWR2944PEVM J8 XDS110 virtual COM = SBL UART (MSS_UARTA). This is where
# uart_uniflash.py talks. Default per radar_setup.md / reflash_chip.py.
DEFAULT_SBL_PORT = "COM11"

# FT4232H VID + candidate PIDs (some EVMs are EEPROM-reprogrammed).
FTDI_VID = 0x0403
FTDI_PIDS = (0x6011, 0x6014)

# Port C bit 6 = nRESET (active LOW). Other bits driven HIGH so any chip
# function we don't control is safely de-asserted.
PORTC_DIR  = 0xFF
PORTC_NRST_DEASSERTED = 0xFF        # nRESET high = chip running
PORTC_NRST_ASSERTED   = 0xFF & ~(1 << 6)  # nRESET low = chip in reset

# Port D bits: SOP0 = D2, SOP1 = D3, SOP2 = D4.
# AWR2944P boot modes (SPRUJ22C §2.10 / mcu_plus_sdk boot.h):
#   SOP[2:0] = 001 (decimal 1) -> Functional / QSPI flash boot
#   SOP[2:0] = 101 (decimal 5) -> SBL UART boot (= flash-programming mode)
PORTD_DIR  = 0xFF
PORTD_FUNCTIONAL = (0xE0 | (1 << 2))                   # 001: D2=1, D3=0, D4=0
PORTD_FLASHMODE  = (0xE0 | (1 << 2) | (1 << 4))        # 101: D2=1, D3=0, D4=1

# Reset pulse width: AWR2944P PORZ minimum is 10 us; 100 ms is well past
# the SoC + PMIC re-power-up settling envelope and matches mmW Studio's
# ar1.SOPControl() implementation in mmwl_port_ftdi.c.
RESET_PULSE_S      = 0.10
POST_RESET_WAIT_S  = 0.50    # let ROM bootloader come up before XMODEM probe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_banner(title: str) -> None:
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def _detect_xds110_port() -> Optional[str]:
    """Return the AWR2944PEVM J8 XDS110 SBL-UART COM port if visible.

    The XDS110 enumerates as two consecutive virtual COM ports; the higher-
    numbered one is the user UART (SBL-UART). VID/PID = 0451:bef3 (TI XDS110).
    Falls back to None if not found — caller can use --port to override.
    """
    candidates = []
    for p in serial.tools.list_ports.comports():
        if p.vid == 0x0451 and p.pid in (0xBEF3, 0xBEF4):
            candidates.append(p.device)
    if not candidates:
        return None
    # XDS110 'Class' channel is the one we want; pick the higher COM index.
    candidates.sort(key=lambda d: int("".join(c for c in d if c.isdigit()) or "0"))
    return candidates[-1]


def _wait_for_uart_boot(port: str, timeout_s: float = 8.0) -> bool:
    """Listen for the ROM's XMODEM-CRC 'C' (0x43) handshake on the SBL UART."""
    print(f"    Probing {port} for ROM XMODEM-CRC handshake ('C' bytes)...")
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except Exception as e:
        print(f"    ERROR: cannot open {port}: {e}")
        return False
    try:
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
                if 0x43 in buf:
                    return True
            else:
                time.sleep(0.05)
        return False
    finally:
        ser.close()


def _wait_for_functional_banner(port: str, timeout_s: float = 12.0) -> bool:
    """After flash + functional reset, listen for any sign of life from
    the firmware's boot prints. Recognizes either mmw_studio_cli (custom
    minimal FW) or mmw_demoDDM (TI SDK demo + Seeker patches) tokens.
    """
    print(f"    Probing {port} for firmware boot banner...")
    try:
        ser = serial.Serial(port, 115200, timeout=0.5)
    except Exception as e:
        print(f"    ERROR: cannot open {port}: {e}")
        return False
    try:
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
                text = buf.decode("ascii", errors="replace")
                # mmw_studio_cli boot tokens, mmw_demoDDM SDK boot tokens,
                # plus the SEEKER PATCH boot marker that proves PAD_BYPASS
                # ran on the patched-demoDDM fork.
                for needle in ("mmw_radar", "mmw_studio", "mmw_lvds",
                               "MMWave_init", "MMW Demo",
                               "MMWDemo", "PAD_BYPASS",
                               "Init Calibration Status"):
                    if needle in text:
                        print(f"    Saw banner token: {needle!r}")
                        return True
            else:
                time.sleep(0.1)
        # Print whatever we did see for diagnostic value.
        if buf:
            print("    Got bytes but no recognized banner. Tail of buffer:")
            print("    " + repr(buf[-240:]))
        return False
    finally:
        ser.close()


# ---------------------------------------------------------------------------
# FTDI electronic SOP / nRESET driver
# ---------------------------------------------------------------------------

class FtdiSopDriver:
    """Drives FT4232H Port C (nRESET) and Port D (SOP[2:0]) on the AWR EVM J10.

    Idempotent open() that returns False if FTDI is unavailable rather than
    raising — the caller falls back to manual operator instructions in that
    case.
    """

    def __init__(self) -> None:
        self._ftdi_c = None
        self._ftdi_d = None

    def open(self) -> bool:
        try:
            from pyftdi.ftdi import Ftdi
        except ImportError:
            print("    pyftdi not installed (pip install pyftdi). Skipping electronic SOP.")
            return False

        for pid in FTDI_PIDS:
            try:
                Ftdi.add_custom_product(FTDI_VID, pid)
            except Exception:
                pass

        last_err: Optional[Exception] = None
        for pid in FTDI_PIDS:
            url_c = f"ftdi://0x{FTDI_VID:04x}:0x{pid:04x}/3"   # interface 3 = ch C
            url_d = f"ftdi://0x{FTDI_VID:04x}:0x{pid:04x}/4"   # interface 4 = ch D
            try:
                ftdi_c = Ftdi()
                ftdi_c.open_bitbang_from_url(url_c, direction=PORTC_DIR)
                ftdi_d = Ftdi()
                ftdi_d.open_bitbang_from_url(url_d, direction=PORTD_DIR)
            except Exception as e:
                last_err = e
                continue
            self._ftdi_c = ftdi_c
            self._ftdi_d = ftdi_d
            print(f"    FT4232H opened (VID 0x{FTDI_VID:04x} PID 0x{pid:04x}).")
            return True

        print(f"    FT4232H not opened. Last error: {last_err!r}")
        print("    Causes: J10 USB not plugged, Zadig drivers not installed on")
        print("    interfaces 2/3, mmW Studio or another process holding the")
        print("    device. Falling back to manual SOP workflow.")
        return False

    def close(self) -> None:
        for ftdi in (self._ftdi_c, self._ftdi_d):
            if ftdi is not None:
                try:
                    ftdi.close()
                except Exception:
                    pass
        self._ftdi_c = None
        self._ftdi_d = None

    def _write_d(self, value: int) -> None:
        assert self._ftdi_d is not None
        self._ftdi_d.write_data(bytes([value & 0xFF]))

    def _write_c(self, value: int) -> None:
        assert self._ftdi_c is not None
        self._ftdi_c.write_data(bytes([value & 0xFF]))

    def reset_into(self, mode: str) -> None:
        """Set SOP, pulse nRESET, release. mode = 'flash' or 'functional'."""
        if mode == "flash":
            sop_value = PORTD_FLASHMODE
            label = "SOP[2:0]=101 (UART boot / flash)"
        elif mode == "functional":
            sop_value = PORTD_FUNCTIONAL
            label = "SOP[2:0]=001 (QSPI functional)"
        else:
            raise ValueError(f"bad mode: {mode}")

        # 1. Assert reset (drive nRESET low). SOP value isn't sampled until
        #    the rising edge of nRESET, so order: nRESET low, then set SOP,
        #    hold, then nRESET high.
        self._write_c(PORTC_NRST_ASSERTED)
        time.sleep(0.005)

        # 2. Set SOP to the requested value while the chip is in reset.
        self._write_d(sop_value)
        time.sleep(RESET_PULSE_S)

        # 3. Release reset (drive nRESET high). Chip latches SOP at this
        #    rising edge and proceeds to ROM bootloader.
        self._write_c(PORTC_NRST_DEASSERTED)
        print(f"    Released nRESET with {label}")

        # 4. Hold the SOP value steady afterwards so any drift doesn't
        #    re-strap the chip on a stray glitch.
        time.sleep(POST_RESET_WAIT_S)


# ---------------------------------------------------------------------------
# Manual fallback
# ---------------------------------------------------------------------------

def _manual_set_flash_mode() -> None:
    print()
    print("    >>> MANUAL ACTION REQUIRED: set EVM to FLASH mode <<<")
    print()
    print("    AWR2944PEVM jumpers (SPRUJ22C §2.10.2 Table 2-12):")
    print("      Flash mode (SOP[2:0]=101): J17 CLOSED, J18 OPEN, J20 CLOSED")
    print()
    print("    From functional state (J17 open, J18 open, J20 closed),")
    print("    add a shunt to J17 — that's the only physical change.")
    print()
    print("    Then power-cycle: pull the 12V barrel and plug it back in.")
    input("    Press ENTER once jumper is set and chip is power-cycled...")


def _manual_set_functional_mode() -> None:
    print()
    print("    >>> MANUAL ACTION REQUIRED: set EVM back to FUNCTIONAL mode <<<")
    print()
    print("    AWR2944PEVM jumpers:")
    print("      Functional (SOP[2:0]=001): J17 OPEN, J18 OPEN, J20 CLOSED")
    print()
    print("    Remove the shunt you added to J17. Then power-cycle:")
    print("    pull the 12V barrel and plug it back in.")
    input("    Press ENTER once jumper is removed and chip is power-cycled...")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default=None,
                    help=f"XDS110 SBL-UART COM port (default: auto-detect, "
                         f"fallback {DEFAULT_SBL_PORT})")
    ap.add_argument("--manual", action="store_true",
                    help="Use manual jumper workflow even if FT4232H is available.")
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip post-flash functional-mode verification.")
    ap.add_argument("--appimage", default=None,
                    help=f"Path to the .appimage to flash. Default: "
                         f"{APPIMAGE}. Override to flash a different fork "
                         f"(e.g. firmware/mmw_demoDDM_patched/awr2x44P_mmw_demoDDM.appimage).")
    ap.add_argument("--cfg", default=None,
                    help=f"Path to the flash .cfg consumed by uart_uniflash.py. "
                         f"Default: {FLASH_CFG}. Override when flashing a "
                         f"different fork whose .cfg points at a different appimage.")
    args = ap.parse_args()

    appimage = Path(args.appimage).resolve() if args.appimage else APPIMAGE
    flash_cfg = Path(args.cfg).resolve() if args.cfg else FLASH_CFG
    fork_label = appimage.parent.name  # e.g. "mmw_studio_cli" or "mmw_demoDDM_patched"

    _print_banner(f"AWR2944P one-shot flash — {fork_label}")
    print(f"    Appimage : {appimage}")
    print(f"    Cfg      : {flash_cfg}")
    print(f"    Flasher  : {UART_UNIFLASH}")

    # Sanity checks.
    for path in (appimage, flash_cfg, UART_UNIFLASH):
        if not path.exists():
            print(f"    ERROR: missing required path: {path}")
            return 10

    # Resolve SBL UART port.
    port = args.port or _detect_xds110_port() or DEFAULT_SBL_PORT
    print(f"    SBL UART : {port}")

    # ----- Step 1: get chip into UART boot mode --------------------------
    _print_banner("Step 1 — put chip in UART boot mode (SOP[2:0]=101)")

    used_ftdi = False
    driver: Optional[FtdiSopDriver] = None
    if not args.manual:
        driver = FtdiSopDriver()
        if driver.open():
            print("    Driving SOP via FT4232H...")
            driver.reset_into("flash")
            used_ftdi = True
        else:
            driver.close()
            driver = None

    if not used_ftdi:
        _manual_set_flash_mode()

    if not _wait_for_uart_boot(port):
        print()
        print("    ERROR: chip is not in UART boot mode (no XMODEM 'C' on UART).")
        if used_ftdi:
            print("    The FTDI strap may not have taken — try --manual to set the")
            print("    physical jumpers and retry.")
        else:
            print("    Verify J17 is CLOSED, J18 is OPEN, J20 is CLOSED, and the")
            print("    12V barrel was actually unplugged + replugged AFTER the")
            print("    jumper change.")
        if driver is not None:
            driver.close()
        return 20
    print("    OK: chip is in UART boot mode.")

    # ----- Step 2: run uart_uniflash --------------------------------------
    _print_banner("Step 2 — flash via uart_uniflash.py")
    cmd = [
        sys.executable, str(UART_UNIFLASH),
        "--serial-port", port,
        "--cfg", str(flash_cfg),
    ]
    print(f"    {' '.join(cmd)}")
    print()
    proc = subprocess.run(cmd, cwd=str(UART_UNIFLASH.parent))
    if proc.returncode != 0:
        print()
        print(f"    ERROR: uart_uniflash.py exited {proc.returncode}.")
        print("    Chip is left in UART boot mode (SOP=101). Re-run this script")
        print("    or manually move J17 back to OPEN to recover.")
        if driver is not None:
            driver.close()
        return 30
    print("    OK: flash complete.")

    # ----- Step 3: back to functional mode --------------------------------
    _print_banner("Step 3 — return chip to functional mode (SOP[2:0]=001)")

    if used_ftdi and driver is not None:
        print("    Driving SOP via FT4232H...")
        driver.reset_into("functional")
    else:
        _manual_set_functional_mode()

    if not args.no_verify:
        if not _wait_for_functional_banner(port):
            print()
            print("    WARN: did not see mmw_studio_cli boot banner within timeout.")
            print("    Possible causes:")
            print("      - chip booted but boot prints muted (some logging configs)")
            print("      - chip is in a reset loop (check D9 LED on EVM)")
            print("      - SOP didn't return to functional mode")
            if driver is not None:
                driver.close()
            return 40
        print("    OK: chip is up in functional mode.")

    if driver is not None:
        driver.close()

    _print_banner("DONE — radar is flashed and running")
    print("    Next steps:")
    print("      1. Plug DCA1000 J1 USB if not already (FT4232H power)")
    print("      2. Connect Ethernet to DCA, ping 192.168.33.180")
    print("      3. Start the seeker_bench radar pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Flash the AWR2944P with the patched mmw_demoDDM.

Walks the operator through the SOP-jumper change, runs uart_uniflash,
verifies the chip boots into mmw_demo on functional mode, and exits.
Tolerates the operator forgetting to power-cycle by re-prompting.

Usage:
    python scripts/reflash_chip.py             # flash patched
    python scripts/reflash_chip.py --stock     # roll back to stock
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import serial
import serial.tools.list_ports

UART_UNIFLASH = Path(
    "C:/ti/mcu_plus_sdk_awr2x44p_10_02_00_04/tools/boot/uart_uniflash.py"
)
SCRIPTS_DIR = Path(__file__).resolve().parent
PATCHED_CFG = SCRIPTS_DIR / "flash_patched_fw.cfg"
STOCK_CFG = SCRIPTS_DIR / "flash_stock_fw.cfg"


def _confirm(msg: str) -> None:
    print(f"\n>>> {msg}")
    input("    Press Enter when done... ")


def _wait_for_uart_boot(timeout_s: float = 10.0) -> bool:
    """Listen on COM11 for the ROM's 'C' XMODEM-CRC handshake. Returns
    True if seen, False on timeout. The ROM emits 'C' (0x43) every
    1-2 sec while in UART boot mode."""
    print("    Probing COM11 for ROM XMODEM handshake...")
    try:
        ser = serial.Serial("COM11", 115200, timeout=0.5)
    except Exception as e:
        print(f"    COM11 not available: {e}")
        return False
    try:
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
                if 0x43 in buf:  # 'C'
                    return True
            else:
                time.sleep(0.05)
        return False
    finally:
        ser.close()


def _wait_for_functional_boot(timeout_s: float = 12.0) -> bool:
    """After flashing + SOP back to functional + power-cycle, the chip
    should boot mmw_demo. Listen for the 'AWR2X44P MMW Demo' banner."""
    print("    Probing COM11 for mmw_demo banner...")
    try:
        ser = serial.Serial("COM11", 115200, timeout=0.5)
    except Exception as e:
        print(f"    COM11 not available: {e}")
        return False
    try:
        deadline = time.monotonic() + timeout_s
        buf = bytearray()
        while time.monotonic() < deadline:
            if ser.in_waiting:
                buf.extend(ser.read(ser.in_waiting))
                text = buf.decode("ascii", errors="replace")
                if "MMW Demo" in text:
                    return True
            else:
                time.sleep(0.1)
        return False
    finally:
        ser.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stock", action="store_true",
                    help="Flash STOCK firmware instead of PATCHED (rollback path)")
    args = ap.parse_args()

    cfg_path = STOCK_CFG if args.stock else PATCHED_CFG
    label = "STOCK" if args.stock else "PATCHED"
    print(f"=== Flashing {label} firmware ===")
    print(f"    Config: {cfg_path}")

    # Step 1 — UART boot mode (SOP jumper).
    _confirm(
        "STEP 1: Set EVM to UART boot mode.\n"
        "    Add a jumper on SOP2 (so SOP0 + SOP2 are both closed,\n"
        "    SOP1 stays open).\n"
        "    Then PULL the 12V barrel and PLUG IT BACK IN.\n"
        "    LED on EVM should be solid."
    )
    if not _wait_for_uart_boot():
        print("    ERROR: chip is not in UART boot mode (no 'C' on COM11).")
        print("           Verify: SOP0 closed, SOP1 open, SOP2 closed,")
        print("                   AND chip was power-cycled AFTER setting SOP.")
        return 1
    print("    OK: chip is in UART boot mode.")

    # Step 2 — run uart_uniflash.
    print(f"\nSTEP 2: Running uart_uniflash with {cfg_path.name}...")
    cmd = [
        sys.executable, str(UART_UNIFLASH),
        "--serial-port", "COM11",
        "--cfg", str(cfg_path),
    ]
    proc = subprocess.run(cmd, cwd=Path(UART_UNIFLASH).parent)
    if proc.returncode != 0:
        print(f"    ERROR: flash failed (exit {proc.returncode}).")
        return 2
    print("    OK: flash complete.")

    # Step 3 — back to functional mode.
    _confirm(
        "STEP 3: Set EVM back to functional (QSPI) mode.\n"
        "    REMOVE the SOP2 jumper (so SOP0 stays closed but SOP2 is\n"
        "    open again — back to one jumper on SOP0 only).\n"
        "    Then PULL the 12V barrel and PLUG IT BACK IN."
    )
    if not _wait_for_functional_boot():
        print("    ERROR: chip is not booting mmw_demo.")
        print("           Verify SOP2 jumper is removed AND chip was power-cycled.")
        return 3
    print(f"    OK: chip is booting {label} mmw_demoDDM.")

    print("\n=== DONE — radar is ready. Start Seeker normally. ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Interactive Maestro servo calibration.

Walks you through picking the µs values that correspond to each
servo's mechanical endpoints, then prints a YAML snippet you can
paste into ``config/app_config.yaml`` under ``gimbal.*_calibration``.

Flow, per servo:
    1. Tool sends mid pulse (1500 µs).
    2. You use +/- keys to nudge the pulse until the servo points
       at the MIN angle (for pan: full left; for tilt: horizon).
    3. Enter that angle in degrees, press ENTER to record.
    4. Repeat at the MAX angle (pan: full right; tilt: 22° up).
    5. Script spits out the calibration block.

Run:
    python -m scripts.gimbal_calibrate
"""
from __future__ import annotations

import sys
import time

from common.logging_setup import configure
from gimbal.maestro_driver import MaestroDriver


def _ask_angle(prompt: str) -> float:
    while True:
        s = input(prompt).strip()
        try:
            return float(s)
        except ValueError:
            print("  (enter a number like 0 or 22)")


def _tune_endpoint(drv: MaestroDriver, channel: int, label: str) -> float:
    """Nudge servo with +/- until user says 'ok'. Returns chosen µs."""
    us = 1500.0
    drv.set_target_us(channel, us)
    print(f"\n── tuning {label} on channel {channel} ──")
    print("  '+'/'-' = ±10 µs    ']'/'[' = ±1 µs    ENTER = accept")
    while True:
        drv.set_target_us(channel, us)
        s = input(f"  {label} = {us:.0f} µs > ").strip()
        if s == "":
            return us
        for ch in s:
            if ch == "+":   us = min(2500, us + 10)
            elif ch == "-": us = max( 500, us - 10)
            elif ch == "]": us = min(2500, us +  1)
            elif ch == "[": us = max( 500, us -  1)


def calibrate_servo(drv: MaestroDriver, servo_name: str, default_ch: int,
                    suggested_min: float, suggested_max: float) -> dict:
    print(f"\n=== {servo_name.upper()} CALIBRATION ===")
    ch_s = input(f"Channel [{default_ch}]: ").strip()
    channel = int(ch_s) if ch_s else default_ch

    print(f"Step 1: get the gimbal to the {servo_name} MIN position "
          f"(suggested: {suggested_min}°).")
    us_min = _tune_endpoint(drv, channel, f"{servo_name}-min")
    a_min = _ask_angle(f"Enter the actual angle in degrees [{suggested_min}]: ") if False else suggested_min
    a_min_in = input(f"Actual min angle in degrees [{suggested_min}]: ").strip()
    a_min = float(a_min_in) if a_min_in else suggested_min

    print(f"\nStep 2: get the gimbal to the {servo_name} MAX position "
          f"(suggested: {suggested_max}°).")
    us_max = _tune_endpoint(drv, channel, f"{servo_name}-max")
    a_max_in = input(f"Actual max angle in degrees [{suggested_max}]: ").strip()
    a_max = float(a_max_in) if a_max_in else suggested_max

    return {
        "channel": channel,
        "min_deg": a_min,
        "max_deg": a_max,
        "us_at_min_deg": round(us_min, 1),
        "us_at_max_deg": round(us_max, 1),
        "invert": False,
    }


def main() -> int:
    configure(level="INFO")
    drv = MaestroDriver()
    if not drv.open():
        print("ERROR: no Pololu Maestro found on any COM port.")
        print("Plug in the USB cable and re-run.")
        return 1
    print(f"Maestro on {drv.port}")

    try:
        pan  = calibrate_servo(drv, "pan",  default_ch=0,
                               suggested_min=-90.0, suggested_max=90.0)
        tilt = calibrate_servo(drv, "tilt", default_ch=1,
                               suggested_min=  0.0, suggested_max=22.0)

        print("\n========= PASTE INTO config/app_config.yaml =========")
        print("gimbal:")
        print("  pan_calibration:")
        for k, v in pan.items():
            print(f"    {k}: {v}")
        print("  tilt_calibration:")
        for k, v in tilt.items():
            print(f"    {k}: {v}")
        print("=====================================================")
    finally:
        # Float servos so they stop holding
        drv.release_all([pan.get("channel", 0) if 'pan' in dir() else 0,
                         tilt.get("channel", 1) if 'tilt' in dir() else 1])
        drv.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

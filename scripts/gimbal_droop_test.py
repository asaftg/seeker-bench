"""Diagnose the tilt 'loses power after 5s' droop.

Symptoms reported 2026-04-25 by operator: gimbal moves to commanded
tilt, then visibly sags downward over the next ~5 s. Question: is
this software (we stop sending pulses), firmware (Maestro serial
timeout), or hardware (servo torque insufficient against gravity at
this tilt with this load)?

This test isolates the software/firmware question from the hardware
one. It runs THREE scenarios on tilt channel 1, in sequence:

  1. CONTINUOUS-HOLD: command 60° tilt, then re-send the same pulse
     every 16ms (60Hz, matching the production manager) for 30 s.
     Read back the Maestro's reported position every 500 ms via the
     'Get Position' command. If the reported quarter-µs value stays
     within ±5 of the commanded value, the Maestro is faithfully
     holding the position and any visible droop is mechanical/
     voltage (servo can't fight gravity).

  2. ONE-SHOT-NO-RESEND: command 60°, then STOP sending serial.
     Read back position every 500 ms for 15 s. If the reported
     position drifts toward the floor or pulses stop being emitted,
     the Maestro firmware has a 'Serial timeout' configured —
     we'd need to either disable it (Pololu Control Center) or
     keep our resend rate well above its threshold.

  3. ANGLE-SWEEP-HOLD: command 30°, hold 5 s. Then 60°, hold 5 s.
     Then 90°, hold 5 s. Continuous 60 Hz resend. Read back position
     in each phase. If droop appears only at one specific angle,
     that's the gravity-load fingerprint — the servo's holding
     torque at that mechanical advantage isn't enough.

Run from project root with the seeker NOT running (only one process
can hold COM4 at a time):

    python scripts/gimbal_droop_test.py [--port COM4] [--channel 1]

Pass ``--no-zero`` to skip the safety park-at-30°-on-exit.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Allow running as a script from project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gimbal.maestro_driver import MaestroDriver


# ── Maestro 'Get Position' compact protocol ────────────────────────
# 0x90, channel  →  device returns 2 bytes (lo, hi) of position in
# quarter-microseconds. So us = (lo | (hi<<8)) / 4.
def maestro_get_position_us(driver: MaestroDriver,
                            channel: int) -> Optional[float]:
    ser = driver._ser  # type: ignore[attr-defined]
    if ser is None or not getattr(ser, "is_open", False):
        return None
    try:
        ser.write(bytes([0x90, int(channel) & 0x7F]))
        ser.flush()
        # Maestro responds 2 bytes. Pyserial timeout is on the Serial
        # object (we configured 0.1s in the driver constructor).
        data = ser.read(2)
        if len(data) != 2:
            return None
        qus = data[0] | (data[1] << 8)
        return qus / 4.0
    except Exception as e:
        print(f"  ! get_position failed: {e}", flush=True)
        return None


# Hard-coded calibration mirror of app_config.yaml as of 2026-04-25.
# We deliberately don't load_config() here so the test can run cold
# without any seeker context. If you change the YAML, mirror here.
TILT_MIN_DEG = 3.0
TILT_MAX_DEG = 90.0
TILT_US_AT_MIN = 1075.0
TILT_US_AT_MAX = 2000.0


def tilt_angle_to_us(angle_deg: float) -> float:
    a = max(TILT_MIN_DEG, min(TILT_MAX_DEG, float(angle_deg)))
    span = TILT_MAX_DEG - TILT_MIN_DEG
    t = (a - TILT_MIN_DEG) / span
    return TILT_US_AT_MIN + t * (TILT_US_AT_MAX - TILT_US_AT_MIN)


# ── Test scenarios ─────────────────────────────────────────────────

def hold_and_poll(driver: MaestroDriver,
                  channel: int,
                  cmd_us: float,
                  duration_s: float,
                  resend: bool,
                  label: str) -> Tuple[float, float, float]:
    """Send `cmd_us` to `channel`, then for `duration_s`:
       - if `resend`: re-send the same pulse every 16 ms (~60 Hz)
       - poll get_position every 500 ms and log it.

    Returns (initial_reported_us, final_reported_us, max_abs_drift_us).
    """
    print(f"\n[{label}] cmd={cmd_us:.0f} µs  resend={resend}  "
          f"duration={duration_s}s", flush=True)
    driver.set_target_us(channel, cmd_us)
    # Give the servo time to physically reach the commanded position
    # before we start sampling.
    time.sleep(0.6)

    initial = maestro_get_position_us(driver, channel)
    if initial is None:
        print(f"  ! initial get_position returned None", flush=True)
        initial = float("nan")
    print(f"  t=0.0s   reported={initial:.0f} µs   diff={initial - cmd_us:+.0f}",
          flush=True)

    t0 = time.time()
    next_poll = t0 + 0.5
    next_resend = t0
    last_reported = initial
    max_drift = 0.0
    while True:
        now = time.time()
        elapsed = now - t0
        if elapsed >= duration_s:
            break
        if resend and now >= next_resend:
            driver.set_target_us(channel, cmd_us)
            next_resend = now + (1.0 / 60.0)
        if now >= next_poll:
            rep = maestro_get_position_us(driver, channel)
            if rep is None:
                print(f"  t={elapsed:4.1f}s  reported=None "
                      f"(maestro stopped responding!)", flush=True)
            else:
                drift = rep - cmd_us
                if abs(drift) > abs(max_drift):
                    max_drift = drift
                arrow = ""
                if last_reported is not None and not (rep != rep):  # nan
                    if rep < last_reported - 2:
                        arrow = " ↓"
                    elif rep > last_reported + 2:
                        arrow = " ↑"
                print(f"  t={elapsed:4.1f}s  reported={rep:.0f} µs   "
                      f"diff={drift:+.0f}{arrow}", flush=True)
                last_reported = rep
            next_poll = now + 0.5
        # Tight sleep — 1 ms granularity is plenty.
        time.sleep(0.001)

    final = maestro_get_position_us(driver, channel)
    final_v = final if final is not None else float("nan")
    print(f"  END     reported={final_v:.0f} µs   "
          f"diff={final_v - cmd_us:+.0f}   "
          f"max drift over run = {max_drift:+.0f} µs", flush=True)
    return (initial, final_v, max_drift)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None,
                    help="COM port (auto if absent)")
    ap.add_argument("--channel", type=int, default=1,
                    help="Tilt servo channel (default 1)")
    ap.add_argument("--no-zero", action="store_true",
                    help="Skip the safety park-at-30°-on-exit")
    args = ap.parse_args()

    driver = MaestroDriver(port=args.port)
    if not driver.open():
        print("FAIL: could not open Maestro. "
              "Make sure the seeker isn't running.", flush=True)
        return 1
    print(f"Maestro opened on {driver.port}", flush=True)

    # Park at a safe middle tilt before we start so the test starts
    # from a known position.
    safe_us = tilt_angle_to_us(30.0)
    print(f"\nParking at 30° ({safe_us:.0f} µs) before tests...", flush=True)
    driver.set_target_us(args.channel, safe_us)
    time.sleep(1.0)

    results = {}

    # ── Test 1: continuous-hold (production-equivalent) ────────────
    cmd60 = tilt_angle_to_us(60.0)
    init_t1, final_t1, drift_t1 = hold_and_poll(
        driver, args.channel, cmd60, duration_s=20.0,
        resend=True, label="TEST 1: continuous 60Hz resend at 60°")
    results["continuous_hold"] = drift_t1

    # ── Test 2: one-shot-no-resend ─────────────────────────────────
    init_t2, final_t2, drift_t2 = hold_and_poll(
        driver, args.channel, cmd60, duration_s=15.0,
        resend=False, label="TEST 2: one-shot at 60°, NO resends")
    results["one_shot"] = drift_t2

    # ── Test 3: angle sweep ────────────────────────────────────────
    print("\n[TEST 3] Angle sweep with 60Hz resends — droop varies "
          "with mechanical advantage / gravity?", flush=True)
    for angle in (15.0, 45.0, 75.0):
        cmd = tilt_angle_to_us(angle)
        _, _, drift = hold_and_poll(
            driver, args.channel, cmd, duration_s=10.0,
            resend=True, label=f"TEST 3.{angle:.0f}: hold {angle}°")
        results[f"sweep_{int(angle)}"] = drift

    # ── Park back to safe and shut down ────────────────────────────
    if not args.no_zero:
        print(f"\nParking back at 30° ({safe_us:.0f} µs) on exit...",
              flush=True)
        driver.set_target_us(args.channel, safe_us)
        time.sleep(1.0)

    driver.close()

    # ── Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print("RESULTS (max drift in µs from commanded):", flush=True)
    for k, v in results.items():
        print(f"  {k:24s} {v:+.0f} µs", flush=True)
    print("\nINTERPRETATION GUIDE:", flush=True)
    print("  continuous_hold drift > 30 µs  → servo physically "
          "drooping despite our pulses are being sent and ack'd.", flush=True)
    print("                                   Cause: voltage/torque "
          "(hardware), NOT software. Recommend bigger PSU or "
          "stiffer servo.", flush=True)
    print("  one_shot drift > continuous_hold drift  → Maestro "
          "Serial Timeout is firing. Disable in Pololu Control "
          "Center, or raise our resend rate.", flush=True)
    print("  sweep results show drift only at high angles  → "
          "gravity load specific; servo is undersized for the "
          "mounted camera weight at that tilt.", flush=True)
    print("  All drifts < 10 µs but operator still sees droop  → "
          "the droop is inside Maestro's PWM resolution; servo "
          "is doing its best but slipping mechanically (clamp, "
          "gear backlash). Mechanical fix needed.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

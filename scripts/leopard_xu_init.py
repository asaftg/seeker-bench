#!/usr/bin/env python3
"""Pre-stream XU init for IMX568 on Linux.

Disables FX3 trigger mode (XU 0x0b = [0,0]) so the sensor runs free.
Verified by decompiling LeopardCamera.dll: this is what
LPCamera.EnableTriggerMode(false, false) writes.

Note: trigger-disable also gets re-applied inside _v4l2_raw_backend.py
on every open() (after STREAMON) and periodically during streaming.
This script is now mostly informational/preflight.
"""
import sys, time
sys.path.insert(0, "/home/asaftg/seeker-bench")
from eo.leopard_linux import LeopardLinux, _SIZES
_SIZES[0x0b] = 2

DEV = "/dev/video0"
RETRIES = 5

for i in range(RETRIES):
    try:
        c = LeopardLinux(DEV)
        try:
            before = int.from_bytes(c._xu_read(0x0b), "little")
            c._xu_write(0x0b, bytes([0, 0]))
            after = int.from_bytes(c._xu_read(0x0b), "little")
            print("[xu_init] %s trigger_mode 0x%04x -> 0x%04x (free-running)"
                  % (DEV, before, after))
        finally:
            c.close()
        sys.exit(0)
    except Exception as e:
        print("[xu_init] attempt %d/%d failed: %s" % (i+1, RETRIES, e))
        time.sleep(1)
sys.exit(1)

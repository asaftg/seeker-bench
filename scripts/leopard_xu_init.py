#!/usr/bin/env python3
"""Pre-stream XU init for IMX568 on Linux.

The Leopard FX3 firmware boots into XU 0x09 = 0x55aa (soft-trigger
armed mode) — the bridge withholds AE updates until a soft-trigger
fires, leaving the sensor stuck at saturation defaults. Writing
XU 0x09 = 0 disables soft-trigger -> free-running streaming with
bridge AE active. Mirrors Windows behavior where profile_control_
exposure=false and the bridge runs its own AE.

Also leave XU 0x06 (ExposureExt) alone so bridge AE can drive it.
seeker software AE will overwrite later via the patched
_set_leopard_exposure_ext if config enables it.
"""
import sys, time
sys.path.insert(0, "/home/asaftg/seeker-bench")
from eo.leopard_linux import LeopardLinux, _SIZES
_SIZES[0x09] = 2  # soft-trigger / streaming-mode toggle

DEV = "/dev/video0"
RETRIES = 5

for i in range(RETRIES):
    try:
        c = LeopardLinux(DEV)
        try:
            before = int.from_bytes(c._xu_read(0x09), "little")
            c._xu_write(0x09, (0).to_bytes(2, "little"))
            after = int.from_bytes(c._xu_read(0x09), "little")
            print("[xu_init] %s soft-trigger 0x%04x -> 0x%04x  (free-running streaming)"
                  % (DEV, before, after))
        finally:
            c.close()
        sys.exit(0)
    except Exception as e:
        print("[xu_init] attempt %d/%d failed: %s" % (i+1, RETRIES, e))
        time.sleep(1)
sys.exit(1)

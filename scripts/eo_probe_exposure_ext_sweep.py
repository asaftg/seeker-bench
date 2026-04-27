"""Sweep LPCamera.ExposureExt and capture via PyAV after each — does the
sensor actually respond, or is the bridge AE overriding our writes?

Procedure for each test value E:
  1. Spawn 32-bit helper to set ExposureExt = E and exit
  2. Wait 0.5 s for FX3 to settle
  3. Open PyAV capture, grab ~20 frames, compute mean of last 5
  4. Close PyAV
  5. Print mean per E

If means differ across E, ExposureExt controls real sensor exposure and we
have manual control. If means stay at 240 (the AE-converged saturated
value we saw earlier), AE is overriding ExposureExt and we still need
the AE-off command.
"""
from __future__ import annotations
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.imx568_capture import IMX568Capture  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
HELPER = REPO / "eo" / "leopard_sdk_helper.py"
PY32 = REPO / "tools" / "python311-x86" / "python.exe"


def set_exposure_ext(value: int) -> dict:
    cmd = [str(PY32), str(HELPER),
           "--exposure-ext", str(value), "--ae", "off", "--json"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"raw": out.stdout, "err": out.stderr}


def capture_mean(n_grab: int = 20, n_avg: int = 5) -> tuple[float, float, float]:
    """Open PyAV, grab n_grab frames, return (min, mean, max) of last n_avg."""
    cap = IMX568Capture(device_index="auto")
    cap.start()
    means = []
    mins = []
    maxs = []
    for _ in range(n_grab):
        f = cap.grab()
        if f is None:
            continue
        y = f[..., 0] if f.ndim == 3 else f
        means.append(float(y.mean()))
        mins.append(int(y.min()))
        maxs.append(int(y.max()))
    cap.stop()
    if not means:
        return (-1.0, -1.0, -1.0)
    last = means[-n_avg:]
    return (float(np.min(mins[-n_avg:])),
            float(np.mean(last)),
            float(np.max(maxs[-n_avg:])))


def main() -> int:
    # Span ~50x range. IMX568 line time is ~10 us at full-resolution; so
    # 100 lines ~= 1 ms, 5000 lines ~= 50 ms. The exact unit of ExposureExt
    # is unknown but a 50x sweep should show *something* if it controls
    # real exposure.
    # Wider range — if mean stays ~100 across this whole span, AE is
    # overriding and we still need the AE-off vendor command.
    test_values = [10, 50, 200, 1000, 5000, 20000, 50000]

    print(f"{'value':>8s}  {'before':>20s}  {'after':>20s}    "
          f"{'min':>4s} {'mean':>6s} {'max':>4s}")
    for v in test_values:
        rs = set_exposure_ext(v)
        before = rs.get("before", {}).get("ExposureExt", "?")
        after = rs.get("after", {}).get("ExposureExt", "?")
        time.sleep(0.5)
        mn, mean, mx = capture_mean()
        print(f"{v:>8d}  {str(before):>20s}  {str(after):>20s}    "
              f"{int(mn):>4d} {mean:>6.1f} {int(mx):>4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

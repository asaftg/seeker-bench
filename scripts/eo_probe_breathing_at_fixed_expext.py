"""Set ExposureExt to a fixed value, then measure mean every 0.5s for 30s.

If mean drifts (AE breathing) → AE is still active despite ExposureExt write.
If mean is rock-stable → exposure is locked, just at a setpoint we don't
control via ExposureExt alone.
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


def main() -> int:
    fixed_value = 1000
    print(f"setting ExposureExt = {fixed_value} ...")
    subprocess.run(
        [str(PY32), str(HELPER),
         "--exposure-ext", str(fixed_value), "--ae", "off", "--json"],
        capture_output=True, text=True, timeout=30,
    )
    time.sleep(0.5)

    print("opening PyAV and recording 30s ...")
    cap = IMX568Capture(device_index="auto")
    cap.start()
    t0 = time.time()
    samples = []
    last = -1e9
    while time.time() - t0 < 30.0:
        f = cap.grab()
        if f is None:
            continue
        now = time.time()
        if now - last >= 0.5:
            y = f[..., 0] if f.ndim == 3 else f
            samples.append((now - t0, float(y.mean()), int(y.min()), int(y.max())))
            last = now
    cap.stop()

    means = np.array([s[1] for s in samples])
    print(f"\n  t       mean    min   max")
    for t, m, mn, mx in samples:
        print(f"  {t:5.1f}s   {m:5.1f}   {mn:3d}   {mx:3d}")
    print(f"\nspan: max-min = {means.max() - means.min():.2f}  "
          f"(mean = {means.mean():.2f}, std = {means.std():.2f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

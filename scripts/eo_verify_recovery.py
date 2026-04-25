"""Verify the new _to_luma recovery against the captured raw bridge frame.

Reads scripts/eo_snapshots/diagnostic/01_raw_bridge.png (which the
operator already captured) and runs it through the live eo_processor
helpers, saving the result so the developer can confirm the visible
fix without asking the operator to run anything.
"""
from __future__ import annotations
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo.eo_processor import (  # noqa: E402
    _to_luma,
    _looks_like_yuy2_zero_chroma,
    _recover_y_from_yuy2_bgr,
    passthrough,
)


def main() -> int:
    diag = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    raw = cv2.imread(str(diag / "01_raw_bridge.png"), cv2.IMREAD_COLOR)
    if raw is None:
        print("missing raw bridge dump")
        return 2

    flagged = _looks_like_yuy2_zero_chroma(raw)
    print(f"_looks_like_yuy2_zero_chroma: {flagged}")

    y = _to_luma(raw)
    print(f"_to_luma stats: min={y.min()} max={y.max()} "
          f"mean={y.mean():.1f} std={y.std():.1f}")

    # Save the new passthrough output for visual inspection.
    out_pt = passthrough(raw)
    cv2.imwrite(str(diag / "20_new_passthrough.png"), out_pt)

    # Save the bare _to_luma output (single-channel) too.
    cv2.imwrite(str(diag / "21_new_luma.png"),
                cv2.cvtColor(y, cv2.COLOR_GRAY2BGR))
    print("wrote 20_new_passthrough.png and 21_new_luma.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())

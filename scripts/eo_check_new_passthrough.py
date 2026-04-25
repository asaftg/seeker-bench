"""Run the new passthrough on the saved diagnostic raw frame."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eo.eo_processor import passthrough  # noqa


def main():
    diag = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    raw = cv2.imread(str(diag / "01_raw_bridge.png"), cv2.IMREAD_COLOR)
    if raw is None:
        print("missing raw bridge dump")
        return 2
    out = passthrough(raw)
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    print(f"passthrough min={g.min()} max={g.max()} "
          f"mean={g.mean():.1f} std={g.std():.1f}")
    cv2.imwrite(str(diag / "30_new_passthrough.png"), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

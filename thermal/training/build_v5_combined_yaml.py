"""Build a YOLO data.yaml that combines antiuav_visible + thermal_drone.

Both are single-class drone datasets. We list both image directories
under the same `train` and `val` keys (ultralytics accepts either a
folder or a list of folders).

The combined set roughly doubles the per-batch distribution coverage:
  - antiuav_visible: ~32k visible-light DJI Phantom against sky
  - thermal_drone: ~30k thermal-IR DJI Phantom against various backgrounds

Output: datasets/seeker_eo_v5_drone.yaml
"""
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "datasets" / "seeker_eo_v5_drone.yaml"

CONTENT = f"""# Auto-generated: combined visible (antiuav) + thermal (thermal_drone)
# drone-only training data. 1 class: 0 = drone.
path: {(REPO / 'datasets').as_posix()}
train:
  - antiuav_visible/images/train
  - thermal_drone/images/train
val:
  - antiuav_visible/images/val
  - thermal_drone/images/val
names:
  0: drone
"""

OUT.write_text(CONTENT, encoding="utf-8")
print(f"wrote {OUT}")
print(CONTENT)

"""
Train YOLOv8n on the Anti-UAV thermal drone dataset.

Reads from datasets/thermal_drone/data.yaml (produced by prepare_antiuav_drone.py).
Promotes runs/train/seeker_thermal_drone/weights/best.pt → models/seeker_thermal.pt
so the drone classifier picks it up automatically.

Run: python scripts/train_thermal_drone.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "thermal_drone" / "data.yaml"
MODELS = ROOT / "models"
MODELS.mkdir(parents=True, exist_ok=True)
OUT = MODELS / "seeker_thermal.pt"


def main():
    if not DATA.exists():
        raise SystemExit(f"[train_drone] missing {DATA} — run prepare_antiuav_drone.py first")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train_drone] device={device}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train_drone] gpu={torch.cuda.get_device_name(0)}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="seeker_thermal_drone",
        exist_ok=True,
        workers=4,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"[train_drone] best.pt missing at {best}")
    shutil.copy2(best, OUT)
    print(f"[train_drone] promoted: {best} -> {OUT}")

    metrics = model.val(data=str(DATA), device=device, verbose=False)
    print("[train_drone] FINAL  "
          f"mAP50={metrics.box.map50:.4f}  "
          f"mAP50-95={metrics.box.map:.4f}  "
          f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}")


if __name__ == "__main__":
    main()

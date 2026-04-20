"""
Train YOLOv8n on the HIT-UAV thermal person+vehicle dataset.

Run: python scripts/train_thermal_hv.py
"""
from __future__ import annotations
from pathlib import Path
import shutil
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "seeker_hv" / "data.yaml"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUT = MODELS_DIR / "seeker_thermal_hv.pt"


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[train_hv] device={device}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[train_hv] gpu={torch.cuda.get_device_name(0)}")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=str(DATA),
        epochs=50,
        imgsz=640,
        batch=16,
        patience=10,
        device=device,
        project=str(ROOT / "runs" / "train"),
        name="seeker_thermal_hv",
        exist_ok=True,
        workers=4,
        verbose=True,
    )

    run_dir = Path(results.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"[train_hv] best.pt not found at {best}")
    shutil.copy2(best, OUT)
    print(f"[train_hv] promoted: {best} -> {OUT}")

    # Final val metrics
    metrics = model.val(data=str(DATA), device=device, verbose=False)
    print(
        "[train_hv] FINAL  "
        f"mAP50={metrics.box.map50:.4f}  "
        f"mAP50-95={metrics.box.map:.4f}  "
        f"P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}"
    )


if __name__ == "__main__":
    main()

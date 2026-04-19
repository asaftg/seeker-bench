"""
Fine-tune YOLOv8n on a Seeker thermal dataset.

Usage:

    python -m thermal.training.train \\
        --data datasets/hand_v1/data.yaml \\
        --epochs 50 --imgsz 640

The resulting best weights land at:
    runs/train/<run-name>/weights/best.pt

Then run `python -m thermal.training.promote_model <path-to-best.pt>`
to copy it into `models/seeker_thermal.pt`, which the runtime
classifier prefers over the stock yolov8n.pt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to data.yaml")
    p.add_argument("--weights", default="models/yolov8n.pt", help="Starting weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="0", help="cuda device id, or 'cpu'")
    p.add_argument("--name", default="seeker_v1", help="Run name under runs/train/")
    args = p.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}", file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        print(f"ERROR: ultralytics not installed ({e}).", file=sys.stderr)
        print("       pip install ultralytics", file=sys.stderr)
        return 3

    model = YOLO(args.weights)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        project="runs/train",
    )
    best = Path("runs/train") / args.name / "weights" / "best.pt"
    print(f"Training complete. Best weights: {best}")
    print(f"Promote with: python -m thermal.training.promote_model {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Thermal v2 — drone-only thermal detector trained on combined
antiuav_thermal + thermal_drone.

Why this exists (2026-05-06):
    seeker_eo_v4_drone_only (a.k.a. our thermal-trained v4) hits 100%
    recall on airborne1 thermal frames but only ~72% precision because
    its training set (thermal_drone) contained drones almost
    exclusively against clean sky. Run on a real outdoor scene with
    palm trees, lampposts, and buildings (airborne1), the model fires
    on every warm-vegetation blob.

    antiuav_thermal (extracted from Anti-UAV-RGBT infrared.mp4 files)
    has the same drones but in much more varied backgrounds — exactly
    the hard negatives needed.

Output: models/seeker_thermal_v2_drone.pt
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_TRAIN = ROOT / "runs" / "train"
MODELS = ROOT / "models"


def say(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    say("========= THERMAL v2 COMBINED TRAIN START =========")
    data_yaml = ROOT / "datasets" / "seeker_thermal_v2_drone.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    base_pt = MODELS / "seeker_eo_v4_drone_only.pt"
    if not base_pt.exists():
        say(f"WARN: {base_pt} missing — using yolov8n.pt")
        base_pt = MODELS / "yolov8n.pt"
        if not base_pt.exists():
            base_pt = Path("yolov8n.pt")

    out_name = "seeker_thermal_v2_combined"
    target_pt = MODELS / "seeker_thermal_v2_drone.pt"

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        say(f"ERROR: ultralytics import failed: {e}")
        return 3

    say(f"cuda={torch.cuda.is_available()}  data={data_yaml}")
    say(f"base={base_pt}  -> target={target_pt.name}")

    try:
        model = YOLO(str(base_pt))
        model.train(
            data=str(data_yaml),
            epochs=5,
            imgsz=640,
            batch=16,
            device=0,
            project=str(RUNS_TRAIN),
            name=out_name,
            exist_ok=True,
            verbose=True,
            patience=2,
            workers=4,
            amp=True,
            cache=False,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=1,
            hsv_h=0.0,
            hsv_s=0.2,
            hsv_v=0.6,
            scale=0.7,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.7,
            close_mosaic=2,
            erasing=0.4,
            degrees=5.0,
            val=True,
            plots=True,
        )
    except Exception as e:
        say(f"FULL TRAIN FAILED: {e}")
        traceback.print_exc()
        return 4

    best = RUNS_TRAIN / out_name / "weights" / "best.pt"
    if not best.exists():
        say(f"ERROR: best.pt not found at {best}")
        return 5

    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_pt)
    say(f"PROMOTED: {best} -> {target_pt}")
    say(f"========= THERMAL v2 COMBINED TRAIN DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

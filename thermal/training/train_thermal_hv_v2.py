"""Thermal person+vehicle detector v2 — trained on FLIR-ADAS v2 thermal.

Why this exists (2026-05-07):
    seeker_thermal_hv.pt (currently in production) misses obvious
    pedestrians and vehicles in airborne scenes — verified on the
    "night tracking a bit flickery" recording where two clearly
    bright pedestrians and a parked car produce 0 detections.

    Root cause: the existing model was either trained on a tiny
    seeker-curated split or uses stock COCO weights with a remap.
    Either way, vehicles and people in the FLIR Boson distribution
    of our recordings are out-of-distribution.

    FLIR-ADAS v2 thermal has 10,742 train + 1,144 val images of
    pedestrians, cars, trucks, buses captured by a real automotive
    thermal sensor — much closer to our FLIR Boson distribution than
    anything we've trained on. 127k positive objects.

Output: models/seeker_thermal_hv_v2.pt
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
    say("========= THERMAL HV v2 TRAIN START =========")
    data_yaml = ROOT / "datasets" / "flir_adas_hv_thermal" / "data.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    # Cold start from yolov8n.pt — the existing thermal_hv.pt may be
    # COCO with a remap, no benefit from carrying its weights.
    base_pt = MODELS / "yolov8n.pt"
    if not base_pt.exists():
        base_pt = Path("yolov8n.pt")

    out_name = "seeker_thermal_hv_v2"
    target_pt = MODELS / "seeker_thermal_hv_v2.pt"

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
            epochs=8,
            imgsz=640,
            batch=16,
            device=0,
            project=str(RUNS_TRAIN),
            name=out_name,
            exist_ok=True,
            verbose=True,
            patience=3,
            workers=4,
            amp=True,
            cache=False,
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            warmup_epochs=1,
            hsv_h=0.0,
            hsv_s=0.2,
            hsv_v=0.6,
            scale=0.6,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.8,
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
    say(f"========= THERMAL HV v2 TRAIN DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

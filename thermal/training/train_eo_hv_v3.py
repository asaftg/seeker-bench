"""EO h/v v3 — FLIR-ADAS RGB + LLVIP visible (unified).

Continues from seeker_eo_hv_v2.pt. Output: models/seeker_eo_hv_v3.pt.
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
    say("========= EO HV v3 (FLIR+LLVIP) START =========")
    data_yaml = ROOT / "datasets" / "seeker_eo_hv_v3.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    base_pt = MODELS / "seeker_eo_hv_v2.pt"
    if not base_pt.exists():
        say(f"WARN: {base_pt} missing — using yolov8n.pt cold start")
        base_pt = MODELS / "yolov8n.pt"
        if not base_pt.exists():
            base_pt = Path("yolov8n.pt")

    out_name = "seeker_eo_hv_v3"
    target_pt = MODELS / "seeker_eo_hv_v3.pt"

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        say(f"ERROR: ultralytics import failed: {e}")
        return 3

    say(f"cuda={torch.cuda.is_available()}  data={data_yaml}")
    say(f"base={base_pt}")

    try:
        model = YOLO(str(base_pt))
        model.train(
            data=str(data_yaml),
            epochs=10,
            imgsz=640,
            batch=16,
            device=0,
            project=str(RUNS_TRAIN),
            name=out_name,
            exist_ok=True,
            verbose=True,
            patience=4,
            workers=4,
            amp=True,
            cache=False,
            optimizer="AdamW",
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=1,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.5,
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
        return 5
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_pt)
    say(f"PROMOTED: {best} -> {target_pt}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

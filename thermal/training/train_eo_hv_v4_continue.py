"""Continue v4 training after the unexplained 2026-05-08 mid-epoch-7 kill.

Picks up from runs/train/seeker_eo_hv_v4/weights/best.pt (epoch 6
weights with val mAP50=0.767), continues for 8 more epochs at the
slightly-decayed LR. Output: same target models/seeker_eo_hv_v4.pt.
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
    say("========= EO HV v4 CONTINUE START =========")
    data_yaml = ROOT / "datasets" / "flir_adas_hv_rgb" / "data.yaml"

    base_pt = RUNS_TRAIN / "seeker_eo_hv_v4" / "weights" / "best.pt"
    if not base_pt.exists():
        say(f"ERROR: {base_pt} missing")
        return 2

    out_name = "seeker_eo_hv_v4_cont"
    target_pt = MODELS / "seeker_eo_hv_v4.pt"

    try:
        from ultralytics import YOLO
    except Exception as e:
        say(f"ERROR: {e}")
        return 3

    say(f"base={base_pt}  -> target={target_pt.name}")
    try:
        model = YOLO(str(base_pt))
        model.train(
            data=str(data_yaml),
            epochs=8,
            imgsz=1024,
            batch=8,
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
            lr0=0.0003,
            lrf=0.01,
            warmup_epochs=0,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.5,
            scale=0.9,
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
        say(f"FAILED: {e}")
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

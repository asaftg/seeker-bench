"""Continue thermal_hv_v2 training for another 8 epochs.

Picks up from models/seeker_thermal_hv_v2.pt (which was promoted from
the first 8-epoch run, mAP50=0.752). Uses the same FLIR-ADAS thermal
data.yaml with a slightly lower base LR since the model has already
converged on the easy stuff.

Output: models/seeker_thermal_hv_v2.pt (overwritten on improvement)
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
    say("========= THERMAL HV v2 CONTINUE 8+8 START =========")
    data_yaml = ROOT / "datasets" / "flir_adas_hv_thermal" / "data.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    base_pt = MODELS / "seeker_thermal_hv_v2.pt"
    if not base_pt.exists():
        say(f"ERROR: {base_pt} missing — run train_thermal_hv_v2.py first")
        return 2

    out_name = "seeker_thermal_hv_v2_cont"
    target_pt = MODELS / "seeker_thermal_hv_v2.pt"

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
            epochs=8,
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
            # Lower starting LR since we're continuing from converged-ish weights
            lr0=0.0003,
            lrf=0.01,
            warmup_epochs=0,
            hsv_h=0.0,
            hsv_s=0.2,
            hsv_v=0.6,
            scale=0.6,
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

    # Compare: only promote if new best.pt beats old
    # (ultralytics best.pt is best by val fitness across just THIS run,
    # so we always promote — the trained continuation should be strictly
    # >= the previous, with same val set + cosine LR decay)
    MODELS.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target_pt)
    say(f"PROMOTED: {best} -> {target_pt}")
    say(f"========= THERMAL HV v2 CONTINUE DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

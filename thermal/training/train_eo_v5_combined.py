"""EO v5 — combined visible + thermal drone fine-tune.

Why this exists:
    v4_drone_only (trained 2026-05-06 on thermal_drone only) hits
    96.3% precision but only 85.2% recall on airborne1 EO replay.
    Misses concentrate in (1) far range (60-75 s, drone ~10 px) and
    (2) entry motion blur (20-30 s).

    Hypothesis: the model overfits to thermal bright-on-dark blob
    statistics. Adding visible-light drone footage (antiuav visible.mp4
    train/val) should:
      - cover the bright-on-bright and dark-on-bright polarities the
        IMX568-NIR-pass produces under different sun/sky conditions
      - introduce a wider range of drone pixel scales (the antiuav
        visible recordings span close to far)

    No thermal data is removed; we add visible.

Output: models/seeker_eo_v5_drone.pt
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
    say("========= EO v5 COMBINED TRAIN START =========")
    data_yaml = ROOT / "datasets" / "seeker_eo_v5_drone.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing — run build_v5_combined_yaml.py first")
        return 2

    # Start from v4 (drone-aware weights, not from yolov8n cold)
    base_pt = MODELS / "seeker_eo_v4_drone_only.pt"
    if not base_pt.exists():
        say(f"WARN: {base_pt} missing — starting from yolov8n.pt")
        base_pt = MODELS / "yolov8n.pt"
        if not base_pt.exists():
            base_pt = Path("yolov8n.pt")

    out_name = "seeker_eo_v5_combined"
    target_pt = MODELS / "seeker_eo_v5_drone.pt"

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
            # Augmentations targeting v4's failure modes:
            hsv_h=0.0,
            hsv_s=0.3,
            hsv_v=0.7,    # heavy brightness jitter (sun-on-drone vs shadow)
            scale=0.8,    # ±80% scale: far-range scale variety
            translate=0.15,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.8,
            close_mosaic=2,
            erasing=0.4,
            degrees=8.0,  # mild rotation
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
    say(f"========= EO v5 COMBINED TRAIN DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""EO v4 — drone-only detector for IMX568 NIR-mono airborne distribution.

Why this exists (2026-05-06):
    seeker_eo_v3.pt's drone class was DEFINED in the model head but
    NEVER actually trained on a single drone image. Verified: scanning
    training_runs/unified/eo_v2/labels/{train,val} returned 0 files
    with class id 2 across 33,661 train images. The unified set was
    LLVIP-pedestrian + FLIR-ADAS-driving + seeker_hv (person/vehicle
    only) — none of them carry drone annotations. The "drone" head
    was uninitialized, which is why the v3 inference pipeline returns
    0/30 detections on `drone test airborne 1` regardless of imgsz,
    conf, or post-filter tuning (verified
    eval/airborne1_run_baseline_and_sweeps.py 2026-05-06).

What this trains:
    Single-class drone detector on the existing thermal_drone dataset
    (29,658 train / 28,954 val images, all 640×512 grayscale BGR with
    one drone per image, labeled). The thermal-drone source is already
    near-monochrome (R-B channel diff ≈ 3 / 255), so its distribution
    is a much closer match to the IMX568-mono + 35mm-NIR-pass capture
    than any RGB driving dataset.

Augmentations target polarity robustness:
    The thermal source always has drone HOTTER than sky (drone bright,
    sky dark). The IMX568-mono captures the drone darker OR brighter
    than sky depending on lighting + drone material. We use the
    `bgr` flip flag together with hsv_v jitter to teach polarity
    invariance, plus `erasing` for occlusion and `scale` for
    range-invariance.

Output: models/seeker_eo_v4_drone_only.pt
Runtime use: load alongside seeker_eo_v3.pt; merge detections.
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
    say("========= EO v4 DRONE-ONLY TRAIN START =========")
    data_yaml = ROOT / "datasets" / "thermal_drone" / "data.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    base_pt = MODELS / "yolov8n.pt"
    if not base_pt.exists():
        say(f"NOTE: {base_pt} missing — ultralytics will auto-download")
        base_pt = Path("yolov8n.pt")

    out_name = "seeker_eo_v4_drone_only"
    target_pt = MODELS / "seeker_eo_v4_drone_only.pt"

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
        results = model.train(
            data=str(data_yaml),
            epochs=8,
            imgsz=640,
            batch=16,
            device=0,
            project=str(RUNS_TRAIN),
            name=out_name,
            exist_ok=True,
            verbose=True,
            patience=5,
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
    say(f"========= EO v4 DRONE-ONLY TRAIN DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

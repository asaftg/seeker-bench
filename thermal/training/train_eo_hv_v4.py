"""EO h/v v4 — tile-aware close-foreground retrain.

Why this exists (2026-05-08)
----------------------------
v2 was trained at imgsz=640 on full FLIR-ADAS RGB driving frames.
The new tiled inference path feeds it 1232×1032 tiles letterboxed to
imgsz=832, and we want native resolution into the model so 500m
vehicles get ~31 px on the detector instead of ~16. There's a
train/test mismatch when v2 sees that input distribution.

Recipe:
  - imgsz=1024 (was 640) — matches the per-tile budget more closely
    and trains the model to handle the slightly larger receptive
    field that tiles present.
  - scale=0.9 (was 0.6) — aggressive random scale lets a single
    image fill the frame during training. Specifically targets the
    "missed prominent foreground vehicle" pattern v2 had on
    working_trial t=44s.
  - mosaic=0.8, close_mosaic=3 — last 3 epochs disable mosaic for
    clean settle.
  - Drop LLVIP visible from training (per the LLVIP-IR-as-thermal
    lesson learned 2026-05-08 — that regressed thermal v3; we expect
    LLVIP visible to similarly bias the model toward urban-pedestrian
    framing rather than the open-field objects we care about).
  - 15 epochs, AdamW lr0=0.0005, patience=5.

Output: models/seeker_eo_hv_v4.pt (kept alongside v2 as rollback).
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
    say("========= EO HV v4 (FLIR-ADAS RGB only, tile-aware) START =========")
    data_yaml = ROOT / "datasets" / "flir_adas_hv_rgb" / "data.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing")
        return 2

    # Start from v2 — it already knows person/vehicle on FLIR-ADAS;
    # we're tuning it for higher resolution + close-foreground.
    base_pt = MODELS / "seeker_eo_hv_v2.pt"
    if not base_pt.exists():
        say(f"WARN: {base_pt} missing, falling back to yolov8n.pt")
        base_pt = MODELS / "yolov8n.pt"
        if not base_pt.exists():
            base_pt = Path("yolov8n.pt")

    out_name = "seeker_eo_hv_v4"
    target_pt = MODELS / "seeker_eo_hv_v4.pt"

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
            epochs=15,
            imgsz=1024,             # was 640 in v2 — match tile budget
            batch=8,                # smaller batch fits imgsz=1024 in 16 GB
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
            lr0=0.0005,
            lrf=0.01,
            warmup_epochs=1,
            hsv_h=0.015,
            hsv_s=0.4,
            hsv_v=0.5,
            scale=0.9,              # AGGRESSIVE — was 0.6
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,
            mosaic=0.8,
            close_mosaic=3,
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
    say("========= EO HV v4 DONE =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

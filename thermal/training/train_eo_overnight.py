"""EO-only overnight fine-tune for Seeker-01.

Why this exists separately from train_overnight.py:
- The prior unified overnight run was killed during the THERMAL smoke and
  never reached EO training. Thermal already has a deployed model
  (models/seeker_thermal_hv.pt). Repeating thermal first risks the same
  fate again, and the user's current request is specifically about
  EO vehicles + humans (low-light testing).

Strategy:
- yolov8s.pt as base (medium accuracy / size; fits A4000 16GB easily).
- Trains on the existing seeker_eo_v2.yaml unified dataset:
    LLVIP visible (~12k low-light city + pedestrians)
  + FLIR ADAS RGB (~9.6k driving, vehicles + pedestrians)
  + seeker_hv (~12k seeker's own captures).
- Augmentation knobs tuned for low-light EO (more HSV V variation,
  hue-stable, no flipud, modest mosaic).
- 60 epochs, patience 15, batch 16, AMP on, cache disk.
- Tee stdout to training_runs/eo_overnight.log via shell redirection.
- Promote best.pt -> models/seeker_eo_v2.pt only if it actually finishes.

Run from project root:
  python thermal/training/train_eo_overnight.py
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_TRAIN = ROOT / "runs" / "train"
REPORT = ROOT / "training_runs" / "report.md"
MODELS = ROOT / "models"
SAMPLES_E = ROOT / "training_runs" / "samples_eo"


def say(msg: str) -> None:
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def append_report(text: str) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def main() -> int:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

    say("========= EO OVERNIGHT START =========")
    data_yaml = ROOT / "datasets" / "seeker_eo_v2.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing. Run build_unified.py first.")
        return 2

    out_name = "seeker_eo_v2"
    target_pt = MODELS / "seeker_eo_v2.pt"

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        say(f"ERROR: ultralytics import failed: {e}")
        return 3

    say(f"cuda={torch.cuda.is_available()}  data={data_yaml.name}")

    # Smoke first — proves label scan + one full epoch works on this dataset
    # under the chosen settings. If it crashes, we abort instead of burning
    # hours on a doomed config.
    try:
        say("SMOKE: 2 epochs @ batch=16 cache=disk")
        YOLO("yolov8s.pt").train(
            data=str(data_yaml), epochs=2, imgsz=640, batch=16,
            device=0, project=str(RUNS_TRAIN), name=out_name + "_smoke",
            exist_ok=True, verbose=True, patience=2,
            workers=8, amp=True, cache="disk",
        )
        say("SMOKE passed")
    except Exception as e:
        say(f"SMOKE FAILED: {e}")
        traceback.print_exc()
        append_report(f"\n## {out_name}\nSMOKE FAILED: {e}\n")
        return 4

    # Full run with low-light-friendly augmentation. Notes:
    #   hsv_v=0.5 (default 0.4): more brightness jitter — exposure varies
    #     wildly outside, the user is hand-tuning gain/exposure during use.
    #   hsv_s=0.6 (default 0.7): slightly less saturation thrash; LLVIP
    #     night scenes are already low-saturation, more jitter just blurs
    #     class boundaries.
    #   flipud=0.0: people/vehicles never appear upside-down in surveillance.
    #   mosaic=1.0: keep, helps with multi-target scenes.
    #   close_mosaic=10: standard YOLOv8 default.
    last_err = None
    chosen = None
    # OOM ladder
    for weights, batch in [("yolov8s.pt", 16), ("yolov8s.pt", 8),
                           ("yolov8n.pt", 16), ("yolov8n.pt", 8)]:
        try:
            say(f"FULL: weights={weights} batch={batch} epochs=60")
            model = YOLO(weights)
            model.train(
                data=str(data_yaml),
                epochs=60,
                imgsz=640,
                batch=batch,
                device=0,
                project=str(RUNS_TRAIN),
                name=out_name,
                exist_ok=True,
                verbose=True,
                patience=15,
                workers=8,
                amp=True,
                cache="disk",
                hsv_h=0.015,
                hsv_s=0.6,
                hsv_v=0.5,
                flipud=0.0,
                fliplr=0.5,
                mosaic=1.0,
                close_mosaic=10,
                erasing=0.4,
            )
            chosen = (weights, batch, model)
            break
        except torch.cuda.OutOfMemoryError as e:
            say(f"OOM at {weights}/{batch}: {e}")
            try: torch.cuda.empty_cache()
            except Exception: pass
            last_err = e
            continue
        except Exception as e:
            say(f"train error at {weights}/{batch}: {e}")
            traceback.print_exc()
            last_err = e
            continue

    if chosen is None:
        append_report(f"\n## {out_name}\nFAILED: {last_err}\n")
        return 5

    weights, batch, _ = chosen
    best_pt = RUNS_TRAIN / out_name / "weights" / "best.pt"
    say(f"best.pt = {best_pt}")

    # Validate
    try:
        val_model = YOLO(str(best_pt))
        metrics = val_model.val(
            data=str(data_yaml), imgsz=640, device=0,
            project=str(RUNS_TRAIN), name=out_name + "_val",
            exist_ok=True, verbose=False,
        )
        mAP50 = float(metrics.box.map50)
        mAP5095 = float(metrics.box.map)
        per_class = {}
        try:
            names = metrics.names if hasattr(metrics, "names") else val_model.names
            maps = metrics.box.maps
            for i, m in enumerate(maps):
                per_class[names[i] if isinstance(names, dict) else names[i]] = float(m)
        except Exception:
            pass
    except Exception as e:
        say(f"val failed: {e}")
        traceback.print_exc()
        append_report(
            f"\n## {out_name}\nTRAINED but VAL FAILED: {e}\nbest.pt = {best_pt}\n"
        )
        return 6

    # Promote — only if no existing target_pt (don't clobber a hand-picked weight).
    promoted = False
    if best_pt.exists():
        if target_pt.exists():
            say(f"NOT overwriting existing {target_pt}; new best.pt at {best_pt}")
        else:
            shutil.copy2(best_pt, target_pt)
            promoted = True
            say(f"PROMOTED best.pt -> {target_pt}")

    append_report(f"""
## {out_name}  ({dt.datetime.now().isoformat(timespec='seconds')})
- weights base: `{weights}`  batch: `{batch}`  epochs: `60`
- data: `{data_yaml.name}`
- best.pt: `{best_pt}`
- **mAP50**: `{mAP50:.4f}`
- **mAP50-95**: `{mAP5095:.4f}`
- promoted to `{target_pt}`: `{promoted}`
- per-class mAP50-95: {per_class}
""")

    say("========= EO OVERNIGHT END =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

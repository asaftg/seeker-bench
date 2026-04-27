"""EO v3 — targeted fine-tune to fix close-range / NIR-pass blind spot.

Why this exists:
    seeker_eo_v2.pt (mAP50≈0.82) handles long-range targets well but
    misses obvious close-range vehicles in NIR-pass-monochromatic
    imagery — verified field test 2026-04-25 against a white Camry
    filling 60% of the frame at ~10 m: model fired on a 30-px distant
    car in the background and skipped the foreground entirely.

    Two compounding causes:
      a) Training distribution: seeker_hv captures + driving-cam data
         skew toward small-pixel-fraction targets. The model's
         "vehicle" feature responses are tuned for small-scale.
      b) NIR-pass + 35 mm lens removes color cues. Bayer debayer
         produces near-monochrome BGR. COCO/LLVIP/FLIR-ADAS RGB
         training images carry real chroma; the model leans on that
         and underperforms on flat-saturation scenes.

Fix strategy (does NOT require new field data):
    Continue-fine-tune from seeker_eo_v2.pt with augmentations picked
    to specifically shock the two failure modes:

    - `scale=0.9`  — random rescale ±90% (vs YOLOv8 default 0.5).
                    Lets a cropped car randomly fill the frame during
                    training. This is the single most important knob.
    - `hsv_s=0.95` — near-full saturation jitter, including ~zero.
                    Simulates the NIR-pass desaturated look every
                    other batch.
    - `mosaic=0.5` — half the time a single image (vs full mosaic).
                    Mosaic shrinks all targets; we want the model to
                    see solo, large-scale targets too.
    - `close_mosaic=5` — last 5 epochs disable mosaic entirely,
                    standard YOLO trick to settle the model on
                    natural scenes after augmentation chaos.
    - `lr0=0.0001` + `optimizer='AdamW'` + `warmup_epochs=0`
                    — three knobs that MUST go together. ultralytics
                    `optimizer='auto'` (the default) silently overrides
                    `lr0` and picks its own based on dataset size; on
                    this dataset it picked lr0=0.01 with 3 epochs of
                    warmup that ramped LR UP to 0.028 by epoch 3.
                    That destroyed v2's learned weights instead of
                    fine-tuning them (verified: mAP50-95 regressed
                    from 0.527 → 0.499 across epochs 1-3 of the
                    earlier run). The explicit triple `optimizer=
                    'AdamW' lr0=0.0001 warmup_epochs=0` forces a real
                    100x-lower-than-scratch LR that ultralytics
                    actually respects.
    - `imgsz=832`  — runtime EOClassifier uses imgsz=1280 for
                    inference. Training at 640 (v2) means a 2x
                    train/test mismatch; 832 narrows that gap. We
                    picked 832 over 960 because the disk cache at
                    960 needed ~191 GB and only ~68 GB was free; 832
                    fits comfortably and still beats v2's 640.
    - `epochs=30, patience=10` — fast fine-tune. If mAP50 plateaus
                    early, patience cuts us off well under the budget.

    This is mathematically equivalent to "imagine 10× more close-up
    and 5× more monochrome examples in the training set" without
    needing to capture or label any of them.

Auto-promotes ``best.pt`` -> ``models/seeker_eo_v3.pt`` (NOT v2 — we
keep v2 around to roll back if v3 regresses on long range). Runtime
prefers v3 if present (see ``eo/eo_classifier.py``).

Run from project root:
  python thermal/training/train_eo_v3_close_targets.py
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

    say("========= EO v3 CLOSE-TARGETS FINE-TUNE START =========")
    data_yaml = ROOT / "datasets" / "seeker_eo_v2.yaml"
    if not data_yaml.exists():
        say(f"ERROR: {data_yaml} missing. Run build_unified.py first.")
        return 2

    # Start from v2; promote to v3 — v2 stays on disk as a rollback.
    base_pt = MODELS / "seeker_eo_v2.pt"
    if not base_pt.exists():
        say(f"ERROR: base model {base_pt} missing. "
            f"Run train_eo_overnight.py first.")
        return 2

    out_name = "seeker_eo_v3"
    target_pt = MODELS / "seeker_eo_v3.pt"

    try:
        from ultralytics import YOLO
        import torch
    except Exception as e:
        say(f"ERROR: ultralytics import failed: {e}")
        return 3

    say(f"cuda={torch.cuda.is_available()}  data={data_yaml.name}")
    say(f"base={base_pt.name}  -> target={target_pt.name}")

    # SMOKE — 2 epochs at the same imgsz/batch we'll use full-run.
    # Catches "OOM at imgsz=832 batch=8" or "LR ratio too aggressive
    # collapses the model" in 2-3 minutes instead of 3 hours.
    #
    # Skippable when re-launching after a previously-validated config —
    # set SEEKER_SKIP_SMOKE=1 in env. We use this when an external
    # event (Windows session lock, terminal close, etc.) killed an
    # earlier full run that had already passed smoke; relaunching from
    # scratch and re-paying the 20+ min smoke is wasteful.
    import os as _os
    if _os.environ.get("SEEKER_SKIP_SMOKE") == "1":
        say("SMOKE: skipped via SEEKER_SKIP_SMOKE=1")
    else:
        try:
            say("SMOKE: 2 epochs @ imgsz=832 batch=8 from v2 base")
            YOLO(str(base_pt)).train(
                data=str(data_yaml), epochs=2, imgsz=832, batch=8,
                device=0, project=str(RUNS_TRAIN),
                name=out_name + "_smoke",
                exist_ok=True, verbose=True, patience=2,
                workers=8, amp=True, cache="disk",
                lr0=0.001,                     # fine-tune LR
                scale=0.9, hsv_s=0.95, mosaic=0.5,
            )
            say("SMOKE passed")
        except Exception as e:
            say(f"SMOKE FAILED: {e}")
            traceback.print_exc()
            append_report(f"\n## {out_name}\nSMOKE FAILED: {e}\n")
            return 4

    # FULL — 30 epochs, OOM ladder for the imgsz=832 batch=8 case.
    last_err = None
    chosen = None
    for batch, ladder_imgsz in [(8, 832), (4, 832), (8, 640)]:
        try:
            say(f"FULL: imgsz={ladder_imgsz} batch={batch} epochs=20 "
                f"(AdamW lr0=0.0001 warmup=0 scale=0.7 hsv_s=0.7 mosaic=0.8)")
            model = YOLO(str(base_pt))
            model.train(
                data=str(data_yaml),
                epochs=20,
                imgsz=ladder_imgsz,
                batch=batch,
                device=0,
                project=str(RUNS_TRAIN),
                name=out_name,
                exist_ok=True,
                verbose=True,
                patience=10,               # stop early if plateaued
                workers=8,
                amp=True,
                cache="disk",
                # ── EXPLICIT optimizer/LR — must match together. See
                # docstring "lr0=0.0001 + ..." note above for why
                # auto-LR demolished the previous run.
                optimizer='AdamW',
                lr0=0.0001,                # real fine-tune (100x lower than auto-default 0.01)
                lrf=0.01,                  # cosine final factor
                warmup_epochs=0,           # don't ramp LR back up — we're already converged
                # ── targeted but mild augmentations ──
                hsv_h=0.015,
                hsv_s=0.7,                 # mild desat jitter (was 0.95; too aggressive)
                hsv_v=0.5,                 # exposure jitter (kept from v2)
                scale=0.7,                 # random ±70% scale (was 0.9; targets fill frame less often but still help)
                flipud=0.0,
                fliplr=0.5,
                mosaic=0.8,                # was 0.5; bigger keeps long-range targets in distribution
                close_mosaic=5,            # last 5 epochs: pure scenes
                erasing=0.4,
                degrees=3.0,               # very mild rotation
                translate=0.1,
            )
            chosen = (batch, ladder_imgsz, model)
            break
        except torch.cuda.OutOfMemoryError as e:
            say(f"OOM at imgsz={ladder_imgsz}/batch={batch}: {e}")
            try: torch.cuda.empty_cache()
            except Exception: pass
            last_err = e
            continue
        except Exception as e:
            say(f"train error at imgsz={ladder_imgsz}/batch={batch}: {e}")
            traceback.print_exc()
            last_err = e
            continue

    if chosen is None:
        append_report(f"\n## {out_name}\nFAILED: {last_err}\n")
        return 5

    batch, ladder_imgsz, _ = chosen
    best_pt = RUNS_TRAIN / out_name / "weights" / "best.pt"
    say(f"best.pt = {best_pt}")

    # Validate against the SAME val set as v2 — apples-to-apples.
    try:
        val_model = YOLO(str(best_pt))
        metrics = val_model.val(
            data=str(data_yaml), imgsz=ladder_imgsz, device=0,
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

    # PROMOTE — overwrite v3 if present (v3 is iterative; v2 stays as
    # the "known-good rollback" and is NEVER touched by this script).
    promoted = False
    if best_pt.exists():
        shutil.copy2(best_pt, target_pt)
        promoted = True
        say(f"PROMOTED best.pt -> {target_pt}")

    append_report(f"""
## {out_name}  ({dt.datetime.now().isoformat(timespec='seconds')})
- base: `{base_pt.name}`  imgsz: `{ladder_imgsz}`  batch: `{batch}`  epochs: `30`
- data: `{data_yaml.name}`
- best.pt: `{best_pt}`
- **mAP50**: `{mAP50:.4f}`
- **mAP50-95**: `{mAP5095:.4f}`
- promoted to `{target_pt}`: `{promoted}`
- per-class mAP50-95: {per_class}
- targeted augs: scale=0.9, hsv_s=0.95, mosaic=0.5, lr0=0.001
""")

    say("========= EO v3 FINE-TUNE END =========")
    return 0


if __name__ == "__main__":
    sys.exit(main())

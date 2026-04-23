"""Overnight training driver for Seeker-01.

Trains two YOLOv8 models (thermal_v2, eo_v2), validates, writes a report,
copies best.pt to models/ if mAP50 beats legacy, and dumps sample PNGs.

All stdout/stderr is tee'd to training_runs/overnight.log with timestamps.
Run:
  python thermal/training/train_overnight.py
"""
from __future__ import annotations

import datetime as dt
import os
import random
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_TRAIN = ROOT / "runs" / "train"
REPORT = ROOT / "training_runs" / "report.md"
LOG = ROOT / "training_runs" / "overnight.log"
MODELS = ROOT / "models"
SAMPLES_T = ROOT / "training_runs" / "samples_thermal"
SAMPLES_E = ROOT / "training_runs" / "samples_eo"


class TeeStream:
    def __init__(self, *streams): self.streams = streams
    def write(self, s):
        ts = dt.datetime.now().strftime("%H:%M:%S ")
        for st in self.streams:
            try:
                if s and s.strip():
                    st.write(ts + s if not s.startswith(ts) else s)
                else:
                    st.write(s)
                st.flush()
            except Exception:
                pass
    def flush(self):
        for st in self.streams:
            try: st.flush()
            except Exception: pass


def setup_log():
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fh = open(LOG, "a", buffering=1, encoding="utf-8")
    sys.stdout = TeeStream(sys.__stdout__, fh)
    sys.stderr = TeeStream(sys.__stderr__, fh)


def say(msg): print(msg)


def append_report(text: str):
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(text + "\n")


def legacy_map50(weights_path: Path, data_yaml: Path) -> float:
    """Validate a legacy .pt on the new val set to get baseline mAP50.
    Returns -1.0 if anything goes wrong (so new model always wins)."""
    if not weights_path.exists():
        return -1.0
    try:
        from ultralytics import YOLO
        m = YOLO(str(weights_path))
        r = m.val(data=str(data_yaml), imgsz=640, device=0, verbose=False,
                  project=str(RUNS_TRAIN), name="legacy_baseline", exist_ok=True)
        return float(r.box.map50)
    except Exception as e:
        say(f"  [legacy baseline skipped] {weights_path.name}: {e}")
        return -1.0


def train_one(bundle_name: str, data_yaml: Path, epochs: int,
              out_name: str, legacy_pt: Path | None,
              samples_dir: Path, target_pt: Path):
    from ultralytics import YOLO
    import torch

    say(f"\n=== TRAIN {bundle_name} :: {data_yaml.name} ===")
    say(f"  epochs={epochs} imgsz=640 device=0 cuda={torch.cuda.is_available()}")

    # OOM/robustness ladder: (weights, batch)
    ladder = [("yolov8s.pt", 32), ("yolov8s.pt", 16), ("yolov8s.pt", 8), ("yolov8n.pt", 16)]
    results = None
    last_err = None
    chosen = None
    for weights, batch in ladder:
        try:
            say(f"  attempt: weights={weights} batch={batch}")
            model = YOLO(weights)
            results = model.train(
                data=str(data_yaml),
                epochs=epochs,
                imgsz=640,
                batch=batch,
                device=0,
                project=str(RUNS_TRAIN),
                name=out_name,
                exist_ok=True,
                verbose=True,
                patience=15,
                workers=4,
                amp=True,
            )
            chosen = (weights, batch, model)
            break
        except torch.cuda.OutOfMemoryError as e:
            say(f"  OOM at batch={batch}: {e}")
            last_err = e
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        except Exception as e:
            say(f"  train error at {weights}/{batch}: {e}")
            traceback.print_exc()
            last_err = e
            continue

    if chosen is None:
        append_report(f"\n## {out_name}\nFAILED: {last_err}\n")
        return

    weights, batch, model = chosen
    best_pt = RUNS_TRAIN / out_name / "weights" / "best.pt"
    say(f"  best.pt = {best_pt}")

    # Validate
    try:
        val_model = YOLO(str(best_pt))
        metrics = val_model.val(data=str(data_yaml), imgsz=640, device=0,
                                project=str(RUNS_TRAIN), name=out_name + "_val",
                                exist_ok=True, verbose=False)
        mAP50 = float(metrics.box.map50)
        mAP5095 = float(metrics.box.map)
        per_class = {}
        try:
            names = metrics.names if hasattr(metrics, "names") else val_model.names
            maps = metrics.box.maps  # per-class mAP50-95
            for i, m in enumerate(maps):
                per_class[names[i] if isinstance(names, dict) else names[i]] = float(m)
        except Exception:
            pass
    except Exception as e:
        say(f"  val failed: {e}")
        traceback.print_exc()
        append_report(f"\n## {out_name}\nTRAINED but VAL FAILED: {e}\nbest.pt = {best_pt}\n")
        return

    legacy = legacy_map50(legacy_pt, data_yaml) if legacy_pt else -1.0
    promoted = False
    if mAP50 > legacy and best_pt.exists():
        if target_pt.exists():
            say(f"  NOT overwriting existing {target_pt}")
        else:
            shutil.copy2(best_pt, target_pt)
            promoted = True
            say(f"  PROMOTED best.pt -> {target_pt}")

    # Sample inferences
    try:
        import yaml
        dd = yaml.safe_load(data_yaml.read_text())
        val_list_path = Path(dd["val"])
        if not val_list_path.is_absolute():
            val_list_path = (Path(dd["path"]) / val_list_path)
        samples_dir.mkdir(parents=True, exist_ok=True)
        imgs = [l.strip() for l in val_list_path.read_text().splitlines() if l.strip()]
        random.shuffle(imgs)
        picks = imgs[:20]
        for i, p in enumerate(picks):
            try:
                r = val_model.predict(p, imgsz=640, device=0, verbose=False,
                                      save=True, project=str(samples_dir),
                                      name="_", exist_ok=True)
            except Exception as e:
                say(f"    sample {i} failed: {e}")
    except Exception as e:
        say(f"  samples failed: {e}")

    append_report(f"""
## {out_name}
- weights base: `{weights}`  batch: `{batch}`  epochs: `{epochs}`
- data: `{data_yaml.name}`
- best.pt: `{best_pt}`
- **mAP50**: `{mAP50:.4f}`
- **mAP50-95**: `{mAP5095:.4f}`
- legacy baseline mAP50: `{legacy:.4f}` ({'BEAT' if mAP50 > legacy else 'did not beat'})
- promoted to `{target_pt}`: `{promoted}`
- per-class mAP50-95: {per_class}
""")


def main():
    setup_log()
    say(f"\n========= OVERNIGHT START {dt.datetime.now().isoformat(timespec='seconds')} =========")
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    if not REPORT.exists():
        REPORT.write_text(f"# Seeker-01 Overnight Training Report\n\nStarted {dt.datetime.now().isoformat(timespec='seconds')}\n", encoding="utf-8")
    else:
        append_report(f"\n---\n(run started {dt.datetime.now().isoformat(timespec='seconds')})")

    thermal_yaml = ROOT / "datasets" / "seeker_thermal_v2.yaml"
    eo_yaml      = ROOT / "datasets" / "seeker_eo_v2.yaml"

    # Thermal
    try:
        train_one(
            bundle_name="thermal_v2",
            data_yaml=thermal_yaml,
            epochs=80,
            out_name="seeker_thermal_v2",
            legacy_pt=MODELS / "seeker_thermal.pt",
            samples_dir=SAMPLES_T,
            target_pt=MODELS / "seeker_thermal_v2.pt",
        )
    except Exception as e:
        say(f"thermal training top-level error: {e}")
        traceback.print_exc()
        append_report(f"\n## seeker_thermal_v2\nTOP-LEVEL FAILURE: {e}\n")

    # EO
    try:
        train_one(
            bundle_name="eo_v2",
            data_yaml=eo_yaml,
            epochs=60,
            out_name="seeker_eo_v2",
            legacy_pt=None,  # no legacy EO-specific model in models/
            samples_dir=SAMPLES_E,
            target_pt=MODELS / "seeker_eo_v2.pt",
        )
    except Exception as e:
        say(f"eo training top-level error: {e}")
        traceback.print_exc()
        append_report(f"\n## seeker_eo_v2\nTOP-LEVEL FAILURE: {e}\n")

    say(f"\n========= OVERNIGHT END {dt.datetime.now().isoformat(timespec='seconds')} =========")


if __name__ == "__main__":
    main()

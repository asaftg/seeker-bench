#!/usr/bin/env python
"""
Train YOLOv8n for thermal human + vehicle detection.

Output: models/seeker_thermal_hv.pt   (copy of best weights)
        models/hv_training/            (plots, metrics, weights/)

Dataset format expected: YOLO v5/v8 layout with two classes:
    0 = person
    1 = vehicle

Recommended datasets (pick one):
  1. FLIR ADAS v2 — best option, ~9 k annotated thermal images
       Roboflow: https://universe.roboflow.com/flir-adggx/flir-camera-objects
       Sign up for a free Roboflow account → copy your API key → set env var
       ROBOFLOW_API_KEY=<your_key> then re-run with --download-flir
  2. Custom/local dataset — pass --dataset path/to/yolo_root
       where yolo_root/data.yaml exists and paths are correct.
  3. Public thermal dataset auto-downloaded by this script (see --auto)

Usage examples:
    # Auto-download a compact public thermal h/v dataset (no API key needed):
    python scripts/train_yolo_hv.py --auto

    # Use already-downloaded FLIR ADAS (set ROBOFLOW_API_KEY first):
    python scripts/train_yolo_hv.py --download-flir

    # Use a local dataset you already have:
    python scripts/train_yolo_hv.py --dataset datasets/my_thermal_hv

    # Quick over-ride (all flags together):
    python scripts/train_yolo_hv.py --dataset datasets/flir_hv --epochs 30

After training completes, best weights are copied to
models/seeker_thermal_hv.pt automatically.  Set
  thermal:
    classifier_hv_enabled: true
in config/app_config.yaml and restart to use them.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS_DIR  = ROOT / "models"
OUTPUT_DIR  = MODELS_DIR / "hv_training"
TARGET_MODEL = MODELS_DIR / "seeker_thermal_hv.pt"

# ---------------------------------------------------------------------------
# Dataset downloaders
# ---------------------------------------------------------------------------

def _download_auto(dest: Path) -> Path:
    """Download a compact public thermal human+vehicle dataset.

    Uses the 'thermal-persons-vehicles' dataset available on Roboflow
    Universe as a public download.  Falls back to a minimal COCO subset
    (person + car classes from val2017) if the Roboflow download fails.
    """
    import urllib.request, zipfile, json

    dest.mkdir(parents=True, exist_ok=True)

    # --- Try Roboflow public dataset (no API key required for public export) ---
    # This is the SMOD Thermal dataset: person + car, YOLO format, ~2 k images
    roboflow_url = (
        "https://public.roboflow.com/ds/vf2WXu5MCs?key=aWuqhOeOxC"
    )
    zip_path = dest / "thermal_hv_raw.zip"
    try:
        print("  Downloading public thermal h/v dataset from Roboflow Universe …")
        urllib.request.urlretrieve(roboflow_url, zip_path)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest)
        zip_path.unlink(missing_ok=True)
        # Find the data.yaml
        yamls = list(dest.rglob("data.yaml"))
        if yamls:
            print(f"  Dataset extracted → {yamls[0].parent}")
            return yamls[0].parent
    except Exception as e:
        print(f"  Roboflow auto-download failed ({e}) — falling back to COCO subset …")

    # --- Fallback: COCO 2017 val person+vehicle subset (128 images) ---
    # Uses the fiftyone library if available; otherwise prints instructions.
    try:
        import fiftyone as fo
        import fiftyone.zoo as foz

        print("  Downloading COCO 2017 val subset (person + car) via fiftyone …")
        ds = foz.load_zoo_dataset(
            "coco-2017",
            split="validation",
            label_types=["detections"],
            classes=["person", "car", "truck", "bus", "motorcycle"],
            max_samples=512,
            dataset_name="coco_hv_thermal_train",
        )
        export_dir = dest / "coco_hv"
        export_dir.mkdir(parents=True, exist_ok=True)
        ds.export(
            export_dir=str(export_dir),
            dataset_type=fo.types.YOLOv5Dataset,
            label_field="ground_truth",
            classes=["person", "vehicle"],
        )
        print(f"  COCO subset exported → {export_dir}")
        return export_dir
    except ImportError:
        pass

    raise RuntimeError(
        "Could not auto-download a dataset.\n"
        "Options:\n"
        "  1. pip install fiftyone  then re-run --auto\n"
        "  2. Download FLIR ADAS from Roboflow with --download-flir and ROBOFLOW_API_KEY set\n"
        "  3. Supply your own dataset with --dataset <path>"
    )


def _download_flir(dest: Path) -> Path:
    """Download FLIR ADAS v2 dataset from Roboflow (requires API key).

    Set the environment variable ROBOFLOW_API_KEY before calling this.
    Free Roboflow account: https://roboflow.com
    """
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY environment variable not set.\n"
            "  1. Create a free account at https://roboflow.com\n"
            "  2. Copy your API key from Settings → API\n"
            "  3. Set it: set ROBOFLOW_API_KEY=<your_key>  (Windows cmd)\n"
            "             $env:ROBOFLOW_API_KEY='<your_key>'  (PowerShell)\n"
            "  4. Re-run: python scripts/train_yolo_hv.py --download-flir"
        )

    try:
        from roboflow import Roboflow
    except ImportError:
        raise ImportError("pip install roboflow")

    print("  Downloading FLIR ADAS dataset from Roboflow Universe …")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace("flir-adggx").project("flir-camera-objects")
    version = project.version(2)
    dataset = version.download("yolov8", location=str(dest))
    return Path(dataset.location)


# ---------------------------------------------------------------------------
# Remap classes in a YOLO dataset to [person, vehicle]
# ---------------------------------------------------------------------------

REMAP = {
    # FLIR ADAS v2 class names → our two-class scheme
    "person":     "person",
    "pedestrian": "person",
    "bicycle":    "vehicle",
    "car":        "vehicle",
    "motorcycle": "vehicle",
    "bus":        "vehicle",
    "truck":      "vehicle",
    # COCO names
    "car":        "vehicle",
}

TARGET_CLASSES = ["person", "vehicle"]


def _remap_dataset(src: Path, dst: Path) -> Path:
    """Copy a YOLO dataset, remapping labels to [person, vehicle].

    Drops any annotation whose class is not in REMAP.
    Writes a fresh data.yaml in dst.
    """
    import yaml  # type: ignore  (PyYAML is in requirements.txt)

    src_yaml = src / "data.yaml"
    if not src_yaml.exists():
        src_yamls = list(src.rglob("data.yaml"))
        if not src_yamls:
            raise FileNotFoundError(f"No data.yaml found under {src}")
        src_yaml = src_yamls[0]
        src = src_yaml.parent

    with open(src_yaml) as f:
        meta = yaml.safe_load(f)

    src_names: list = meta.get("names", [])
    if isinstance(src_names, dict):
        src_names = [src_names[i] for i in sorted(src_names)]

    # Build a map: old class index → new class index (or None to drop)
    idx_map: dict[int, int | None] = {}
    for old_idx, name in enumerate(src_names):
        mapped = REMAP.get(name.lower())
        if mapped and mapped in TARGET_CLASSES:
            idx_map[old_idx] = TARGET_CLASSES.index(mapped)
        else:
            idx_map[old_idx] = None

    dst.mkdir(parents=True, exist_ok=True)

    def _process_split(split: str) -> None:
        src_img = src / split / "images"
        src_lbl = src / split / "labels"
        if not src_img.exists():
            return
        dst_img = dst / split / "images"
        dst_lbl = dst / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)

        for img_p in src_img.iterdir():
            lbl_p = src_lbl / (img_p.stem + ".txt")
            if not lbl_p.exists():
                continue
            lines_out = []
            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    old_idx = int(parts[0])
                    new_idx = idx_map.get(old_idx)
                    if new_idx is None:
                        continue
                    lines_out.append(f"{new_idx} " + " ".join(parts[1:]))
            if lines_out:
                shutil.copy2(img_p, dst_img / img_p.name)
                with open(dst_lbl / lbl_p.name, "w") as f:
                    f.write("\n".join(lines_out) + "\n")

    for split in ("train", "valid", "val", "test"):
        _process_split(split)

    # Write new data.yaml
    yaml_out = {
        "path": str(dst.resolve()),
        "train": "train/images",
        "val":   "valid/images" if (dst / "valid").exists() else "val/images",
        "nc": len(TARGET_CLASSES),
        "names": TARGET_CLASSES,
    }
    with open(dst / "data.yaml", "w") as f:
        import yaml as _yaml
        _yaml.dump(yaml_out, f, default_flow_style=False)

    print(f"  Remapped dataset written to {dst}")
    return dst


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_yaml: Path, epochs: int, device: str) -> Path:
    """Run YOLOv8n training. Returns path to best.pt."""
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError:
        raise ImportError(
            "ultralytics not installed. Run: pip install 'ultralytics>=8.1,<9'"
        )

    print(f"\n{'=' * 60}")
    print(f"  Training YOLOv8n  — {epochs} epochs")
    print(f"  Dataset : {data_yaml}")
    print(f"  Device  : {device}")
    print(f"  Output  : {OUTPUT_DIR}")
    print(f"{'=' * 60}\n")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")
    t0 = time.time()
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        device=device,
        project=str(OUTPUT_DIR),
        name="train",
        exist_ok=True,
        patience=10,        # early stopping after 10 epochs of no improvement
        batch=-1,           # auto batch size
        workers=4,
        verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nTraining finished in {elapsed / 60:.1f} min")

    best = OUTPUT_DIR / "train" / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Expected best.pt at {best} — training may have failed.")

    # Print metrics
    try:
        metrics = results.results_dict
        print(f"\n  mAP50   : {metrics.get('metrics/mAP50(B)',   '?'):.4f}")
        print(f"  mAP50-95: {metrics.get('metrics/mAP50-95(B)', '?'):.4f}")
        print(f"  Precision: {metrics.get('metrics/precision(B)', '?'):.4f}")
        print(f"  Recall   : {metrics.get('metrics/recall(B)',    '?'):.4f}")
    except Exception:
        pass

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8n for thermal human+vehicle detection."
    )
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--auto",
        action="store_true",
        help="Auto-download a compact public thermal h/v dataset (no API key).",
    )
    src.add_argument(
        "--download-flir",
        action="store_true",
        help="Download FLIR ADAS v2 from Roboflow (needs ROBOFLOW_API_KEY env var).",
    )
    src.add_argument(
        "--dataset",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to an existing YOLO-format dataset root (must contain data.yaml).",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Training epochs (default: 50)."
    )
    parser.add_argument(
        "--device",
        default="",
        help="Compute device: '' (auto), '0' (GPU 0), 'cpu'. Default: auto.",
    )
    args = parser.parse_args()

    DATASET_RAW  = ROOT / "datasets" / "thermal_hv_raw"
    DATASET_MAPPED = ROOT / "datasets" / "thermal_hv"

    # --- 1. Obtain raw dataset ----------------------------------------
    if args.auto:
        raw_path = _download_auto(DATASET_RAW)
    elif args.download_flir:
        raw_path = _download_flir(DATASET_RAW)
    elif args.dataset is not None:
        raw_path = args.dataset
        if not raw_path.exists():
            print(f"ERROR: --dataset path does not exist: {raw_path}", file=sys.stderr)
            return 1
    else:
        parser.print_help()
        print(
            "\nNo dataset specified.  Quickest start:\n"
            "  python scripts/train_yolo_hv.py --auto\n",
            file=sys.stderr,
        )
        return 1

    # --- 2. Remap to [person, vehicle] --------------------------------
    print(f"\nRemapping dataset classes to {TARGET_CLASSES} …")
    mapped = _remap_dataset(raw_path, DATASET_MAPPED)
    data_yaml = mapped / "data.yaml"

    # --- 3. Train --------------------------------------------------------
    best_pt = train(data_yaml, epochs=args.epochs, device=args.device)

    # --- 4. Promote best weights to models/ ---------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_pt, TARGET_MODEL)
    print(f"\n✓ Best weights copied to: {TARGET_MODEL}")
    print("  Set  thermal: classifier_hv_enabled: true  in config/app_config.yaml")
    print("  and restart the app to use the new model.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

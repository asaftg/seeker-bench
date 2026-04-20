"""
Convert FLIR ADAS v2 thermal split to YOLO h/v format.

Expected input: datasets/flir_adas_v2/FLIR_ADAS_v2.zip  (the zip from FLIR.com)

After extraction the layout looks like:
    FLIR_ADAS_v2/
        images_thermal_train/
            data/*.jpg
            coco.json
        images_thermal_val/
            data/*.jpg
            coco.json
        images_rgb_train/ ...   (we ignore RGB)

Class merge to seeker 2-class scheme:
    person                              -> 0 (person)
    car, truck, bus, motor, scooter,
    other vehicle, stroller             -> 1 (vehicle)
    bike, dog, deer, light, hydrant,
    sign, skateboard, train             -> dropped

Output merged directly into datasets/seeker_hv/ with "flir_" prefix.
"""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "datasets" / "flir_adas_v2" / "FLIR_ADAS_v2.zip"
EXTRACT = ROOT / "datasets" / "flir_adas_v2" / "extracted"
DST = ROOT / "datasets" / "seeker_hv"

PERSON_NAMES = {"person"}
VEHICLE_NAMES = {"car", "truck", "bus", "motor", "motorcycle", "scooter",
                 "other vehicle", "stroller"}


def unzip_if_needed():
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[flir] already extracted at {EXTRACT}")
        return
    if not ZIP.exists():
        raise SystemExit(f"[flir] missing zip: {ZIP}")
    EXTRACT.mkdir(parents=True, exist_ok=True)
    print(f"[flir] extracting {ZIP} ({ZIP.stat().st_size/1e9:.1f} GB) ...")
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EXTRACT)
    print("[flir] extracted")


def find_splits(root: Path) -> list[tuple[str, Path]]:
    """Return list of (split_name, split_dir) pairs for thermal splits."""
    splits = []
    for split_name in ("train", "val", "video_thermal_test"):
        # FLIR uses images_thermal_train / images_thermal_val
        candidates = list(root.rglob(f"images_thermal_{split_name}"))
        for c in candidates:
            if (c / "coco.json").exists() or (c / "data").exists():
                splits.append((split_name, c))
    return splits


def convert_split(split_name: str, split_dir: Path, dst_split: str) -> tuple[int, int, int]:
    coco_file = split_dir / "coco.json"
    if not coco_file.exists():
        print(f"[flir] SKIP {split_name}: no coco.json in {split_dir}")
        return 0, 0, 0
    with coco_file.open(encoding="utf-8") as f:
        coco = json.load(f)

    # Build category_id -> class_name map
    cat_map: dict[int, str] = {c["id"]: c["name"].lower().strip() for c in coco["categories"]}

    # Build image_id -> (file_name, W, H)
    images: dict[int, dict] = {}
    for im in coco["images"]:
        images[im["id"]] = {
            "file": im["file_name"],   # e.g. "data/video-xxxx.jpg"
            "w": float(im["width"]),
            "h": float(im["height"]),
        }

    # Accumulate boxes per image
    boxes_by_img: dict[int, list[str]] = defaultdict(list)
    kept = dropped = 0
    for ann in coco["annotations"]:
        cname = cat_map.get(ann["category_id"], "")
        if cname in PERSON_NAMES:
            cls = 0
        elif cname in VEHICLE_NAMES:
            cls = 1
        else:
            dropped += 1
            continue
        img = images.get(ann["image_id"])
        if img is None:
            continue
        x, y, w, h = ann["bbox"]  # COCO: x,y,w,h in absolute px
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / img["w"]
        cy = (y + h / 2.0) / img["h"]
        nw = w / img["w"]
        nh = h / img["h"]
        boxes_by_img[ann["image_id"]].append(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        kept += 1

    img_dst = DST / "images" / dst_split
    lbl_dst = DST / "labels" / dst_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img = 0
    for img_id, lines in boxes_by_img.items():
        im_info = images[img_id]
        src_img = split_dir / im_info["file"]
        if not src_img.exists():
            # sometimes file_name is already relative to split_dir/data
            alt = split_dir / "data" / Path(im_info["file"]).name
            if alt.exists():
                src_img = alt
            else:
                continue
        stem = Path(im_info["file"]).stem
        out_name = f"flir_{stem}"
        shutil.copy2(src_img, img_dst / f"{out_name}.jpg")
        (lbl_dst / f"{out_name}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1

    print(f"[flir] {split_name} -> {dst_split}: images={n_img} boxes_kept={kept} boxes_dropped={dropped}")
    return n_img, kept, dropped


def main():
    unzip_if_needed()
    # Root might be one level deep (FLIR_ADAS_v2/...) or flat
    root = EXTRACT
    if (EXTRACT / "FLIR_ADAS_v2").exists():
        root = EXTRACT / "FLIR_ADAS_v2"
    print(f"[flir] root = {root}")

    splits = find_splits(root)
    if not splits:
        raise SystemExit(f"[flir] could not find any images_thermal_* splits under {root}")
    print(f"[flir] found splits: {[(s[0], str(s[1])) for s in splits]}")

    total_imgs = total_boxes = 0
    for split_name, split_dir in splits:
        # Map FLIR split names to our splits. Use "train" as train, "val"/"video_thermal_test" as val.
        dst_split = "train" if split_name == "train" else "val"
        ni, kb, _ = convert_split(split_name, split_dir, dst_split)
        total_imgs += ni
        total_boxes += kb

    print(f"[flir] TOTAL images={total_imgs} boxes_kept={total_boxes}")


if __name__ == "__main__":
    main()

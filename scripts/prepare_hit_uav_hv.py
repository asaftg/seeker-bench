"""
Convert HIT-UAV thermal dataset to YOLO h/v format for Seeker-01 Ticket 1.

Source classes (from yolo_labels/*.txt):
    0 = Person, 1 = Car, 2 = Bicycle, 3 = OtherVehicle

Target classes:
    0 = person   (from Person)
    1 = vehicle  (from Car + OtherVehicle)
    Bicycle is dropped (ambiguous for drone-detection context).

Splits come from normal_xml/ImageSets/Main/{train,val}.txt (basenames, no ext).

Output layout:
    datasets/hit_uav_hv/
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt
        data.yaml
"""
from __future__ import annotations
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "datasets" / "hit_uav"
DST = Path(__file__).resolve().parents[1] / "datasets" / "hit_uav_hv"

SRC_IMAGES = SRC / "normal_xml" / "JPEGImages"
SRC_LABELS = SRC / "yolo_labels"
SRC_SPLITS = SRC / "normal_xml" / "ImageSets" / "Main"

# source class id -> (new class id or None to drop)
REMAP = {0: 0, 1: 1, 2: None, 3: 1}


def read_split(name: str) -> list[str]:
    path = SRC_SPLITS / f"{name}.txt"
    return [l.strip() for l in path.read_text().splitlines() if l.strip()]


def convert_label(src_txt: Path, dst_txt: Path) -> int:
    kept = 0
    lines_out: list[str] = []
    for line in src_txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        new_cls = REMAP.get(cls)
        if new_cls is None:
            continue
        lines_out.append(f"{new_cls} {parts[1]} {parts[2]} {parts[3]} {parts[4]}")
        kept += 1
    dst_txt.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept


def build_split(split: str) -> tuple[int, int]:
    img_dst = DST / "images" / split
    lbl_dst = DST / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    bases = read_split(split)
    n_img = 0
    n_box = 0
    for base in bases:
        src_img = SRC_IMAGES / f"{base}.jpg"
        src_lbl = SRC_LABELS / f"{base}.txt"
        if not src_img.exists() or not src_lbl.exists():
            continue
        boxes = convert_label(src_lbl, lbl_dst / f"{base}.txt")
        if boxes == 0:
            # skip images with no person/vehicle objects (pure bicycle images)
            (lbl_dst / f"{base}.txt").unlink(missing_ok=True)
            continue
        shutil.copy2(src_img, img_dst / f"{base}.jpg")
        n_img += 1
        n_box += boxes
    return n_img, n_box


def write_yaml() -> Path:
    yaml_path = DST / "data.yaml"
    yaml_path.write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
    )
    return yaml_path


def main():
    DST.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        n_img, n_box = build_split(split)
        print(f"[{split}] images={n_img}  boxes={n_box}")
    yp = write_yaml()
    print(f"data.yaml -> {yp}")


if __name__ == "__main__":
    main()

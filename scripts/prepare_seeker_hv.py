"""
Build the merged Seeker h/v thermal dataset.

Copies HIT-UAV (person+vehicle, aerial) and LLVIP (person, surveillance) into
a single YOLO dataset:
    datasets/seeker_hv/
        images/train  images/val
        labels/train  labels/val
        data.yaml
Classes: 0=person, 1=vehicle.

Run: python scripts/prepare_seeker_hv.py
"""
from __future__ import annotations
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DST = ROOT / "datasets" / "seeker_hv"
HIT = ROOT / "datasets" / "hit_uav_hv"


def copy_hit_uav():
    if not HIT.exists():
        print(f"[seeker_hv] SKIP HIT-UAV (not built: run prepare_hit_uav_hv.py first)")
        return 0, 0
    n_img = n_lbl = 0
    for split in ("train", "val"):
        src_img = HIT / "images" / split
        src_lbl = HIT / "labels" / split
        dst_img = DST / "images" / split
        dst_lbl = DST / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for p in src_img.glob("*.jpg"):
            out = f"hit_{p.stem}"
            shutil.copy2(p, dst_img / f"{out}.jpg")
            n_img += 1
        for p in src_lbl.glob("*.txt"):
            out = f"hit_{p.stem}"
            shutil.copy2(p, dst_lbl / f"{out}.txt")
            n_lbl += 1
    print(f"[seeker_hv] HIT-UAV merged: {n_img} images, {n_lbl} label files")
    return n_img, n_lbl


def count_split(split: str) -> tuple[int, int]:
    img = DST / "images" / split
    lbl = DST / "labels" / split
    return (len(list(img.glob("*.jpg"))) if img.exists() else 0,
            len(list(lbl.glob("*.txt"))) if lbl.exists() else 0)


def write_yaml():
    yaml = DST / "data.yaml"
    yaml.write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
    )
    print(f"[seeker_hv] data.yaml -> {yaml}")


def main():
    DST.mkdir(parents=True, exist_ok=True)
    copy_hit_uav()
    # prepare_llvip_hv.py writes directly into DST/images|labels when run separately
    for split in ("train", "val"):
        ni, nl = count_split(split)
        print(f"[seeker_hv] {split}: images={ni}  label_files={nl}")
    write_yaml()


if __name__ == "__main__":
    main()

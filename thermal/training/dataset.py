"""
YOLO dataset layout for Seeker-01 thermal fine-tuning.

Why this file exists: the class list for training MUST match the
`TargetClass` enum used at runtime. If they drift, the model will
predict indices that get decoded to the wrong label. So we derive
`names` here from `TargetClass` and write it into `data.yaml`.

On-disk layout we produce:

    datasets/<name>/
    ├── data.yaml                 # ultralytics-format config
    ├── images/
    │   ├── train/                # training images (.png)
    │   ├── val/                  # validation images
    │   └── unlabeled/            # captures waiting for labels
    └── labels/
        ├── train/                # YOLO txt files, one per image
        └── val/

`labelImg` / Roboflow can target `images/unlabeled/` and write
YOLO-format txt files next to the images; our `split_unlabeled`
helper then moves labeled pairs into train/val.
"""
from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import yaml

from common.frames import TargetClass


# Ordered list — index in this list is the integer class id written
# into YOLO label files. DO NOT reorder; append new classes at the end.
TRAIN_CLASSES: List[TargetClass] = [
    TargetClass.DRONE,
    TargetClass.HAND,
    TargetClass.BIRD,
]


def class_names() -> List[str]:
    return [c.value for c in TRAIN_CLASSES]


def class_index(cls: TargetClass) -> int:
    return TRAIN_CLASSES.index(cls)


@dataclass
class DatasetLayout:
    root: Path
    name: str = "seeker_thermal"

    @property
    def data_yaml(self) -> Path:     return self.root / "data.yaml"
    @property
    def images_train(self) -> Path:  return self.root / "images" / "train"
    @property
    def images_val(self) -> Path:    return self.root / "images" / "val"
    @property
    def images_unlabeled(self) -> Path: return self.root / "images" / "unlabeled"
    @property
    def labels_train(self) -> Path:  return self.root / "labels" / "train"
    @property
    def labels_val(self) -> Path:    return self.root / "labels" / "val"

    def all_dirs(self) -> List[Path]:
        return [
            self.images_train, self.images_val, self.images_unlabeled,
            self.labels_train, self.labels_val,
        ]


def create(root: Path, name: str = "seeker_thermal") -> DatasetLayout:
    """Create the full directory tree and a fresh `data.yaml`."""
    layout = DatasetLayout(root=Path(root), name=name)
    for d in layout.all_dirs():
        d.mkdir(parents=True, exist_ok=True)
    write_data_yaml(layout)
    return layout


def write_data_yaml(layout: DatasetLayout) -> None:
    """Overwrite `data.yaml` using the current TargetClass enum."""
    cfg = {
        "path": str(layout.root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(TRAIN_CLASSES),
        "names": class_names(),
    }
    layout.data_yaml.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


def split_unlabeled(
    layout: DatasetLayout,
    val_fraction: float = 0.2,
    seed: int = 0,
) -> Tuple[int, int]:
    """Move labeled (image + .txt) pairs from `unlabeled/` into
    `train/` and `val/` with a deterministic random split.

    An image is considered labeled if a YOLO-format `.txt` with the
    same stem exists alongside it in the unlabeled dir. Unlabeled
    images are left untouched.

    Returns (num_train, num_val).
    """
    rng = random.Random(seed)
    imgs = sorted(layout.images_unlabeled.glob("*.png"))
    pairs = [(img, img.with_suffix(".txt")) for img in imgs
             if img.with_suffix(".txt").exists()]
    rng.shuffle(pairs)
    n_val = int(round(len(pairs) * val_fraction))
    val_set = set(p[0].stem for p in pairs[:n_val])

    nt, nv = 0, 0
    for img, label in pairs:
        if img.stem in val_set:
            dst_img, dst_lbl = layout.images_val / img.name, layout.labels_val / label.name
            nv += 1
        else:
            dst_img, dst_lbl = layout.images_train / img.name, layout.labels_train / label.name
            nt += 1
        shutil.move(str(img), str(dst_img))
        shutil.move(str(label), str(dst_lbl))
    return nt, nv

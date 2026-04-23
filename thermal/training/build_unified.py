"""Build unified Seeker v2 datasets for YOLO training.

Generates:
  datasets/seeker_thermal_v2.yaml
  datasets/seeker_eo_v2.yaml

Strategy:
  For each source/split, we write YOLO-format label .txt files into a
  unified cache at  training_runs/unified/<bundle>/labels/<split>/<src>/
  and we hardlink (fallback: file copy-by-reference is avoided; we just
  keep the ORIGINAL image path in the list and ensure ultralytics can
  locate the label by placing a SHADOW label tree matching the image
  tree).  We use ultralytics' rule: replace `/images/` with `/labels/`.
  To guarantee this works for every source regardless of its native
  layout, we build a fresh unified tree with hardlinks to images:

      training_runs/unified/<bundle>/images/<split>/<src>/<name>.jpg
      training_runs/unified/<bundle>/labels/<split>/<src>/<name>.txt

  This costs ~0 disk (hardlinks) and makes ultralytics happy.

Run:
  python thermal/training/build_unified.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "datasets"
OUT = ROOT / "training_runs" / "unified"
LOG = ROOT / "training_runs" / "build_unified.log"

THERMAL_BUNDLE = "thermal_v2"
EO_BUNDLE      = "eo_v2"

# Unified classes (must match common/frames.TargetClass ordering we chose)
CLASSES = ["person", "vehicle", "drone"]  # 0,1,2

_log_fh = None
def log(msg: str) -> None:
    global _log_fh
    if _log_fh is None:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        _log_fh = open(LOG, "w", encoding="utf-8")
    print(msg)
    _log_fh.write(msg + "\n")
    _log_fh.flush()


def link_or_copy(src: Path, dst: Path) -> bool:
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)          # hardlink — no extra disk on same volume
        return True
    except OSError:
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            log(f"  [skip] copy failed {src} -> {dst}: {e}")
            return False


def ensure_dirs(bundle: str, split: str, src: str):
    img_dir = OUT / bundle / "images" / split / src
    lbl_dir = OUT / bundle / "labels" / split / src
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    return img_dir, lbl_dir


def emit_pair(bundle: str, split: str, src: str,
              img_src: Path, lines: list[str]) -> str | None:
    """Create hardlink for image + write label .txt. Return new img path."""
    img_dir, lbl_dir = ensure_dirs(bundle, split, src)
    img_dst = img_dir / img_src.name
    lbl_dst = lbl_dir / (img_src.stem + ".txt")
    if not link_or_copy(img_src, img_dst):
        return None
    try:
        lbl_dst.write_text("\n".join(lines), encoding="utf-8")
    except Exception as e:
        log(f"  [skip] label write failed {lbl_dst}: {e}")
        return None
    return str(img_dst).replace("\\", "/")


# ──────────────────────────────────────────────────────────────
# Source adapters — each yields (split, img_path, yolo_lines)
# ──────────────────────────────────────────────────────────────

def src_hit_uav(bundle="thermal_v2"):
    base = DS / "hit_uav"
    remap = {0: 0, 1: 1, 2: 0, 3: 1}  # person, car->veh, bicycle->person, other_vehicle->veh
    split_map = {"train": "train", "val": "val", "test": "val"}  # test folded into val
    for native, split in split_map.items():
        img_dir = base / "normal_json" / native
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl_src = base / "yolo_labels" / (img.stem + ".txt")
            if not lbl_src.exists():
                continue
            lines = []
            try:
                for ln in lbl_src.read_text().splitlines():
                    p = ln.strip().split()
                    if len(p) < 5:
                        continue
                    c = int(p[0])
                    if c not in remap:
                        continue
                    lines.append(f"{remap[c]} {p[1]} {p[2]} {p[3]} {p[4]}")
            except Exception as e:
                log(f"  [skip] hit_uav label parse {lbl_src}: {e}")
                continue
            if not lines:
                continue
            yield split, img, lines


def src_hit_uav_hv(bundle="thermal_v2"):
    base = DS / "hit_uav_hv"
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl_src = lbl_dir / (img.stem + ".txt")
            if not lbl_src.exists():
                continue
            try:
                lines = [ln for ln in lbl_src.read_text().splitlines() if ln.strip()]
            except Exception:
                continue
            if lines:
                yield split, img, lines  # already 0=person, 1=vehicle


def src_thermal_drone(bundle="thermal_v2"):
    base = DS / "thermal_drone"
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl_src = lbl_dir / (img.stem + ".txt")
            if not lbl_src.exists():
                continue
            try:
                lines = []
                for ln in lbl_src.read_text().splitlines():
                    p = ln.strip().split()
                    if len(p) < 5:
                        continue
                    # remap 0->2 (drone)
                    if int(p[0]) == 0:
                        lines.append(f"2 {p[1]} {p[2]} {p[3]} {p[4]}")
            except Exception:
                continue
            if lines:
                yield split, img, lines


# FLIR COCO cat remap: cat_id -> unified class
FLIR_REMAP = {1: 0, 2: 0, 3: 1, 4: 0, 6: 1, 8: 1, 78: 1}


def _flir_iter(coco_json: Path, img_dir: Path, split: str):
    try:
        d = json.loads(coco_json.read_text())
    except Exception as e:
        log(f"  [skip] flir parse {coco_json}: {e}")
        return
    by_img = {}
    for ann in d["annotations"]:
        cid = ann["category_id"]
        if cid not in FLIR_REMAP:
            continue
        by_img.setdefault(ann["image_id"], []).append(ann)
    for img in d["images"]:
        anns = by_img.get(img["id"], [])
        if not anns:
            continue
        p = img_dir / img["file_name"]
        if not p.exists():
            continue
        W, H = img["width"], img["height"]
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            nw = w / W
            nh = h / H
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"{FLIR_REMAP[ann['category_id']]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            yield split, p, lines


def src_flir_thermal():
    root = DS / "flir_adas_v2" / "extracted"
    for native, split in (("images_thermal_train", "train"),
                          ("images_thermal_val", "val")):
        cj = root / native / "coco.json"
        if cj.exists():
            yield from _flir_iter(cj, root / native, split)


def src_flir_rgb():
    root = DS / "flir_adas_v2" / "extracted"
    for native, split in (("images_rgb_train", "train"),
                          ("images_rgb_val", "val")):
        cj = root / native / "coco.json"
        if cj.exists():
            yield from _flir_iter(cj, root / native, split)


def _llvip_iter(img_root: Path, ann_root: Path, split: str):
    """LLVIP: PASCAL VOC XML per image. One class: person."""
    for img in img_root.glob("*.jpg"):
        xml_path = ann_root / (img.stem + ".xml")
        if not xml_path.exists():
            continue
        try:
            root = ET.parse(xml_path).getroot()
            size = root.find("size")
            W = int(size.find("width").text)
            H = int(size.find("height").text)
        except Exception:
            continue
        lines = []
        for obj in root.findall("object"):
            name = obj.find("name").text.strip().lower()
            if name != "person":
                continue
            bb = obj.find("bndbox")
            xmin = float(bb.find("xmin").text)
            ymin = float(bb.find("ymin").text)
            xmax = float(bb.find("xmax").text)
            ymax = float(bb.find("ymax").text)
            cx = (xmin + xmax) / 2 / W
            cy = (ymin + ymax) / 2 / H
            nw = (xmax - xmin) / W
            nh = (ymax - ymin) / H
            if nw <= 0 or nh <= 0:
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if lines:
            yield split, img, lines


def src_llvip_ir():
    base = DS / "llvip" / "extracted" / "LLVIP"
    ann = base / "Annotations"
    for native, split in (("train", "train"), ("test", "val")):
        d = base / "infrared" / native
        if d.is_dir():
            yield from _llvip_iter(d, ann, split)


def src_llvip_vis():
    base = DS / "llvip" / "extracted" / "LLVIP"
    ann = base / "Annotations"
    for native, split in (("train", "train"), ("test", "val")):
        d = base / "visible" / native
        if d.is_dir():
            yield from _llvip_iter(d, ann, split)


def src_seeker_hv():
    base = DS / "seeker_hv"
    for split in ("train", "val"):
        img_dir = base / "images" / split
        lbl_dir = base / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.jpg"):
            lbl_src = lbl_dir / (img.stem + ".txt")
            if not lbl_src.exists():
                continue
            try:
                lines = [ln for ln in lbl_src.read_text().splitlines() if ln.strip()]
            except Exception:
                continue
            if lines:
                yield split, img, lines


# ──────────────────────────────────────────────────────────────
# Build
# ──────────────────────────────────────────────────────────────

THERMAL_SOURCES = [
    ("hit_uav",       src_hit_uav),
    ("hit_uav_hv",    src_hit_uav_hv),
    ("thermal_drone", src_thermal_drone),
    ("flir_thermal",  src_flir_thermal),
    ("llvip_ir",      src_llvip_ir),
]

EO_SOURCES = [
    ("llvip_vis",  src_llvip_vis),
    ("flir_rgb",   src_flir_rgb),
    ("seeker_hv",  src_seeker_hv),
]


def build_bundle(bundle: str, sources, yaml_out: Path,
                 cap_per_source_train=12000, cap_per_source_val=2000):
    log(f"\n=== Building {bundle} ===")
    train_list = OUT / bundle / f"{bundle}_train.txt"
    val_list   = OUT / bundle / f"{bundle}_val.txt"
    train_list.parent.mkdir(parents=True, exist_ok=True)
    t_fh = open(train_list, "w", encoding="utf-8")
    v_fh = open(val_list, "w", encoding="utf-8")
    totals = {}
    for src_name, gen in sources:
        log(f"  [source] {src_name}")
        counts = {"train": 0, "val": 0}
        try:
            for split, img, lines in gen():
                cap = cap_per_source_train if split == "train" else cap_per_source_val
                if counts[split] >= cap:
                    continue
                new_img = emit_pair(bundle, split, src_name, img, lines)
                if new_img is None:
                    continue
                (t_fh if split == "train" else v_fh).write(new_img + "\n")
                counts[split] += 1
        except Exception as e:
            log(f"  [source-error] {src_name}: {e}")
        log(f"    -> train={counts['train']}  val={counts['val']}")
        totals[src_name] = counts
    t_fh.close(); v_fh.close()

    yaml_out.write_text(
        "# Auto-generated by thermal/training/build_unified.py\n"
        f"path: {(OUT / bundle).as_posix()}\n"
        f"train: {train_list.as_posix()}\n"
        f"val: {val_list.as_posix()}\n"
        "names:\n"
        "  0: person\n"
        "  1: vehicle\n"
        "  2: drone\n",
        encoding="utf-8",
    )
    log(f"  wrote {yaml_out}")
    log(f"  totals: {totals}")
    return totals


def main():
    DS.mkdir(exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    thermal_yaml = DS / "seeker_thermal_v2.yaml"
    eo_yaml      = DS / "seeker_eo_v2.yaml"
    build_bundle(THERMAL_BUNDLE, THERMAL_SOURCES, thermal_yaml)
    build_bundle(EO_BUNDLE,      EO_SOURCES,      eo_yaml)
    log("\nDONE.")


if __name__ == "__main__":
    main()

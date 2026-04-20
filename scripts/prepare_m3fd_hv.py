"""
Convert M3FD thermal (IR) fusion dataset to YOLO h/v format.

Expected input: one of
    datasets/m3fd/M3FD.zip
    datasets/m3fd/M3FD_Fusion.zip
    datasets/m3fd/M3FD_Detection.zip

M3FD_Detection has YOLO-style labels already; M3FD_Fusion only has images.
Annotations (M3FD_Detection layout):
    Annotation/*.xml    VOC format, classes: People, Car, Bus, Motorcycle, Lamp, Truck
    Ir/*.png            thermal images (one per annotation)
    Vis/*.png           RGB (ignored)

Class merge:
    people / person          -> 0 (person)
    car / bus / motorcycle /
    motor / truck            -> 1 (vehicle)
    lamp                     -> dropped

Output merged into datasets/seeker_hv/ with "m3fd_" prefix.
All M3FD goes to train split — it's too small to carve a val partition.
"""
from __future__ import annotations
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "m3fd"
EXTRACT = SRC / "extracted"
DST = ROOT / "datasets" / "seeker_hv"

PERSON_NAMES = {"people", "person", "pedestrian"}
VEHICLE_NAMES = {"car", "bus", "motorcycle", "motor", "truck"}


def unzip_all_if_needed():
    zips = list(SRC.glob("*.zip"))
    if not zips:
        raise SystemExit(f"[m3fd] no *.zip in {SRC} — download M3FD_Detection.zip from "
                         f"https://github.com/JinyuanLiu-CV/M3FD")
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[m3fd] already extracted at {EXTRACT}")
        return
    EXTRACT.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print(f"[m3fd] extracting {z.name} ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(EXTRACT)
    print("[m3fd] extracted")


def find_ir_and_ann() -> tuple[Path, Path]:
    # Look for Ir + Annotation dirs anywhere under EXTRACT
    ir_dir = None
    ann_dir = None
    for p in EXTRACT.rglob("Ir"):
        if p.is_dir():
            ir_dir = p
            break
    for p in EXTRACT.rglob("Annotation"):
        if p.is_dir():
            ann_dir = p
            break
    if ir_dir is None or ann_dir is None:
        raise SystemExit(f"[m3fd] could not find Ir/ and Annotation/ under {EXTRACT}. "
                         f"Detected: ir={ir_dir}  ann={ann_dir}")
    return ir_dir, ann_dir


def voc_to_yolo(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    out = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower().strip()
        if name in PERSON_NAMES:
            cls = 0
        elif name in VEHICLE_NAMES:
            cls = 1
        else:
            continue
        bb = obj.find("bndbox")
        x1 = float(bb.find("xmin").text)
        y1 = float(bb.find("ymin").text)
        x2 = float(bb.find("xmax").text)
        y2 = float(bb.find("ymax").text)
        cx = (x1 + x2) / 2.0 / W
        cy = (y1 + y2) / 2.0 / H
        w = (x2 - x1) / W
        h = (y2 - y1) / H
        if w <= 0 or h <= 0:
            continue
        out.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def main():
    unzip_all_if_needed()
    ir_dir, ann_dir = find_ir_and_ann()
    print(f"[m3fd] ir={ir_dir}  ann={ann_dir}")

    img_dst = DST / "images" / "train"
    lbl_dst = DST / "labels" / "train"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Also drop 10% into val
    n_img = n_box = 0
    for idx, img in enumerate(sorted(ir_dir.glob("*.png")) + sorted(ir_dir.glob("*.jpg"))):
        xml = ann_dir / (img.stem + ".xml")
        if not xml.exists():
            continue
        lines = voc_to_yolo(xml)
        if not lines:
            continue
        split = "val" if (idx % 10 == 0) else "train"
        out_img = DST / "images" / split
        out_lbl = DST / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)
        out_name = f"m3fd_{img.stem}"
        shutil.copy2(img, out_img / f"{out_name}{img.suffix}")
        (out_lbl / f"{out_name}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(lines)
    print(f"[m3fd] images={n_img} boxes={n_box}")


if __name__ == "__main__":
    main()

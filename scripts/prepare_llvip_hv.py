"""
Extract LLVIP and convert thermal (infrared) pedestrian annotations to YOLO h/v format.

LLVIP layout (after unzip):
    LLVIP/
        infrared/train/*.jpg
        infrared/test/*.jpg
        visible/train/*.jpg
        visible/test/*.jpg
        Annotations/*.xml   (PASCAL VOC, pedestrian class only)

Output is merged into datasets/seeker_hv/{images,labels}/{train,val}/...
Class 0 = person (LLVIP has only pedestrians).
"""
from __future__ import annotations
import shutil
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZIP = ROOT / "datasets" / "llvip" / "LLVIP.zip"
EXTRACT = ROOT / "datasets" / "llvip" / "extracted"
DST = ROOT / "datasets" / "seeker_hv"


def unzip_if_needed():
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[llvip] already extracted at {EXTRACT}")
        return
    EXTRACT.mkdir(parents=True, exist_ok=True)
    print(f"[llvip] extracting {ZIP} ...")
    with zipfile.ZipFile(ZIP) as zf:
        zf.extractall(EXTRACT)
    print("[llvip] extracted")


def voc_to_yolo(xml_path: Path) -> list[str]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    W = float(size.find("width").text)
    H = float(size.find("height").text)
    out = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        if name != "person":
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
        out.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def find_llvip_root() -> Path:
    # Handle common archive layouts (LLVIP.zip unpacks to LLVIP/... or directly)
    if (EXTRACT / "LLVIP" / "infrared").exists():
        return EXTRACT / "LLVIP"
    if (EXTRACT / "infrared").exists():
        return EXTRACT
    # search
    for p in EXTRACT.rglob("infrared"):
        if p.is_dir() and (p / "train").exists():
            return p.parent
    raise SystemExit(f"[llvip] could not locate infrared/ dir under {EXTRACT}")


def build_split(llvip_root: Path, src_split: str, dst_split: str) -> tuple[int, int]:
    ir_dir = llvip_root / "infrared" / src_split
    ann_dir = llvip_root / "Annotations"
    img_dst = DST / "images" / dst_split
    lbl_dst = DST / "labels" / dst_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for img in sorted(ir_dir.glob("*.jpg")):
        xml = ann_dir / (img.stem + ".xml")
        if not xml.exists():
            continue
        lines = voc_to_yolo(xml)
        if not lines:
            continue
        out_name = f"llvip_{img.stem}"
        (lbl_dst / f"{out_name}.txt").write_text("\n".join(lines) + "\n")
        shutil.copy2(img, img_dst / f"{out_name}.jpg")
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def main():
    unzip_if_needed()
    root = find_llvip_root()
    print(f"[llvip] root = {root}")
    n_t, b_t = build_split(root, "train", "train")
    n_v, b_v = build_split(root, "test", "val")
    print(f"[llvip] train images={n_t} boxes={b_t}")
    print(f"[llvip] val   images={n_v} boxes={b_v}")


if __name__ == "__main__":
    main()

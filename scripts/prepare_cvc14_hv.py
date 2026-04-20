"""
Convert CVC-14 far-IR pedestrian dataset to YOLO h/v format (person only).

CVC-14 structure (after unzip):
    CVC-14-Day/
        Day/FIR/NewTest/FramesPos/*.tif   (thermal images)
        Day/FIR/NewTest/Annotations/*.txt (bbox per line)
        Day/FIR/Train/FramesPos/*.tif
        Day/FIR/Train/Annotations/*.txt
    CVC-14-Night/
        Night/FIR/...

Annotation txt format (one box per line):
    x y w h  (absolute pixels)   [plus optional fields we ignore]

Class: 0 = person only.

Output merged into datasets/seeker_hv/ with "cvc_" prefix.
"""
from __future__ import annotations
import shutil
import zipfile
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "cvc14"
EXTRACT = SRC / "extracted"
DST = ROOT / "datasets" / "seeker_hv"


def unzip_all():
    zips = list(SRC.glob("*.zip"))
    if not zips:
        print(f"[cvc14] no *.zip in {SRC} — skipping")
        return False
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[cvc14] already extracted at {EXTRACT}")
        return True
    EXTRACT.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print(f"[cvc14] extracting {z.name}")
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(EXTRACT)
        except zipfile.BadZipFile:
            print(f"[cvc14] {z.name} not a zip — skipped")
    return True


def parse_ann(txt: Path) -> list[tuple[float, float, float, float]]:
    out = []
    for line in txt.read_text(errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        try:
            x, y, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        # CVC-14 is sometimes cx,cy,w,h — detect heuristically by whether box would
        # exceed image bounds. We normalize later regardless.
        if w > 0 and h > 0:
            out.append((x, y, w, h))
    return out


def find_splits(root: Path) -> list[tuple[Path, Path, str]]:
    """Return list of (frames_dir, ann_dir, split_tag)."""
    splits = []
    for fp in root.rglob("FramesPos"):
        if not fp.is_dir():
            continue
        ann = fp.parent / "Annotations"
        if not ann.is_dir():
            continue
        # Guess split from path
        path_str = str(fp).lower()
        dst = "val" if "test" in path_str else "train"
        splits.append((fp, ann, dst))
    return splits


def convert(frames: Path, anns: Path, dst_split: str) -> tuple[int, int]:
    img_dst = DST / "images" / dst_split
    lbl_dst = DST / "labels" / dst_split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for img_path in sorted(frames.iterdir()):
        if img_path.suffix.lower() not in {".tif", ".tiff", ".png", ".jpg"}:
            continue
        ann_path = anns / (img_path.stem + ".txt")
        if not ann_path.exists():
            continue
        boxes = parse_ann(ann_path)
        if not boxes:
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        # CVC-14 ann format: x,y are top-left corner in px.
        lines = []
        for x, y, w, h in boxes:
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            if not (0 < nw <= 1 and 0 < nh <= 1 and 0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        if not lines:
            continue
        # Save as PNG (uniform with other prefixes) — convert TIF to 8-bit PNG.
        if img.dtype != "uint8":
            img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
        else:
            img8 = img
        if img8.ndim == 2:
            img8 = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        out_stem = f"cvc_{img_path.stem}"
        cv2.imwrite(str(img_dst / f"{out_stem}.png"), img8)
        (lbl_dst / f"{out_stem}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def main():
    if not unzip_all():
        return
    splits = find_splits(EXTRACT)
    if not splits:
        print(f"[cvc14] no FramesPos/Annotations found in {EXTRACT}")
        return
    total_i = total_b = 0
    for fp, ann, tag in splits:
        ni, nb = convert(fp, ann, tag)
        print(f"[cvc14] {fp} -> {tag}: images={ni} boxes={nb}")
        total_i += ni
        total_b += nb
    print(f"[cvc14] TOTAL images={total_i} boxes={total_b}")


if __name__ == "__main__":
    main()

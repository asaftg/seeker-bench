"""
Convert Anti-UAV thermal tracking sequences to a YOLO drone-detection dataset.

Anti-UAV (any variant: Anti-UAV, Anti-UAV410, Anti-UAV600, 3rd Anti-UAV
Challenge) distributes video-like sequences with per-frame JSON labels.

Expected layout after extraction of any Anti-UAV archive:
    <root>/<sequence_name>/IR/*.jpg
    <root>/<sequence_name>/IR_label.json   (or similar)
    or
    <root>/<sequence_name>/infrared/*.jpg
    <root>/<sequence_name>/infrared.json

JSON schema (common):
    {
      "exist": [0/1, 0/1, ...],
      "gt_rect": [[x,y,w,h], [x,y,w,h], ...]
    }

Writes to datasets/thermal_drone/ with "antiuav_<seq>_<frame>" prefix.
Class 0 = drone.  Every 10th sequence becomes val split.
"""
from __future__ import annotations
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "antiuav"
EXTRACT = SRC / "extracted"
DST = ROOT / "datasets" / "thermal_drone"

IR_DIR_NAMES = {"IR", "infrared", "ir", "IR_img"}
JSON_NAMES = {"IR_label.json", "infrared.json", "IR.json", "infrared_label.json"}


def unzip_all():
    zips = list(SRC.glob("*.zip"))
    if not zips:
        print(f"[antiuav] no *.zip in {SRC} — skipping extraction")
        return
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[antiuav] already extracted at {EXTRACT}")
        return
    EXTRACT.mkdir(parents=True, exist_ok=True)
    for z in zips:
        print(f"[antiuav] extracting {z.name} ({z.stat().st_size/1e9:.2f} GB) ...")
        with zipfile.ZipFile(z) as zf:
            zf.extractall(EXTRACT)
    print("[antiuav] extracted")


def find_sequences(root: Path) -> list[Path]:
    """Return paths to every sequence dir (containing an IR subdir + JSON)."""
    seqs = []
    seen = set()
    for ir_name in IR_DIR_NAMES:
        for ir in root.rglob(ir_name):
            if not ir.is_dir():
                continue
            seq = ir.parent
            if seq in seen:
                continue
            # Does it have a JSON sibling?
            json_path = None
            for jn in JSON_NAMES:
                cand = seq / jn
                if cand.exists():
                    json_path = cand
                    break
            if json_path is None:
                # Sometimes json is inside the IR dir
                for jn in JSON_NAMES:
                    cand = ir / jn
                    if cand.exists():
                        json_path = cand
                        break
            if json_path is not None:
                seqs.append(seq)
                seen.add(seq)
    return seqs


def load_json_robust(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"[antiuav] json parse failed for {path}: {e}")
            return None


def get_json_for_seq(seq: Path) -> Path | None:
    for jn in JSON_NAMES:
        for loc in (seq, *[seq / n for n in IR_DIR_NAMES]):
            cand = loc / jn
            if cand.exists():
                return cand
    # Fallback: any .json in the seq dir
    for j in seq.glob("*.json"):
        if "rgb" not in j.name.lower() and "visible" not in j.name.lower():
            return j
    return None


def get_ir_dir(seq: Path) -> Path | None:
    for n in IR_DIR_NAMES:
        p = seq / n
        if p.is_dir():
            return p
    return None


def convert_sequence(seq: Path, split: str) -> tuple[int, int]:
    ir_dir = get_ir_dir(seq)
    json_path = get_json_for_seq(seq)
    if ir_dir is None or json_path is None:
        return 0, 0
    data = load_json_robust(json_path)
    if not data:
        return 0, 0

    exist = data.get("exist") or []
    rects = data.get("gt_rect") or data.get("gt") or []
    if not rects:
        return 0, 0

    frames = sorted(ir_dir.glob("*.jpg")) + sorted(ir_dir.glob("*.png"))
    n_frames = min(len(frames), len(rects))
    if n_frames == 0:
        return 0, 0

    img_dst = DST / "images" / split
    lbl_dst = DST / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    # Read first frame to get W/H
    import cv2
    first = cv2.imread(str(frames[0]))
    if first is None:
        return 0, 0
    H, W = first.shape[:2]

    n_img = n_box = 0
    seq_name = seq.name
    # Only keep every 5th frame — adjacent frames in tracking seqs are near-duplicates.
    for i in range(0, n_frames, 5):
        if exist and i < len(exist) and not exist[i]:
            continue
        rect = rects[i]
        if rect is None or len(rect) != 4:
            continue
        x, y, w, h = rect
        if w <= 0 or h <= 0:
            continue
        cx = (x + w / 2.0) / W
        cy = (y + h / 2.0) / H
        nw = w / W
        nh = h / H
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < nw < 1 and 0 < nh < 1):
            continue
        frame_path = frames[i]
        out_name = f"antiuav_{seq_name}_{frame_path.stem}"
        shutil.copy2(frame_path, img_dst / f"{out_name}{frame_path.suffix}")
        (lbl_dst / f"{out_name}.txt").write_text(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
        n_img += 1
        n_box += 1
    return n_img, n_box


# ---------------------------------------------------------------------------
# DUT Anti-UAV layout support: flat `img/` + `xml/` (or `annotations/`) dirs.
# DUT typically ships as:
#     DUT Anti-UAV/
#         train/img/*.jpg
#         train/xml/*.xml       (VOC, class=drone or UAV)
#         val/img/*.jpg
#         val/xml/*.xml
#         test/img/*.jpg
#         test/xml/*.xml
# Also tolerates txt label files (one bbox per line, "x y w h" absolute px).
# ---------------------------------------------------------------------------
import cv2
import xml.etree.ElementTree as ET


def find_flat_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    """Return list of (img_dir, ann_dir, split_hint)."""
    pairs = []
    seen = set()
    # Look for `img` dirs with a sibling annotations dir
    for img_dir in root.rglob("img"):
        if not img_dir.is_dir():
            continue
        parent = img_dir.parent
        if parent in seen:
            continue
        ann_dir = None
        for ann_name in ("xml", "annotations", "Annotations", "anno", "label", "labels"):
            cand = parent / ann_name
            if cand.is_dir():
                ann_dir = cand
                break
        if ann_dir is None:
            continue
        # Infer split from parent dir name
        split = "val" if any(tag in parent.name.lower() for tag in ("val", "test")) else "train"
        pairs.append((img_dir, ann_dir, split))
        seen.add(parent)
    return pairs


def parse_dut_ann(ann_path: Path, W: int, H: int) -> list[str]:
    """Parse a DUT-style annotation file -> YOLO lines.  Accepts VOC XML or plain txt."""
    lines: list[str] = []
    if ann_path.suffix.lower() == ".xml":
        try:
            root = ET.parse(ann_path).getroot()
        except Exception:
            return []
        # Try to pull width/height from XML if present (overrides passed W/H)
        size = root.find("size")
        if size is not None:
            try:
                W = int(float(size.find("width").text)) or W
                H = int(float(size.find("height").text)) or H
            except Exception:
                pass
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            try:
                x1 = float(bb.find("xmin").text)
                y1 = float(bb.find("ymin").text)
                x2 = float(bb.find("xmax").text)
                y2 = float(bb.find("ymax").text)
            except Exception:
                continue
            cx = (x1 + x2) / 2.0 / W
            cy = (y1 + y2) / 2.0 / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            if not (0 < w < 1 and 0 < h < 1 and 0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    else:
        # txt: one box per line, whitespace-separated.
        for ln in ann_path.read_text(errors="ignore").splitlines():
            parts = ln.strip().split()
            if len(parts) < 4:
                continue
            try:
                nums = [float(p) for p in parts[:4]]
            except ValueError:
                continue
            x, y, w, h = nums
            if w <= 0 or h <= 0:
                continue
            cx = (x + w / 2.0) / W
            cy = (y + h / 2.0) / H
            nw = w / W
            nh = h / H
            if not (0 < nw < 1 and 0 < nh < 1 and 0 < cx < 1 and 0 < cy < 1):
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return lines


def convert_flat(img_dir: Path, ann_dir: Path, split: str) -> tuple[int, int]:
    img_dst = DST / "images" / split
    lbl_dst = DST / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    prefix = f"dut_{img_dir.parent.name}_"
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        stem = img_path.stem
        # Try matching annotation: same stem, any supported extension
        ann_path = None
        for ext in (".xml", ".txt"):
            cand = ann_dir / f"{stem}{ext}"
            if cand.exists():
                ann_path = cand
                break
        if ann_path is None:
            continue
        # Read image dims (needed for txt format; xml may override)
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        H, W = im.shape[:2]
        lines = parse_dut_ann(ann_path, W, H)
        if not lines:
            continue
        out_name = f"{prefix}{stem}"
        shutil.copy2(img_path, img_dst / f"{out_name}{img_path.suffix}")
        (lbl_dst / f"{out_name}.txt").write_text("\n".join(lines) + "\n")
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def find_video_sequences(root: Path) -> list[tuple[Path, Path, str]]:
    """Find (infrared.mp4, infrared.json, split) triples — Anti-UAV-RGBT layout.

    Structure:
        <root>/<split_dir>/<sequence_name>/infrared.mp4
        <root>/<split_dir>/<sequence_name>/infrared.json
    split_dir is test/train/val — used to decide our YOLO split.
    """
    out = []
    for mp4 in root.rglob("infrared.mp4"):
        js = mp4.parent / "infrared.json"
        if not js.exists():
            continue
        # Derive split from whichever ancestor dir is named train/val/test
        split = "train"
        for anc in mp4.parents:
            name = anc.name.lower()
            if name in ("train",):
                split = "train"; break
            if name in ("val", "test"):
                split = "val"; break
        out.append((mp4, js, split))
    return out


def convert_video_sequence(mp4: Path, json_path: Path, split: str) -> tuple[int, int]:
    data = load_json_robust(json_path)
    if not data:
        return 0, 0
    exist = data.get("exist") or []
    rects = data.get("gt_rect") or data.get("gt") or []
    if not rects:
        return 0, 0

    cap = cv2.VideoCapture(str(mp4))
    if not cap.isOpened():
        print(f"[antiuav] video open failed: {mp4}")
        return 0, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n_frames = min(total_frames, len(rects))
    if n_frames == 0:
        cap.release()
        return 0, 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0

    img_dst = DST / "images" / split
    lbl_dst = DST / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    seq_name = mp4.parent.name
    # Keep every 5th frame
    n_img = n_box = 0
    i = 0
    while i < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % 5 == 0:
            if not (exist and i < len(exist) and not exist[i]):
                rect = rects[i] if i < len(rects) else None
                if rect and len(rect) == 4:
                    x, y, w, h = rect
                    if w > 0 and h > 0 and W > 0 and H > 0:
                        cx = (x + w / 2.0) / W
                        cy = (y + h / 2.0) / H
                        nw = w / W
                        nh = h / H
                        if 0 < cx < 1 and 0 < cy < 1 and 0 < nw < 1 and 0 < nh < 1:
                            out_name = f"rgbt_{seq_name}_{i:06d}"
                            cv2.imwrite(str(img_dst / f"{out_name}.jpg"), frame,
                                        [cv2.IMWRITE_JPEG_QUALITY, 90])
                            (lbl_dst / f"{out_name}.txt").write_text(
                                f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n"
                            )
                            n_img += 1
                            n_box += 1
        i += 1
    cap.release()
    return n_img, n_box


def write_yaml():
    DST.mkdir(parents=True, exist_ok=True)
    yaml = DST / "data.yaml"
    yaml.write_text(
        f"path: {DST.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: drone\n"
    )
    print(f"[antiuav] data.yaml -> {yaml}")


def main():
    unzip_all()
    if not EXTRACT.exists():
        print("[antiuav] nothing to prep")
        return

    total_img = total_box = 0

    # Path 1: sequence format (CVPR Anti-UAV, Anti-UAV410, Anti-UAV600)
    seqs = find_sequences(EXTRACT)
    print(f"[antiuav] sequences found: {len(seqs)}")
    for i, seq in enumerate(sorted(seqs)):
        split = "val" if (i % 10 == 0) else "train"
        ni, nb = convert_sequence(seq, split)
        total_img += ni
        total_box += nb
        if (i + 1) % 20 == 0:
            print(f"[antiuav] seq progress {i+1}/{len(seqs)}  total_imgs={total_img}")

    # Path 1b: video-sequence format (Anti-UAV-RGBT: infrared.mp4 + infrared.json)
    vids = find_video_sequences(EXTRACT)
    print(f"[antiuav] video sequences found: {len(vids)}")
    for i, (mp4, js, split) in enumerate(sorted(vids)):
        ni, nb = convert_video_sequence(mp4, js, split)
        total_img += ni
        total_box += nb
        if (i + 1) % 20 == 0:
            print(f"[antiuav] video progress {i+1}/{len(vids)}  total_imgs={total_img}")

    # Path 2: flat img+annotations format (DUT Anti-UAV)
    pairs = find_flat_pairs(EXTRACT)
    print(f"[antiuav] flat img/ann pairs found: {len(pairs)}")
    for img_dir, ann_dir, split in pairs:
        ni, nb = convert_flat(img_dir, ann_dir, split)
        print(f"[antiuav] flat {img_dir.parent.name}/{img_dir.name} -> {split}: imgs={ni} boxes={nb}")
        total_img += ni
        total_box += nb

    print(f"[antiuav] TOTAL images={total_img} boxes={total_box}")
    write_yaml()


if __name__ == "__main__":
    main()

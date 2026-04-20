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
    seqs = find_sequences(EXTRACT)
    print(f"[antiuav] found {len(seqs)} sequences")
    if not seqs:
        return

    total_img = total_box = 0
    for i, seq in enumerate(sorted(seqs)):
        split = "val" if (i % 10 == 0) else "train"
        ni, nb = convert_sequence(seq, split)
        total_img += ni
        total_box += nb
        if (i + 1) % 20 == 0:
            print(f"[antiuav] progress: {i+1}/{len(seqs)} sequences  imgs={total_img}")
    print(f"[antiuav] TOTAL images={total_img} boxes={total_box}")
    write_yaml()


if __name__ == "__main__":
    main()

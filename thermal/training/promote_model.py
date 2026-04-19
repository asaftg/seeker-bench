"""
Promote a freshly trained weights file into the runtime classifier slot.

Usage:

    python -m thermal.training.promote_model runs/train/seeker_v1/weights/best.pt

This copies the given weights to `models/seeker_thermal.pt`, which
`thermal/drone_classifier.py` prefers over the stock `models/yolov8n.pt`
on the next Seeker-01 restart. No code changes required to swap in
your own fine-tuned model — drop-in replacement by design.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


DEFAULT_DEST = Path("models") / "seeker_thermal.pt"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", help="Path to a trained YOLO .pt file (e.g. best.pt)")
    p.add_argument("--dest", default=str(DEFAULT_DEST),
                   help=f"Destination path (default: {DEFAULT_DEST})")
    p.add_argument("--force", action="store_true",
                   help="Overwrite dest without prompting")
    args = p.parse_args()

    src = Path(args.source)
    dst = Path(args.dest)

    if not src.exists():
        print(f"ERROR: source not found: {src}", file=sys.stderr)
        return 2

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not args.force:
        reply = input(f"{dst} already exists. Overwrite? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted.")
            return 1

    shutil.copy2(src, dst)
    print(f"Promoted {src} -> {dst}")
    print("Restart Seeker-01 to use the new model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

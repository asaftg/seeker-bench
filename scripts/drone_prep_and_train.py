"""
Run drone prep + training directly (no Downloads polling).

Use this when the Anti-UAV zips are already in datasets/antiuav/ and you just
want to re-process them and retrain. Safe to re-run; skips already-extracted.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs" / "drone_prep_and_train.log"
LOG.parent.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], label: str, timeout_h: float) -> bool:
    log(f"[{label}] START")
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n==== {label} ====\n"); f.flush()
            r = subprocess.run(cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                               timeout=int(timeout_h * 3600))
        log(f"[{label}] rc={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        log(f"[{label}] EXCEPTION: {e}")
        return False


def main():
    LOG.write_text("")
    log("drone_prep_and_train: start")
    # Nuke stale thermal_drone output so leftover zero-frame prep from the
    # previous run doesn't pollute the new dataset.
    stale = ROOT / "datasets" / "thermal_drone"
    if stale.exists():
        import shutil
        shutil.rmtree(stale)
        log(f"removed stale {stale}")
    prep_ok = run([sys.executable, str(ROOT / "scripts" / "prepare_antiuav_drone.py")],
                  "prep", timeout_h=3.0)
    if not prep_ok:
        log("prep FAILED — aborting")
        return
    train_ok = run([sys.executable, str(ROOT / "scripts" / "train_thermal_drone.py")],
                   "train", timeout_h=6.0)
    summary = ROOT / "logs" / "drone_prep_and_train_summary.txt"
    img_train = len(list((ROOT / "datasets/thermal_drone/images/train").glob("*.*"))) if (ROOT / "datasets/thermal_drone/images/train").exists() else 0
    img_val = len(list((ROOT / "datasets/thermal_drone/images/val").glob("*.*"))) if (ROOT / "datasets/thermal_drone/images/val").exists() else 0
    summary.write_text(
        "Drone prep + train summary\n"
        "===========================\n"
        f"train images: {img_train}\n"
        f"val images:   {img_val}\n"
        f"prep ok:      {prep_ok}\n"
        f"train ok:     {train_ok}\n"
        f"log:          {LOG}\n"
    )
    log(f"wrote summary {summary}")
    log("done")


if __name__ == "__main__":
    main()

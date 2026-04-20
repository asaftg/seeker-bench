"""
Overnight orchestrator: wait for FLIR ADAS zip, try public downloads, prep everything,
retrain both h/v and drone models, promote best.pt, write a summary.

Safe to re-run: every step is idempotent (re-uses already-extracted dirs, already-built
data.yaml, etc.).  Failures in any prep step are logged and skipped, never fatal.

Usage:
    python scripts/overnight_retrain.py

What it does (in order):
    1. Poll Downloads/ for FLIR_ADAS_v2.zip; when it appears and is stable,
       move it into datasets/flir_adas_v2/.
    2. Try autonomous downloads of CVC-14 + Anti-UAV mirrors.
    3. Run each prep script that has source data ready.
    4. Update datasets/seeker_hv/data.yaml.
    5. Train h/v model -> models/seeker_thermal_hv.pt.
    6. If Anti-UAV zip is present, train drone model -> models/seeker_thermal.pt.
    7. Write logs/overnight_summary.txt.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = LOG_DIR / "overnight_summary.txt"
MAIN_LOG = LOG_DIR / "overnight_retrain.log"


# ---------------------------------------------------------------- logging
def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with MAIN_LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(path: Path, label: str) -> bool:
    if not path.exists():
        log(f"[{label}] script missing: {path}")
        return False
    log(f"[{label}] running {path.name}")
    try:
        r = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=60 * 60,  # 1 h per prep script
        )
        with MAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(f"---- {label} stdout ----\n{r.stdout}\n")
            if r.stderr:
                f.write(f"---- {label} stderr ----\n{r.stderr}\n")
        if r.returncode != 0:
            log(f"[{label}] FAILED (rc={r.returncode}) — see {MAIN_LOG}")
            return False
        log(f"[{label}] OK")
        return True
    except Exception as e:
        log(f"[{label}] EXCEPTION: {e}")
        return False


def run_training(script: Path, label: str) -> bool:
    """Training has its own progress output — stream to log."""
    if not script.exists():
        log(f"[{label}] missing {script}")
        return False
    log(f"[{label}] starting training (this takes hours)")
    try:
        with MAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n==== {label} training START ====\n")
            f.flush()
            r = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=6 * 60 * 60,  # 6 h cap per training
            )
        log(f"[{label}] training finished rc={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        log(f"[{label}] training EXCEPTION: {e}")
        return False


# ---------------------------------------------------------------- FLIR wait
def wait_for_flir(timeout_min: int = 120) -> Path | None:
    """Block until FLIR ADAS zip arrives in Downloads/, then move it into the repo."""
    dst_dir = ROOT / "datasets" / "flir_adas_v2"
    dst = dst_dir / "FLIR_ADAS_v2.zip"
    if dst.exists() and dst.stat().st_size > 10 * 1024**3:  # > 10 GB already moved
        log(f"[flir-wait] zip already in place: {dst}")
        return dst

    log(f"[flir-wait] polling {DOWNLOADS} for FLIR_ADAS_v2*.zip (timeout {timeout_min} min)")
    deadline = time.time() + timeout_min * 60
    last_size = -1
    stable_ticks = 0
    candidate: Path | None = None

    while time.time() < deadline:
        matches = list(DOWNLOADS.glob("FLIR_ADAS_v2*.zip")) + list(DOWNLOADS.glob("*FLIR_ADAS*.zip"))
        matches = [m for m in matches if m.is_file() and not m.name.endswith(".crdownload")]
        if matches:
            candidate = max(matches, key=lambda p: p.stat().st_mtime)
            sz = candidate.stat().st_size
            if sz == last_size and sz > 1024**3:  # stable for 1 poll, >1 GB
                stable_ticks += 1
                if stable_ticks >= 2:
                    break
            else:
                stable_ticks = 0
                last_size = sz
            log(f"[flir-wait] seen {candidate.name} @ {sz/1e9:.2f} GB (stable {stable_ticks}/2)")
        else:
            log("[flir-wait] not yet present (still .crdownload or not started)")
        time.sleep(60)

    if candidate is None:
        log("[flir-wait] TIMEOUT — no FLIR zip found. Continuing without it.")
        return None

    dst_dir.mkdir(parents=True, exist_ok=True)
    log(f"[flir-wait] moving {candidate} -> {dst}")
    try:
        shutil.move(str(candidate), str(dst))
    except Exception as e:
        log(f"[flir-wait] move failed ({e}) — trying copy")
        shutil.copy2(str(candidate), str(dst))
    return dst


# ---------------------------------------------------------------- main
def main():
    MAIN_LOG.write_text("")  # reset
    log("=" * 60)
    log("overnight_retrain: start")
    log("=" * 60)

    # 1. Wait for FLIR ADAS zip
    flir_zip = wait_for_flir(timeout_min=120)

    # 2. Try autonomous downloads (CVC-14, Anti-UAV mirrors)
    run_script(ROOT / "scripts" / "download_public_datasets.py", "download-public")

    # 3. Run every prep script that has source data
    prep_scripts = [
        ("hit-uav",  ROOT / "scripts" / "prepare_hit_uav_hv.py"),
        ("llvip",    ROOT / "scripts" / "prepare_llvip_hv.py"),
        ("flir",     ROOT / "scripts" / "prepare_flir_adas_hv.py"),
        ("m3fd",     ROOT / "scripts" / "prepare_m3fd_hv.py"),
        ("cvc14",    ROOT / "scripts" / "prepare_cvc14_hv.py"),
        ("merge",    ROOT / "scripts" / "prepare_seeker_hv.py"),  # writes data.yaml
    ]
    for label, path in prep_scripts:
        run_script(path, label)

    # Count final h/v dataset
    img_train = len(list((ROOT / "datasets/seeker_hv/images/train").glob("*.*")))
    img_val   = len(list((ROOT / "datasets/seeker_hv/images/val").glob("*.*")))
    log(f"[merge] final seeker_hv: train={img_train} val={img_val}")

    # 4. Train h/v (seeker_thermal_hv.pt)
    hv_ok = run_training(ROOT / "scripts" / "train_thermal_hv.py", "train-hv")

    # 5. Drone side: prep + train (only if Anti-UAV zips are present)
    drone_done = False
    antiuav_zips = list((ROOT / "datasets" / "antiuav").glob("*.zip"))
    if antiuav_zips:
        run_script(ROOT / "scripts" / "prepare_antiuav_drone.py", "antiuav-prep")
        if (ROOT / "datasets" / "thermal_drone" / "data.yaml").exists():
            drone_done = run_training(ROOT / "scripts" / "train_thermal_drone.py", "train-drone")
    else:
        log("[drone] no Anti-UAV zips in datasets/antiuav/ — skipping drone retrain")

    # 6. Summary
    summary_lines = [
        "Seeker-01 overnight retrain summary",
        "=" * 50,
        f"h/v dataset: train={img_train}  val={img_val} images",
        f"h/v training succeeded: {hv_ok}",
        f"drone training succeeded: {drone_done}",
        "",
        f"FLIR ADAS: {'picked up' if flir_zip else 'NOT PRESENT'}",
        f"Full log: {MAIN_LOG}",
    ]
    SUMMARY.write_text("\n".join(summary_lines) + "\n")
    log("wrote summary: " + str(SUMMARY))
    log("overnight_retrain: done")


if __name__ == "__main__":
    main()

"""
Drone retrain orchestrator.

Polls ~/Downloads for any thermal-drone dataset zips (DUT Anti-UAV train/val/test,
Anti-UAV CVPR challenge, etc.), waits for them to finish downloading,
moves them to datasets/antiuav/, then runs the drone prep + training pipeline.

Matches on filenames containing any of: anti, uav, dut, drone (case-insensitive),
excluding anything with "adas" (FLIR ADAS uses "UAV" in no URL but adding guard).

Run:
    python scripts/drone_retrain.py

Safe to re-run; only unmoved zips are touched. Logs to logs/drone_retrain.log.
"""
from __future__ import annotations
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOWNLOADS = Path.home() / "Downloads"
DST = ROOT / "datasets" / "antiuav"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOG_DIR / "drone_retrain.log"

MATCH_RE = re.compile(r"(anti|uav|dut|drone)", re.IGNORECASE)
SKIP_RE = re.compile(r"(adas|flir_adas)", re.IGNORECASE)

# Stability: must be same size across two consecutive polls at `poll_interval_s`
# before we consider the download complete and move the file.
POLL_INTERVAL_S = 60
STABLE_POLLS_NEEDED = 2
# If no new zips appear for this many polls after at least one has moved, fire training.
QUIET_POLLS_NEEDED = 5
TIMEOUT_MIN = 240  # 4 h absolute ceiling


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def candidate_zips(folder: Path) -> list[Path]:
    out = []
    for p in folder.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() != ".zip":
            continue
        if SKIP_RE.search(p.name):
            continue
        if not MATCH_RE.search(p.name):
            continue
        out.append(p)
    return out


def wait_and_move_all() -> list[Path]:
    """Return the list of zips that ended up in DST after polling."""
    DST.mkdir(parents=True, exist_ok=True)
    moved: list[Path] = []
    sizes: dict[Path, int] = {}
    stable_count: dict[Path, int] = {}
    quiet_polls = 0
    deadline = time.time() + TIMEOUT_MIN * 60

    while time.time() < deadline:
        found = candidate_zips(DOWNLOADS)
        crdownloads = list(DOWNLOADS.glob("*.crdownload"))
        new_this_round = False
        for p in found:
            sz = p.stat().st_size
            if sizes.get(p) == sz and sz > 5 * 1024**2:  # same size, > 5 MB
                stable_count[p] = stable_count.get(p, 0) + 1
                if stable_count[p] >= STABLE_POLLS_NEEDED:
                    dst = DST / p.name
                    if dst.exists():
                        continue
                    try:
                        shutil.move(str(p), str(dst))
                        log(f"moved {p.name} -> {dst}  ({sz/1e9:.2f} GB)")
                        moved.append(dst)
                        new_this_round = True
                    except Exception as e:
                        log(f"move failed {p}: {e}")
            else:
                stable_count[p] = 0
                sizes[p] = sz

        # Also count zips already in DST from previous runs
        in_dst = list(DST.glob("*.zip"))

        if not crdownloads and not candidate_zips(DOWNLOADS):
            # No matching pending downloads AND no matching zip still in Downloads
            if in_dst:
                quiet_polls += 1
                log(f"no new drone zips; quiet {quiet_polls}/{QUIET_POLLS_NEEDED}  "
                    f"(have {len(in_dst)} in {DST.name})")
                if quiet_polls >= QUIET_POLLS_NEEDED:
                    break
            else:
                log("no drone zips anywhere yet; still waiting")
                quiet_polls = 0
        else:
            quiet_polls = 0 if new_this_round else quiet_polls
            log(f"polling: found={len(found)} crdownload={len(crdownloads)} in_dst={len(in_dst)}")

        time.sleep(POLL_INTERVAL_S)

    return list(DST.glob("*.zip"))


def run_step(cmd: list[str], label: str, timeout_h: float = 6.0) -> bool:
    log(f"[{label}] START: {' '.join(cmd)}")
    try:
        with LOG.open("a", encoding="utf-8") as f:
            f.write(f"\n==== {label} output ====\n")
            f.flush()
            r = subprocess.run(
                cmd, cwd=str(ROOT), stdout=f, stderr=subprocess.STDOUT,
                timeout=int(timeout_h * 3600),
            )
        log(f"[{label}] DONE rc={r.returncode}")
        return r.returncode == 0
    except Exception as e:
        log(f"[{label}] EXCEPTION: {e}")
        return False


def main():
    LOG.write_text("")  # reset log
    log("=" * 60)
    log("drone_retrain: start")
    log(f"Downloads dir: {DOWNLOADS}")
    log(f"Target dir:    {DST}")
    log("=" * 60)

    zips = wait_and_move_all()
    log(f"wait phase done: {len(zips)} zip(s) in {DST}:")
    for z in zips:
        log(f"  - {z.name} ({z.stat().st_size/1e9:.2f} GB)")
    if not zips:
        log("ERROR no drone zips found after timeout - aborting")
        return

    prep_ok = run_step(
        [sys.executable, str(ROOT / "scripts" / "prepare_antiuav_drone.py")],
        "antiuav-prep", timeout_h=2.0,
    )
    if not prep_ok:
        log("prep failed - aborting before training")
        return

    data_yaml = ROOT / "datasets" / "thermal_drone" / "data.yaml"
    if not data_yaml.exists():
        log(f"ERROR missing {data_yaml} after prep - aborting")
        return

    train_ok = run_step(
        [sys.executable, str(ROOT / "scripts" / "train_thermal_drone.py")],
        "train-drone", timeout_h=6.0,
    )

    summary = LOG_DIR / "drone_retrain_summary.txt"
    summary.write_text(
        "Drone retrain summary\n"
        "=====================\n"
        f"zips processed:        {len(zips)}\n"
        f"prep succeeded:        {prep_ok}\n"
        f"training succeeded:    {train_ok}\n"
        f"model promoted to:     {ROOT / 'models' / 'seeker_thermal.pt'}\n"
        f"full log:              {LOG}\n"
    )
    log(f"wrote {summary}")
    log("drone_retrain: done")


if __name__ == "__main__":
    main()

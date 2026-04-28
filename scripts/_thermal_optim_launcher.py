"""
Auto-resuming launcher for the thermal optimization harness.

Wraps ``_thermal_optim_harness.py`` in a retry loop that:
  - Checks for a STOP sentinel before each launch.
  - Runs the harness with --resume so a partial run picks up where
    it left off.
  - Sleeps + retries on transient failures (camera not yet ready,
    transient I/O errors, etc.).
  - Bails after `max_consecutive_failures` to avoid an infinite
    retry storm when something is fundamentally broken.

Designed for overnight unattended runs: launch this, walk away.
The user can drop a `STOP` file in the pose's output directory to
abort cleanly, or kill the Python process directly.

Usage::

    python scripts/_thermal_optim_launcher.py --pose garage_overnight
    python scripts/_thermal_optim_launcher.py --pose garage_overnight \\
            --max-failures 5 --sleep 30
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime


def _here() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def _harness_path() -> str:
    return os.path.join(_here(), "_thermal_optim_harness.py")


def _state_path(pose: str) -> str:
    return os.path.join("recordings", "optim", pose, "STATE.json")


def _stop_path(pose: str) -> str:
    return os.path.join("recordings", "optim", pose, "STOP")


def _read_state(pose: str) -> dict:
    sp = _state_path(pose)
    if not os.path.exists(sp):
        return {}
    try:
        with open(sp) as f:
            return json.load(f)
    except Exception:
        return {}


def _is_complete(pose: str) -> bool:
    return _read_state(pose).get("tier") == "complete"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", required=True,
                    help="Pose tag for output dir (recordings/optim/<pose>/)")
    ap.add_argument("--max-failures", type=int, default=5,
                    help="Bail after N consecutive harness failures")
    ap.add_argument("--sleep", type=float, default=30.0,
                    help="Seconds to wait between failed retries")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--use-existing-npz", type=str, default=None,
                    help="Skip live capture and use a saved .npz")
    args = ap.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(_here(), ".."))
    os.chdir(repo_root)

    if _is_complete(args.pose):
        print(f"[launcher] pose {args.pose} already complete — nothing to do")
        return 0

    consec_failures = 0
    attempt = 0
    while True:
        attempt += 1
        if os.path.exists(_stop_path(args.pose)):
            print(f"[launcher] STOP sentinel present, exiting")
            return 0

        cmd = [
            sys.executable, _harness_path(),
            "--pose", args.pose,
            "--frames", str(args.frames),
            "--resume",
        ]
        if args.use_existing_npz:
            cmd += ["--use-existing-npz", args.use_existing_npz]

        ts = datetime.now().isoformat(timespec="seconds")
        print(f"[launcher] attempt {attempt} at {ts}: {' '.join(cmd)}")
        rc = subprocess.call(cmd)
        if rc == 0:
            if _is_complete(args.pose):
                print(f"[launcher] pose {args.pose} complete (rc=0)")
                return 0
            print(f"[launcher] harness returned rc=0 but state != complete; "
                  f"resuming after {args.sleep}s")
            time.sleep(args.sleep)
            consec_failures = 0
            continue
        if rc == 130:
            print(f"[launcher] harness was stopped (rc=130 / KeyboardInterrupt)")
            return 130
        consec_failures += 1
        print(f"[launcher] harness failed with rc={rc} "
              f"(consec_failures={consec_failures}/{args.max_failures})")
        if consec_failures >= args.max_failures:
            print(f"[launcher] giving up after {consec_failures} consecutive failures")
            return rc
        time.sleep(args.sleep)


if __name__ == "__main__":
    sys.exit(main())

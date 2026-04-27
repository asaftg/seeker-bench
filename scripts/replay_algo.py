"""
Replay an algorithm against a captured JSONL session.

v1: tracking predictor only. The design is plugin-style so future
algorithms (classifier, fusion, FPS) can be added by registering a
new ``--algo NAME`` adapter.

The first thing this script must do — and the gate that the recorder
verification plan calls out — is **prove parity** between the live
gimbal manager and the pure ``algorithms.track_predictor.step``. With
no variant overrides the replay output should match the recorded
gimbal/state setpoints (target_pan_deg, target_tilt_deg) within ε
modulo the controller's slew/clamp logic which we DON'T re-run here.

Usage:
    python scripts/replay_algo.py --latest --algo predictor
    python scripts/replay_algo.py --latest --algo predictor \
        --variant lead_time_s=0,vel_alpha=0.2

Variants are KEY=VALUE pairs joined by commas. Keys map to fields on
``algorithms.track_predictor.PredictorParams``. Default behaviour
loads the params from the session's config_snapshot in the JSONL
header so replay matches the live config exactly.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import io
import json
import os
import sys
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

# Add project root to sys.path so we can `import algorithms.*` when
# running the script from the repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from algorithms.track_predictor import (
    PredictorParams,
    PredictorState,
    step as predictor_step,
)
from scripts.replay_inspect import find_latest, iter_records


def _params_from_config(cfg_snap: Dict[str, Any]) -> PredictorParams:
    """Reconstruct PredictorParams from the recorded config snapshot.

    Falls back to dataclass defaults for any missing key, so an old
    recording without the latest knob still replays.
    """
    g = (cfg_snap or {}).get("gimbal", {}) or {}
    p = PredictorParams()
    if "track_lead_time_s" in g:
        p.lead_time_s = float(g["track_lead_time_s"])
    if "track_vel_alpha" in g:
        p.vel_alpha = float(g["track_vel_alpha"])
    if "track_predict_warmup_n" in g:
        p.predict_warmup_n = int(g["track_predict_warmup_n"])
    if "track_predict_cap_deg" in g:
        p.predict_cap_deg = float(g["track_predict_cap_deg"])
    if "track_gimbal_settled_dps" in g:
        p.gimbal_settled_dps = float(g["track_gimbal_settled_dps"])
    if "track_extrap_horizon_s" in g:
        p.extrap_horizon_s = float(g["track_extrap_horizon_s"])
    if "track_vel_clip_dps" in g:
        p.vel_clip_dps = float(g["track_vel_clip_dps"])
    return p


def _apply_variant(p: PredictorParams, spec: str) -> PredictorParams:
    """Override fields on ``p`` from a "k=v,k=v" variant spec."""
    if not spec or spec.lower() == "default":
        return p
    field_types = {f.name: f.type for f in dataclasses.fields(p)}
    for chunk in spec.split(","):
        if "=" not in chunk:
            continue
        k, v = chunk.split("=", 1)
        k = k.strip()
        v = v.strip()
        if k not in field_types:
            sys.stderr.write(f"unknown variant field: {k}\n")
            continue
        cur = getattr(p, k)
        try:
            cast = type(cur)(v)
        except Exception:
            cast = float(v)
        setattr(p, k, cast)
    return p


# ──────────────────────────────────────────────────────────────
# Predictor adapter
# ──────────────────────────────────────────────────────────────
def replay_predictor(
    path: str,
    track_id: Optional[int],
    params: PredictorParams,
    out_csv: Optional[str] = None,
) -> int:
    """Re-run track_predictor.step against captured fused observations.

    ``track_id`` may be None — in that case we pick whichever id appears
    most often in the session's track_predictor_step events (a sensible
    default for "the only target the user actually tracked").
    """
    # Pass 1: scan the recording, collect:
    #   * config snapshot from the header (param defaults)
    #   * for each tick where a fused observation is fresh for the
    #     target track id, build (ts_ns, cur_pan, cur_tilt, az, el)
    #   * also retain the recorded predictor diag for ε comparison
    cfg_snap: Dict[str, Any] = {}
    fused_by_ts: List[Tuple[int, Dict[str, Any]]] = []
    gimbal_by_ts: List[Tuple[int, Dict[str, Any]]] = []
    recorded_steps: List[Tuple[int, Dict[str, Any]]] = []  # (ts_ns, payload)

    for rec in iter_records(path):
        ch = rec.get("channel")
        ts = int(rec.get("ts_ns") or 0)
        msg = rec.get("msg") or {}
        if ch == "session/header":
            cfg_snap = msg.get("config_snapshot") or {}
        elif ch == "fusion/tracks":
            fused_by_ts.append((ts, msg))
        elif ch == "gimbal/state":
            gimbal_by_ts.append((ts, msg))
        elif ch == "events" and msg.get("type") == "track_predictor_step":
            recorded_steps.append((ts, msg.get("payload") or {}))

    if not recorded_steps:
        sys.stderr.write("no track_predictor_step events in this recording\n")
        return 1

    if track_id is None:
        from collections import Counter
        c = Counter(int(p.get("tracked_id", -1)) for _, p in recorded_steps)
        track_id = c.most_common(1)[0][0]
        sys.stderr.write(f"--track-id not given; using {track_id}\n")

    # Apply config-snapshot defaults THEN the variant override.
    params = _params_from_config(cfg_snap) if not getattr(params, "_explicit_", False) \
        else params

    # Helper: nearest gimbal sample for a given ts
    g_ts = [t for t, _ in gimbal_by_ts]
    g_data = [m for _, m in gimbal_by_ts]
    def _nearest_gimbal(target: int) -> Optional[Dict[str, Any]]:
        if not g_ts:
            return None
        best_i, best_d = 0, abs(g_ts[0] - target)
        for i in range(1, len(g_ts)):
            d = abs(g_ts[i] - target)
            if d < best_d:
                best_d, best_i = d, i
        return g_data[best_i]

    # Build an (ts_ns, cur_pan, cur_tilt, obs_az, obs_el, fresh) input
    # series. We use the recorded track_predictor_step events as the
    # tick beats — that's exactly when the live manager called step(),
    # so timing is faithful.
    out_rows: List[Dict[str, Any]] = []
    state = PredictorState()
    last_recorded_world_az = None
    diff_counts = {"sp_pan_max_abs": 0.0, "sp_tilt_max_abs": 0.0,
                   "settled_mismatch": 0, "n": 0}

    for ts, recorded in recorded_steps:
        if int(recorded.get("tracked_id", -1)) != int(track_id):
            continue
        cur_pan = float(recorded.get("cur_pan") or 0.0)
        cur_tilt = float(recorded.get("cur_tilt") or 0.0)
        fresh = bool(recorded.get("fresh_fused"))
        # Recover the camera-frame az/el the live manager used by
        # de-projecting obs_world_az -- obs_world_az = cur_pan + obs_az.
        obs_world_az = recorded.get("obs_world_az")
        obs_world_el = recorded.get("obs_world_el")
        if fresh and obs_world_az is not None and obs_world_el is not None:
            obs_az = float(obs_world_az) - cur_pan
            obs_el = float(obs_world_el) - cur_tilt
        else:
            obs_az = obs_el = None

        sp_pan, sp_tilt, diag = predictor_step(
            state,
            now=float(recorded.get("now") or (ts / 1e9)),
            cur_pan=cur_pan,
            cur_tilt=cur_tilt,
            obs_az_deg=obs_az,
            obs_el_deg=obs_el,
            fresh_fused=fresh,
            params=params,
        )

        # ε-compare against what the live manager recorded.
        rec_sp_pan = recorded.get("sp_pan")
        rec_sp_tilt = recorded.get("sp_tilt")
        d_sp_pan = (None if rec_sp_pan is None or sp_pan is None
                    else float(sp_pan) - float(rec_sp_pan))
        d_sp_tilt = (None if rec_sp_tilt is None or sp_tilt is None
                     else float(sp_tilt) - float(rec_sp_tilt))
        if d_sp_pan is not None:
            diff_counts["sp_pan_max_abs"] = max(diff_counts["sp_pan_max_abs"], abs(d_sp_pan))
        if d_sp_tilt is not None:
            diff_counts["sp_tilt_max_abs"] = max(diff_counts["sp_tilt_max_abs"], abs(d_sp_tilt))
        if bool(recorded.get("settled")) != bool(diag.get("settled")):
            diff_counts["settled_mismatch"] += 1
        diff_counts["n"] += 1

        out_rows.append({
            "ts_ns": ts,
            "fresh": fresh,
            "cur_pan": cur_pan, "cur_tilt": cur_tilt,
            "settled": diag.get("settled"),
            "lead": diag.get("lead"),
            "world_az_dot": diag.get("world_az_dot"),
            "world_el_dot": diag.get("world_el_dot"),
            "sp_pan_replay": sp_pan, "sp_tilt_replay": sp_tilt,
            "sp_pan_recorded": rec_sp_pan, "sp_tilt_recorded": rec_sp_tilt,
            "d_sp_pan": d_sp_pan, "d_sp_tilt": d_sp_tilt,
        })

    if not out_rows:
        sys.stderr.write(f"no ticks for tracked_id={track_id}\n")
        return 1

    # Write the CSV
    cols = list(out_rows[0].keys())
    fh: io.IOBase
    if out_csv:
        fh = open(out_csv, "w", encoding="utf-8", newline="")
    else:
        fh = sys.stdout  # type: ignore[assignment]
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for r in out_rows:
        w.writerow(r)
    if out_csv:
        fh.close()

    sys.stderr.write(
        f"\nparity: ticks={diff_counts['n']} "
        f"max|Δsp_pan|={diff_counts['sp_pan_max_abs']:.4f}° "
        f"max|Δsp_tilt|={diff_counts['sp_tilt_max_abs']:.4f}° "
        f"settled_mismatch={diff_counts['settled_mismatch']}\n")
    sys.stderr.write(f"params: {params}\n")
    return 0


# ──────────────────────────────────────────────────────────────
ALGOS = {"predictor": replay_predictor}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", help="JSONL file (default: --latest)")
    p.add_argument("--latest", action="store_true",
                   help="Open the newest recordings/seeker_*.jsonl")
    p.add_argument("--dir", default="recordings")
    p.add_argument("--algo", default="predictor", choices=list(ALGOS.keys()))
    p.add_argument("--variant", default="default",
                   help='K=V,K=V overrides for the algo params (e.g. '
                        '"lead_time_s=0,vel_alpha=0.2"). "default" = use '
                        "config snapshot from the recording.")
    p.add_argument("--track-id", type=int, default=None,
                   help="Tracked id to replay; defaults to the most-seen id.")
    p.add_argument("--out", default=None,
                   help="CSV path; default = stdout")
    args = p.parse_args()

    if args.latest or not args.file:
        path = find_latest(args.dir)
        if path is None:
            sys.stderr.write(f"no seeker_*.jsonl in {args.dir!r}\n")
            return 2
    else:
        path = args.file

    if args.algo == "predictor":
        # Build params: defaults; the adapter loads config_snapshot
        # internally before applying the variant.
        params = PredictorParams()
        if args.variant and args.variant.lower() != "default":
            params._explicit_ = True  # type: ignore[attr-defined]
            params = _apply_variant(params, args.variant)
        return replay_predictor(path, args.track_id, params, out_csv=args.out)

    sys.stderr.write(f"unknown algo: {args.algo}\n")
    return 2


if __name__ == "__main__":
    sys.exit(main())

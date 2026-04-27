"""Optical truth — what did the camera actually see, regardless of what the
controller said.

Why this exists:
    The recorded gimbal/state stream reports the COMMANDED pan/tilt — i.e.
    what the controller asked the servo to do. With analog hobby servos
    under gravity load there is no position feedback, so the actual
    mechanical pose can drift several degrees while the recorded angle
    stays flat. Same idea for synth-target BB centering: the geometric
    pixel→angle math can be exact, but if the gimbal mechanically lands
    short, the bbox content will not be at image centre.

This tool reads the thermal/eo JPEG frames out of a recording and tracks
visual content over time, reporting drift in pixels and (using the FOV
recorded in the frame metadata) in degrees.

Modes:
    --drift t0 [t1]
        From frame at t0, sample features and track them through the rest
        of the recording (or up to t1). Output per-frame (dx_px, dy_px,
        daz_deg, del_deg) of the median feature.

    --bb-residual
        For every synthetic_target_drawn event in the recording, take the
        bbox content as the anchor template, locate it in the post-settle
        frame (default: t_draw + 1.5 s), and report:
          - draw-time bbox centre (image px)
          - post-settle bbox-content position (image px)
          - distance from image centre after settle (px and deg)
        This is the truth about whether the gimbal centred the BB.

Both modes use a base recording from --recording PATH (or pick latest by
mtime in `recordings/`).

CSV output via --out PATH; otherwise human-readable to stdout.
"""
from __future__ import annotations

import argparse
import base64
import csv
import glob
import io
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def latest_recording(directory: str = "recordings") -> Optional[str]:
    matches = sorted(glob.glob(os.path.join(directory, "*.jsonl")),
                     key=os.path.getmtime)
    return matches[-1] if matches else None


# ──────────────────────────────────────────────────────────────────
# JSONL streaming helpers
# ──────────────────────────────────────────────────────────────────
def iter_records(path: str):
    with io.open(path, "r", encoding="utf-8") as fh:
        for ln, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                sys.stderr.write(f"[optical] {path}:{ln}: {e}\n")
                continue


def decode_jpeg_b64(b64: str) -> Optional[np.ndarray]:
    raw = base64.b64decode(b64)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    return img


def _to_gray(b64: str) -> Optional[np.ndarray]:
    if not b64:
        return None
    try:
        return decode_jpeg_b64(b64)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────
# Pass 1 — index the recording
# ──────────────────────────────────────────────────────────────────
def index_recording(path: str, channel: str
                    ) -> Tuple[List[Tuple[float, str]], List[Dict[str, Any]],
                               List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return:
        frame_idx     [(t_rel_s, raw_jpeg_b64)]
        frame_meta    [{t, w, h, hfov, vfov, zoom}]      one per frame
        events        [{t, type, payload}]
        gimbal_states [{t, pan, tilt, mode, tid}]
    """
    frame_idx: List[Tuple[float, str]] = []
    frame_meta: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    gimbal_states: List[Dict[str, Any]] = []
    t0_ns: Optional[int] = None
    with io.open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = rec.get("ts_ns")
            if ts is not None and t0_ns is None:
                t0_ns = int(ts)
            t_rel = (int(ts) - t0_ns) / 1e9 if (ts and t0_ns) else 0.0
            ch = rec.get("channel", "")
            msg = rec.get("msg") or {}
            if ch == channel:
                frame_idx.append((t_rel, msg.get("jpeg_b64") or ""))
                frame_meta.append({
                    "t": t_rel,
                    "w": msg.get("width"),
                    "h": msg.get("height"),
                    "hfov": msg.get("hfov_deg"),
                    "vfov": msg.get("vfov_deg"),
                    "zoom": msg.get("zoom_preset"),
                })
            elif ch == "events":
                events.append({"t": t_rel,
                               "type": msg.get("type"),
                               "payload": msg.get("payload") or {}})
            elif ch == "gimbal/state":
                gimbal_states.append({"t": t_rel,
                                      "pan": msg.get("pan_deg"),
                                      "tilt": msg.get("tilt_deg"),
                                      "mode": msg.get("mode"),
                                      "tid": msg.get("tracked_target_id")})
    return frame_idx, frame_meta, events, gimbal_states


def find_frame_near(frame_idx: List[Tuple[float, str]], t: float
                    ) -> Optional[Tuple[float, int, str]]:
    """Return (t_rel, idx_in_list, jpeg_b64) for the frame with t_rel
    closest to t.
    """
    if not frame_idx:
        return None
    best = min(range(len(frame_idx)), key=lambda i: abs(frame_idx[i][0] - t))
    return frame_idx[best][0], best, frame_idx[best][1]


# ──────────────────────────────────────────────────────────────────
# Mode: drift
# ──────────────────────────────────────────────────────────────────
def cmd_drift(path: str, channel: str, t_anchor: float,
              t_end: Optional[float], out_csv: Optional[str]) -> int:
    frame_idx, frame_meta, _, gimbal_states = index_recording(path, channel)
    if not frame_idx:
        print(f"no {channel} frames in {path}")
        return 2

    anchor_lookup = find_frame_near(frame_idx, t_anchor)
    if anchor_lookup is None:
        print("no anchor frame found")
        return 2
    t_a, anchor_i, anchor_b64 = anchor_lookup
    anchor_img = _to_gray(anchor_b64)
    if anchor_img is None:
        print("anchor frame failed to decode")
        return 2

    h, w = anchor_img.shape
    print(f"anchor frame t={t_a:.2f}s  size={w}x{h}  "
          f"hfov={frame_meta[anchor_i]['hfov']}  "
          f"vfov={frame_meta[anchor_i]['vfov']}  "
          f"zoom={frame_meta[anchor_i]['zoom']}")

    # Sample features from a centred ROI to avoid edges
    roi_pad = int(min(w, h) * 0.2)
    mask = np.zeros_like(anchor_img)
    mask[roi_pad:h - roi_pad, roi_pad:w - roi_pad] = 255
    p0 = cv2.goodFeaturesToTrack(anchor_img, maxCorners=80,
                                 qualityLevel=0.01, minDistance=14, mask=mask)
    if p0 is None or len(p0) == 0:
        print("no features found in anchor frame ROI")
        return 2
    print(f"sampled {len(p0)} features from anchor ROI")

    # Iterate forward
    rows = [("t_rel", "n_kept", "dx_px", "dy_px", "daz_deg", "del_deg",
             "cmd_pan", "cmd_tilt")]
    prev_img = anchor_img
    prev_pts = p0
    for i in range(anchor_i + 1, len(frame_idx)):
        t_i, b64_i = frame_idx[i]
        if t_end is not None and t_i > t_end:
            break
        img = _to_gray(b64_i)
        if img is None or img.shape != anchor_img.shape:
            continue

        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_img, img, prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        good = (status.flatten() == 1) if status is not None else None
        if good is None or good.sum() < 8:
            # Lost too many features; reseed
            p0_new = cv2.goodFeaturesToTrack(
                anchor_img, maxCorners=80, qualityLevel=0.01,
                minDistance=14, mask=mask)
            if p0_new is None:
                break
            prev_pts = p0_new
            prev_img = anchor_img
            # Skip emitting a row this tick
            continue

        kept_anchor = p0[good]
        kept_now = nxt[good]
        delta = kept_now - kept_anchor
        median_dx = float(np.median(delta[:, 0, 0]))
        median_dy = float(np.median(delta[:, 0, 1]))

        meta_i = frame_meta[i]
        daz = (median_dx / w) * (meta_i["hfov"] or 0)
        del_ = -(median_dy / h) * (meta_i["vfov"] or 0)

        # match a gimbal state at this time
        gs = min(gimbal_states, key=lambda g: abs(g["t"] - t_i)) if gimbal_states else None
        rows.append((f"{t_i:.3f}", int(good.sum()),
                     f"{median_dx:+.2f}", f"{median_dy:+.2f}",
                     f"{daz:+.4f}", f"{del_:+.4f}",
                     f"{(gs and gs['pan']) or 0:+.2f}",
                     f"{(gs and gs['tilt']) or 0:+.2f}"))

        prev_img = img
        prev_pts = nxt  # propagate

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w_csv = csv.writer(f)
            w_csv.writerows(rows)
        print(f"\nwrote {len(rows) - 1} rows → {out_csv}")
    else:
        # print every Nth row
        n = max(1, (len(rows) - 1) // 30)
        print(f"\n{'t':>7s} {'kept':>4s} {'dx_px':>7s} {'dy_px':>7s} "
              f"{'daz_deg':>9s} {'del_deg':>9s} {'cmd_pan':>8s} {'cmd_tilt':>9s}")
        for r in rows[1::n]:
            print(f"{r[0]:>7s} {r[1]:>4} {r[2]:>7s} {r[3]:>7s} "
                  f"{r[4]:>9s} {r[5]:>9s} {r[6]:>8s} {r[7]:>9s}")
    return 0


# ──────────────────────────────────────────────────────────────────
# Mode: bb-residual
# ──────────────────────────────────────────────────────────────────
def cmd_bb_residual(path: str, channel: str, settle_s: float,
                    out_csv: Optional[str]) -> int:
    frame_idx, frame_meta, events, gimbal_states = index_recording(path, channel)
    bb_draws = [e for e in events if e["type"] == "synthetic_target_drawn"]
    if not bb_draws:
        print("no synthetic_target_drawn events in recording")
        return 2
    rows = [("idx", "t_draw", "bbox", "img_w", "img_h", "hfov", "vfov",
             "draw_cx_px", "draw_cy_px",
             "post_cx_px", "post_cy_px",
             "post_cx_off_px", "post_cy_off_px",
             "post_cx_off_deg", "post_cy_off_deg",
             "n_features", "settle_lag_s")]
    for k, ev in enumerate(bb_draws):
        t_draw = ev["t"]
        bbox = ev["payload"].get("bbox") or [0, 0, 0, 0]
        # Frame at draw time and at settle
        anc = find_frame_near(frame_idx, t_draw)
        post = find_frame_near(frame_idx, t_draw + settle_s)
        if anc is None or post is None:
            continue
        t_a, ai, b64_a = anc
        t_p, pi, b64_p = post
        anchor_img = _to_gray(b64_a)
        post_img = _to_gray(b64_p)
        if anchor_img is None or post_img is None:
            continue
        h_, w_ = anchor_img.shape

        x, y, ww, hh = (int(v) for v in bbox)
        # Clamp bbox into image bounds
        x = max(0, min(w_ - 2, x))
        y = max(0, min(h_ - 2, y))
        ww = max(2, min(w_ - x, ww))
        hh = max(2, min(h_ - y, hh))
        cx = x + ww * 0.5
        cy = y + hh * 0.5

        # Sample features INSIDE the bbox in the anchor frame
        mask = np.zeros_like(anchor_img)
        mask[y:y + hh, x:x + ww] = 255
        p0 = cv2.goodFeaturesToTrack(anchor_img, maxCorners=60,
                                     qualityLevel=0.01, minDistance=8,
                                     mask=mask)
        if p0 is None or len(p0) < 6:
            print(f"  BB#{k+1} t_draw={t_draw:.2f}: too few features inside "
                  f"bbox ({0 if p0 is None else len(p0)}); skipping")
            continue

        # Chain LK frame-by-frame from anchor to post-settle to handle
        # large gimbal motions (anchor→post direct fails on slews >100 px).
        prev_img = anchor_img
        prev_pts = p0
        kept_count = len(p0)
        success = True
        for j in range(ai + 1, pi + 1):
            t_j, b64_j = frame_idx[j]
            img_j = _to_gray(b64_j)
            if img_j is None or img_j.shape != anchor_img.shape:
                continue
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_img, img_j, prev_pts, None,
                winSize=(31, 31), maxLevel=4,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                          30, 0.01),
            )
            if status is None:
                success = False; break
            good = (status.flatten() == 1)
            if good.sum() < 4:
                success = False; break
            prev_pts = nxt[good].reshape(-1, 1, 2)
            prev_img = img_j
            kept_count = int(good.sum())
        if not success or kept_count < 4:
            print(f"  BB#{k+1} t_draw={t_draw:.2f}: feature tracking lost "
                  f"during chain (last kept {kept_count})")
            continue

        kept_now = prev_pts
        post_cx = float(np.median(kept_now[:, 0, 0]))
        post_cy = float(np.median(kept_now[:, 0, 1]))
        good = np.ones(len(kept_now), dtype=bool)
        # The BB content's median position post-settle
        # Offset from image centre = pixel residual after gimbal "centred"
        post_cx_off = post_cx - w_ / 2.0
        post_cy_off = post_cy - h_ / 2.0
        meta_p = frame_meta[pi]
        post_cx_off_deg = (post_cx_off / w_) * (meta_p["hfov"] or 0)
        post_cy_off_deg = -(post_cy_off / h_) * (meta_p["vfov"] or 0)
        rows.append((k + 1, f"{t_draw:.2f}",
                     str(bbox), w_, h_,
                     f"{meta_p['hfov']:.2f}", f"{meta_p['vfov']:.2f}",
                     f"{cx:.1f}", f"{cy:.1f}",
                     f"{post_cx:.1f}", f"{post_cy:.1f}",
                     f"{post_cx_off:+.1f}", f"{post_cy_off:+.1f}",
                     f"{post_cx_off_deg:+.3f}", f"{post_cy_off_deg:+.3f}",
                     int(good.sum()), f"{t_p - t_draw:.2f}"))

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w_csv = csv.writer(f)
            w_csv.writerows(rows)
        print(f"wrote {len(rows) - 1} BB rows → {out_csv}")
    else:
        if len(rows) <= 1:
            print("no usable BB events")
            return 0
        # Pretty-print
        print()
        for header_row, *data_rows in [rows]:
            pass
        hdr = rows[0]
        print(" | ".join(f"{h:>13s}" if isinstance(h, str) else f"{h:>13}"
                         for h in hdr))
        for r in rows[1:]:
            print(" | ".join(f"{c!s:>13s}" for c in r))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--recording", "-r", default=None,
                   help="JSONL recording (defaults to latest in recordings/)")
    p.add_argument("--channel", default="thermal/frame",
                   choices=("thermal/frame", "eo/frame"),
                   help="which sensor's frames to use")
    p.add_argument("--out", default=None, help="CSV output path")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp_drift = sub.add_parser("drift",
                              help="Optical drift from anchor frame")
    sp_drift.add_argument("--t0", type=float, required=True,
                          help="Anchor frame time (s rel. recording start)")
    sp_drift.add_argument("--t1", type=float, default=None,
                          help="End time (s); default: end of recording")

    sp_bb = sub.add_parser("bb-residual",
                           help="Where did BB content land after settle?")
    sp_bb.add_argument("--settle-s", type=float, default=1.5,
                       help="Settle delay after draw (default 1.5 s)")

    args = p.parse_args(argv)

    rec = args.recording or latest_recording()
    if rec is None or not os.path.exists(rec):
        sys.stderr.write(f"recording not found: {rec}\n")
        return 2
    print(f"recording: {rec}")
    print(f"channel:   {args.channel}")

    if args.cmd == "drift":
        return cmd_drift(rec, args.channel, args.t0, args.t1, args.out)
    elif args.cmd == "bb-residual":
        return cmd_bb_residual(rec, args.channel, args.settle_s, args.out)
    else:
        p.print_help()
        return 2


if __name__ == "__main__":
    sys.exit(main())

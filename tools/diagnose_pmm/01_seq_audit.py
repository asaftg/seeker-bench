"""Phase-0 diagnostic 0.2: packet-sequence integrity audit.

Reads <recording>_radar.csv (one row per UDP packet), computes the gaps
in seq_num across consecutive packets, buckets packets by frame index
(via cumulative byte_offset against bytes_per_frame), and reports any
frame that lost more than `--threshold-pct` of its expected packets.

The radar transport falls back through 64 MB → 16 MB → 8 MB recv
buffers silently. All three recordings show socket_recv_buffer_bytes:
8388608, the smallest fallback. The user has separately confirmed the
live radar GUI is at 7-9 fps (vs 20 fps configured), which means the
Python pipeline CAN'T drain UDP fast enough at live capture time. So
finding drops here is the expected outcome; the question is "where".

Usage:
    python -m tools.diagnose_pmm.01_seq_audit drone_fly
    python -m tools.diagnose_pmm.01_seq_audit airborne1
    python -m tools.diagnose_pmm.01_seq_audit background
    python -m tools.diagnose_pmm.01_seq_audit <path/to/meta.yaml>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS.parents[1]))

from tools.diagnose_pmm._common import resolve_recording  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("recording", help="Friendly name (drone_fly|airborne1|background) or path to .meta.yaml")
    ap.add_argument("--threshold-pct", type=float, default=0.1,
                    help="Flag frames missing more than this %% of expected packets")
    ap.add_argument("--out", type=str, default="runs/seq_audit",
                    help="Output dir (relative to repo root)")
    args = ap.parse_args()

    repo_root = _THIS.parents[1]
    out_dir = repo_root / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = resolve_recording(args.recording)
    print(f"Recording: {rec.name}")
    print(f"  bin:    {rec.bin_path} ({rec.bin_path.stat().st_size/1e6:.0f} MB)")
    print(f"  csv:    {rec.csv_path}")
    print(f"  socket recv buffer: {rec.dims.socket_recv_buffer_bytes/1e6:.0f} MB")

    if not rec.csv_path.exists():
        print(f"FATAL: csv {rec.csv_path} missing")
        return 1

    # CSV columns: ts_host_ns, byte_offset, payload_len, seq_num, chunk_offset
    print(f"  reading csv...", end=" ", flush=True)
    arr = np.genfromtxt(rec.csv_path, delimiter=",", skip_header=1, dtype=np.int64)
    print(f"{arr.shape[0]:,} packets")

    if arr.shape[0] == 0:
        print("  EMPTY csv. nothing to audit.")
        return 1

    ts_ns       = arr[:, 0]
    byte_offset = arr[:, 1]
    payload_len = arr[:, 2]
    seq_num     = arr[:, 3]
    chunk_off   = arr[:, 4]

    # ---- 1. seq-number gaps -----------------------------------------------
    # gap[i] = seq[i] - seq[i-1] - 1.  >0 = drops.  <0 = out-of-order.
    seq_diff = np.diff(seq_num)
    gaps = seq_diff - 1
    n_drops_total      = int(np.maximum(gaps, 0).sum())
    n_outoforder       = int((gaps < 0).sum())
    n_zero_gap         = int((gaps == 0).sum())   # contiguous packets
    n_packets_received = arr.shape[0]
    n_packets_expected = int(seq_num[-1] - seq_num[0]) + 1
    drop_pct = 100.0 * n_drops_total / max(n_packets_expected, 1)
    print(f"  seq:    first={seq_num[0]:,}  last={seq_num[-1]:,}")
    print(f"  packets received   {n_packets_received:,}")
    print(f"  packets expected   {n_packets_expected:,}")
    print(f"  packets dropped    {n_drops_total:,} ({drop_pct:.3f}%)")
    print(f"  contiguous pairs   {n_zero_gap:,}")
    print(f"  out-of-order pairs {n_outoforder}")

    # ---- 2. bucket drops by frame -----------------------------------------
    bpf = rec.dims.bytes_per_frame
    # frame_idx of each packet = (cumulative bytes BEFORE this packet) / bpf
    # The DCA wire byte_offset is reset at the start of each capture, so
    # byte_offset[k] is the cumulative byte position. Use it directly.
    pkt_frame_idx = byte_offset // bpf
    n_frames = int(pkt_frame_idx.max()) + 1
    drops_per_frame = np.zeros(n_frames, dtype=np.int64)
    # Distribute each gap to the frame the gap-end packet lands in
    # (i.e. the frame that started receiving after the missing range).
    # Approximation: assume each gap belongs to the frame that begins at
    # seq[i] (= pkt_frame_idx[i+1]).
    pos = np.maximum(gaps, 0)
    nz = np.flatnonzero(pos)
    for k in nz:
        drops_per_frame[pkt_frame_idx[k + 1]] += int(pos[k])

    # Expected packets per frame: bpf / 1456 (typical UDP payload). We use
    # the median payload_len as the actual chunk size.
    chunk_bytes = int(np.median(payload_len))
    if chunk_bytes <= 0:
        chunk_bytes = 1456
    expected_per_frame = bpf / chunk_bytes
    drop_pct_per_frame = 100.0 * drops_per_frame / max(expected_per_frame, 1)
    bad_mask = drop_pct_per_frame > args.threshold_pct
    n_bad = int(bad_mask.sum())
    worst = int(drop_pct_per_frame.argmax()) if n_frames else 0

    print(f"  bytes_per_frame    {bpf:,}")
    print(f"  median chunk bytes {chunk_bytes}")
    print(f"  expected pkts/frame ~{expected_per_frame:.0f}")
    print(f"  frames audited     {n_frames}")
    print(f"  frames over {args.threshold_pct:.2f}%% drop: {n_bad} ({100.0*n_bad/max(n_frames,1):.1f}%)")
    if n_frames:
        print(f"  worst frame: idx={worst} t={worst*rec.dims.framePeriodicity_s:.2f}s drop={drop_pct_per_frame[worst]:.2f}%")

    # ---- 3. write outputs --------------------------------------------------
    summary_path = out_dir / f"{rec.name}_seq_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Recording: {rec.name}\n")
        f.write(f"bin:    {rec.bin_path}\n")
        f.write(f"csv:    {rec.csv_path}\n")
        f.write(f"socket_recv_buffer_bytes: {rec.dims.socket_recv_buffer_bytes}\n\n")
        f.write(f"packets_received: {n_packets_received}\n")
        f.write(f"packets_expected: {n_packets_expected}\n")
        f.write(f"packets_dropped:  {n_drops_total} ({drop_pct:.3f}%)\n")
        f.write(f"out_of_order:     {n_outoforder}\n")
        f.write(f"frames_audited:   {n_frames}\n")
        f.write(f"frames_over_{args.threshold_pct:.2f}pct: {n_bad}\n")
        if n_frames:
            f.write(f"worst_frame_idx: {worst}\n")
            f.write(f"worst_frame_drop_pct: {drop_pct_per_frame[worst]:.3f}\n")

    csv_out = out_dir / f"{rec.name}_drops_per_frame.csv"
    with open(csv_out, "w") as f:
        f.write("frame_idx,t_s,drops,drop_pct\n")
        for i in range(n_frames):
            f.write(f"{i},{i*rec.dims.framePeriodicity_s:.3f},{drops_per_frame[i]},"
                    f"{drop_pct_per_frame[i]:.4f}\n")

    print(f"\nWrote:")
    print(f"  {summary_path}")
    print(f"  {csv_out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 4))
        t = np.arange(n_frames) * rec.dims.framePeriodicity_s
        ax.plot(t, drop_pct_per_frame, lw=0.6, color="crimson")
        ax.axhline(args.threshold_pct, color="k", ls="--", lw=0.6,
                   label=f"{args.threshold_pct}% threshold")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("packet drop %")
        ax.set_title(f"{rec.name}  drops_total={n_drops_total:,} "
                     f"({drop_pct:.3f}%)  worst_frame={drop_pct_per_frame.max():.2f}%")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        png_out = out_dir / f"{rec.name}_drops.png"
        fig.tight_layout()
        fig.savefig(png_out, dpi=120)
        plt.close(fig)
        print(f"  {png_out}")
    except ImportError:
        print("  (matplotlib not available; skipping PNG)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

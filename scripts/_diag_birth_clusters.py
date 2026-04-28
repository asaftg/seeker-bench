"""Cluster fused_track_born events by world position to estimate
how many DISTINCT physical targets the recording actually saw vs
how many IDs fusion issued.

If the same physical target is being reborn under multiple IDs,
births within ~2° of each other will dominate the histogram.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def main(path: str) -> None:
    births = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("channel") != "events":
                continue
            msg = r.get("msg") or {}
            if msg.get("type") != "fused_track_born":
                continue
            pl = msg.get("payload") or {}
            births.append((float(pl.get("az", 0.0)),
                           float(pl.get("el", 0.0)),
                           int(pl.get("id", 0)),
                           pl.get("primary", "?"),
                           pl.get("class", "?")))
    print(f"total fused_track_born events: {len(births)}")
    # Greedy cluster: for each birth, place into existing cluster if
    # within 2° in az & el of cluster's centroid; otherwise new cluster.
    clusters: list[dict] = []
    for az, el, tid, prim, cls in births:
        best_i, best_d = -1, 9.9
        for i, c in enumerate(clusters):
            daz = az - c["az"] / max(1, c["n"])
            dele = el - c["el"] / max(1, c["n"])
            d = max(abs(daz), abs(dele))
            if d < best_d:
                best_d, best_i = d, i
        if best_i >= 0 and best_d < 2.0:
            c = clusters[best_i]
            c["az"] += az; c["el"] += el; c["n"] += 1
            c["ids"].append(tid)
        else:
            clusters.append({"az": az, "el": el, "n": 1,
                             "ids": [tid], "class": cls, "primary": prim})

    print(f"distinct world clusters (within 2°): {len(clusters)}")
    print(f"\n  {'#':>3}  {'cluster_az':>10} {'cluster_el':>10}  {'n_births':>9}  ids")
    for i, c in enumerate(sorted(clusters, key=lambda x: -x["n"])):
        cx, cy = c["az"]/c["n"], c["el"]/c["n"]
        print(f"  {i:>3}  {cx:>10.2f} {cy:>10.2f}  {c['n']:>9}  "
              f"{c['ids'][:8]}{'...' if len(c['ids'])>8 else ''}")


if __name__ == "__main__":
    main(sys.argv[1])

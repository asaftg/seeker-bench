"""EO image-quality diagnostic — captures one live frame and dumps it
through every processing mode side-by-side, plus a JSON report of the
pixel statistics, so the developer can SEE the actual sensor output and
iterate on processing without asking the operator to take screenshots.

Run::

    python -m scripts.eo_diagnostic

Outputs land in ``scripts/eo_snapshots/diagnostic/`` —

    01_raw_bridge.png        ← exactly what the FX3 bridge handed back
    02_passthrough.png       ← BGR→GRAY→BGR (current default render)
    03_percentile_only.png   ← + percentile [0.5, 99.5] stretch
    04_percentile_gamma.png  ← + gamma 0.85
    05_denoise_percentile.png← median 3×3 → percentile
    06_full_chain.png        ← everything (denoise + percentile + gamma + clahe + unsharp)
    report.json              ← per-stage min/max/mean/std/p1/p99

The point: the developer reads the PNGs directly (the Read tool renders
images), looks at what each stage actually does to a real frame from
THIS camera in THIS lighting, and picks the right defaults. The
operator does not have to be the human eyeball.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Allow `python scripts/eo_diagnostic.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eo.eo_processor import (  # noqa: E402
    enhance,
    passthrough,
)


def _grab_settled_frame(cap, warmup_frames: int = 8):
    """Pull a few frames so the bridge AE has time to settle, return the last."""
    last = None
    for _ in range(warmup_frames):
        ok, fr = cap.read()
        if ok and fr is not None:
            last = fr
        time.sleep(0.05)
    return last


def _stats(img: np.ndarray) -> dict:
    """Per-channel and luma stats so we can SEE if a stage did anything."""
    gray = (
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.ndim == 3
        else img
    )
    flat = gray.reshape(-1)
    p1, p50, p99 = np.percentile(flat, [1, 50, 99]).tolist()
    return {
        "shape": list(img.shape),
        "min": int(flat.min()),
        "max": int(flat.max()),
        "mean": round(float(flat.mean()), 2),
        "std": round(float(flat.std()), 2),
        "p1": round(p1, 2),
        "p50": round(p50, 2),
        "p99": round(p99, 2),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--device", type=int, default=0,
                    help="cv2 device index (default 0 — the IMX568 lives there on this rig)")
    ap.add_argument("--no-camera", action="store_true",
                    help="Skip camera, use a synthetic gradient + noise frame "
                         "instead (lets us verify the processing chain on a "
                         "laptop with no IMX568 attached).")
    args = ap.parse_args()

    out_dir = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.no_camera:
        # 720p gradient + 8% gaussian noise — same shape the bridge would give us.
        h, w = 720, 1280
        col = np.linspace(20, 180, w, dtype=np.float32)
        frame = np.tile(col, (h, 1))
        frame += np.random.normal(0, 12, frame.shape).astype(np.float32)
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        raw_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        source_label = "synthetic gradient+noise (no camera)"
    else:
        # Real IMX568 path — match imx568_capture.py exactly so what we see
        # here is what the live app sees.
        cap = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"FAILED to open device {args.device}. Try --no-camera.", file=sys.stderr)
            return 2
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUY2"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2472)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2064)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # AUTO

        # Side-quest A: probe raw-YUY2 mode separately so the developer
        # can SEE whether CAP_PROP_CONVERT_RGB=0 actually works on this
        # bridge. Save the raw-Y plane (if available) before falling
        # back to the BGR-decoded path.
        try:
            if cap.set(cv2.CAP_PROP_CONVERT_RGB, 0):
                _grab_settled_frame(cap, warmup_frames=4)  # let mode settle
                ok_raw, raw_buf = cap.read()
                if ok_raw and raw_buf is not None:
                    raw_path = out_dir / "00_raw_yuy2_y_plane.png"
                    if raw_buf.ndim == 3 and raw_buf.shape[2] == 2:
                        y_plane = raw_buf[:, :, 0].copy()
                    elif raw_buf.ndim == 2 and raw_buf.shape[1] == 2 * 2472:
                        y_plane = raw_buf[:, 0::2].copy()
                    else:
                        y_plane = None
                    if y_plane is not None:
                        # Match the live downscale.
                        if y_plane.shape[1] > 1236:
                            scale = 1236 / y_plane.shape[1]
                            y_plane = cv2.resize(
                                y_plane,
                                (1236, int(round(y_plane.shape[0] * scale))),
                                interpolation=cv2.INTER_AREA,
                            )
                        cv2.imwrite(str(raw_path), y_plane)
                        print(f"  raw-YUY2 mode WORKED: wrote {raw_path.name} "
                              f"min={y_plane.min()} max={y_plane.max()} "
                              f"mean={y_plane.mean():.1f} std={y_plane.std():.1f}")
                    else:
                        print("  raw-YUY2 mode: bridge accepted flag but "
                              f"shape was {raw_buf.shape} — not the format "
                              "we expected; falling back to BGR")
                else:
                    print("  raw-YUY2 mode: opened but no frames")
            else:
                print("  raw-YUY2 mode: bridge rejected CAP_PROP_CONVERT_RGB=0")
        except Exception as e:
            print(f"  raw-YUY2 mode: probe threw {e!r}")
        finally:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # back to BGR for the main capture

        raw_bgr = _grab_settled_frame(cap)
        cap.release()
        if raw_bgr is None:
            print("Bridge returned no frames. Is the camera plugged in / "
                  "is CameraTool closed?", file=sys.stderr)
            return 3
        source_label = (
            f"IMX568 device={args.device} {raw_bgr.shape[1]}x{raw_bgr.shape[0]}"
        )

        # Match the live pipeline's downscale so the diagnostic shows what
        # the GUI shows, not the raw 2472-wide buffer (which would compress
        # to a tiny 1236-wide JPEG anyway).
        if raw_bgr.shape[1] > 1236:
            scale = 1236 / raw_bgr.shape[1]
            new_w = 1236
            new_h = int(round(raw_bgr.shape[0] * scale))
            raw_bgr = cv2.resize(raw_bgr, (new_w, new_h),
                                 interpolation=cv2.INTER_AREA)

    # Build all six variants from the same raw frame.
    variants = {
        "01_raw_bridge.png": raw_bgr,
        "02_passthrough.png": passthrough(raw_bgr),
        "03_percentile_only.png": enhance(
            raw_bgr, gamma=1.0,  # gamma 1.0 BUT clahe>0 forces stretch on
            clahe_clip=0.0, unsharp_amount=0.0,
            low_pct=0.5, high_pct=99.5,
            denoise_ksize=0,
        ),
        # gamma!=1 forces percentile stretch path:
        "04_percentile_gamma.png": enhance(
            raw_bgr, gamma=0.85,
            clahe_clip=0.0, unsharp_amount=0.0,
            low_pct=0.5, high_pct=99.5,
            denoise_ksize=0,
        ),
        "05_denoise_percentile.png": enhance(
            raw_bgr, gamma=1.0001,  # nudge off 1.0 to trigger stretch
            clahe_clip=0.0, unsharp_amount=0.0,
            low_pct=0.5, high_pct=99.5,
            denoise_ksize=3,
        ),
        "06_full_chain.png": enhance(
            raw_bgr, gamma=0.85,
            clahe_clip=1.5, clahe_grid=8,
            unsharp_amount=0.4, unsharp_radius=1.0,
            low_pct=0.5, high_pct=99.5,
            denoise_ksize=3,
        ),
    }

    # 03 needs special treatment: the enhance() chain skips percentile-stretch
    # if gamma==1 AND clahe==0 AND unsharp==0 (bypass-fast-path). To force a
    # pure percentile-only result, do it inline here.
    from eo.eo_processor import _percentile_stretch, _to_luma  # noqa
    y = _to_luma(raw_bgr)
    y = _percentile_stretch(y, 0.5, 99.5)
    variants["03_percentile_only.png"] = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)

    # Write images
    for name, img in variants.items():
        cv2.imwrite(str(out_dir / name), img)

    # Stats report — one block per variant + the source.
    report = {
        "source": source_label,
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {name: _stats(img) for name, img in variants.items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    print("\nEO diagnostic complete.")
    print(f"Source: {source_label}")
    print(f"Wrote {len(variants)} PNGs + report.json to:")
    print(f"  {out_dir}")
    print("\nstage                          mean   std    p1     p99")
    print("-" * 60)
    for name, st in report["stages"].items():
        print(f"{name:30s} {st['mean']:6.1f} {st['std']:5.1f} "
              f"{st['p1']:5.1f}  {st['p99']:5.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

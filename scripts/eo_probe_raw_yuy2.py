"""Exhaustive probe: which OpenCV combo, if any, hands us raw YUY2 bytes
on the LI-IMX568 FX3 bridge (so we get a clean mono Y plane instead of
DirectShow's destructive YUY2->BGR auto-decode)?

Tries every cell of the matrix:
   backend  ∈ {DSHOW, MSMF}
   flag-order ∈ {flag-before-open-properties,
                 flag-after-fourcc-and-size,
                 flag-after-first-grab}
   re-fourcc-after-flag ∈ {True, False}

For each cell:
   1. open
   2. set props in chosen order
   3. grab one frame
   4. report shape, dtype, channel count, mean per channel, whether the
      frame "smells like raw YUY2" (HxWx2 packed or HxW*2 flat) and what
      the extracted Y plane mean+std look like.

Anything that produces (H, 2*W) or (H, W, 2) is a WIN — we slice the
even bytes and we have the clean Y plane that Leopard sees.

This script does NOT modify imx568_capture.py. It just tells us which
combo works on this specific OpenCV build / FX3 firmware so we can
plumb the right one in next.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

NATIVE_W = 2472
NATIVE_H = 2064

BACKENDS = [
    (cv2.CAP_DSHOW, "DSHOW"),
    (cv2.CAP_MSMF, "MSMF"),
]


def _shape_signature(buf: np.ndarray) -> str:
    if buf is None:
        return "None"
    return f"shape={tuple(buf.shape)} dtype={buf.dtype}"


def _is_raw_yuy2(buf: np.ndarray) -> bool:
    if buf is None:
        return False
    if buf.ndim == 3 and buf.shape[2] == 2 and buf.shape[1] == NATIVE_W:
        return True
    if buf.ndim == 2 and buf.shape[1] == 2 * NATIVE_W and buf.shape[0] == NATIVE_H:
        return True
    return False


def _extract_y(buf: np.ndarray) -> np.ndarray:
    if buf.ndim == 3 and buf.shape[2] == 2:
        return buf[..., 0].copy()
    if buf.ndim == 2:
        return buf[:, 0::2].copy()
    raise ValueError(f"buf shape {buf.shape} not raw YUY2")


def _means(buf: np.ndarray) -> str:
    if buf is None:
        return "n/a"
    if buf.ndim == 3:
        if buf.shape[2] == 3:
            b, g, r = buf[..., 0].mean(), buf[..., 1].mean(), buf[..., 2].mean()
            return f"B={b:.1f} G={g:.1f} R={r:.1f}"
        if buf.shape[2] == 2:
            return f"ch0={buf[..., 0].mean():.1f} ch1={buf[..., 1].mean():.1f}"
    return f"mean={buf.mean():.1f}"


def _try_combo(
    backend_id: int,
    backend_name: str,
    flag_phase: str,
    refourcc: bool,
    device_index: int,
) -> dict:
    label = f"{backend_name}/flag-{flag_phase}/refourcc={refourcc}"
    res = {"label": label, "opened": False, "raw_yuy2": False}

    cap = cv2.VideoCapture(device_index, backend_id)
    if not cap.isOpened():
        res["error"] = "isOpened=False"
        return res
    res["opened"] = True

    fourcc = cv2.VideoWriter_fourcc(*"YUY2")
    try:
        if flag_phase == "before-everything":
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
        elif flag_phase == "after-format":
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        elif flag_phase == "after-grab":
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
            cap.read()
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        else:
            res["error"] = f"unknown flag_phase {flag_phase}"
            cap.release()
            return res

        if refourcc:
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Settle a few frames; some bridges need a moment after a mode change.
        for _ in range(3):
            cap.read()
            time.sleep(0.05)

        ok, buf = cap.read()
        if not ok or buf is None:
            res["error"] = "read returned None"
            cap.release()
            return res

        res["sig"] = _shape_signature(buf)
        res["means"] = _means(buf)
        res["raw_yuy2"] = _is_raw_yuy2(buf)
        if res["raw_yuy2"]:
            y = _extract_y(buf)
            res["y_min"] = int(y.min())
            res["y_max"] = int(y.max())
            res["y_mean"] = float(y.mean())
            res["y_std"] = float(y.std())
            # Save it so we can eyeball.
            out = Path(__file__).resolve().parent / "eo_snapshots" / "diagnostic"
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(
                str(out / f"probe_{backend_name}_{flag_phase}_refourcc{refourcc}.png"),
                y,
            )
    finally:
        cap.release()
    return res


def main() -> int:
    device_index = 0
    print(f"Probing device index {device_index}\n")

    matrix = []
    for backend_id, backend_name in BACKENDS:
        for flag_phase in ("before-everything", "after-format", "after-grab"):
            for refourcc in (False, True):
                matrix.append((backend_id, backend_name, flag_phase, refourcc))

    print(f"{'combo':<55s} {'opened':<6s} {'raw_yuy2':<8s} info")
    print("-" * 120)
    winners = []
    for backend_id, backend_name, flag_phase, refourcc in matrix:
        res = _try_combo(backend_id, backend_name, flag_phase, refourcc, device_index)
        info = res.get("error", "")
        if "sig" in res:
            info = f"{res['sig']}  {res['means']}"
        if res["raw_yuy2"]:
            info += (f"  RAW YUY2 ✓  Y min={res['y_min']} max={res['y_max']} "
                     f"mean={res['y_mean']:.1f} std={res['y_std']:.1f}")
            winners.append(res["label"])
        print(f"{res['label']:<55s} {str(res['opened']):<6s} "
              f"{str(res['raw_yuy2']):<8s} {info}")
        # tiny gap so the bridge doesn't get poked back-to-back
        time.sleep(0.4)

    print("\nWINNERS:" if winners else "\nno winners — every combo gave broken BGR decode")
    for w in winners:
        print(f"  {w}")

    return 0 if winners else 1


if __name__ == "__main__":
    sys.exit(main())

"""One-off: 2-epoch smoke for the v3 fine-tune config.

Captures all output (stdout + stderr) and prints loud markers at every
phase boundary so a silent crash is impossible to misread.

Notes from earlier failure:
    - Windows ultralytics requires the train call live under
      ``if __name__ == '__main__'`` because the dataloader worker spawn
      tries to re-import this module. Without the guard the workers
      raise "An attempt has been made to start a new process before
      the current process has finished its bootstrapping phase".
    - imgsz=960 + cache='disk' wants ~191 GB on the cache drive; only
      ~68 GB was free. Dropped to imgsz=832 which still beats v2's 640
      and fits cache comfortably.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path


def main() -> int:
    print("[SMOKE] starting", flush=True)
    print(f"[SMOKE] cwd={Path('.').resolve()}", flush=True)

    try:
        import torch
        print(f"[SMOKE] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
        from ultralytics import YOLO
        import ultralytics
        print(f"[SMOKE] ultralytics={ultralytics.__version__}", flush=True)
    except Exception as e:
        print(f"[SMOKE] FAIL imports: {e}", flush=True)
        traceback.print_exc()
        return 1

    ROOT = Path(__file__).resolve().parents[1]
    data_yaml = ROOT / "datasets" / "seeker_eo_v2.yaml"
    base_pt = ROOT / "models" / "seeker_eo_v2.pt"
    print(f"[SMOKE] data={data_yaml} exists={data_yaml.exists()}", flush=True)
    print(f"[SMOKE] base={base_pt} exists={base_pt.exists()}", flush=True)

    try:
        model = YOLO(str(base_pt))
        print(f"[SMOKE] model loaded; class names={model.names}", flush=True)
    except Exception as e:
        print(f"[SMOKE] FAIL model load: {e}", flush=True)
        traceback.print_exc()
        return 2

    print("[SMOKE] starting train(epochs=2 imgsz=832 batch=8)", flush=True)
    try:
        model.train(
            data=str(data_yaml), epochs=2, imgsz=832, batch=8,
            device=0, project=str(ROOT/'runs/train'), name='seeker_eo_v3_smoke',
            exist_ok=True, verbose=True, patience=2,
            workers=8, amp=True, cache='disk',
            lr0=0.001, scale=0.9, hsv_s=0.95, mosaic=0.5,
        )
        print("[SMOKE] PASSED", flush=True)
        return 0
    except Exception as e:
        print(f"[SMOKE] FAIL train: {e}", flush=True)
        traceback.print_exc()
        return 3


if __name__ == "__main__":
    sys.exit(main())

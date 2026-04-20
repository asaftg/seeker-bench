#!/usr/bin/env python
"""
Seeker-01 end-to-end smoke test.

Runs a quick sanity check of the full pipeline WITHOUT real hardware:

    fake thermal source -> thermal manager -> frame bus -> sensor bridge -> JSON

Exit code 0 = all checks pass. Non-zero = something is broken.

Usage:
    python scripts/smoke_test.py          # from the project root
    python -m scripts.smoke_test          # if your CWD is seeker_bench
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f" -- {detail}"
        print(msg)


def main() -> int:
    print("=" * 50)
    print(" Seeker-01 Smoke Test")
    print("=" * 50)

    # ── 1. Imports ─────────────────────────────────────
    print("\n1. Core imports...")
    try:
        from common.config import load_config
        from common.frame_bus import BUS
        from common.frames import Topic, ThermalFrame
        from gui.sensor_bridge import thermal_to_wire
        from thermal.thermal_manager import ThermalManager
        check("Core imports", True)
    except Exception as e:
        check("Core imports", False, str(e))
        return 1

    # ── 2. Config ──────────────────────────────────────
    print("\n2. Config...")
    try:
        cfg = load_config()
        check("app_config.yaml loads", True)
        check("thermal section exists", "thermal" in cfg)
        check("heat_detector section exists", "heat_detector" in cfg)
        check("gui section exists", "gui" in cfg)
    except Exception as e:
        check("Config load", False, str(e))

    # ── 3. Thermal manager (fake source) ───────────────
    print("\n3. Thermal pipeline (fake source, 2 seconds)...")
    tm = ThermalManager(use_fake=True, enable_classifier=True)
    tm.start()
    time.sleep(2.5)

    tf = BUS.get_latest(Topic.THERMAL)
    check("Frame published on bus", tf is not None)

    if tf is not None:
        check("Frame is connected", tf.connected)
        check("Frame has display image", tf.agc8 is not None)
        if tf.agc8 is not None:
            h, w = tf.agc8.shape[:2]
            check(f"Display is 640x512 (got {w}x{h})", w == 640 and h == 512)
        check("Frame ID > 0", tf.frame_id > 0, f"got {tf.frame_id}")
        check("hfov_deg is set", tf.hfov_deg > 0)
        check("zoom_preset is set", len(tf.zoom_preset) > 0)

    # ── 4. Zoom presets ────────────────────────────────
    print("\n4. Zoom presets...")
    for preset in ("wide", "mid", "narrow", "full"):
        ok = tm.set_zoom_preset(preset)
        check(f"set_zoom_preset('{preset}')", ok)

    time.sleep(0.5)
    tf2 = BUS.get_latest(Topic.THERMAL)
    if tf2 is not None:
        check("Frame after zoom change", tf2.zoom_preset == "full",
              f"got {tf2.zoom_preset}")

    # ── 5. Sensor bridge serialization ─────────────────
    print("\n5. Sensor bridge (JSON serialization)...")
    if tf is not None:
        try:
            wire = thermal_to_wire(tf)
            check("thermal_to_wire() succeeds", True)
            check("Wire has 'connected'", "connected" in wire)
            check("Wire has 'jpeg_b64'", "jpeg_b64" in wire)
            check("Wire has 'detections'", "detections" in wire)
            check("Wire has 'zoom_preset'", "zoom_preset" in wire)
            # Verify it's JSON-serializable
            json_str = json.dumps(wire)
            check("Wire is JSON-serializable", len(json_str) > 100)
        except Exception as e:
            check("Sensor bridge", False, str(e))

    # ── 6. Classifier / YOLO ───────────────────────────
    print("\n6. Classifier...")
    model_path = ROOT / "models" / "seeker_thermal.pt"
    check("seeker_thermal.pt exists", model_path.exists())
    if model_path.exists():
        try:
            from thermal.drone_classifier import Classifier
            c = Classifier(
                enable_yolo=True,
                trained_model_path=str(model_path),
                conf_threshold=0.4,
                coco_to_target={0: "drone"},
            )
            check("Classifier loads YOLO", c.yolo_active)
        except Exception as e:
            check("Classifier init", False, str(e))

    # ── 7. H/V Classifier (Ticket 1) ──────────────────
    print("\n7. Human+Vehicle classifier (classifier_hv)...")
    try:
        import numpy as np
        from thermal.classifier_hv import HumanVehicleClassifier

        clf_hv = HumanVehicleClassifier(
            model_path="models/seeker_thermal_hv.pt",   # may not exist yet → COCO fallback
            conf_threshold=0.01,                        # very low so COCO fallback can return results
        )
        check("HumanVehicleClassifier instantiates", True)
        check("HumanVehicleClassifier.active", clf_hv.active,
              "ultralytics not available or model load failed")

        if clf_hv.active:
            # Test 1: blank image → should return None (nothing to detect)
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            result_blank = clf_hv.classify(blank)
            check(
                "blank image → None or low-conf result",
                result_blank is None or float(result_blank.get("conf", 0)) < 0.5,
                f"got {result_blank}",
            )

            # Test 2: person-like patch (medium grey rectangle, person shape)
            person_roi = np.full((128, 64, 3), 120, dtype=np.uint8)
            result_person = clf_hv.classify(person_roi)
            # We can't guarantee the label on a synthetic patch, but the call
            # must not raise and must return None or a valid dict
            check(
                "person-like ROI → valid output shape",
                result_person is None or (
                    isinstance(result_person, dict)
                    and "class" in result_person
                    and "conf" in result_person
                ),
                f"got {result_person}",
            )

            # Test 3: vehicle-like patch (wide dark rectangle)
            vehicle_roi = np.full((64, 128, 3), 80, dtype=np.uint8)
            result_vehicle = clf_hv.classify(vehicle_roi)
            check(
                "vehicle-like ROI → valid output shape",
                result_vehicle is None or (
                    isinstance(result_vehicle, dict)
                    and "class" in result_vehicle
                    and "conf" in result_vehicle
                ),
                f"got {result_vehicle}",
            )

    except Exception as e:
        check("HV classifier tests", False, str(e))

    # ── 8. TargetClass enum has PERSON + VEHICLE ────────
    print("\n8. TargetClass enum (Phase B extensions)...")
    try:
        from common.frames import TargetClass
        check("TargetClass.PERSON exists",  hasattr(TargetClass, "PERSON"))
        check("TargetClass.VEHICLE exists", hasattr(TargetClass, "VEHICLE"))
        check("TargetClass.PERSON value",   TargetClass.PERSON  == "person")
        check("TargetClass.VEHICLE value",  TargetClass.VEHICLE == "vehicle")
    except Exception as e:
        check("TargetClass extensions", False, str(e))

    # ── 9. Cleanup ─────────────────────────────────────
    tm.stop()

    # ── Summary ────────────────────────────────────────
    total = PASS + FAIL
    print("\n" + "=" * 50)
    print(f" Results: {PASS}/{total} passed, {FAIL} failed")
    print("=" * 50)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

import json

from common.frames import (
    BBox,
    ClassificationResult,
    TargetClass,
    ThermalDetection,
    ThermalFrame,
    Topic,
)


def test_target_class_json_serializable():
    # TargetClass inherits from str so it JSON-encodes naturally
    payload = {"cls": TargetClass.HAND}
    assert json.dumps(payload) == '{"cls": "hand"}'


def test_bbox_tuple_roundtrip():
    b = BBox(x=1, y=2, w=3, h=4)
    assert b.as_tuple() == (1, 2, 3, 4)


def test_thermal_frame_defaults():
    tf = ThermalFrame(timestamp=1.0, frame_id=0, connected=True)
    assert tf.detections == []
    assert tf.raw16 is None
    assert tf.agc8 is None
    assert tf.zoom_preset == "full"


def test_detection_with_classification():
    det = ThermalDetection(
        bbox=BBox(10, 20, 5, 5),
        area_px=25,
        contrast=1200.0,
        classification=ClassificationResult(
            target_class=TargetClass.HAND,
            confidence=0.92,
            classifier_used="shape_heuristic",
        ),
    )
    assert det.classification.target_class == TargetClass.HAND


def test_topic_constants():
    assert Topic.THERMAL == "thermal"
    assert Topic.RADAR == "radar"

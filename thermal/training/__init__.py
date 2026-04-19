"""Seeker-01 thermal fine-tuning scaffold.

This package contains everything needed to:

    1. Record a labeled-ready dataset from the live thermal feed
       (`record_for_training`)
    2. Define the on-disk YOLO dataset layout with the class list
       pulled from `common.frames.TargetClass` so it can never drift
       (`dataset`)
    3. Train a YOLOv8n variant on that dataset (`train`)
    4. Promote the trained weights into the runtime classifier slot
       (`promote_model`)

Labeling itself is out of scope — use `labelImg` or Roboflow; see
`label_tool_readme.md`.
"""

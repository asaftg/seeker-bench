# Labeling Seeker-01 thermal data

Seeker-01 does not ship an in-app labeler. Use an external tool —
pick one of the two below — and point it at
`datasets/<your-dataset>/images/unlabeled/`.

## Option 1 — labelImg (offline, simple)

```bash
pip install labelImg
labelImg datasets/hand_v1/images/unlabeled
```

1. Menu: `View → Auto Save mode` (on).
2. Menu: `PascalVOC` → switch to `YOLO` (top-left of the toolbar).
3. Draw a box around each target, type the class name. The class
   list comes from `thermal.training.dataset.class_names()`. As of
   this writing it is: `drone`, `hand`, `bird`.
4. labelImg writes `<stem>.txt` next to each `<stem>.png`. When
   you're done, go to step 3 below.

## Option 2 — Roboflow (browser, team-friendly)

1. Create a project, choose "Object Detection".
2. Upload the contents of `images/unlabeled/`.
3. Define the classes **in the same order** as
   `thermal.training.dataset.TRAIN_CLASSES` (drone, hand, bird).
4. Label the images.
5. Export as `YOLOv8` format, download the zip, and unzip the
   `labels/` folder contents back next to the images in
   `images/unlabeled/` so each `.png` has a paired `.txt`.

## Step 3 — Split into train/val

Once every image has a paired `.txt`:

```python
from pathlib import Path
from thermal.training.dataset import DatasetLayout, split_unlabeled

layout = DatasetLayout(root=Path("datasets/hand_v1"))
n_train, n_val = split_unlabeled(layout, val_fraction=0.2)
print(f"Moved {n_train} train / {n_val} val")
```

Then train:

```bash
python -m thermal.training.train --data datasets/hand_v1/data.yaml --epochs 50
python -m thermal.training.promote_model runs/train/seeker_v1/weights/best.pt
```

Restart Seeker-01 — the runtime classifier will pick up
`models/seeker_thermal.pt` automatically.

## Labeling tips

- Draw the bbox tightly around the warm signature, not the object's
  visible silhouette.
- A partially-occluded hand is still a `hand`.
- Sky-background frames with no target need an **empty** `.txt` file
  (0 bytes) — they become negative examples and reduce false alarms.
- Aim for balance: roughly equal counts per class, across lighting /
  background conditions.
- Start small: 100–200 labeled frames per class is enough to see a
  clear improvement over stock COCO.

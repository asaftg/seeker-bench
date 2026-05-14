"""Fine-tune seeker_eo_v3.pt on NightOwls pedestrian data (grayscale, night).

Strategy (per user choice): FREEZE BACKBONE, fine-tune detection head only.
- Vehicle/drone class weights frozen → no regression on those classes.
- Person class head adapts to the night-grayscale distribution.
- ultralytics convention: backbone is layers 0..9, head is 10+ (for yolov8n).
  We freeze [0..10] which spans the backbone + neck connector layers.
"""
import os, datetime as dt
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
BASE_WEIGHTS = Path('C:/Users/asaf.ruf.BLUERIVERTECH/Desktop/Seeker01/seeker_bench/models/seeker_eo_v3.pt')
DATA_YAML = ROOT / 'yolo_dataset' / 'data.yaml'
RUN_NAME = 'seeker_eo_v3_human_finetune'
OUT_DIR = ROOT / 'runs'

assert BASE_WEIGHTS.exists(), f'base weights not found: {BASE_WEIGHTS}'
assert DATA_YAML.exists(),    f'data.yaml not found: {DATA_YAML}'

print(f'base: {BASE_WEIGHTS}')
print(f'data: {DATA_YAML}')

model = YOLO(str(BASE_WEIGHTS))
print(f'classes: {model.names}')

# Freeze backbone — yolov8 head starts at layer ~10. Freeze 0..10 inclusive.
FREEZE_LAYERS = 10

t0 = dt.datetime.now()
print(f'\nstart: {t0.strftime("%H:%M:%S")}')
results = model.train(
    data=str(DATA_YAML),
    epochs=10,
    imgsz=832,                # matches deployed engine input size
    batch=16,                 # A4000 has 17 GB VRAM, comfortable
    device=0,
    project=str(OUT_DIR),
    name=RUN_NAME,
    exist_ok=True,
    freeze=FREEZE_LAYERS,     # freeze backbone

    # Optimization — gentle fine-tune (head only)
    optimizer='AdamW',
    lr0=5e-4,                 # head-only fine-tune → modest LR
    lrf=0.01,                 # cosine decay to lr0*0.01
    warmup_epochs=1.0,
    weight_decay=5e-4,

    # Augmentation — night-NIR distribution match. Grayscale is already
    # done at conversion time (R=G=B). Add modest geometry + intensity aug.
    hsv_h=0.0,                # no hue (grayscale already)
    hsv_s=0.0,                # no saturation
    hsv_v=0.4,                # brightness jitter — simulates AGC variation
    fliplr=0.5,
    flipud=0.0,
    scale=0.4,                # mild scale jitter
    translate=0.1,
    mosaic=0.5,               # half-time mosaic (denser pedestrians)
    close_mosaic=2,
    erasing=0.2,               # random erasing — handles occlusion

    val=True,
    plots=True,
    verbose=True,
    patience=10,
)

t1 = dt.datetime.now()
print(f'\nend: {t1.strftime("%H:%M:%S")}  ({t1-t0})')

best_pt = OUT_DIR / RUN_NAME / 'weights' / 'best.pt'
print(f'\nbest weights: {best_pt}')
if best_pt.exists():
    # Copy as seeker_eo_v4.pt
    target = ROOT / 'seeker_eo_v4.pt'
    import shutil as _sh
    _sh.copy2(best_pt, target)
    print(f'promoted to: {target}')

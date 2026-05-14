"""Convert NightOwls COCO-style JSON annotations to ultralytics YOLO format.

Output:
  yolo_dataset/
    images/train/  *.png
    images/val/    *.png
    labels/train/  *.txt
    labels/val/    *.txt
    data.yaml

Class mapping for our seeker_eo_v3 (3 classes: person/vehicle/drone):
  NightOwls pedestrian (id=1) → 0 (person)
  NightOwls bicycledriver (id=2), motorbikedriver (id=3) → SKIP (not in our schema)
  NightOwls ignore (id=4) → SKIP

Grayscale conversion happens here too — the deployed sensor produces mono
NIR, so we train on grayscale-converted PNGs (still 3-channel for the model
but R=G=B). This is the cheapest distribution-match we can do without
synthetic NIR augmentation.

YOLO label format per .txt: "class cx cy w h" all normalized 0..1.
"""
import json, os, shutil, cv2, random
from collections import defaultdict

ANNS = 'nightowls/validation.json'
IMAGES_DIR = 'nightowls/images'
OUT = 'yolo_dataset'
VAL_FRACTION = 0.10   # 10% for val, 90% for train
RANDOM_SEED = 42
PERSON_CLASS = 0      # matches seeker_eo_v3.pt 0=person

random.seed(RANDOM_SEED)
os.makedirs(f'{OUT}/images/train', exist_ok=True)
os.makedirs(f'{OUT}/images/val',   exist_ok=True)
os.makedirs(f'{OUT}/labels/train', exist_ok=True)
os.makedirs(f'{OUT}/labels/val',   exist_ok=True)

with open(ANNS) as f:
    data = json.load(f)

id_to_img = {im['id']: im for im in data['images']}
img_anns = defaultdict(list)
for a in data['annotations']:
    if a['category_id'] != 1:   # only pedestrians
        continue
    if a.get('ignore'):
        continue
    img_anns[a['image_id']].append(a)

candidate_ids = sorted(img_anns.keys())
random.shuffle(candidate_ids)
n_val = int(len(candidate_ids) * VAL_FRACTION)
val_ids = set(candidate_ids[:n_val])
train_ids = set(candidate_ids[n_val:])
print(f'split: {len(train_ids)} train / {len(val_ids)} val (total {len(candidate_ids)} ped-images)')

# Process each image
n_written = n_missing = 0
for img_id in candidate_ids:
    im = id_to_img[img_id]
    fn = im['file_name']
    src = os.path.join(IMAGES_DIR, fn)
    if not os.path.isfile(src):
        n_missing += 1
        continue
    split = 'val' if img_id in val_ids else 'train'
    # Read image, convert to grayscale-as-3channel
    img = cv2.imread(src)
    if img is None:
        n_missing += 1
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    dst_img = f'{OUT}/images/{split}/{fn}'
    cv2.imwrite(dst_img, gray3)

    W, H = im['width'], im['height']
    # Write YOLO label
    lbl_path = f'{OUT}/labels/{split}/{fn.rsplit(".",1)[0]}.txt'
    with open(lbl_path, 'w') as lf:
        for a in img_anns[img_id]:
            x, y, w, h = a['bbox']  # COCO format: top-left + w+h, pixels
            cx = (x + w/2) / W
            cy = (y + h/2) / H
            nw = w / W
            nh = h / H
            # Clamp
            cx = max(0, min(1, cx)); cy = max(0, min(1, cy))
            nw = max(0, min(1, nw)); nh = max(0, min(1, nh))
            lf.write(f'{PERSON_CLASS} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n')
    n_written += 1

print(f'wrote {n_written} image+label pairs; missing source: {n_missing}')

# Write data.yaml
with open(f'{OUT}/data.yaml', 'w') as f:
    f.write(f"""path: {os.path.abspath(OUT)}
train: images/train
val: images/val

# Keep the SAME class indices as the base model (seeker_eo_v3.pt).
# We only have person annotations in this dataset; vehicle/drone won't
# get supervised here (frozen backbone preserves their existing weights).
nc: 3
names:
  0: person
  1: vehicle
  2: drone
""")
print(f'wrote {OUT}/data.yaml')

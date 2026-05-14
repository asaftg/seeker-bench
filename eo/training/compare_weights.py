"""Compare baseline (seeker_eo_v3.pt) vs fine-tuned (seeker_eo_v4.pt) on the
exact frames we pulled from the 'good vehicles not good enough humans' recording.
"""
import os, cv2, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'seeker_bench'))
from ultralytics import YOLO

BASE   = 'C:/Users/asaf.ruf.BLUERIVERTECH/Desktop/Seeker01/seeker_bench/models/seeker_eo_v3.pt'
TUNED  = str(Path(__file__).resolve().parent / 'seeker_eo_v4.pt')
FRAMES = [
    ('2x', 'C:/Users/asaf.ruf.BLUERIVERTECH/AppData/Local/Temp/pick_2x_05_fid1655.jpg'),
    ('4x', 'C:/Users/asaf.ruf.BLUERIVERTECH/AppData/Local/Temp/pick_4x_05_fid1962.jpg'),
    ('8x', 'C:/Users/asaf.ruf.BLUERIVERTECH/AppData/Local/Temp/pick_8x_00_fid1729.jpg'),
]

def test_one(model_path, label):
    m = YOLO(model_path)
    print(f'\n========== {label}: {model_path} ==========')
    for fz, fp in FRAMES:
        if not os.path.exists(fp):
            print(f'  {fz}: missing {fp}'); continue
        img = cv2.imread(fp)
        if img is None:
            print(f'  {fz}: decode fail'); continue
        # Convert to grayscale-as-3channel to match the fine-tune distribution
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        # Run both at conf=0.01 to see the FULL conf landscape
        r = m.predict(img_gray, conf=0.01, imgsz=832, verbose=False)
        res = r[0]
        n = len(res.boxes) if res.boxes is not None else 0
        if n == 0:
            print(f'  {fz}: 0 dets'); continue
        confs = res.boxes.conf.cpu().numpy()
        clss  = res.boxes.cls.cpu().numpy().astype(int)
        names = [res.names.get(int(c)) for c in clss]
        persons = sorted([float(c) for n_, c in zip(names, confs) if n_ == 'person'], reverse=True)
        vehicles = sorted([float(c) for n_, c in zip(names, confs) if n_ == 'vehicle'], reverse=True)
        print(f'  {fz}: persons={len(persons)} top={[round(c,3) for c in persons[:5]]}  '
              f'vehicles={len(vehicles)} top={[round(c,3) for c in vehicles[:5]]}')

test_one(BASE, 'BASELINE')
test_one(TUNED, 'FINE-TUNED')

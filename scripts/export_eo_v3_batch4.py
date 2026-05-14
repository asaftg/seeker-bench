import sys, os, shutil
sys.path.insert(0, '/home/asaftg/seeker-bench')
from ultralytics import YOLO
FINAL = '/home/asaftg/seeker-bench/models/seeker_eo_v3.engine.batch4'
m = YOLO('/home/asaftg/seeker-bench/models/seeker_eo_v3.pt')
print('starting export imgsz=832 batch=4 half=True device=0', flush=True)
out_path = m.export(format='engine', imgsz=832, batch=4, half=True, device=0, verbose=False)
print(f'EXPORT_OUT={out_path}', flush=True)
if os.path.exists(str(out_path)):
    shutil.move(str(out_path), FINAL)
    print(f'DONE: moved engine to {FINAL}', flush=True)
else:
    print(f'ERROR: expected output not found at {out_path}', flush=True)
    sys.exit(2)

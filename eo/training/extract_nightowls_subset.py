"""Download only the pedestrian-containing images from NightOwls validation zip.
Uses HTTP range requests via remotezip — pulls ~6 GB instead of 50 GB."""
import json, os, sys, time
from remotezip import RemoteZip

URL = 'http://thor.robots.ox.ac.uk/~vgg/data/nightowls/python/nightowls_validation.zip'
ANNS_PATH = 'nightowls/validation.json'
OUT_DIR = 'nightowls/images'

os.makedirs(OUT_DIR, exist_ok=True)

# Load annotations, find images with at least 1 pedestrian (category_id=1)
with open(ANNS_PATH) as f:
    data = json.load(f)
id_to_filename = {im['id']: im['file_name'] for im in data['images']}
ped_image_ids = set(a['image_id'] for a in data['annotations']
                    if a['category_id'] == 1)
target_files = sorted({id_to_filename[i] for i in ped_image_ids
                       if i in id_to_filename})
print(f'targeting {len(target_files)} pedestrian images (~{len(target_files) * 1.0:.0f} MB each estimate)')

# Open remote zip once, fetch each target file
zip_prefix = 'nightowls_validation/'
already = set(os.listdir(OUT_DIR))
remaining = [fn for fn in target_files if fn not in already]
print(f'already on disk: {len(already)}; remaining to download: {len(remaining)}')

t0 = time.time()
n_done = 0
bytes_total = 0

with RemoteZip(URL) as zip_:
    for fn in remaining:
        path_in_zip = zip_prefix + fn
        try:
            data_bytes = zip_.read(path_in_zip)
        except KeyError:
            print(f'  MISS: {fn}')
            continue
        out_path = os.path.join(OUT_DIR, fn)
        with open(out_path, 'wb') as f:
            f.write(data_bytes)
        n_done += 1
        bytes_total += len(data_bytes)
        if n_done % 50 == 0 or n_done == len(remaining):
            elapsed = time.time() - t0
            rate_mb = bytes_total / 1e6 / max(elapsed, 0.1)
            eta_min = (len(remaining) - n_done) / max(n_done/elapsed if elapsed > 0 else 1, 1) / 60
            print(f'  [{n_done}/{len(remaining)}]  {bytes_total/1e6:.0f} MB total  {rate_mb:.1f} MB/s  ETA ~{eta_min:.1f} min')

print(f'\nDone. {n_done} files downloaded, {bytes_total/1e9:.2f} GB total')

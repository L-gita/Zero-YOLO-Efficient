import os
import glob

from PIL import Image

LIB = r"C:\Users\12983\Documents\repro\低照图库"
bad = []
for p in glob.glob(os.path.join(LIB, "**", "*.*"), recursive=True):
    if not p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
        continue
    try:
        with Image.open(p) as im:
            im.load()
    except Exception as e:
        bad.append((os.path.basename(p), os.path.getsize(p), str(e)[:70]))
print(f"严格解码失败 {len(bad)} 个")
for n, s, e in bad:
    print(f"  {n}\t{s}\t{e}")

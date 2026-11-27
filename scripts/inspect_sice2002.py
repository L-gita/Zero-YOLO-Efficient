"""Inspect the 2002-image 'train_data' set found in community repos.
Prints brightness histogram and builds a contact sheet for visual identification."""
import glob
import os
import numpy as np
import cv2

SRC = r"C:\Users\12983\Documents\repro\sice_repo\src\data\train_data"
OUT = r"C:\Users\12983\Documents\repro\compare\train_data_sheet.png"

files = sorted(glob.glob(os.path.join(SRC, "*.jpg")), key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
means, sizes = [], []
for f in files:
    img = cv2.imread(f)
    sizes.append((img.shape[1], img.shape[0]))
    means.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean() / 255.0))
means = np.array(means)
print(f"count={len(files)}")
print(f"brightness mean={means.mean():.3f} median={np.median(means):.3f} "
      f"p10={np.percentile(means,10):.3f} p90={np.percentile(means,90):.3f}")
bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
h, _ = np.histogram(means, bins)
for i in range(len(h)):
    print(f"  {bins[i]:.1f}-{bins[i+1]:.1f}: {h[i]:5d}  {'#' * int(h[i] / max(1, h.max()) * 40)}")
print("sizes (w x h) top-5:", [(f"{w}x{h}", sizes.count((w, h))) for w, h in
                               sorted(set(sizes), key=lambda s: sizes.count(s), reverse=True)[:5]])

# contact sheet: 6x6 sample spanning the brightness range
idx = np.linspace(0, len(files) - 1, 36).astype(int)
cell = 200
sheet = np.zeros((cell * 6, cell * 6, 3), np.uint8)
for k, i in enumerate(idx):
    img = cv2.resize(cv2.imread(files[i]), (cell, cell), interpolation=cv2.INTER_AREA)
    cv2.putText(img, f"{os.path.basename(files[i])} b={means[i]:.2f}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
    sheet[(k // 6) * cell:(k // 6 + 1) * cell, (k % 6) * cell:(k % 6 + 1) * cell] = img
os.makedirs(os.path.dirname(OUT), exist_ok=True)
cv2.imwrite(OUT, sheet)
print("sheet ->", OUT)

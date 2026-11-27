"""Arrange LOL dataset into the folder layout expected by Li-Chongyi/Zero-DCE.

Source : lol_repo/  (clone of 23225114YKW/Low-light-image-enhancement-with-LoL-dataset)
Target : Zero-DCE/Zero-DCE_code/data/{train_data,test_data/...}
         Zero-DCE/Zero-DCE_code/snapshots/Epoch99.pth

Original dataloader only globs "*.jpg", and the repo's released train_data is
jpg too, so we re-encode the PNG pairs to JPEG (quality 95).
"""
import os
import shutil
import sys
from glob import glob

from PIL import Image

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "lol_repo")
CODE = os.path.join(BASE, "Zero-DCE", "Zero-DCE_code")


def convert(src_dir, dst_dir, ext=".jpg", quality=95, limit=None):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(
        f for f in glob(os.path.join(src_dir, "*"))
        if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp")
    )
    if limit:
        files = files[:limit]
    n = 0
    for i, f in enumerate(files):
        dst = os.path.join(dst_dir, os.path.splitext(os.path.basename(f))[0] + ext)
        if os.path.exists(dst):
            n += 1
            continue
        with Image.open(f) as im:
            im = im.convert("RGB")
            im.save(dst, quality=quality)
        n += 1
        if (i + 1) % 50 == 0:
            print(f"  {os.path.basename(dst_dir)}: {i + 1}/{len(files)}", flush=True)
    print(f"{dst_dir}: {n} images", flush=True)
    return n


def main():
    print("BASE =", BASE, flush=True)
    # 1) training low-light images -> data/train_data/*.jpg  (485)
    convert(os.path.join(SRC, "traindata", "low"), os.path.join(CODE, "data", "train_data"))
    # 2) paired normal-light ground truth (only needed for PSNR/SSIM on the train split)
    convert(os.path.join(SRC, "traindata", "high"), os.path.join(CODE, "data", "train_high"))
    # 3) unpaired test folders (LIME 10, DICM 64)
    for name in ("LIME", "DICM"):
        src = os.path.join(SRC, "test_data", name)
        if os.path.isdir(src):
            convert(src, os.path.join(CODE, "data", "test_data", name))
    # 4) pre-trained snapshot
    snap_dir = os.path.join(CODE, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    pth = os.path.join(SRC, "Epoch.pth")
    if os.path.exists(pth):
        shutil.copyfile(pth, os.path.join(snap_dir, "Epoch99.pth"))
        print("snapshot -> snapshots/Epoch99.pth", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())

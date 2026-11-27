r"""为低照图库生成预览联系表 + 亮度统计。

用法：python scripts\make_lightbox.py
输出：低照图库\预览_*.png
"""
import os
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LIB = r"C:\Users\12983\Documents\repro\低照图库"
FONT = r"C:\Windows\Fonts\msyh.ttc"
COLS = 8
TH = 230          # 缩略图高度
PAD = 8
LBL = 26


def sheet(files, title, out):
    n = len(files)
    rows = (n + COLS - 1) // COLS
    # 先用第一张估宽度
    im0 = Image.open(files[0]); im0.thumbnail((10 ** 4, TH))
    cw = im0.width
    head = 44
    W = COLS * (cw + PAD) + PAD
    H = head + rows * (TH + LBL + PAD) + PAD
    canvas = Image.new("RGB", (W, H), (24, 24, 28))
    d = ImageDraw.Draw(canvas)
    d.text((12, 10), title, font=ImageFont.truetype(FONT, 24), fill=(255, 255, 255))
    f = ImageFont.truetype(FONT, 15)
    st = []
    for i, p in enumerate(files):
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im)
        m = arr.mean()
        st.append(m)
        im.thumbnail((10 ** 4, TH), Image.LANCZOS)
        r, c = divmod(i, COLS)
        x = PAD + c * (cw + PAD); y = head + r * (TH + LBL + PAD)
        canvas.paste(im, (x, y))
        d.text((x, y + TH + 3), f"{os.path.basename(p)}  亮度 {m:.0f}", font=f, fill=(160, 230, 170))
    canvas.save(out)
    print(f"{os.path.basename(out)}  {W}x{H}  n={n}  亮度 {np.mean(st):.1f} (min {np.min(st):.0f}, max {np.max(st):.0f})")
    return np.mean(st), np.min(st), np.max(st)


def main():
    for d in sorted(os.listdir(LIB)):
        full = os.path.join(LIB, d)
        if not os.path.isdir(full) or d.startswith("预览") or d in ("试跑结果",):
            continue
        files = sorted(glob.glob(os.path.join(full, "*.*")))
        files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"))]
        if files:
            sheet(files, f"{d}（{len(files)} 张低照原图）", os.path.join(LIB, f"预览_{d}.png"))


if __name__ == "__main__":
    main()

r"""把 demo_pairs 的单张对比图拼成总览，并生成 eval15 数据集的低照/正常光配对联系表。

输出：
  C:\Users\12983\Documents\repro\demo_pairs\overview_eval15_pairs.png     # 15 对低照/GT 缩略图
  C:\Users\12983\Documents\repro\demo_pairs\overview_6cases.png           # 6 个案例的 5 列对比长图
"""
import os
import glob
from PIL import Image, ImageDraw, ImageFont

ROOT = r"C:\Users\12983\Documents\repro"
LOW = os.path.join(ROOT, "lol_test", "eval15", "low")
HIGH = os.path.join(ROOT, "lol_test", "eval15", "high")
OUT = os.path.join(ROOT, "demo_pairs")
FONT = r"C:\Windows\Fonts\msyh.ttc"

PICKS = ["1.png", "111.png", "493.png", "748.png", "23.png", "179.png"]


def contact_sheet():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LOW, "*.png")))
    tw, th = 236, 157
    cols = 5
    rows = (len(names) + cols - 1) // cols
    cell_h = th * 2 + 40
    canvas = Image.new("RGB", (cols * tw, rows * cell_h + 30), (16, 16, 16))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 15)
    d.text((8, 6), "LOL eval15 测试集：每格上=低照输入，下=正常光参考", font=ImageFont.truetype(FONT, 18),
           fill=(240, 240, 240))
    for i, n in enumerate(names):
        r, c = divmod(i, cols)
        x, y = c * tw, 30 + r * cell_h
        lo = Image.open(os.path.join(LOW, n)).convert("RGB").resize((tw - 6, th), Image.LANCZOS)
        hi = Image.open(os.path.join(HIGH, n)).convert("RGB").resize((tw - 6, th), Image.LANCZOS)
        canvas.paste(lo, (x + 3, y + 18))
        canvas.paste(hi, (x + 3, y + 18 + th))
        d.text((x + 4, y + 1), n, font=f, fill=(150, 220, 150))
    p = os.path.join(OUT, "overview_eval15_pairs.png")
    canvas.save(p)
    print("saved", p, canvas.size)


def overview_strips():
    strips = []
    for n in PICKS:
        p = os.path.join(OUT, f"pair_{n}")
        if os.path.exists(p):
            im = Image.open(p)
            w = 1500
            strips.append(im.resize((w, int(im.height * w / im.width)), Image.LANCZOS))
    W = strips[0].width
    H = sum(s.height for s in strips) + 8 * (len(strips) - 1)
    canvas = Image.new("RGB", (W, H), (16, 16, 16))
    y = 0
    for s in strips:
        canvas.paste(s, (0, y))
        y += s.height + 8
    p = os.path.join(OUT, "overview_6cases.png")
    canvas.save(p)
    print("saved", p, canvas.size)


if __name__ == "__main__":
    contact_sheet()
    overview_strips()

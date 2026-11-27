r"""把多张图拼成网格（用于交付总览图）。

用法：python scripts\pack_grid.py out.png 列数 目标宽度 图片1 图片2 ...
"""
import sys
from PIL import Image, ImageDraw, ImageFont

FONT = r"C:\Windows\Fonts\msyh.ttc"


def main():
    out, cols, target_w = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    paths = sys.argv[4:]
    ims = [Image.open(p).convert("RGB") for p in paths]
    cw = target_w // cols
    scaled = []
    for p, im in zip(paths, ims):
        h = max(1, round(im.height * cw / im.width))
        scaled.append((p, im.resize((cw, h), Image.LANCZOS)))
    rowh = [max(s[1].height for s in scaled[r * cols:(r + 1) * cols]) for r in range((len(scaled) + cols - 1) // cols)]
    pad, lbl = 6, 22
    W = cols * (cw + pad) + pad
    H = sum(rowh) + pad * (len(rowh) + 1) + lbl * len(rowh)
    canvas = Image.new("RGB", (W, H), (22, 22, 26))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 16)
    y = pad
    for r in range(len(rowh)):
        row = scaled[r * cols:(r + 1) * cols]
        d.text((pad, y), "", font=f, fill=(255, 255, 255))
        for c, (p, im) in enumerate(row):
            x = pad + c * (cw + pad)
            canvas.paste(im, (x, y))
        y += rowh[r] + lbl
    canvas.save(out)
    print(f"{out}  {canvas.width}x{canvas.height}")


if __name__ == "__main__":
    main()

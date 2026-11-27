"""用"新默认"（保留颗粒：普通暗图 s=0.20 + 不碰色度）批量跑图，并与原版基线对比。

用法：
  python scripts\\grain_default_test.py <图片或文件夹>... [-o DIR] [--rows N] [--zoom-ref 图]

输出：
  <out>\\sheet_<组名>.png    联系表：低照输入 / 原版基线 / 新默认（每行一张图）
  <out>\\metrics.csv
  <out>\\summary.md
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import enhance as EN          # noqa: E402
import preprocess as pp       # noqa: E402
import preprocess_v4 as pv4   # noqa: E402

FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 20)
FONT_T = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)


def chroma_noise(u8):
    c = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    return float(np.mean(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5))))


def luma_std8(u8):
    g = np.asarray(Image.fromarray(u8).convert("L"), dtype=np.float32)
    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    return float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def default_pipeline(net, low):
    """复刻 enhance.py 的新默认：auto + 普通暗图 s=0.20 / 近黑 1.0 + chroma-mode none"""
    m = float(np.mean(np.asarray(Image.fromarray(low).convert("L"), dtype=np.float32)))
    s_eff = 1.0 if m < 5.0 else 0.20
    info = pp.auto_preprocess(low, strength=s_eff)
    e, _ = EN.enhance_image(net, Image.fromarray(info["img"]))
    u8 = np.asarray(e)
    if info["mean"] < 5.0:
        u8 = pp.chroma_guided_smooth(u8, strength=0.5 * s_eff)
        u8 = pp.quantize8(pp.limit_dark_gain(u8, np.asarray(info["img"]), l0=3.0, l1=12.0,
                                             gamma=0.55, strength=0.80 * s_eff,
                                             smooth_radius=8, dither=0.0), dither=0.4)
    else:
        u8 = pp.quantize8(u8.astype(np.float32), dither=0.4)
    return u8


def _label(img, text, font=FONT):
    c = Image.new("RGB", (img.shape[1], img.shape[0] + 28), (26, 26, 26))
    c.paste(Image.fromarray(img), (0, 28))
    ImageDraw.Draw(c).text((8, 3), text, font=font, fill=(238, 238, 238))
    return np.asarray(c)


def make_sheet(items, width=380, group_name=""):
    """items: [(name, low, base, new)]"""
    rows = []
    for name, low, base, new in items:
        h = int(round(width * low.shape[0] / low.shape[1]))
        row = []
        for tag, im in (("低照输入", low), ("原版基线", base), ("新默认（保颗粒）", new)):
            r = cv2.resize(im, (width, h), interpolation=cv2.INTER_AREA)
            row.append(_label(r, f"{name}  {tag}", FONT_SZ))
        rows.append(np.hstack(row))
    head = _label(np.zeros((8, rows[0].shape[1], 3), np.uint8), group_name, FONT_T)
    return np.vstack([head] + rows)


FONT_SZ = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 15)


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                out += [os.path.join(r, f) for f in fs
                        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
        else:
            out.append(p)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "demo_mosaic", "grain_default"))
    ap.add_argument("--group", default="dark")
    ap.add_argument("--sheet-width", type=int, default=360)
    req = ap.parse_args()
    os.makedirs(req.out, exist_ok=True)
    files = collect(req.inputs)
    net = EN.get_net(EN.MODELS["sice"])
    rows, items = [], []
    for f in files:
        low = np.asarray(Image.open(f).convert("RGB"))
        if low.shape[0] * low.shape[1] > 6_000_000:
            low = cv2.resize(low, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        base, _ = EN.enhance_image(net, Image.fromarray(low))
        base = np.asarray(base)
        new = default_pipeline(net, low)
        mb, mn = dict(bright=float(np.mean(np.asarray(Image.fromarray(base).convert("L")))),
                      chroma_noise=chroma_noise(base), luma_std8=luma_std8(base),
                      artificial_edges=pv4.artificial_edges(base)), \
                 dict(bright=float(np.mean(np.asarray(Image.fromarray(new).convert("L")))),
                      chroma_noise=chroma_noise(new), luma_std8=luma_std8(new),
                      artificial_edges=pv4.artificial_edges(new))
        name = os.path.splitext(os.path.basename(f))[0]
        rows.append(dict(file=name, in_bright=float(np.mean(np.asarray(Image.fromarray(low).convert("L")))),
                         base_bright=mb["bright"], new_bright=mn["bright"],
                         base_chroma=mb["chroma_noise"], new_chroma=mn["chroma_noise"],
                         base_std8=mb["luma_std8"], new_std8=mn["luma_std8"],
                         base_edges=mb["artificial_edges"], new_edges=mn["artificial_edges"],
                         psnr_base_new=psnr(base, new), ssim_base_new=float(cv2.matchTemplate(
                             cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32),
                             cv2.cvtColor(new, cv2.COLOR_RGB2GRAY).astype(np.float32),
                             cv2.TM_CCOEFF_NORMED)[0][0]) if base.shape == new.shape else 0.0))
        items.append((name, low, base, new))
    sheet = make_sheet(items, req.sheet_width, f"{req.group}：低照输入 / 原版基线 / 新默认（s=0.20，不碰色度）")
    Image.fromarray(sheet).save(os.path.join(req.out, f"sheet_{req.group}.png"))
    with open(os.path.join(req.out, f"metrics_{req.group}.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    # summary
    def avg(k, tag):
        return sum(r[k] for r in rows) / len(rows)
    lines = [f"# {req.group} 组：原版基线 vs 新默认（保颗粒）", "",
             f"共 {len(rows)} 张。新默认 = auto + 普通暗图 s=0.20 + 不碰色度；近黑图仍走中性抬亮。", "",
             "| 指标 | 原版基线 | 新默认 |", "| --- | --- | --- |",
             f"| 平均亮度 | {avg('base_bright',''):.1f} | {avg('new_bright',''):.1f} |",
             f"| 色噪 | {avg('base_chroma',''):.2f} | {avg('new_chroma',''):.2f} |",
             f"| 细节 std8 | {avg('base_std8',''):.2f} | {avg('new_std8',''):.2f} |",
             f"| 人造边界(/千像素) | {avg('base_edges',''):.1f} | {avg('new_edges',''):.1f} |",
             f"| 两者差异 PSNR | — | {avg('psnr_base_new',''):.2f} dB |", ""]
    with open(os.path.join(req.out, f"summary_{req.group}.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("→", req.out)


if __name__ == "__main__":
    main()

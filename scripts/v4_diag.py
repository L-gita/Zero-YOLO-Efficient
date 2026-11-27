"""v4 vs v4b 诊断：为什么 v4 的分界更明显，v4b 是否修好。

对比 5 个版本：
  原版基线 / 保守默认 0.45 / v4（盒式羽化+盒式平滑） / v4b-gauss（高斯羽化+高斯平滑）
  / v4b-guided（高斯羽化 + 亮度引导滤波平滑）

指标：
  色噪       = 色度高频残差（越低越干净）
  细节 std8  = 8×8 局部亮度 std（越高越保纹理）
  人造边界   = 每千像素"色度跳变≥3 且 亮度跳变<1"的边界数（越低越少色块分界线）
用法：python scripts\\v4_diag.py 图... [-o DIR]
"""
import argparse
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


def chroma_noise(u8):
    c = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    return float(np.mean(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5))))


def luma_std8(u8):
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    return float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))


def run_all(net, low, s_cons=0.45):
    out = {}
    b, _ = EN.enhance_image(net, Image.fromarray(low))
    out["原版基线"] = np.asarray(b)
    info = pp.auto_preprocess(low, strength=s_cons)
    e, _ = EN.enhance_image(net, Image.fromarray(info["img"]))
    e = np.asarray(e)
    if info["mean"] < 5:
        cur = pp.quantize8(np.asarray(pp.limit_dark_gain(
            np.asarray(pp.chroma_guided_smooth(e, strength=0.5 * s_cons), dtype=np.uint8),
            np.asarray(info["img"]), l0=3.0, l1=12.0, gamma=0.55,
            strength=0.80 * s_cons, smooth_radius=8, dither=0.0), dtype=np.float32), dither=0.4)
    else:
        cur = pp.quantize8(np.asarray(pp.chroma_denoise_flat(
            e, low, info["noise"], strength=min(1.2, max(0.0, info["k"] * 0.45 * s_cons)),
            target_amp=None), dtype=np.float32), dither=0.4)
    out["保守默认"] = cur
    # v4 旧版
    out["v4 旧(盒式)"] = pv4.chroma_denoise_v4(e, strength=0.9, tau=5.0, beta=1.6,
                                               guide_radius=16, smooth_radius=3)
    # v4b 两种平滑
    out["v4b(高斯)"] = pv4.chroma_denoise_v4b(e, strength=0.9, tau=5.0, beta=1.6,
                                              sigma_feather=8.0, sigma_chroma=2.0, smoother="gauss")
    out["v4b(引导)"] = pv4.chroma_denoise_v4b(e, strength=0.9, tau=5.0, beta=1.6,
                                              sigma_feather=8.0, sigma_chroma=2.0, smoother="guided")
    out["v4c(只缩不抹)"] = pv4.chroma_desat_noisy(e, strength=1.0, tau=5.0, beta=1.6,
                                                sigma_feather=6.0, max_desat=0.55)
    # v5：完全不碰色度（只做亮度降噪），色度原样保留 = 基线的色度
    out["v5(不碰色度)"] = e.copy()
    # v5d：先不碰色度，再做一次"全局、等比"的轻度去饱和（不产生局部平台）
    ycc = cv2.cvtColor(e.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    ycc[..., 1] = 128 + (ycc[..., 1] - 128) * 0.85
    ycc[..., 2] = 128 + (ycc[..., 2] - 128) * 0.85
    out["v5d(全局×0.85)"] = np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0, 255).astype(np.uint8)
    return out


def sheet(variants, box, zoom=5, crop_pad=8):
    x, y, w, h = box
    tops, bots = [], []
    for name, im in variants.items():
        tops.append(_label(im, name))
        y0, x0 = max(0, y - crop_pad), max(0, x - crop_pad)
        y1, x1 = min(im.shape[0], y + h + crop_pad), min(im.shape[1], x + w + crop_pad)
        z = cv2.resize(im[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        bots.append(_label(z, name + f" ×{zoom}", FONT))
    top = np.hstack(tops)
    bot = np.hstack(bots)
    if bot.shape[1] != top.shape[1]:
        s = top.shape[1] / bot.shape[1]
        bot = cv2.resize(bot, (top.shape[1], int(round(bot.shape[0] * s))), interpolation=cv2.INTER_CUBIC)
    return np.vstack([top, bot])


def _label(img, text, font=None):
    font = font or FONT
    c = Image.new("RGB", (img.shape[1], img.shape[0] + 30), (26, 26, 26))
    c.paste(Image.fromarray(img), (0, 30))
    ImageDraw.Draw(c).text((8, 4), text, font=font, fill=(240, 240, 240))
    return np.asarray(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "demo_mosaic", "v4b"))
    ap.add_argument("--box", default="", help="放大区域 x,y,w,h（默认自动取色噪最重处）")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    net = EN.get_net(EN.MODELS["sice"])
    rows = []
    for f in args.inputs:
        low = np.asarray(Image.open(f).convert("RGB"))
        if low.shape[0] * low.shape[1] > 6_000_000:
            low = cv2.resize(low, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        vs = run_all(net, low)
        base = vs["原版基线"]
        if args.box:
            x, y, w, h = [int(v) for v in args.box.split(",")]
        else:
            c = cv2.cvtColor(base, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
            m = cv2.blur(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5)).max(axis=2), (64, 64))
            _, _, _, (x, y) = cv2.minMaxLoc(m)
            w, h = 96, 72
        name = os.path.splitext(os.path.basename(f))[0]
        Image.fromarray(sheet(vs, (x, y, w, h))).save(os.path.join(args.out, name + "_v4b对比.png"))
        for k, im in vs.items():
            rows.append(dict(file=name, variant=k, bright=float(cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).mean()),
                             chroma_noise=chroma_noise(im), luma_std8=luma_std8(im),
                             artificial_edges=pv4.artificial_edges(im)))
        print(f"[{name}] box=({x},{y},{w},{h})")
        for r in rows[-len(vs):]:
            print(f"   {r['variant']:<12} 色噪 {r['chroma_noise']:5.2f} | 细节 {r['luma_std8']:5.2f} | "
                  f"人造边界 {r['artificial_edges']:6.1f} | 亮度 {r['bright']:5.1f}")
    import csv
    with open(os.path.join(args.out, "v4b_metrics.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader(); wr.writerows(rows)
    print("→", args.out)


if __name__ == "__main__":
    main()

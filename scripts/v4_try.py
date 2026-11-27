"""v4 原型对比：原版基线 / 当前默认（保守 auto 0.45）/ v4（A+B 色度去噪）。

用法：
  python scripts\\v4_try.py 图1 图2 ... [-o DIR]
输出（每张）：
  <名>_v4对比.png   上排 4 列：低照输入 / 原版基线 / 当前默认 / v4
                    下排同区域 4.2× 放大（自动取基线色噪最重的 32×32 块）
  v4_metrics.csv / summary.md
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import enhance as EN              # noqa: E402  (tiled/fp16 降级都在这里)
import preprocess as pp           # noqa: E402
import preprocess_v4 as pv4       # noqa: E402

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
FONT = ImageFont.truetype(FONT_PATH, 22)
FONT_S = ImageFont.truetype(FONT_PATH, 18)


# ---------------------------------------------------------------- 指标
def chroma_noise(u8):
    c = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    return float(np.mean(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5))))


def luma_std8(u8):
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (3, 3))
    m2 = cv2.blur(g * g, (3, 3))
    return float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))


def metrics(u8):
    return dict(bright=float(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).mean()),
                chroma_noise=chroma_noise(u8),
                luma_std8=luma_std8(u8),
                seam=pv4.seam_prominence(u8))


def worst_noise_box(u8, box=48):
    """基线里色噪最重的 box×box 区域"""
    c = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    e = np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5)).max(axis=2)
    m = cv2.blur(e, (box, box))
    _, _, _, (x, y) = cv2.minMaxLoc(m)
    return x, y, box, box


# ---------------------------------------------------------------- 渲染
def label(img, text, font=FONT):
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + 32), (26, 26, 26))
    canvas.paste(Image.fromarray(img), (0, 32))
    ImageDraw.Draw(canvas).text((8, 4), text, font=font, fill=(240, 240, 240))
    return np.asarray(canvas)


def build_sheet(cols, zoom_box, zoom=4.2):
    H = min(c.shape[0] for c, _ in cols)
    top = [label(c[:H], t) for c, t in cols]
    x, y, w, h = zoom_box
    tiles = []
    for (c, t) in cols:
        y0, x0 = max(0, y - 4), max(0, x - 4)
        y1, x1 = min(c.shape[0], y + h + 4), min(c.shape[1], x + w + 4)
        crop = c[y0:y1, x0:x1]
        z = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
        tiles.append(label(z, t + " ×4.2", FONT_S))
    W = min(t.shape[1] for t in tiles)
    tiles = [t[:, :W] for t in tiles]
    top_strip, tile_strip = np.hstack(top), np.hstack(tiles)
    # 两行宽度对齐（下排按上排宽度整体缩放，保持长宽比）
    if tile_strip.shape[1] != top_strip.shape[1]:
        s = top_strip.shape[1] / tile_strip.shape[1]
        tile_strip = cv2.resize(tile_strip, (top_strip.shape[1], int(round(tile_strip.shape[0] * s))),
                                interpolation=cv2.INTER_CUBIC)
    return np.vstack([top_strip, tile_strip])


# ---------------------------------------------------------------- 主流程
def run_v4(net, low_u8, cfg):
    """原版基线之外的两条链路：当前默认（保守 auto）与 v4"""
    # 当前默认
    info = pp.auto_preprocess(low_u8, strength=cfg["cur_strength"])
    e_cur, _ = EN.enhance_image(net, Image.fromarray(info["img"]))
    cur = np.asarray(e_cur)
    if info["mean"] < 5.0:
        cur = pp.quantize8(np.asarray(pp.limit_dark_gain(
            np.asarray(pp.chroma_guided_smooth(cur, strength=0.5 * cfg["cur_strength"]), dtype=np.uint8),
            np.asarray(info["img"]), l0=3.0, l1=12.0, gamma=0.55,
            strength=0.80 * cfg["cur_strength"], smooth_radius=8, dither=0.0), dtype=np.float32),
            dither=0.4)
    else:
        cur = pp.quantize8(np.asarray(pp.chroma_denoise_flat(
            cur, low_u8, info["noise"],
            strength=min(1.2, max(0.0, info["k"] * 0.45 * cfg["cur_strength"])),
            target_amp=None), dtype=np.float32), dither=0.4)

    # v4：预处理沿用"只降亮度噪"的轻处理，色度交给 A+B
    info4 = pp.auto_preprocess(low_u8, strength=cfg["v4_pre"])
    e4, _ = EN.enhance_image(net, Image.fromarray(info4["img"]))
    e4 = np.asarray(e4)
    v4 = pv4.chroma_denoise_v4(e4, strength=cfg["v4_strength"], tau=cfg["tau"],
                               beta=cfg["beta"], guide_radius=cfg["guide"],
                               smooth_radius=cfg["smooth"])
    if info4["mean"] < 5.0:      # 极暗图仍保留"中性抬亮"，否则彩色斑回来
        v4 = pp.quantize8(np.asarray(pp.limit_dark_gain(v4, np.asarray(info4["img"]),
                                                       l0=3.0, l1=12.0, gamma=0.55,
                                                       strength=0.80, smooth_radius=8,
                                                       dither=0.0), dtype=np.float32),
                          dither=0.4)
    else:
        v4 = pp.quantize8(v4.astype(np.float32), dither=0.4)
    return cur, v4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "demo_mosaic", "v4"))
    ap.add_argument("--ckpt", default=EN.MODELS["sice"])
    ap.add_argument("--cur-strength", type=float, default=0.45)
    ap.add_argument("--v4-pre", type=float, default=0.45, help="v4 预处理强度（只影响亮度降噪）")
    ap.add_argument("--v4-strength", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--beta", type=float, default=1.6)
    ap.add_argument("--guide", type=int, default=16)
    ap.add_argument("--smooth", type=int, default=3)
    args = ap.parse_args()
    cfg = dict(cur_strength=args.cur_strength, v4_pre=args.v4_pre,
               v4_strength=args.v4_strength, tau=args.tau, beta=args.beta,
               guide=args.guide, smooth=args.smooth)
    os.makedirs(args.out, exist_ok=True)

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            for r, _, fs in os.walk(p):
                files += [os.path.join(r, f) for f in fs
                          if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))]
        else:
            files.append(p)
    files = sorted(set(files))
    print(f"[模型] {args.ckpt}")
    print(f"[v4] cur_strength={cfg['cur_strength']} v4_pre={cfg['v4_pre']} "
          f"v4_strength={cfg['v4_strength']} tau={cfg['tau']} beta={cfg['beta']} "
          f"guide={cfg['guide']} smooth={cfg['smooth']}")

    net = EN.get_net(args.ckpt)
    rows = []
    for i, f in enumerate(files, 1):
        low = np.asarray(Image.open(f).convert("RGB"))
        w0, h0 = low.shape[1], low.shape[0]
        if w0 * h0 > 6_000_000:
            low = np.asarray(Image.fromarray(low).resize((w0 // 2, h0 // 2), Image.LANCZOS))
        base, _ = EN.enhance_image(net, Image.fromarray(low))
        base = np.asarray(base)
        cur, v4 = run_v4(net, low, cfg)
        mb, mc, mv = metrics(base), metrics(cur), metrics(v4)
        for tag, m in [("原版基线", mb), ("当前默认", mc), ("v4", mv)]:
            rows.append(dict(file=os.path.basename(f), variant=tag, **m))
        sheet = build_sheet([(low, "低照输入"), (base, "原版基线 Zero-DCE"),
                             (cur, f"当前默认 auto {cfg['cur_strength']}"),
                             (v4, "v4（A+B）")],
                            worst_noise_box(base))
        name = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(args.out, name + "_v4对比.png")
        Image.fromarray(sheet).save(out)
        print(f"[{i}/{len(files)}] {os.path.basename(f)}: "
              f"色噪 {mb['chroma_noise']:.2f}/{mc['chroma_noise']:.2f}/{mv['chroma_noise']:.2f} | "
              f"细节 {mb['luma_std8']:.2f}/{mc['luma_std8']:.2f}/{mv['luma_std8']:.2f} | "
              f"分界 {mb['seam']:.1f}/{mc['seam']:.1f}/{mv['seam']:.1f}")
        torch.cuda.empty_cache()

    csv_p = os.path.join(args.out, "v4_metrics.csv")
    with open(csv_p, "w", newline="", encoding="utf-8-sig") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    names = []
    for r in rows:
        if r["file"] not in names:
            names.append(r["file"])
    lines = ["# v4 原型对比（原版基线 / 当前默认 / v4 A+B）", "",
             "指标：色噪=色度高频残差（越低越干净）；细节 std8=8×8 局部亮度标准差（越高越保纹理）；"
             "分界=亮度平坦区色度梯度 p95（越低越少人造边界）", ""]
    for nm in names:
        sub = [r for r in rows if r["file"] == nm]
        lines += [f"## {nm}", "", "| 版本 | 亮度 | 色噪 | 细节 std8 | 分界 |", "| --- | --- | --- | --- | --- |"]
        for r in sub:
            lines.append(f"| {r['variant']} | {r['bright']:.1f} | {r['chroma_noise']:.2f} | "
                         f"{r['luma_std8']:.2f} | {r['seam']:.1f} |")
        lines.append("")
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"\n完成 → {args.out}")


if __name__ == "__main__":
    main()

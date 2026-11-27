"""保守档对比：同一张图，跑 原版基线 / auto 1.0 / auto 0.5 / auto 0.35 / light，
输出「一行多列」的对比长图 + 指标表。

用法：
    python scripts/conservative_sweep.py 图1.png 图2.jpg ... [-o 输出目录]
"""
import argparse
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
import preprocess as pp  # noqa: E402
import enhance as EN  # noqa: E402

FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 22)

VARIANTS = [
    ("原版基线", None),
    ("auto 1.0（现默认）", dict(kind="auto", strength=1.0)),
    ("auto 0.5", dict(kind="auto", strength=0.5)),
    ("auto 0.35", dict(kind="auto", strength=0.35)),
    ("light 轻处理", dict(kind="light")),
]


def chroma_noise(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    c = ycc[..., 1:]
    blur = cv2.GaussianBlur(c, (5, 5), 0)
    return float(np.mean(np.abs(c - blur)))


def luma_std8(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (8, 8))
    m2 = cv2.blur(g * g, (8, 8))
    return float(np.mean(np.sqrt(np.maximum(m2 - m * m, 0))))


def chroma_flat_share(rgb, step=4):
    """色度大片度：把色度量化到 step 级，最大同值连通域占全图比例。"""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.int16)
    q = ((ycc[..., 1] // step) * 1000 + (ycc[..., 2] // step)).astype(np.int32)
    uniq, counts = np.unique(q, return_counts=True)
    return float(counts.max() / q.size)


def boundary_prominence(diff):
    """处理前后差异图的‘边界突兀度’：5×5 内梯度峰值 / 局部均值。"""
    d = diff.astype(np.float32)
    gx = np.abs(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3))
    g = gx + gy
    local = cv2.blur(g, (15, 15)) + 1e-3
    return float(np.mean(g / local))


def run_variant(net, low_u8, cfg):
    if cfg is None:
        out, _ = EN.enhance_image(net, Image.fromarray(low_u8))
        return np.asarray(out).astype(np.uint8)

    if cfg["kind"] == "light":
        pre = pp.preprocess(low_u8, denoise=0.4, deblock_strength=0.0, prelift_amount=0.06)
        raw, _ = EN.enhance_image(net, Image.fromarray(pre))
        u8 = np.asarray(raw).astype(np.uint8)
        u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3, strength=0.3)
        u8 = pp.post_smooth(u8, 0.6)
        return pp.quantize8(u8.astype(np.float32), dither=0.6)

    # auto 模式
    s = cfg["strength"]
    info = pp.auto_preprocess(low_u8, strength=cfg["strength"])
    raw, _ = EN.enhance_image(net, Image.fromarray(info["img"]))
    u8 = np.asarray(raw).astype(np.uint8)
    if info["mean"] < 5.0:
        u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3, strength=0.5 * s)
        u8 = pp.quantize8(pp.limit_dark_gain(u8, low_u8, l0=3.0, l1=12.0, gamma=0.55,
                                             strength=0.80 * s, smooth_radius=8, dither=0.0),
                          dither=0.6)
    else:
        u8 = pp.chroma_denoise_flat(u8, low_u8, info["noise"],
                                    strength=min(1.2, max(0.0, info["k"] * 0.45 * s)),
                                    target_amp=None)
        u8 = pp.quantize8(u8, dither=0.6)
    return u8


def make_strip(low_u8, outs, zoom_center=None, zoom=2.6):
    tiles = []
    for name, o in outs:
        tiles.append((name, o))
    cell_h = 300
    panels = []
    for name, im in tiles:
        h, w = im.shape[:2]
        tw = int(round(cell_h * w / h))
        z = cv2.resize(im, (tw, cell_h), interpolation=cv2.INTER_AREA)
        c = Image.new("RGB", (tw, cell_h + 30), (24, 24, 24))
        c.paste(Image.fromarray(z), (0, 30))
        ImageDraw.Draw(c).text((8, 4), name, font=FONT, fill=(240, 240, 240))
        panels.append(np.asarray(c))
    top = np.hstack(panels)

    if zoom_center is not None:
        cx, cy = zoom_center
        bw, bh = 110, 78
        x0 = int(np.clip(cx - bw // 2, 0, low_u8.shape[1] - bw))
        y0 = int(np.clip(cy - bh // 2, 0, low_u8.shape[0] - bh))
        zp = []
        for name, im in tiles:
            crop = im[y0:y0 + bh, x0:x0 + bw]
            z = cv2.resize(crop, None, fx=zoom * 1.6, fy=zoom * 1.6, interpolation=cv2.INTER_NEAREST)
            c = Image.new("RGB", (z.shape[1], z.shape[0] + 30), (24, 24, 24))
            c.paste(Image.fromarray(z), (0, 30))
            ImageDraw.Draw(c).text((8, 4), f"{name} ×{zoom*1.6:.1f}", font=FONT, fill=(240, 240, 240))
            zp.append(np.asarray(c))
        bottom = np.hstack(zp)
        # 两行总宽可能差几像素（缩放取整）→ 把下排整体缩放到与上排同宽
        if bottom.shape[1] != top.shape[1]:
            newh = max(1, int(round(bottom.shape[0] * top.shape[1] / bottom.shape[1])))
            bottom = cv2.resize(bottom, (top.shape[1], newh), interpolation=cv2.INTER_AREA)
        return np.vstack([top, bottom])
    return top


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    files = []
    for p in args.inputs:
        if os.path.isdir(p):
            for dp, _, fns in os.walk(p):
                for fn in fns:
                    if fn.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")):
                        files.append(os.path.join(dp, fn))
        else:
            files.append(p)
    files = [f for f in files if os.path.isfile(f)]
    if not files:
        raise SystemExit("没有找到图片")

    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(files[0])), "保守档对比")
    os.makedirs(out_dir, exist_ok=True)

    net = EN.get_net(EN.MODELS["sice"])
    rows = []
    for f in files:
        low_u8 = np.asarray(Image.open(f).convert("RGB"))
        outs = []
        for name, cfg in VARIANTS:
            u8 = run_variant(net, low_u8, cfg)
            outs.append((name, u8))
            if cfg is not None:
                diff = np.abs(u8.astype(np.int16) - outs[0][1].astype(np.int16)).max(axis=2)
                rows.append(dict(
                    file=os.path.basename(f), variant=name,
                    bright=float(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).mean()),
                    chroma_noise=chroma_noise(u8),
                    luma_std8=luma_std8(u8),
                    flat_share=chroma_flat_share(u8),
                    boundary=boundary_prominence(diff),
                ))
            else:
                rows.append(dict(
                    file=os.path.basename(f), variant=name,
                    bright=float(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).mean()),
                    chroma_noise=chroma_noise(u8),
                    luma_std8=luma_std8(u8),
                    flat_share=chroma_flat_share(u8),
                    boundary=0.0,
                ))
        # 放大窗口：默认取“基线里色噪最重”的块（16×16 网格里 chroma 噪声最大）
        ycc = cv2.cvtColor(outs[0][1], cv2.COLOR_RGB2YCrCb).astype(np.float32)
        c = ycc[..., 1:]
        nz = np.abs(c - cv2.GaussianBlur(c, (5, 5), 0)).mean(axis=2)
        nz = cv2.blur(nz, (32, 32))
        _, _, _, maxloc = cv2.minMaxLoc(nz)
        strip = make_strip(low_u8, outs, zoom_center=maxloc)
        of = os.path.join(out_dir, os.path.splitext(os.path.basename(f))[0] + "_保守档对比.png")
        Image.fromarray(strip).save(of)
        print(f"[ok] {of}")

    # summary
    import csv
    csv_path = os.path.join(out_dir, "conservative_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    lines = ["# 保守档对比（auto 1.0 / 0.5 / 0.35 / light）", "",
             "指标说明：`chroma_noise` 色度高频残差（越低越干净）；`luma_std8` 8×8 局部亮度标准差（越高细节越多）；"
             "`flat_share` 色度大片度（越低越不容易出现大色块）；`boundary` 处理前后差异的边界突兀度（越低越自然）。", ""]
    files_sorted = sorted({r["file"] for r in rows})
    for fn in files_sorted:
        rs = [r for r in rows if r["file"] == fn]
        lines.append(f"## {fn}")
        lines.append("| 版本 | 亮度 | 色噪 | 细节std8 | 色度大片度 | 边界突兀度 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for r in rs:
            lines.append(f"| {r['variant']} | {r['bright']:.1f} | {r['chroma_noise']:.2f} | "
                         f"{r['luma_std8']:.2f} | {r['flat_share']:.4f} | {r['boundary']:.2f} |")
        lines.append("")
    open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(lines))
    print("[ok]", csv_path)


if __name__ == "__main__":
    main()

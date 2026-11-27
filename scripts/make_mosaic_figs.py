#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""马赛克对照实验的配图：指标柱状图 + 管线视觉对比条带 + 放大细节。

依赖 demo_mosaic/ablation/metrics.csv（由 scripts/mosaic_ablation.py 生成），
会重跑 2 个代表案例的 E0~E5 以生成视觉条带。

用法: python scripts\\make_mosaic_figs.py
"""
import csv
import os
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

ROOT = r"C:\Users\12983\Documents\repro"
ABL = os.path.join(ROOT, "demo_mosaic", "ablation")
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import mosaic_ablation as ma  # noqa: E402

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

PIPES = ["E1", "E2", "E3", "E4", "E5"]
COLORS = ["#6B8EAE", "#C4A35A", "#8FBC8F", "#D08C7A", "#9B7FBF"]
GROUP_LABEL = {"eval15": "eval15", "LIME": "LIME", "extreme": "极端暗图", "DICM": "DICM"}


def group_title(rows, g):
    n = len({r["name"] for r in rows if r["group"] == g})
    extra = "，有 GT" if g == "eval15" else ""
    return f"{GROUP_LABEL.get(g, g)}（{n} 张{extra}）"


def load_csv():
    with open(os.path.join(ABL, "metrics.csv"), encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def avg(rows, group, pipe, key):
    vals = [float(r[key]) for r in rows if r["group"] == group and r["pipeline"] == pipe and r[key] != ""]
    return float(np.mean(vals)) if vals else np.nan


def fig_metrics(rows):
    groups = [g for g in ("eval15", "LIME", "extreme", "DICM") if any(r["group"] == g for r in rows)]
    specs = [("chroma_noise", "色度噪声 std（越小越干净）", None),
             ("blockiness", "块效应比（8×8 网格，越接近 1 越好）", None),
             ("M_gain", "梯度放大 M(输出)/M(输入)", None)]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    x = np.arange(len(groups))
    w = 0.15
    for ax, (key, title, _) in zip(axes, specs):
        for j, (pipe, col) in enumerate(zip(PIPES, COLORS)):
            vals = [avg(rows, g, pipe, key) for g in groups]
            bars = ax.bar(x + (j - 2) * w, vals, w, label=pipe, color=col)
            if key == "blockiness":
                ax.axhline(1.0, color="gray", ls="--", lw=1)
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}" if key != "chroma_noise" else f"{v:.2f}",
                            ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels([group_title(rows, g) for g in groups], fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.suptitle("图 A｜同一模型 + 不同预处理顺序：色噪 / 块效应 / 梯度放大的变化（E1 基线 → E5 本文方案）",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(ABL, "fig_A_pipeline_metrics.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[ok]", out)


def fig_reduction(rows):
    groups = [g for g in ("eval15", "LIME", "extreme", "DICM") if any(r["group"] == g for r in rows)]
    keys = [("chroma_noise", "色度噪声"), ("blockiness", "块效应比"), ("M_gain", "梯度放大")]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(groups))
    w = 0.26
    for j, (key, lab) in enumerate(keys):
        red = []
        for g in groups:
            e1, e5 = avg(rows, g, "E1", key), avg(rows, g, "E5", key)
            red.append((e1 - e5) / e1 * 100.0 if e1 else np.nan)
        bars = ax.bar(x + (j - 1) * w, red, w, label=f"{lab} 下降", color=["#6B8EAE", "#C4A35A", "#8FBC8F"][j])
        for b, v in zip(bars, red):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:+.1f}%", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([group_title(rows, g) for g in groups], fontsize=9)
    ax.set_ylabel("相对 E1 基线的下降比例（%）")
    ax.set_title("图 B｜本文方案（E5）相对基线（E1）的改善幅度：负值=更小更好", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ABL, "fig_B_reduction.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print("[ok]", out)


def _label(img, text, h=30):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT_PATH, 20)
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    d.text((8, 4), text, font=f, fill=(255, 255, 255))
    return im


def _best_zoom_box(rgb, box_w=100, box_h=72):
    """在 E1 输出上找“色噪最重”的区域（用 Cb/Cr 高频残差做积分图）。"""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    energy = np.zeros(rgb.shape[:2], np.float32)
    for c in (1, 2):
        ch = ycc[:, :, c]
        energy += np.abs(ch - cv2.blur(ch, (5, 5)))
    integ = cv2.integral(energy)
    h, w = energy.shape
    best, best_val = (0, 0), -1.0
    for y in range(0, max(1, h - box_h), 16):
        for x in range(0, max(1, w - box_w), 16):
            s = integ[y + box_h, x + box_w] - integ[y, x + box_w] - integ[y + box_h, x] + integ[y, x]
            if s > best_val:
                best_val, best = s, (x, y)
    x, y = best
    return x, y, min(x + box_w, w), min(y + box_h, h)


def fig_strips(cases):
    for group, name, low_path in cases:
        rgb = np.asarray(Image.open(low_path).convert("RGB"))
        outs = {"E0": rgb}
        for p in PIPES:
            outs[p] = ma.run_pipeline(p, rgb)
        box = _best_zoom_box(outs["E1"])
        h, w = rgb.shape[:2]
        cw, ch = box[2] - box[0], box[3] - box[1]
        scale = max(2, int(360 / max(cw, ch)))
        top, bottom = [], []
        short = {"E0": "低照原图", "E1": "基线 Zero-DCE", "E2": "先降噪→增强",
                 "E3": "增强→后降噪", "E4": "弱降噪+平滑", "E5": "本文方案"}
        for p in ["E0"] + PIPES:
            t = Image.fromarray(outs[p]).resize((260, int(260 * h / w)), Image.LANCZOS)
            top.append(_label(np.asarray(t), f"{p} {short[p]}"))
            crop = Image.fromarray(outs[p][box[1]:box[3], box[0]:box[2]]).resize(
                (cw * scale, ch * scale), Image.NEAREST)
            bottom.append(_label(np.asarray(crop), f"{p} 放大 {scale}×"))
        W = sum(i.width for i in top) + 8 * (len(top) - 1)
        sheet = Image.new("RGB", (W, max(i.height for i in top) + max(i.height for i in bottom) + 8), (18, 18, 18))
        x = 0
        for i in top:
            sheet.paste(i, (x, 0))
            x += i.width + 8
        x = 0
        for i in bottom:
            sheet.paste(i, (x, max(t.height for t in top) + 8))
            x += i.width + 8
        out = os.path.join(ABL, f"strip_{group}_{name}.png")
        sheet.save(out)
        print("[ok]", out)


def main():
    rows = load_csv()
    fig_metrics(rows)
    fig_reduction(rows)
    cases = [
        ("extreme", "126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
        ("eval15", "748", os.path.join(ROOT, "lol_test", "eval15", "low", "748.png")),
        ("LIME", "1", os.path.join(ROOT, "低照图库", "02_LIME", "1.bmp")),
    ]
    fig_strips([c for c in cases if os.path.exists(c[2])])


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AB 对比：输入一张（或一批）**原始低照图**，同时用两个模型增强，输出对比与指标。

  A = 原版 Zero-DCE（不带边缘损失训练）
  B = 改进版（训练时加论文 (4.9)~(4.12) 的 Sobel 边缘损失）

用法：
    python scripts\\ab_enhance_compare.py 低照图.jpg
    python scripts\\ab_enhance_compare.py data\\test_data\\LIME -o 结果\\我的AB
    python scripts\\ab_enhance_compare.py 照片文件夹 --base xxx.pth --edge yyy.pth

输出（默认写到第一个输入旁边的 `AB_对比_<时间>`，或 -o 指定）：
    <名字>_原版增强.png   只用原版 Zero-DCE 的增强结果（可直接当日常成片看）
    <名字>_edge增强.png   加边缘损失的改进版增强结果（可直接当日常成片看）
    <名字>_AB对比.png     每张图：原图 | 原版增强 | 改进版增强（下排同区域放大 2×）
    <名字>_指标.png       每张图：亮度 / 梯度幅值 M / L_edge / 梯度一致性 柱状对比
    overview_AB.png       全部图的总览联系表
    metrics.csv           逐图数值
    summary.md            均值表 + 结论口径（可直接贴报告）

指标口径与 pair_compare.py / 论文一致：
    L_edge(vs 原图) = (1/(3N))·Σ_c Σ_i |M(I_low^c)_i − M(I_enh^c)_i|   ← 论文 (4.12)，越小越好
    cos / corr     = 增强图与**原图**的梯度方向一致性 / 幅值相关，越大越好
    PSNR/SSIM(A,B) = 两个模型的增强结果之间的一致性（越接近 1 说明改动越小）
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 让 `import model` 找到 Zero-DCE_code 下的 model.py
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
import model as dce_model  # noqa: E402

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
C_LOW, C_BASE, C_EDGE = "#9D9D9D", "#4C78A8", "#E45756"


# --------------------------------------------------------------------------- 模型
def get_net(ckpt):
    net = dce_model.enhance_net_nopool()
    try:
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
    except TypeError:  # 老版本 torch 没有 weights_only 参数
        state = torch.load(ckpt, map_location="cpu")
    net.load_state_dict(state)
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def run_net(net, arr):
    """arr: HxWx3 float32 [0,1] -> float32 HxWx3 [0,1]"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        _, out, _ = net(t)
    out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return out


# --------------------------------------------------------------------------- 指标
def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a, b):
    try:
        import cv2
    except ImportError:
        return float("nan")
    a = a.astype(np.float64) * 255.0
    b = b.astype(np.float64) * 255.0
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ga, gb = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    gaa = cv2.GaussianBlur(a * a, (11, 11), 1.5)
    gbb = cv2.GaussianBlur(b * b, (11, 11), 1.5)
    gab = cv2.GaussianBlur(a * b, (11, 11), 1.5)
    va, vb, cab = gaa - ga * ga, gbb - gb * gb, gab - ga * gb
    m = ((2 * ga * gb + C1) * (2 * cab + C2)) / ((ga ** 2 + gb ** 2 + C1) * (va + vb + C2))
    return float(m.mean())


def sobel(x):
    """论文 (4.9) Sobel，(4.11) 幅值 |Ix|+|Iy|。x: HxWx3 float [0,1]。"""
    try:
        import cv2
    except ImportError:
        raise SystemExit("需要 opencv-python：pip install opencv-python")
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gx = np.zeros_like(x, dtype=np.float32)
    gy = np.zeros_like(x, dtype=np.float32)
    for c in range(3):
        gx[:, :, c] = cv2.filter2D(x[:, :, c], cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
        gy[:, :, c] = cv2.filter2D(x[:, :, c], cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
    return gx, gy, np.abs(gx) + np.abs(gy)


def measure(low, enh):
    """low/enh: HxWx3 float [0,1] → 与"原图"对比的一组指标。"""
    gxl, gyl, ml = sobel(low)
    gxe, gye, me = sobel(enh)
    num = gxl * gxe + gyl * gye
    na, nb = np.sqrt(gxl ** 2 + gyl ** 2), np.sqrt(gxe ** 2 + gye ** 2)
    cos_global = float(num.sum() / (math.sqrt(float((na ** 2).sum()) * float((nb ** 2).sum())) + 1e-12))
    ok = (na > 1e-8) & (nb > 1e-8)
    cos_pixel = float(np.mean(num[ok] / (na[ok] * nb[ok]))) if ok.any() else float("nan")
    a, b = ml.ravel().astype(np.float64), me.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    den = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    corr = float((a * b).sum() / den) if den > 1e-12 else float("nan")
    return {
        "bright": float(enh.mean() * 255.0),
        "M": float(me.mean()),
        "M_low": float(ml.mean()),
        "L_edge": float(np.abs(ml - me).mean()),   # = 1/(3N)·Σ_c Σ_i |M_low − M_enh|
        "cos_global": cos_global,
        "cos_pixel": cos_pixel,
        "corr": corr,
    }


# --------------------------------------------------------------------------- 画图
def _font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def _fit(im, w=480):
    h = int(round(im.height * w / im.width))
    return im.resize((w, h), Image.LANCZOS)


def _zoom_box(ml, frac=0.34):
    """在梯度最强的位置取一个窗口（用于放大对比）。ml: HxW（已按通道合并）。"""
    h, w = ml.shape
    win = max(48, int(min(h, w) * frac))
    integral = np.cumsum(np.cumsum(ml, axis=0), axis=1)
    pad = np.zeros((h + 1, w + 1), dtype=np.float64)
    pad[1:, 1:] = integral
    best, box = -1.0, (0, 0, win, win)
    step = max(8, win // 6)
    for y in range(0, h - win + 1, step):
        for x in range(0, w - win + 1, step):
            s = pad[y + win, x + win] - pad[y, x + win] - pad[y + win, x] + pad[y, x]
            if s > best:
                best, box = s, (x, y, x + win, y + win)
    return box


def draw_pair_figure(low_pil, base_pil, edge_pil, name, out_png, tag="原图"):
    """上排：原图 | 原版增强 | 改进版增强；下排：同区域放大 2×。"""
    w = 520
    imgs = [_fit(p, w) for p in (low_pil, base_pil, edge_pil)]
    ml = sobel(np.asarray(low_pil, np.float32) / 255.0)[2].sum(axis=2)   # 三通道幅值相加 → HxW
    x1, y1, x2, y2 = _zoom_box(ml)
    zoom = []
    for p in (low_pil, base_pil, edge_pil):
        c = p.crop((x1, y1, x2, y2))
        zoom.append(c.resize((c.width * 2, c.height * 2), Image.LANCZOS))
    zw = min(int(w * 1.55), max(z.width for z in zoom))
    zoom = [_fit(z, zw) for z in zoom]

    labels = [f"低照输入", "原版 Zero-DCE", "改进版（+边缘损失）"]
    bar = 34
    W = w * 3 + 40
    H = bar + imgs[0].height + bar + zoom[0].height + 20
    sheet = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(sheet)
    f = _font(19)
    fs = _font(15)
    for i, (im, lab) in enumerate(zip(imgs, labels)):
        sheet.paste(im, (10 + i * (w + 10), bar))
        d.text((14 + i * (w + 10), 8), lab, fill="black", font=f)
    zb = bar + imgs[0].height + bar
    for i, z in enumerate(zoom):
        sheet.paste(z, (10 + i * (zw + 10), zb))
        d.text((14 + i * (zw + 10), bar + imgs[0].height + 6), f"同区域放大 2×", fill="gray", font=fs)
    d.text((10, H - 16), name, fill="gray", font=fs)
    sheet.save(out_png, format="PNG")
    return out_png


def draw_metric_figure(low, base, edge, name, out_png):
    m_low = measure(low, low)
    m_base = measure(low, base)
    m_edge = measure(low, edge)
    keys = [("bright", "亮度 (0-255)"), ("M", "平均梯度幅值 M"), ("L_edge", "L_edge (4.12)\n越小越\"贴\"原图"),
            ("cos_global", "梯度方向 cos"), ("corr", "梯度幅值 corr")]
    fig, axes = plt.subplots(1, len(keys), figsize=(15.5, 3.6))
    for ax, (k, title) in zip(axes, keys):
        vals = [m_base[k], m_edge[k]]
        bars = ax.bar(["原版", "改进版"], vals, color=[C_BASE, C_EDGE], width=0.55)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}" if v < 10 else f"{v:.2f}",
                    ha="center", va="bottom", fontsize=11)
        lo = min(vals)
        hi = max(vals)
        pad = (hi - lo) * 0.35 if hi > lo else abs(hi) * 0.05 + 1e-6
        ax.set_ylim(max(0, lo - pad), hi + pad)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"{name}｜原版 vs 改进版（都与低照原图比，纵轴截断以放大差异）   "
                 f"PSNR(原版,改进)={psnr(to_u8(base), to_u8(edge)):.2f} dB", fontsize=12)
    fig.subplots_adjust(top=0.80, bottom=0.14, wspace=0.32)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png, m_base, m_edge, m_low


def to_u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 主流程
def collect(paths):
    files = []
    for p in paths:
        p = os.path.abspath(p.strip('"'))
        if os.path.isdir(p):
            for ext in EXTS:
                files += glob.glob(os.path.join(p, "**", "*" + ext), recursive=True)
                files += glob.glob(os.path.join(p, "**", "*" + ext.upper()), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[跳过] 找不到: {p}")
    out, seen = [], set()
    for f in sorted(files):
        if os.path.basename(f).startswith(".") or f in seen:
            continue
        seen.add(f)
        out.append(f)
    return out


def main():
    ap = argparse.ArgumentParser(description="原版 vs 改进版 Zero-DCE 一键 AB 对比（输入低照原图）")
    ap.add_argument("inputs", nargs="+", help="低照图文件或文件夹（可多个）")
    ap.add_argument("-o", "--out", default=None, help="输出目录")
    ap.add_argument("--base", default=os.path.join(CODE, "snapshots_sice", "Epoch49.pth"),
                    help="A：原版权重（默认 Epoch49：与改进版权重同为 50 轮 / 12,500 iter，公平对比）")
    ap.add_argument("--edge", default=os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth"), help="B：改进版权重")
    ap.add_argument("--limit", type=int, default=0, help="最多处理几张（0=全部）")
    ap.add_argument("--no-fig", action="store_true", help="只存增强图与 CSV，不画对比图")
    args = ap.parse_args()

    files = collect(args.inputs)
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit("没有找到图片")

    if args.out:
        outdir = os.path.abspath(args.out)
    else:
        first = files[0]
        base_dir = first if os.path.isdir(first) else os.path.dirname(first)
        outdir = os.path.join(base_dir, "AB_对比_" + time.strftime("%Y%m%d_%H%M"))
    os.makedirs(outdir, exist_ok=True)

    print(f"[模型] A(原版)  = {args.base}")
    print(f"[模型] B(改进)  = {args.edge}")
    net_a, net_b = get_net(args.base), get_net(args.edge)
    if not os.path.exists(os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth")):
        print("[warn] 改进版权重不存在，请先训练或改 --edge")

    rows = []
    strips = []
    for i, f in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(f))[0]
        im = Image.open(f)
        try:
            from PIL import ImageOps

            im = ImageOps.exif_transpose(im)
        except Exception:
            pass
        low_pil = im.convert("RGB")
        low = np.asarray(low_pil, np.float32) / 255.0

        base = run_net(net_a, low)
        edge = run_net(net_b, low)
        base_pil = Image.fromarray(to_u8(base))
        edge_pil = Image.fromarray(to_u8(edge))

        m_base = measure(low, base)
        m_edge = measure(low, edge)
        row = {
            "file": os.path.relpath(f, os.path.commonpath(files) if os.path.isdir(os.path.commonpath(files)) else os.path.dirname(f)),
            "size": f"{low_pil.width}x{low_pil.height}",
            "bright_in": float(low.mean() * 255.0),
            "bright_base": m_base["bright"],
            "bright_edge": m_edge["bright"],
            "M_in": m_base["M_low"],
            "M_base": m_base["M"],
            "M_edge": m_edge["M"],
            "Ledge_base": m_base["L_edge"],
            "Ledge_edge": m_edge["L_edge"],
            "cos_base": m_base["cos_global"],
            "cos_edge": m_edge["cos_global"],
            "corr_base": m_base["corr"],
            "corr_edge": m_edge["corr"],
            "psnr_base_edge": psnr(to_u8(base), to_u8(edge)),
            "ssim_base_edge": ssim(base, edge),
        }
        rows.append(row)
        print(f"[{i}/{len(files)}] {name}  亮度 {row['bright_in']:.1f}→{row['bright_base']:.1f}/{row['bright_edge']:.1f}"
              f"  L_edge {row['Ledge_base']:.4f}→{row['Ledge_edge']:.4f}")

        if not args.no_fig:
            p1 = os.path.join(outdir, name + "_AB对比.png")
            p2 = os.path.join(outdir, name + "_指标.png")
            draw_pair_figure(low_pil, base_pil, edge_pil, name, p1)
            draw_metric_figure(low, base, edge, name, p2)
            strips.append(p1)

        try:
            base_pil.save(os.path.join(outdir, name + "_原版增强.png"))
            edge_pil.save(os.path.join(outdir, name + "_edge增强.png"))
        except Exception as e:
            print(f"[warn] 保存增强图失败: {e}")

    # ---------- CSV + 汇总
    csv_path = os.path.join(outdir, "metrics.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        wr = csv.DictWriter(fp, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)

    def mean(k):
        vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not math.isnan(r[k])]
        return float(np.mean(vals)) if vals else float("nan")

    n = len(rows)
    mb = mean("bright_base")
    me = mean("bright_edge")
    lb = mean("Ledge_base")
    le = mean("Ledge_edge")
    imp = (lb - le) / lb * 100.0 if lb else float("nan")
    lines = [
        "# 原版 vs 改进版 Zero-DCE —— AB 对比汇总",
        "",
        f"- 输入：{n} 张低照原图",
        f"- A 原版权重：`{args.base}`",
        f"- B 改进版权重：`{args.edge}`",
        f"- 输出目录：`{outdir}`",
        "",
        "| 指标（与低照原图比） | 原版 Zero-DCE | 改进版（+边缘损失） | 变化 |",
        "| --- | --- | --- | --- |",
        f"| 平均亮度 | {mb:.2f} | {me:.2f} | {(me - mb) / mb * 100:+.1f}% |",
        f"| 平均梯度幅值 M | {mean('M_base'):.4f} | {mean('M_edge'):.4f} | "
        f"{(mean('M_edge') - mean('M_base')) / mean('M_base') * 100:+.1f}% |",
        f"| L_edge (4.12) ↓ | {lb:.4f} | {le:.4f} | {imp:+.1f}% |",
        f"| 梯度方向 cos ↑ | {mean('cos_base'):.4f} | {mean('cos_edge'):.4f} | "
        f"{(mean('cos_edge') - mean('cos_base')) * 100:+.2f}% |",
        f"| 梯度幅值 corr ↑ | {mean('corr_base'):.4f} | {mean('corr_edge'):.4f} | "
        f"{(mean('corr_edge') - mean('corr_base')) * 100:+.2f}% |",
        "",
        f"- 两个模型输出之间：PSNR {mean('psnr_base_edge'):.2f} dB、SSIM {mean('ssim_base_edge'):.4f}"
        f"（差异不大说明改进是「温和」的；越小说明两模型差异越大）",
        "",
        "## 怎么读",
        "- **L_edge 下降** = 改进版增强后的梯度场更贴近原图（论文 (4.12) 的直接目标）。",
        "- **cos / corr 上升** = 增强图的结构方向、幅值结构更像原图（边缘没被哄抬/抹平）。",
        "- **亮度 / M 下降** = 代价：因为 LE 曲线本来会把暗区梯度放大，边缘损失在往回压，所以整体会略暗、略平。",
        "- 每张图直接可用的成片：`*_原版增强.png`（原版 Zero-DCE）与 `*_edge增强.png`（加边缘损失）。",
        "- 每张图的 `*_AB对比.png`：上排原图|原版|改进，下排同区域放大 2×（看轮廓/噪点）。",
        "- 每张图的 `*_指标.png`：五项指标柱状对比。",
    ]
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    # ---------- 联系表
    if strips:
        try:
            thumbs = [_fit(Image.open(p).convert("RGB"), 1100) for p in strips]
            W = max(t.width for t in thumbs) + 20
            H = sum(t.height + 8 for t in thumbs) + 10
            sheet = Image.new("RGB", (W, H), "white")
            y = 5
            for t in thumbs:
                sheet.paste(t, (10, y))
                y += t.height + 8
            sheet.save(os.path.join(outdir, "overview_AB.png"), format="PNG")
        except Exception as e:
            print(f"[warn] 总览拼图失败: {e}")

    print(f"\n[完成] {n} 张 → {outdir}")
    print(f"  汇总：{os.path.join(outdir, 'summary.md')}")
    print(f"  L_edge 平均 {lb:.4f} → {le:.4f}（{imp:+.1f}%）  亮度 {mb:.1f} → {me:.1f}")


if __name__ == "__main__":
    main()

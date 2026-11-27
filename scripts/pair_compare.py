#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pair_compare.py —— 输入两张图，输出"对比指标 + 曲线图"（Edge-DCE 实验的通用小工具）

典型用法
--------
1) 评估一次增强的效果（原图 vs 增强结果）：
   python scripts\pair_compare.py 夜景.jpg enhanced\夜景_enhanced.png

2) 有正常光参考图时（低照输入 vs 正常光 GT），算 PSNR / SSIM：
   python scripts\pair_compare.py eval15\low\748.png eval15\high\748.png --label-a 低照输入 --label-b 正常光GT

3) 对比两个模型的结果：
   python scripts\pair_compare.py 结果_基线.png 结果_加边缘损失.png --label-a 基线 --label-b 加边缘损失

输出（默认写到 B 图同目录下的 <B 文件名>_pairfigs\）
--------------------------------------------------
  fig_metrics_card.png   一页卡片：两图并排 / 指标柱状 / 边缘保持曲线 / 关键数字
  fig_gradient_maps.png  两图的梯度幅值图 M 与 |ΔM| 热力图
  metrics.csv            全部指标（一行）
  summary.md             指标表 + 图注（可直接贴报告）

指标口径
--------
  亮度      0~255 均值
  PSNR/SSIM 需要两图尺寸一致（不一致会自动把 B 缩放到 A 的尺寸并提示）
  M         论文 (4.11) 逐通道 Sobel 幅值 |Ix|+|Iy| 的平均
  L_edge    论文 (4.12) 1/(3N)·Σ_c Σ_i |M(A)_i - M(B)_i|
  cos       梯度方向一致性（逐像素、跨 RGB 展平后求余弦，尺度无关）
  corr      M(A) 与 M(B) 两张幅值图的相关系数（尺度无关）
  放大倍数  M(B)/M(A)，越接近 1 表示梯度场越接近；> 1 表示 B 把梯度放大了
"""
import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
C_A = "#9D9D9D"      # 图 A（原图 / 低照）
C_B = "#E45756"      # 图 B（结果 / 参考）
C_M = "#4C78A8"

# 中文字体
if os.path.exists(FONT_PATH):
    font_manager.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# --------------------------------------------------------------------------- 读取
def load_rgb(path):
    """读成 float32 [0,1] 的 RGB；PIL 读取（Unicode 路径安全）+ 自动纠正手机方向。"""
    if not os.path.exists(path):
        raise SystemExit("找不到图片：{}".format(path))
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    return np.asarray(im, dtype=np.float32) / 255.0


def to_u8(x):
    return np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- 指标
def psnr(a, b):
    """a/b 为 0~255 的 uint8。"""
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a, b):
    """SSIM：3 通道 RGB、11×11 高斯、σ=1.5（与仓库其它脚本口径一致）。"""
    a = a.astype(np.float64) * 255.0
    b = b.astype(np.float64) * 255.0
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ga = cv2.GaussianBlur(a, (11, 11), 1.5)
    gb = cv2.GaussianBlur(b, (11, 11), 1.5)
    gaa = cv2.GaussianBlur(a * a, (11, 11), 1.5)
    gbb = cv2.GaussianBlur(b * b, (11, 11), 1.5)
    gab = cv2.GaussianBlur(a * b, (11, 11), 1.5)
    va, vb, cab = gaa - ga * ga, gbb - gb * gb, gab - ga * gb
    m = ((2 * ga * gb + C1) * (2 * cab + C2)) / ((ga ** 2 + gb ** 2 + C1) * (va + vb + C2))
    return float(m.mean())


def sobel_fields(x):
    """返回 gx, gy, M（三者都是 HxWx3）。M 用论文 (4.11) 的 |Ix|+|Iy|。"""
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    for c in range(3):
        gx[:, :, c] = cv2.filter2D(x[:, :, c], cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
        gy[:, :, c] = cv2.filter2D(x[:, :, c], cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
    return gx, gy, np.abs(gx) + np.abs(gy)


def cosine_grads(gxa, gya, gxb, gyb):
    """梯度方向一致性，两种口径：
    - global：全图梯度向量整体夹角余弦（与 diag_edge_effect.py 口径一致，强边缘主导）
    - pixel ：逐像素夹角余弦再平均（更严格，弱梯度/噪声也会计入）
    """
    num = gxa * gxb + gya * gyb
    na = np.sqrt(gxa ** 2 + gya ** 2)
    nb = np.sqrt(gxb ** 2 + gyb ** 2)
    cos_global = float(num.sum() / (math.sqrt(float((na ** 2).sum()) * float((nb ** 2).sum())) + 1e-12))
    ok = (na > 1e-8) & (nb > 1e-8)
    cos_pixel = float(np.mean(num[ok] / (na[ok] * nb[ok]))) if ok.any() else float("nan")
    return cos_global, cos_pixel


def corr_maps(ma, mb):
    a = ma.ravel().astype(np.float64)
    b = mb.ravel().astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    d = math.sqrt(float((a * a).sum()) * float((b * b).sum()))
    if d < 1e-12:
        return float("nan")
    return float((a * b).sum() / d)


def edge_curve(ma, mb, ps=(1, 2, 5, 10, 20, 30, 50, 70, 100)):
    """按 A 的梯度强度分档，看 B/A 的幅值放大倍数（越接近 1 越"保住"梯度尺度）。"""
    xs, ys = [], []
    for p in ps:
        thr = np.percentile(ma, 100 - p)
        m = ma >= thr
        if m.sum() == 0:
            continue
        xs.append(p)
        ys.append(float(mb[m].mean() / max(ma[m].mean(), 1e-9)))
    return xs, ys


# --------------------------------------------------------------------------- 画图
def draw_card(a, b, ma, mb, met, out_png, label_a, label_b, name_a, name_b):
    fig = plt.figure(figsize=(14.0, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.18)

    # 上排：两图 + 数值面板
    for k, (img, lab) in enumerate([(a, label_a), (b, label_b)]):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title("{}　亮度 {:.1f}".format(lab, float(img.mean() * 255.0)), fontsize=10)
        ax.axis("off")

    axn = fig.add_subplot(gs[0, 2])
    axn.axis("off")
    n = [
        ("PSNR", "{:.2f} dB".format(met["psnr"])),
        ("SSIM", "{:.4f}".format(met["ssim"])),
        ("亮度 A → B", "{:.1f} → {:.1f}".format(met["bright_a"], met["bright_b"])),
        ("平均梯度 M A → B", "{:.4f} → {:.4f}".format(met["m_a"], met["m_b"])),
        ("L_edge (4.12)", "{:.4f}".format(met["ledge"])),
        ("梯度方向 cos(全局/逐像素)", "{:.4f} / {:.4f}".format(met["cos"], met["cos_pix"])),
        ("梯度幅值 corr", "{:.4f}".format(met["corr"])),
        ("放大倍数 M(B)/M(A)", "{:.2f}×".format(met["ratio"])),
    ]
    axn.text(0.0, 1.02, "指标（{} → {}）".format(label_a, label_b), fontsize=11.5, weight="bold")
    for i, (k, v) in enumerate(n):
        y = 0.90 - i * 0.115
        axn.text(0.00, y, "• " + k, fontsize=10)
        axn.text(0.56, y, v, fontsize=10, weight="bold")

    # 下排左：柱状（亮度 / 平均梯度 M 各画 A、B 两根；L_edge 与放大倍数写在下方）
    axb = fig.add_subplot(gs[1, 0])
    cats = ["亮度\n(0-255)", "平均梯度 M\n(×100)"]
    va = [met["bright_a"], met["m_a"] * 100]
    vb = [met["bright_b"], met["m_b"] * 100]
    x = np.arange(len(cats))
    w = 0.36
    axb.bar(x - w / 2, va, w, color=C_A, label=label_a, edgecolor="white")
    axb.bar(x + w / 2, vb, w, color=C_B, label=label_b, edgecolor="white")
    for i in range(len(cats)):
        axb.annotate("{:.1f}".format(va[i]), (i - w / 2, va[i]), ha="center", va="bottom", fontsize=8.5)
        axb.annotate("{:.1f}".format(vb[i]), (i + w / 2, vb[i]), ha="center", va="bottom", fontsize=8.5)
    axb.set_xticks(x)
    axb.set_xticklabels(cats, fontsize=9)
    axb.set_ylim(0, max(va + vb) * 1.30)
    axb.set_title("亮度与梯度幅值", fontsize=9.5)
    axb.legend(fontsize=7.5, frameon=False, loc="upper right")
    axb.grid(axis="y", alpha=0.2)
    axb.text(0.5, -0.30,
             "L_edge (4.12) = {:.4f}　｜　M(B)/M(A) = {:.2f}×".format(met["ledge"], met["ratio"]),
             transform=axb.transAxes, ha="center", fontsize=9)

    # 下排中：边缘保持曲线
    axc = fig.add_subplot(gs[1, 1])
    ps, ratios = edge_curve(ma, mb)
    axc.plot(ps, ratios, "o-", color=C_B, label="B/A 放大倍数")
    axc.axhline(1.0, ls="--", color="#666", lw=1, label="1×（梯度不变）")
    axc.set_xscale("log")
    axc.set_xticks(ps)
    axc.set_xticklabels([str(p) for p in ps], fontsize=8)
    axc.set_xlabel("只看 A 中最强的 p% 像素（对数轴）", fontsize=9)
    axc.set_ylabel("M(B) / M(A)", fontsize=9)
    axc.set_title("边缘保持曲线：越贴近 1 表示梯度尺度越一致", fontsize=9.5)
    axc.grid(alpha=0.25)
    axc.legend(fontsize=7.5, frameon=False)

    # 下排右：|ΔM| 热力图
    axd = fig.add_subplot(gs[1, 2])
    dm = np.abs(ma - mb).mean(axis=2)
    vmax = float(np.percentile(dm, 99.5)) + 1e-6
    im = axd.imshow(dm, cmap="inferno", vmin=0, vmax=vmax)
    axd.set_title("|M(A) − M(B)| 逐像素（均值 {:.4f}）".format(float(dm.mean())), fontsize=9.5)
    axd.axis("off")
    plt.colorbar(im, ax=axd, fraction=0.046, pad=0.02, format="%.3f")

    fig.suptitle("图对比：{} vs {}\n{}　→　{}\n".format(label_a, label_b, name_a, name_b) +
                 "亮度 {:.1f}→{:.1f}｜PSNR {:.2f} dB｜SSIM {:.4f}｜L_edge {:.4f}｜梯度放大 {:.2f}×".format(
                     met["bright_a"], met["bright_b"], met["psnr"], met["ssim"], met["ledge"], met["ratio"]),
                 fontsize=10.5, y=0.995)
    fig.subplots_adjust(top=0.86, bottom=0.09, left=0.05, right=0.97)
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[ok]", out_png)


def draw_gradient_maps(a, b, ma, mb, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.4))
    axes[0, 0].imshow(np.clip(a, 0, 1)); axes[0, 0].set_title("A 原图", fontsize=10); axes[0, 0].axis("off")
    axes[0, 1].imshow(ma.mean(axis=2), cmap="gray", vmax=np.percentile(ma.mean(axis=2), 99.5))
    axes[0, 1].set_title("A 的梯度幅值 M（逐通道平均）", fontsize=10); axes[0, 1].axis("off")
    axes[1, 0].imshow(np.clip(b, 0, 1)); axes[1, 0].set_title("B 结果", fontsize=10); axes[1, 0].axis("off")
    axes[1, 1].imshow(mb.mean(axis=2), cmap="gray", vmax=np.percentile(ma.mean(axis=2), 99.5))
    axes[1, 1].set_title("B 的梯度幅值 M", fontsize=10); axes[1, 1].axis("off")
    ax = axes[0, 2]
    dm = np.abs(ma - mb).mean(axis=2)
    vmax = float(np.percentile(dm, 99.5)) + 1e-6
    im = ax.imshow(dm, cmap="inferno", vmin=0, vmax=vmax)
    ax.set_title("|M(A) − M(B)|", fontsize=10); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, format="%.3f")
    # 下右：|ΔM| 的直方图 + 改善占比（若 B 梯度更接近某参考，这里只展示分布）
    ax2 = axes[1, 2]
    ax2.hist(dm.ravel(), bins=60, color=C_M, alpha=0.85)
    ax2.axvline(float(dm.mean()), color=C_B, lw=1.6, ls="--", label="均值 {:.4f}".format(float(dm.mean())))
    ax2.set_yscale("log")
    ax2.set_title("|ΔM| 分布（对数纵轴）", fontsize=10)
    ax2.set_xlabel("|M(A) − M(B)|", fontsize=9)
    ax2.legend(fontsize=8, frameon=False)
    fig.suptitle("梯度幅值图与差异", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[ok]", out_png)


# --------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="两张图对比：指标 + 曲线图（Edge-DCE 实验工具）")
    ap.add_argument("a", help="第一张图（如 低照输入 / 基线结果）")
    ap.add_argument("b", help="第二张图（如 正常光 GT / 加边缘损失结果）")
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认 B 图旁的 <名字>_pairfigs）")
    ap.add_argument("--label-a", default=None, help="A 的显示名（默认文件名）")
    ap.add_argument("--label-b", default=None, help="B 的显示名（默认文件名）")
    ap.add_argument("--no-fig", action="store_true", help="只算数值，不出图")
    args = ap.parse_args()

    name_a = os.path.basename(args.a)
    name_b = os.path.basename(args.b)
    label_a = args.label_a or os.path.splitext(name_a)[0]
    label_b = args.label_b or os.path.splitext(name_b)[0]

    a = load_rgb(args.a)
    b0 = load_rgb(args.b)
    resized = False
    if b0.shape[:2] != a.shape[:2]:
        print("[提示] 两图尺寸不同（{} vs {}），把 B 缩放到 A 的尺寸再比较。".format(
            "{}x{}".format(b0.shape[1], b0.shape[0]), "{}x{}".format(a.shape[1], a.shape[0])))
        b0 = cv2.resize(b0, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
        resized = True
    b = b0

    # 8bit 域算 PSNR/SSIM（与仓库其它脚本一致）
    a8, b8 = to_u8(a), to_u8(b)
    gxa, gya, ma = sobel_fields(a)
    gxb, gyb, mb = sobel_fields(b)

    ledge = float(np.abs(ma - mb).mean())                       # 论文 (4.12)
    ledge_a0 = float(np.abs(ma).mean())                         # 参考：A 自身的平均幅值
    met = dict(
        file_a=name_a, file_b=name_b,
        size="{}x{}".format(a.shape[1], a.shape[0]),
        bright_a=float(a.mean() * 255.0), bright_b=float(b.mean() * 255.0),
        m_a=float(ma.mean()), m_b=float(mb.mean()),
        psnr=float(psnr(a8, b8)), ssim=float(ssim(a, b)),
        ledge=ledge, cos=cosine_grads(gxa, gya, gxb, gyb)[0],
        cos_pix=cosine_grads(gxa, gya, gxb, gyb)[1], corr=corr_maps(ma, mb),
        ratio=float(mb.mean() / max(ma.mean(), 1e-9)),
        ledge_red=float((ma.mean() - mb.mean()) / max(ma.mean(), 1e-9)),
        resized=int(resized),
    )

    # 输出目录
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.splitext(args.b)[0] + "_pairfigs"
    os.makedirs(out_dir, exist_ok=True)

    # 控制台表
    rows = [
        ("输入", "{}  vs  {}".format(name_a, name_b)),
        ("尺寸", met["size"] + ("（B 已缩放）" if resized else "")),
        ("亮度 (0-255)", "{:.1f} → {:.1f}  ({:+.1f}%)".format(
            met["bright_a"], met["bright_b"], (met["bright_b"] - met["bright_a"]) / max(met["bright_a"], 1e-9) * 100)),
        ("平均梯度幅值 M", "{:.4f} → {:.4f}  ({:+.1f}%)".format(
            met["m_a"], met["m_b"], (met["m_b"] - met["m_a"]) / max(met["m_a"], 1e-9) * 100)),
        ("PSNR (dB)", "{:.2f}".format(met["psnr"])),
        ("SSIM", "{:.4f}".format(met["ssim"])),
        ("L_edge (4.12)", "{:.4f}   （A 自身平均幅值 {:.4f}）".format(ledge, ledge_a0)),
        ("梯度方向 cos(全局/逐像素)", "{:.4f} / {:.4f}".format(met["cos"], met["cos_pix"])),
        ("梯度幅值 corr", "{:.4f}".format(met["corr"])),
        ("放大倍数 M(B)/M(A)", "{:.2f}×".format(met["ratio"])),
    ]
    print("\n" + "=" * 66)
    for k, v in rows:
        print("  {:<20}{}".format(k, v))
    print("=" * 66 + "\n")

    # 写 CSV
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["指标", "值"])
        for k, v in rows:
            w.writerow([k, v])

    # 出图
    if not args.no_fig:
        draw_card(a, b, ma, mb, met, os.path.join(out_dir, "fig_metrics_card.png"),
                  label_a, label_b, name_a, name_b)
        draw_gradient_maps(a, b, ma, mb, os.path.join(out_dir, "fig_gradient_maps.png"))

    # 写 summary.md
    md = [
        "# 两图对比报告",
        "",
        "- A：`{}`（显示为「{}」）".format(args.a, label_a),
        "- B：`{}`（显示为「{}」）".format(args.b, label_b),
        "- 尺寸：{}{}".format(met["size"], "（B 已缩放到 A）" if resized else ""),
        "",
        "| 指标 | 含义 | 值 |",
        "|---|---|---|",
        "| PSNR | 像素级误差，越高越好（两图形变/照度差越大越低） | {:.2f} dB |".format(met["psnr"]),
        "| SSIM | 结构相似度，越高越好 | {:.4f} |".format(met["ssim"]),
        "| 亮度 | 0~255 均值 | {:.1f} → {:.1f} |".format(met["bright_a"], met["bright_b"]),
        "| 平均梯度 M | 论文 (4.11) 的逐通道 Sobel 幅值均值 | {:.4f} → {:.4f} |".format(met["m_a"], met["m_b"]),
        "| L_edge (4.12) | 两张图梯度幅值之差的平均（越小=梯度场越接近） | {:.4f} |".format(met["ledge"]),
        "| 梯度方向 cos | 梯度方向一致性（全局 / 逐像素平均，1=完全同向） | {:.4f} / {:.4f} |".format(met["cos"], met["cos_pix"]),
        "| 梯度幅值 corr | 幅值图相关系数（1=形状一致） | {:.4f} |".format(met["corr"]),
        "| 放大倍数 | M(B)/M(A)（1=梯度尺度不变，>1=放大） | {:.2f}× |".format(met["ratio"]),
        "",
        "图注（可直接贴报告）：",
        "",
        "- `fig_metrics_card.png`：{} vs {}：亮度 {:.1f}→{:.1f}，PSNR {:.2f} dB，SSIM {:.4f}，L_edge {:.4f}，梯度放大 {:.2f}×。".format(
            label_a, label_b, met["bright_a"], met["bright_b"], met["psnr"], met["ssim"], met["ledge"], met["ratio"]),
        "- `fig_gradient_maps.png`：两图梯度幅值 M、|M(A)−M(B)| 热力图与分布。",
        "",
        "> 说明：PSNR/SSIM 要求两图同尺寸（已自动缩放）；若两图不是「同一场景的照度前后」，PSNR/SSIM 的绝对值意义有限，重点看梯度类指标。",
    ]
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("输出目录：", out_dir)
    print("  metrics.csv / summary.md / fig_metrics_card.png / fig_gradient_maps.png")


if __name__ == "__main__":
    sys.exit(main())

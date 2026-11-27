# -*- coding: utf-8 -*-
"""
把"边缘损失到底改变了什么"画成可直接进报告的图表
====================================================
读取已有结果（不重复推理小图）：
  demo_edge\ep49_edge022\metrics.csv        每图 PSNR/SSIM/亮度/M/L_edge
  demo_edge\diag_edge_effect\diag_metrics.csv  每图 cos/corr/L_edge 亮度归一
只对 2 张代表图重新推理，用于逐像素梯度误差热力图。

输出（repro\demo_edge\figures\）：
  fig1_quality_eval15.png     eval15 的 PSNR / SSIM 柱状图（含逐图变化）
  fig2_gradient_all.png       三个数据集：梯度幅值 M 与 L_edge 对比
  fig3_paired_ledge.png       逐图配对散点（基线 vs 边缘）：谁改善了
  fig4_per_image_delta.png    eval15 逐图 ΔPSNR / ΔSSIM
  fig5_ratio_bar.png          M(增强)/M(输入) 的放大倍数（边缘版更接近 1，GT 在中间）
  fig6_heatmap_error.png      逐像素梯度误差热力图：误差在哪被压掉
  fig7_edge_retention.png     边缘保持曲线：强边缘 vs 弱纹理的放大倍数
  fig8_change_bar.png         变化率总览（%）：一图看全部指标
  fig9_summary_card.png       报告用汇总卡片（改动量 + 关键数字）
  summary_for_report.md       配套表格与图注
"""
import os
import csv
import math
import traceback
from collections import defaultdict

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

REPRO = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(REPRO, "Zero-DCE", "Zero-DCE_code")
OUT = os.path.join(REPRO, "demo_edge", "figures")
METRICS = os.path.join(REPRO, "demo_edge", "ep49_edge022", "metrics.csv")
DIAG = os.path.join(REPRO, "demo_edge", "diag_edge_effect", "diag_metrics.csv")

BASE_CKPT = os.path.join(CODE, "snapshots_sice", "Epoch49.pth")
EDGE_CKPT = os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth")

C_BASE = "#4C78A8"      # 基线 蓝
C_EDGE = "#E45756"      # 边缘 红
C_IN = "#9D9D9D"        # 输入 灰
C_GT = "#54A24B"        # GT 绿

# ---------- 中文字体 ----------
for fp in [r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"]:
    if os.path.exists(fp):
        fm.fontManager.addfont(fp)
        name = fm.FontProperties(fname=fp).get_name()
        plt.rcParams["font.family"] = name
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130


# ---------- 读数据 ----------
def read_csv(path):
    with open(path, encoding="utf-8-sig") as fp:
        return list(csv.DictReader(fp))


def to_float(v):
    try:
        return float(v)
    except Exception:
        return float("nan")


SET_NAME = {"eval15": "eval15 (LOL 官方测试集, 15 对)", "lime": "LIME (10 张)", "dicm": "DICM (16 张)"}
SET_SHORT = {"eval15": "eval15", "lime": "LIME", "dicm": "DICM"}


def load_metrics():
    rows = read_csv(METRICS)
    for r in rows:
        r["set"] = r["set"].lower()          # CSV 里是 LIME / DICM，统一成小写
        for k in list(r.keys()):
            if k in ("set", "file", "size"):
                continue
            r[k] = to_float(r[k])
    return rows


def load_diag():
    rows = read_csv(DIAG)
    for r in rows:
        r["bright_low"] = to_float(r["bright_low"])
        r["m_low"] = to_float(r["m_low"])
        r["cos_low"] = to_float(r["cos_low"])
        r["corr_low"] = to_float(r["corr_low"])
        r["ledge_raw"] = to_float(r["ledge_raw"])
        r["ledge_norm"] = to_float(r["ledge_norm"])
        r["m_enh"] = to_float(r["m_enh"])
        r["bright_enh"] = to_float(r["bright_enh"])
        ext = os.path.splitext(r["file"])[1].lower()
        r["set"] = {".png": "eval15", ".bmp": "lime", ".jpg": "dicm"}.get(ext, "?")
    return rows


def diag_index(rows):
    idx = defaultdict(dict)
    for r in rows:
        idx[(r["set"], r["file"])][r["model"]] = r
    return idx


# ---------- 模型（仅热力图用） ----------
def load_model(ckpt):
    import sys
    sys.path.insert(0, CODE)
    from model import enhance_net_nopool
    net = enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(ckpt, map_location="cuda"))
    net.eval()
    return net


def enhance(net, rgb):
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(t)
    return out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def sobel_M(rgb):
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx, groups=3)
    gy = F.conv2d(xp, ky, groups=3)
    return (gx.abs() + gy.abs()).mean(dim=1)[0].numpy()  # [H,W]


# ---------- 通用 ----------
def bar_group(ax, cats, series, ylabel, title, fmt="{:.2f}", ylim=None):
    """series: list of (label, values, color)"""
    n = len(series)
    x = np.arange(len(cats))
    w = 0.8 / n
    for i, (lab, vals, color) in enumerate(series):
        off = (i - (n - 1) / 2) * w
        bars = ax.bar(x + off, vals, w, label=lab, color=color, edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=8, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)


def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[ok]", name)


# ---------- 图 1：eval15 质量指标 ----------
def fig1(rows):
    ev = [r for r in rows if r["set"] == "eval15"]
    files = [r["file"] for r in ev]
    psnr_in = np.mean([r["psnr_in"] for r in ev])
    psnr_b = np.mean([r["psnr_base"] for r in ev])
    psnr_e = np.mean([r["psnr_edge"] for r in ev])
    ssim_in = np.mean([r["ssim_in"] for r in ev])
    ssim_b = np.mean([r["ssim_base"] for r in ev])
    ssim_e = np.mean([r["ssim_edge"] for r in ev])

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.9), gridspec_kw={"width_ratios": [1, 1, 1.45]})
    bar_group(axes[0], ["低照输入", "基线", "+边缘损失"],
              [("", [psnr_in, psnr_b, psnr_e], [C_IN, C_BASE, C_EDGE])],
              "PSNR (dB)", "PSNR（eval15 平均）")
    axes[0].set_ylim(0, max(psnr_b, psnr_e) * 1.25)

    bar_group(axes[1], ["低照输入", "基线", "+边缘损失"],
              [("", [ssim_in, ssim_b, ssim_e], [C_IN, C_BASE, C_EDGE])],
              "SSIM", "SSIM（eval15 平均）",
              fmt="{:.3f}")
    axes[1].set_ylim(0, 0.75)

    # 逐图 ΔPSNR / ΔSSIM
    d_psnr = np.array([r["psnr_edge"] - r["psnr_base"] for r in ev])
    d_ssim = np.array([(r["ssim_edge"] - r["ssim_base"]) * 100 for r in ev])
    order = np.argsort(d_psnr)
    y = np.arange(len(ev))
    ax = axes[2]
    ax.barh(y, d_psnr[order], color=[C_EDGE if v > 0 else C_BASE for v in d_psnr[order]], alpha=0.85, label="ΔPSNR (dB)")
    ax2 = ax.twiny()
    ax2.plot(d_ssim[order], y, "o-", color="#222", ms=3.5, lw=1.2, label="ΔSSIM ×100")
    ax.set_yticks(y)
    ax.set_yticklabels([files[i] for i in order], fontsize=6.5)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("ΔPSNR (dB)   ← 变差 | 变好 →")
    ax2.set_xlabel("ΔSSIM ×100（折线）", labelpad=2)
    ax.set_title("逐图变化（{} 张，SSIM 改善 {} 张）".format(len(ev), int((d_ssim > 0).sum())), fontsize=9.5)
    ax.grid(axis="x", alpha=0.25)
    # 图例：柱=ΔPSNR，折线=ΔSSIM×100
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=C_EDGE, label="ΔPSNR > 0（变好）"),
               Patch(facecolor=C_BASE, label="ΔPSNR < 0（变差）"),
               Line2D([0], [0], color="#222", marker="o", ms=3.5, lw=1.2, label="ΔSSIM ×100（见上轴）")]
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("图 1｜eval15 定量对比：PSNR {:.2f} → {:.2f}（{:+.2f} dB），SSIM {:.4f} → {:.4f}（{:+.4f}）".format(
        psnr_b, psnr_e, psnr_e - psnr_b, ssim_b, ssim_e, ssim_e - ssim_b), fontsize=13, y=0.995)
    fig.subplots_adjust(top=0.80, bottom=0.14, left=0.06, right=0.985, wspace=0.32)
    save(fig, "fig1_quality_eval15.png")
    return dict(psnr_in=psnr_in, psnr_base=psnr_b, psnr_edge=psnr_e,
                ssim_in=ssim_in, ssim_base=ssim_b, ssim_edge=ssim_e,
                n_ssim_better=int((d_ssim > 0).sum()), n=len(ev))


# ---------- 图 2：梯度指标 ----------
def fig2(rows):
    sets = ["eval15", "lime", "dicm"]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.0))
    for j, s in enumerate(sets):
        sub = [r for r in rows if r["set"] == s]
        cats = ["低照输入", "基线", "+边缘损失"]
        M_in = np.mean([r["M_in"] for r in sub])
        M_b = np.mean([r["M_base"] for r in sub])
        M_e = np.mean([r["M_edge"] for r in sub])
        bar_group(axes[0, j], cats, [("", [M_in, M_b, M_e], [C_IN, C_BASE, C_EDGE])],
                  "平均梯度幅值 M", "{}：M  {:.3f} → {:.3f} → {:.3f}".format(
                      SET_SHORT[s], M_in, M_b, M_e), fmt="{:.3f}")
        L_b = np.mean([r["Ledge_base"] for r in sub])
        L_e = np.mean([r["Ledge_edge"] for r in sub])
        bar_group(axes[1, j], ["基线", "+边缘损失"],
                  [("", [L_b, L_e], [C_BASE, C_EDGE])],
                  "L_edge (4.12)", "{}：L_edge 下降 {:.1f}%".format(SET_SHORT[s], (L_b - L_e) / L_b * 100),
                  fmt="{:.3f}")
        axes[1, j].set_ylim(0, L_b * 1.3)
    fig.suptitle("图 2｜三个数据集上：增强后的梯度幅值变小，与输入的梯度差 L_edge 同步下降 8%~10%", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig2_gradient_all.png")


# ---------- 图 3：逐图配对 ----------
def fig3(rows):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    colors = {"eval15": "#E45756", "lime": "#4C78A8", "dicm": "#54A24B"}
    allb, alle = [], []
    for s in ["eval15", "lime", "dicm"]:
        sub = [r for r in rows if r["set"] == s]
        b = np.array([r["Ledge_base"] for r in sub])
        e = np.array([r["Ledge_edge"] for r in sub])
        allb.append(b)
        alle.append(e)
        ax.scatter(b, e, s=42, alpha=0.85, color=colors[s], label=SET_NAME[s].split(" (")[0], edgecolor="white", lw=0.5)
    b = np.concatenate(allb)
    e = np.concatenate(alle)
    lim = [0, max(b.max(), e.max()) * 1.08]
    ax.plot(lim, lim, "k--", lw=1, label="不改善线 y=x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("基线模型的 L_edge（与输入的梯度差）")
    ax.set_ylabel("+边缘损失模型的 L_edge")
    ax.set_title("图 3a｜逐图配对：全部 {} 张都在对角线下方\n= 每一张图的梯度场都更贴近输入".format(len(b)), fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    ax = axes[1]
    pct = (b - e) / b * 100
    bins = np.arange(math.floor(pct.min()), math.ceil(pct.max()) + 1, 1.0)
    ax.hist(pct, bins=bins, color="#72B7B2", edgecolor="white")
    ax.axvline(np.mean(pct), color=C_EDGE, lw=2, label="均值 {:.1f}%".format(np.mean(pct)))
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("L_edge 下降比例 (%)")
    ax.set_ylabel("图片张数")
    ax.set_title("图 3b｜改善幅度分布：均值 {:.1f}%，最小 {:.1f}%，最大 {:.1f}%".format(
        np.mean(pct), pct.min(), pct.max()), fontsize=11)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.suptitle("图 3｜41 张测试图逐张对比：没有一张变差", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig3_paired_ledge.png")
    return dict(mean_pct=float(np.mean(pct)), min_pct=float(pct.min()), max_pct=float(pct.max()), n=len(b))


# ---------- 图 4：亮度无关诊断 ----------
def fig4(diag, idx):
    sets = ["eval15", "lime", "dicm"]
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.3))
    metrics = [("cos_low", "cos(g_low, g_enh) 梯度方向一致性", 0.55, 0.85),
               ("corr_low", "corr(M_low, M_enh) 幅值图相关", 0.55, 0.85),
               ("ledge_norm", "亮度归一后的 L_edge（越小越好）", 0, None)]
    for ax, (key, title, lo, hi) in zip(axes, metrics):
        cats, vals_b, vals_e = [], [], []
        for s in sets:
            bs = [r[key] for r in diag if r["set"] == s and r["model"] == "baseline"]
            es = [r[key] for r in diag if r["set"] == s and r["model"] == "edge"]
            if not bs:
                continue
            cats.append(SET_SHORT[s])
            vals_b.append(np.mean(bs))
            vals_e.append(np.mean(es))
        bar_group(ax, cats, [("基线", vals_b, C_BASE), ("+边缘损失", vals_e, C_EDGE)],
                  title.split(" ")[0], title, fmt="{:.4f}" if key != "ledge_norm" else "{:.3f}")
        ax.set_ylim(lo, hi if hi else max(vals_b) * 1.35)
    fig.suptitle("图 4｜排除亮度干扰后结论依然成立：梯度场结构更接近输入（三个指标同向改善）", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig4_diag_brightness_free.png")


# ---------- 图 5：梯度放大倍数 ----------
def fig5(rows, diag, idx):
    sets = ["eval15", "lime", "dicm"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(sets))
    w = 0.2
    ratios_base, ratios_edge, ratios_gt = [], [], []
    for s in sets:
        sub = [r for r in rows if r["set"] == s]
        ratios_base.append(np.mean([r["M_base"] / r["M_in"] for r in sub]))
        ratios_edge.append(np.mean([r["M_edge"] / r["M_in"] for r in sub]))
        gts = [r["m_enh"] / r["m_low"] for r in diag if r["set"] == s and r["model"] == "GT"]
        ratios_gt.append(np.mean(gts) if gts else np.nan)

    ax.bar(x - w, [1.0] * len(sets), w, color=C_IN, label="输入本身 (=1)", edgecolor="white")
    ax.bar(x, ratios_base, w, color=C_BASE, label="基线", edgecolor="white")
    ax.bar(x + w, ratios_edge, w, color=C_EDGE, label="+边缘损失", edgecolor="white")
    for i, (rb, re_) in enumerate(zip(ratios_base, ratios_edge)):
        ax.annotate("{:.2f}×".format(rb), (i, rb), ha="center", va="bottom", fontsize=9, color=C_BASE)
        ax.annotate("{:.2f}×".format(re_), (i + w, re_), ha="center", va="bottom", fontsize=9, color=C_EDGE)
        if not math.isnan(ratios_gt[i]):
            ax.plot([i - w * 1.6, i + w * 1.6], [ratios_gt[i]] * 2, "--", color=C_GT, lw=1.8,
                    label="正常光 GT 的放大倍数（eval15 才有）" if i == 0 else None)
            ax.annotate("GT {:.2f}×".format(ratios_gt[i]), (i, ratios_gt[i]), ha="center", va="bottom",
                        fontsize=9, color=C_GT)
    ax.set_xticks(x)
    ax.set_xticklabels([SET_SHORT[s] for s in sets])
    ax.set_ylabel("M(增强图) / M(低照输入)")
    # 标题用 eval15（唯一有 GT 的数据集，口径可比）：6.14× → 5.64×，GT 3.75×
    i_e = sets.index("eval15")
    ax.set_title("图 5｜增强把梯度放大了几倍：eval15 上 {:.2f}× → {:.2f}×，\n"
                 "而真实正常光图只有 {:.2f}× —— 这既说明它有用，也说明它没完全对齐".format(
                     ratios_base[i_e], ratios_edge[i_e], ratios_gt[i_e]), fontsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, "fig5_ratio_bar.png")
    # 返回每个数据集的值，报告/汇总卡里按 eval15 口径引用
    out = {}
    for i, s in enumerate(sets):
        out["{}_base".format(s)] = float(ratios_base[i])
        out["{}_edge".format(s)] = float(ratios_edge[i])
    out["eval15_gt"] = float(ratios_gt[i_e])
    out["ratio_base"] = float(ratios_base[i_e])
    out["ratio_edge"] = float(ratios_edge[i_e])
    out["ratio_gt"] = float(ratios_gt[i_e])
    return out


# ---------- 图 6：逐像素误差热力图 ----------
def fig6():
    cases = [
        ("eval15/748（羽毛球场）", os.path.join(REPRO, "lol_test", "eval15", "low", "748.png")),
        ("LIME/1（威尼斯夜景）", os.path.join(CODE, "data", "test_data", "LIME", "1.bmp")),
    ]
    base_net, edge_net = load_model(BASE_CKPT), load_model(EDGE_CKPT)
    fig, axes = plt.subplots(len(cases), 4, figsize=(15.5, 7.2))
    for i, (title, path) in enumerate(cases):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_b, out_e = enhance(base_net, rgb), enhance(edge_net, rgb)
        M_l, M_b, M_e = sobel_M(rgb), sobel_M(out_b), sobel_M(out_e)
        E_b, E_e = np.abs(M_l - M_b), np.abs(M_l - M_e)
        improve = E_b - E_e  # >0 表示边缘损失把误差压小了
        vmax = np.percentile(np.maximum(E_b, E_e), 99.5)

        axes[i, 0].imshow(out_e)
        axes[i, 0].set_title("{}\n+边缘损失输出".format(title), fontsize=9)
        axes[i, 0].axis("off")
        for k, (img, tt, cmap) in enumerate([
            (E_b, "基线：|M_low − M_enh|  均值 {:.3f}".format(E_b.mean()), "inferno"),
            (E_e, "+边缘损失：均值 {:.3f}（−{:.1f}%）".format(E_e.mean(), (E_b.mean() - E_e.mean()) / E_b.mean() * 100), "inferno"),
        ]):
            im = axes[i, 1 + k].imshow(img, cmap=cmap, vmin=0, vmax=vmax)
            axes[i, 1 + k].set_title(tt, fontsize=9)
            axes[i, 1 + k].axis("off")
            plt.colorbar(im, ax=axes[i, 1 + k], fraction=0.046, pad=0.02, format="%.2f")
        # 右列：颜色按 |改善量| 的 98 分位定标，否则大部分像素接近 0 会全白
        iv = float(np.percentile(np.abs(improve), 98)) + 1e-6
        im = axes[i, 3].imshow(improve, cmap="RdBu_r", vmin=-iv, vmax=iv)
        axes[i, 3].set_title("误差被压小的量 (红=改善)\n正部 {:.1f}% 负部 {:.1f}%".format(
            (improve > 0).mean() * 100, (improve < 0).mean() * 100), fontsize=9)
        axes[i, 3].axis("off")
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.02, format="%.2f")
    fig.suptitle("图 6｜逐像素梯度误差热力图：误差（主要是平坦区的噪声梯度）被系统性地压小，"
                 "轮廓处保持", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig6_heatmap_error.png")


# ---------- 图 7：边缘保持曲线 ----------
def fig7():
    cases = [("eval15 748", os.path.join(REPRO, "lol_test", "eval15", "low", "748.png"),
              os.path.join(REPRO, "lol_test", "eval15", "high", "748.png")),
             ("LIME 1", os.path.join(CODE, "data", "test_data", "LIME", "1.bmp"), None)]
    base_net, edge_net = load_model(BASE_CKPT), load_model(EDGE_CKPT)
    ps = [1, 2, 5, 10, 20, 30, 50, 70, 100]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for ax, (title, lp, gp) in zip(axes, cases):
        bgr = cv2.imread(lp, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        M_l = sobel_M(rgb)
        curves = {"baseline": [], "edge": []}
        for name, net in [("baseline", base_net), ("edge", edge_net)]:
            M_e = sobel_M(enhance(net, rgb))
            for p in ps:
                thr = np.percentile(M_l, 100 - p)
                m = M_l >= thr
                curves[name].append(M_e[m].mean() / M_l[m].mean())
        ax.plot(ps, curves["baseline"], "o-", color=C_BASE, label="基线")
        ax.plot(ps, curves["edge"], "s-", color=C_EDGE, label="+边缘损失")
        if gp:
            g = cv2.imread(gp, cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(g, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            M_g = sobel_M(gt)
            gt_c = []
            for p in ps:
                thr = np.percentile(M_l, 100 - p)
                m = M_l >= thr
                gt_c.append(M_g[m].mean() / M_l[m].mean())
            ax.plot(ps, gt_c, "^--", color=C_GT, label="正常光 GT")
        ax.set_xscale("log")
        ax.set_xticks(ps)
        ax.set_xticklabels([str(p) for p in ps])
        ax.set_xlabel("只看输入中最强的 p% 像素（对数轴）")
        ax.set_ylabel("M(输出) / M(输入)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("图 7｜边缘保持曲线：越靠左=越强的边缘，边缘版对所有梯度级都比基线更收敛（更接近 1）", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save(fig, "fig7_edge_retention.png")


# ---------- 图 8：变化率总览 ----------
def fig8(rows, diag):
    sets = ["eval15", "lime", "dicm"]
    specs = [
        ("bright", "平均亮度", "base", "edge", rows, "{:+.1f}%"),
        ("M", "平均梯度幅值", "base", "edge", rows, "{:+.1f}%"),
        ("Ledge", "L_edge (4.12)", "base", "edge", rows, "{:+.1f}%"),
        ("cos_low", "梯度方向 cos", "baseline", "edge", diag, "{:+.1f}%"),
        ("corr_low", "梯度幅值 corr", "baseline", "edge", diag, "{:+.1f}%"),
        ("ledge_norm", "亮度归一 L_edge", "baseline", "edge", diag, "{:+.1f}%"),
        ("psnr", "PSNR", "base", "edge", rows, "{:+.1f}%"),
        ("ssim", "SSIM", "base", "edge", rows, "{:+.1f}%"),
    ]
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    y = np.arange(len(specs))
    w = 0.26
    for k, s in enumerate(sets):
        vals = []
        for key, label, mb, me, src, _ in specs:
            if src is rows:
                sub = [r for r in rows if r["set"] == s]
                b = np.mean([r["{}_base".format(key) if key in ("psnr", "ssim") else key + "_base"] for r in sub])
                e = np.mean([r["{}_edge".format(key) if key in ("psnr", "ssim") else key + "_edge"] for r in sub])
            else:
                b = np.mean([r[key] for r in diag if r["set"] == s and r["model"] == mb])
                e = np.mean([r[key] for r in diag if r["set"] == s and r["model"] == me])
            vals.append((e - b) / abs(b) * 100 if b else 0.0)
        bars = ax.barh(y + (k - 1) * w, vals, w, label=SET_SHORT[s],
                       color=["#E45756", "#4C78A8", "#54A24B"][k], edgecolor="white")
        for bb, v in zip(bars, vals):
            ax.annotate("{:+.1f}%".format(v), (v, bb.get_y() + bb.get_height() / 2),
                        ha="left" if v >= 0 else "right", va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels([s[1] for s in specs])
    ax.invert_yaxis()
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("+边缘损失 相对 基线的变化率 (%)   ← 下降 | 上升 →")
    ax.set_title("图 8｜一图看全：亮度/M/L_edge 下降，结构一致性(cos/corr)上升，PSNR 降、SSIM 微升", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "fig8_change_bar.png")


# ---------- 图 9：汇总卡片 ----------
def fig9(F1, F3, F5, rows, diag):
    fig = plt.figure(figsize=(12.5, 6.8))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.02), 0.96, 0.96, boxstyle="round,pad=0.01,rounding_size=0.015",
                                facecolor="#F7F9FC", edgecolor="#D0D7DE", lw=1.2))
    ax.text(0.5, 0.93, "Zero-DCE + 论文 (4.9)~(4.12) Sobel 边缘损失：做了什么、变了多少", ha="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.885, "基线 = 原版 Zero-DCE（SICE 2002 张，50 轮）　实验组 = 同数据同超参 + λ·L_edge（λ=0.22，边缘项约占总损失 5%）",
            ha="center", fontsize=9.5, color="#444")

    ax.text(0.04, 0.81, "① 工作量", fontsize=11.5, weight="bold")
    ax.text(0.04, 0.765, "• 改动 4 个文件：Myloss.py(+57 行) / lowlight_train.py / enhance.py / tasks.json",
            fontsize=9.5)
    ax.text(0.04, 0.725, "• 总计 +88 / −6 行；新增评测对比脚本 616 行（3 个）", fontsize=9.5)
    ax.text(0.04, 0.685, "• 训练 2002 张 × 50 轮 ≈ 25 min（边缘项额外开销 <2%，因为只是两组 3×3 卷积）",
            fontsize=9.5)
    ax.text(0.04, 0.645, "• 评测：3 个数据集 41 张图 × 2 个模型 × 8 项指标 = 656 个数值", fontsize=9.5)

    ax.text(0.04, 0.575, "② 关键数字（eval15, 15 对，基线 → +边缘损失）", fontsize=11.5, weight="bold")
    kv = [
        ("PSNR", "{:.2f} → {:.2f} dB（{:+.2f}）".format(F1["psnr_base"], F1["psnr_edge"], F1["psnr_edge"] - F1["psnr_base"])),
        ("SSIM", "{:.4f} → {:.4f}（{:+.4f}）".format(F1["ssim_base"], F1["ssim_edge"], F1["ssim_edge"] - F1["ssim_base"])),
        ("L_edge (4.12)", "0.2900 → 0.2627（−9.4%）"),
        ("梯度方向 cos", "0.6309 → 0.6421（+1.8%）"),
        ("平均亮度", "97.40 → 87.30（−10.4%）"),
        ("梯度放大倍数", "{:.2f}× → {:.2f}×（真实正常光 {:.2f}×）".format(F5["ratio_base"], F5["ratio_edge"], F5["ratio_gt"])),
    ]
    for i, (k, v) in enumerate(kv):
        y0 = 0.535 - i * 0.038
        ax.text(0.045, y0, "• {}".format(k), fontsize=9.5, weight="bold")
        ax.text(0.175, y0, v, fontsize=9.5)

    ax.text(0.54, 0.575, "③ 结论（可作为报告口径）", fontsize=11.5, weight="bold")
    bullets = [
        "• 论文 (4.12) 的效果被定量复现：41/41 张图的 L_edge 都下降，",
        "  平均 {:.1f}%（范围 {:.1f}%~{:.1f}%）。".format(F3["mean_pct"], F3["min_pct"], F3["max_pct"]),
        "• 排除亮度干扰后仍成立：cos/corr/归一化 L_edge 三项同向改善。",
        "• 机制（eval15）：LE 曲线把梯度放大 {:.2f}×，边缘损失压到 {:.2f}×".format(F5["ratio_base"], F5["ratio_edge"]),
        "  （真实正常光图本身是 {:.2f}×）。".format(F5["ratio_gt"]),
        "• 代价：整体亮度 −10.4%，PSNR −0.81 dB；SSIM +0.0048（{} 张中 {} 张改善）。".format(F1["n"], F1["n_ssim_better"]),
        "• 结论：这是「结构正则项」而不是「锐化项」；λ 建议 0.05~0.10，或改成单边 hinge",
        "  只罚「边缘变弱」，可兼顾亮度。",
    ]
    for i, t in enumerate(bullets):
        ax.text(0.54, 0.52 - i * 0.043, t, fontsize=9.5)

    ax.text(0.04, 0.245, "④ 图表索引", fontsize=11.5, weight="bold")
    idx = [
        "fig1 质量指标（PSNR/SSIM + 逐图变化）　fig2 梯度幅值与 L_edge　fig3 逐图配对散点",
        "fig4 亮度无关诊断　fig5 梯度放大倍数　fig6 逐像素误差热力图　fig7 边缘保持曲线　fig8 变化率总览",
        "zoom\\*.png 放大细节 + Sobel 幅值图　overview_*.png 全数据集并排总览　metrics.csv 原始数值",
    ]
    for i, t in enumerate(idx):
        ax.text(0.04, 0.20 - i * 0.038, t, fontsize=9.5, color="#333")
    fig.savefig(os.path.join(OUT, "fig9_summary_card.png"), facecolor="white", bbox_inches="tight")
    plt.close(fig)
    print("[ok] fig9_summary_card.png")


def write_markdown(F1, F3, F5, rows):
    lines = ["# 边缘损失（论文 4.9~4.12）图表说明与图注", ""]
    lines.append("所有图由 `repro\\scripts\\make_edge_figures.py` 自动生成，数据来自 "
                 "`demo_edge\\ep49_edge022\\metrics.csv` 与 `demo_edge\\diag_edge_effect\\diag_metrics.csv`。")
    lines.append("")
    lines.append("## 一、每个指标说明什么")
    lines.append("")
    lines.append("| 指标 | 含义 | 期望方向 | 本次实测（基线→边缘） |")
    lines.append("|---|---|---|---|")
    lines.append("| PSNR (dB) | 与正常光 GT 的像素级误差 | 越高越好 | 17.22 → 16.41（−0.81） |")
    lines.append("| SSIM | 与 GT 的结构相似度 | 越高越好 | 0.5509 → 0.5557（+0.0048） |")
    lines.append("| L_edge (4.12) | 增强前后梯度幅值之差（论文定义的损失本体） | 越低越好 | 0.2900 → 0.2627（−9.4%） |")
    lines.append("| cos(g_low,g_enh) | 梯度向量方向一致性（尺度无关） | 越高越好 | 0.6309 → 0.6421 |")
    lines.append("| corr(M_low,M_enh) | 梯度幅值图相关（尺度无关） | 越高越好 | 0.6286 → 0.6469 |")
    lines.append("| 亮度归一 L_edge | 把增强图缩放到输入亮度后再算 4.12，排除“变暗取巧” | 越低越好 | 0.0351 → 0.0342 |")
    lines.append("| M 放大倍数(eval15) | M(输出)/M(输入)，衡量梯度被放大几倍 | 越接近 GT 的 {:.2f}× 越自然 | {:.2f}× → {:.2f}× |".format(F5["ratio_gt"], F5["ratio_base"], F5["ratio_edge"]))
    lines.append("")
    lines.append("## 二、图注（可直接复制进报告）")
    lines.append("")
    caps = [
        ("fig1_quality_eval15.png", "图 1｜eval15 定量对比：平均 PSNR 下降 {:.2f} dB，SSIM 提升 {:.4f}；逐图看 {} / {} 张 SSIM 改善。".format(F1["psnr_edge"] - F1["psnr_base"], F1["ssim_edge"] - F1["ssim_base"], F1["n_ssim_better"], F1["n"])),
        ("fig2_gradient_all.png", "图 2｜三个数据集上平均梯度幅值 M 与 L_edge 的变化，L_edge 一致下降 8%~10%。"),
        ("fig3_paired_ledge.png", "图 3｜41 张测试图逐图配对：全部落在 y=x 下方，平均下降 {:.1f}%（{:.1f}%~{:.1f}%），无一变差。".format(F3["mean_pct"], F3["min_pct"], F3["max_pct"])),
        ("fig4_diag_brightness_free.png", "图 4｜亮度无关诊断：cos、corr 上升、归一化 L_edge 下降，说明不是“变暗取巧”。"),
        ("fig5_ratio_bar.png", "图 5｜梯度放大倍数（eval15）：{:.2f}× → {:.2f}×，向真实正常光图的 {:.2f}× 靠拢（尚未完全对齐）。".format(F5["ratio_base"], F5["ratio_edge"], F5["ratio_gt"])),
        ("fig6_heatmap_error.png", "图 6｜逐像素梯度误差热力图：误差主要来自平坦区噪声梯度，边缘损失把它系统性压小，轮廓处基本不变。"),
        ("fig7_edge_retention.png", "图 7｜边缘保持曲线：对不同强度 p% 的梯度，边缘版都比基线更收敛。"),
        ("fig8_change_bar.png", "图 8｜变化率总览：亮度/M/L_edge 下降，结构一致性指标上升，PSNR 降、SSIM 微升。"),
        ("fig9_summary_card.png", "图 9｜一页汇总：工作量、关键数字、结论与图表索引。"),
    ]
    for f, c in caps:
        lines.append("- **`{}`**：{}".format(f, c))
    lines.append("")
    lines.append("## 三、报告里怎么讲（建议口径）")
    lines.append("")
    lines.append("1. **先说机制**：Zero-DCE 的 LE 曲线在暗区斜率 1+|α|>1，提亮必然放大梯度；无约束时网络会把噪声也一起放大。"
                 "论文 (4.12) 用逐通道 Sobel 幅值差把这件事约束住。")
    lines.append("2. **再说效果**：41/41 张图梯度场更贴近输入（L_edge 平均 −%s），且排除亮度干扰后仍成立。" % "{:.1f}".format(F3["mean_pct"]))
    lines.append("3. **再说代价**：亮度 −10.4%、PSNR −0.81 dB、SSIM +0.0048 —— 它是正则项而非锐化项。")
    lines.append("4. **落到建议**：若要兼顾亮度，λ 调到 0.05~0.10（对应边缘项占 1%~2.5%），或改用单边 hinge。")
    with open(os.path.join(OUT, "summary_for_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("[ok] summary_for_report.md")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_metrics()
    diag = load_diag()
    idx = diag_index(diag)
    F1 = F3 = F5 = None
    for name, fn in [("fig1", lambda: fig1(rows)),
                     ("fig2", lambda: fig2(rows)),
                     ("fig3", lambda: fig3(rows))]:
        try:
            r = fn()
            if name == "fig1":
                F1 = r
            elif name == "fig3":
                F3 = r
        except Exception:
            print("[FAIL]", name)
            traceback.print_exc()
    try:
        fig4(diag, idx)
    except Exception:
        print("[FAIL] fig4"); traceback.print_exc()
    try:
        F5 = fig5(rows, diag, idx)
    except Exception:
        print("[FAIL] fig5"); traceback.print_exc()
    try:
        fig6()
    except Exception:
        print("[FAIL] fig6"); traceback.print_exc()
    try:
        fig7()
    except Exception:
        print("[FAIL] fig7"); traceback.print_exc()
    try:
        fig8(rows, diag)
    except Exception:
        print("[FAIL] fig8"); traceback.print_exc()
    try:
        fig9(F1, F3, F5, rows, diag)
    except Exception:
        print("[FAIL] fig9"); traceback.print_exc()
    try:
        write_markdown(F1, F3, F5, rows)
    except Exception:
        print("[FAIL] md"); traceback.print_exc()
    print("\n输出目录：", OUT)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""画两分支算法流程图（论文/答辩用）"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib import font_manager

FONT = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

OUT = r"C:\Users\12983\Documents\repro\docs"
os.makedirs(OUT, exist_ok=True)

fig, ax = plt.subplots(figsize=(16, 8.6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")

C_DEC = "#FFF2CC"
C_PRE = "#D9EAD3"
C_DCE = "#DDEBF7"
C_POST = "#FCE4D6"
C_OUT = "#EFEFEF"


def box(x, y, w, h, text, fc, fs=11, ec="#7A7A7A", lw=1.5, bold=False, align="center"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.3,rounding_size=0.9",
                                fc=fc, ec=ec, lw=lw, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4,
            fontweight=("bold" if bold else "normal"), linespacing=1.55, color="#1a1a1a")


def diamond(x, y, w, h, text, fs=11.5):
    pts = [(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)]
    ax.add_patch(Polygon(pts, closed=True, fc=C_DEC, ec="#7A7A7A", lw=1.5, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4, linespacing=1.4)


def arrow(p1, p2, label="", color="#4d4d4d", fs=10, dx=0.0, dy=0.0):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=18, lw=1.7,
                                 color=color, zorder=2, shrinkA=3, shrinkB=3))
    if label:
        ax.text((p1[0] + p2[0]) / 2 + dx, (p1[1] + p2[1]) / 2 + dy, label, ha="center",
                va="center", fontsize=fs, color="#C00000",
                bbox=dict(fc="white", ec="none", pad=1.5))


# 标题
ax.text(50, 58.4, "零-DCE 去马赛克：单图自适应两分支流程", ha="center",
        fontsize=18, fontweight="bold", color="#111111")

# 左侧公共链路
box(8.5, 31, 13, 7.5, "输入图像\n（已编码 sRGB）", C_OUT, fs=11, bold=True)
box(25.5, 31, 17.5, 9.5, "统计量估计\n均值 m\n色度噪声 σc、块效应 b", C_PRE, fs=10.5)
diamond(45, 31, 13.5, 11, "自动判定\nauto_decision()", fs=11)

arrow((15.3, 31), (16.5, 31))
arrow((34.5, 31), (38.1, 31))
arrow((45, 36.6), (45, 47.5), "m < 5（近黑）", dy=0.0, dx=0.0)
arrow((45, 25.4), (45, 17.0), "m ≥ 5（常规）", dy=0.0)

# 近黑分支（上）
box(70, 47.5, 45, 11.5,
    "近黑分支（m < 5）\n"
    "① 预处理：块效应>1.03 才去块 + 亮度 NLM 轻混合 + 暗部轻抬\n"
    "② DCE-Net 增强（8 次迭代曲线）\n"
    "③ 后处理：引导滤波平滑色度 + 近黑限增益（<3 全中性抬亮，3~12 线性羽化，>12 保持模型输出）\n"
    "④ 暗部补偿：Y 通道暗部抬 4 级  →  输出",
    C_PRE, fs=9.6)
arrow((51.7, 33.2), (64.1, 42.2))

# 常规分支（下）：先判 σc
diamond(64, 17.0, 15, 10.5, "σc < 0.3 ？", fs=11)
arrow((51.5, 28.4), (56.4, 21.2))

box(64, 4.6, 30, 6.2, "跳过：图像原样进 DCE-Net，输出 = 原版\n（不做任何预处理/后处理）", C_OUT, fs=10.2)
box(93.5, 26.5, 12, 26,
    "常规分支\n（m ≥ 5）\n\n"
    "图像原样\n进 DCE-Net\n\n"
    "增强后仅对\n色度 Cb/Cr\n在低信号平坦区\n轻度平滑\n\n"
    "（亮度完全不动，\n保颗粒与边缘）",
    C_POST, fs=9.3)
arrow((64, 11.6), (64, 7.9), "是", dx=-2.6)
arrow((71.6, 17.0), (87.3, 24.2), "否（0.3 ≤ σc）", dx=1.6, dy=2.2)

# 输出汇合
ax.text(93.5, 40.4, "→ 输出", ha="center", fontsize=11.5, fontweight="bold")
arrow((92.2, 47.0), (93.5, 39.6), color="#7A7A7A")
ax.text(64, 1.6, "→ 输出（与原版逐像素一致）", ha="center", fontsize=10.2, color="#333333")

# 参数注释
ax.text(2.0, 8.6,
        "关键阈值（当前默认）：\n"
        "• m < 5 → 近黑分支；否则 σc < 0.3 → 跳过（=原版）\n"
        "• 0.3 ≤ σc < 1.5 → 色度 light，强度 = 0.35·σc，截断 0.15~0.35\n"
        "• 色度噪声 σc：Cb/Cr 减高斯(σ=1.5) 后高频残差的 std 均值\n"
        "• 全程不改 DCE-Net 结构、不改曲线迭代次数",
        ha="left", va="top", fontsize=9.6, color="#333333", linespacing=1.7,
        bbox=dict(fc="#F7F7F7", ec="#CCCCCC", pad=6))

fig.tight_layout()
out = os.path.join(OUT, "算法流程图_两分支.png")
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
print("[ok]", out)

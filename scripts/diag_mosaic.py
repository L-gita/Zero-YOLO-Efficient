#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""马赛克成因诊断：对极端暗光图，量化“量化断层 / 8x8 块效应 / 色度噪声 / 梯度放大”。

对每张图：
  1. 用 SICE 权重（snapshots_sice/Epoch50.pth）增强；
  2. 统计暗区（输入灰度 < 40）的：
     - 输入灰度级数（unique levels）与每个 level 平均覆盖像素数 → 量化断层风险
     - 8x8 网格边界跳变 vs 非边界跳变（blockiness ratio）→ JPEG 块效应是否被放大
     - 色度高频残差 std（Cb/Cr 减去 5x5 模糊）→ 色噪强度
     - 平均梯度幅值 M 的放大倍数 → 噪声/对比放大
输出一张表 + CSV。

用法: python scripts\\diag_mosaic.py
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)

import torch  # noqa: E402
import model  # noqa: E402

CKPT = os.path.join(CODE, "snapshots_sice", "Epoch50.pth")
CASES = [
    ("01_extreme/126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
    ("01_extreme/101_0", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "101_0_.png")),
    ("03_DICM/01.jpg", os.path.join(ROOT, "低照图库", "03_DICM", "01.jpg")),
    ("02_LIME/1.bmp", os.path.join(ROOT, "低照图库", "02_LIME", "1.bmp")),
]
OUT = os.path.join(ROOT, "demo_mosaic")
os.makedirs(OUT, exist_ok=True)


def load_model():
    net = model.enhance_net_nopool().cuda()
    sd = torch.load(CKPT, map_location="cuda", weights_only=True)
    net.load_state_dict(sd)
    net.eval()
    return net


def enhance(net, rgb_u8):
    x = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        _, out, _ = net(x)
    y = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


def sobel_M(rgb_u8):
    """论文 (4.9)/(4.11)：逐通道 |Ix|+|Iy|，取通道平均。"""
    f = rgb_u8.astype(np.float32) / 255.0
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    M = []
    for c in range(3):
        ch = f[:, :, c]
        ix = cv2.filter2D(ch, -1, kx, borderType=cv2.BORDER_REPLICATE)
        iy = cv2.filter2D(ch, -1, ky, borderType=cv2.BORDER_REPLICATE)
        M.append(np.abs(ix) + np.abs(iy))
    return np.mean(M, axis=0)


def blockiness(gray):
    """8x8 网格边界跳变 / 非边界跳变。>1 说明有块状结构。"""
    g = gray.astype(np.float32)
    dv = np.abs(np.diff(g, axis=1))          # 水平相邻差
    dh = np.abs(np.diff(g, axis=0))
    # 边界列索引：7,15,23...（diff 下标 i 表示 g[:,i+1]-g[:,i]）
    cols = np.arange(dv.shape[1])
    rows = np.arange(dh.shape[0])
    v_edge = dv[:, (cols % 8) == 7]
    v_in = dv[:, (cols % 8) != 7]
    h_edge = dh[(rows % 8) == 7, :]
    h_in = dh[(rows % 8) != 7, :]
    num = v_edge.mean() + h_edge.mean()
    den = v_in.mean() + h_in.mean() + 1e-9
    return float(num / den)


def chroma_noise(rgb_u8, mask):
    ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    out = []
    for c in (1, 2):
        ch = ycc[:, :, c]
        low = cv2.blur(ch, (5, 5))
        resid = (ch - low)[mask]
        out.append(float(resid.std()) if resid.size else 0.0)
    return out


def dark_stats(gray_u8, mask):
    vals = gray_u8[mask]
    if vals.size == 0:
        return 0, 0.0
    levels = np.unique(vals)
    return int(vals.size), int(levels.size)


def main():
    net = load_model()
    rows = []
    for tag, path in CASES:
        if not os.path.exists(path):
            print(f"[skip] {path}")
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        dark = gray < 40                     # 只看暗区（占绝大部分像素）
        enh = enhance(net, rgb)
        gray_e = cv2.cvtColor(enh, cv2.COLOR_RGB2GRAY)

        n_px, n_lv = dark_stats(gray, dark)
        cb, cr = chroma_noise(rgb, dark)
        cb_e, cr_e = chroma_noise(enh, dark)
        m_in = float(sobel_M(rgb)[dark].mean())
        m_e = float(sobel_M(enh)[dark].mean())
        blk_in = blockiness(gray)
        blk_e = blockiness(gray_e)
        rows.append(dict(
            case=tag,
            dark_frac=float(dark.mean()),
            dark_px=int(n_px),
            dark_levels=int(n_lv),
            px_per_level=float(n_px / max(n_lv, 1)),
            chroma_noise_in=cb + cr,
            chroma_noise_out=cb_e + cr_e,
            M_in=m_in, M_out=m_e, M_gain=float(m_e / max(m_in, 1e-6)),
            bright_in=float(gray[dark].mean()) if n_px else 0.0,
            bright_out=float(gray_e[dark].mean()) if n_px else 0.0,
            blockiness_in=blk_in, blockiness_out=blk_e,
        ))
        Image.fromarray(enh).save(os.path.join(OUT, tag.replace("/", "_") + "_enhanced.png"))
        print(f"[ok] {tag}")

    hdr = ["case", "dark_frac", "dark_px", "dark_levels", "px_per_level",
           "chroma_noise_in", "chroma_noise_out", "M_in", "M_out", "M_gain",
           "bright_in", "bright_out", "blockiness_in", "blockiness_out"]
    with open(os.path.join(OUT, "mosaic_metrics.csv"), "w", encoding="utf-8-sig", newline="") as fp:
        fp.write(",".join(hdr) + "\n")
        for r in rows:
            fp.write(",".join(str(r[k]) if not isinstance(r[k], float) else f"{r[k]:.4f}" for k in hdr) + "\n")

    print("\n" + " | ".join(["case", "暗区占比", "暗区级数", "色噪 in→out", "M放大", "亮度 in→out", "块效应 in→out"]))
    for r in rows:
        print(f"{r['case']:<18} | {r['dark_frac']*100:5.1f}% | {r['dark_levels']:>3} | "
              f"{r['chroma_noise_in']:.2f}→{r['chroma_noise_out']:.2f} | {r['M_gain']:.2f}× | "
              f"{r['bright_in']:.1f}→{r['bright_out']:.1f} | {r['blockiness_in']:.3f}→{r['blockiness_out']:.3f}")
    print(f"\n结果: {os.path.join(OUT, 'mosaic_metrics.csv')}")


if __name__ == "__main__":
    main()

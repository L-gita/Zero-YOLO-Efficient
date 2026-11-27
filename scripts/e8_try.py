# -*- coding: utf-8 -*-
"""试验：在近黑区域用“中性 gamma 抬亮”替代模型的强放大，消除彩色大斑块。"""
import os
import sys
import numpy as np
import cv2
from PIL import Image

ROOT = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402
import model  # noqa: E402
import preprocess as pp  # noqa: E402
import diag_color_blobs as dcb  # noqa: E402

CASES = [
    ("extreme_126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
    ("extreme_147_0", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "147_0_.png")),
    ("eval15_748", os.path.join(ROOT, "lol_test", "eval15", "low", "748.png")),
    ("LIME_1", os.path.join(ROOT, "低照图库", "02_LIME", "1.bmp")),
]


def neutral_lift(low_rgb, gamma=0.45, gain=1.0):
    """把低照图按灰度整体 gamma 抬亮（三通道同一映射 → 绝对中性，不会产生彩色斑块）"""
    g = cv2.cvtColor(low_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lift = np.power(np.clip(g, 0, 1), gamma) * 255.0 * gain
    return np.repeat(np.clip(lift, 0, 255)[..., None], 3, axis=2)


def limit_dark_gain(enh_rgb, low_rgb, l0=8.0, l1=28.0, gamma=0.45, strength=1.0):
    """低照亮度 < l0 的地方完全用中性抬亮替代；l0~l1 之间线性过渡。"""
    low = np.ascontiguousarray(low_rgb)
    y = cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = np.clip((l1 - y) / max(l1 - l0, 1e-6), 0.0, 1.0)[..., None] * float(np.clip(strength, 0, 1))
    lift = neutral_lift(low, gamma=gamma)
    out = enh_rgb.astype(np.float32) * (1.0 - m) + lift * m
    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    net = dcb.load_model()
    print(f"{'case':16s} {'pipe':6s} {'最大有色块':>10s} {'有色总面积':>10s} {'亮度':>7s} {'PSNR':>7s} {'SSIM':>7s}")
    for name, path in CASES:
        if not os.path.exists(path):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        gt = None
        gp = path.replace("\\low\\", "\\high\\")
        if "eval15" in path and os.path.exists(gp):
            gt = np.asarray(Image.open(gp).convert("RGB"))
        e1 = dcb.pipe_e1(net, rgb)
        e6 = dcb.pipe_e6(net, rgb)
        e8 = limit_dark_gain(e6, rgb, l0=8, l1=28, gamma=0.45, strength=1.0)
        e8b = limit_dark_gain(e6, rgb, l0=6, l1=20, gamma=0.40, strength=0.85)
        for k, v in [("E1", e1), ("E6", e6), ("E8", e8), ("E8b", e8b)]:
            blob, tot = dcb.colored_blob(v)
            br = cv2.cvtColor(v, cv2.COLOR_RGB2GRAY).mean()
            ps = ss = float("nan")
            if gt is not None:
                ps = dcb_psnr(v, gt)
                ss = dcb_ssim(v, gt)
            print(f"{name:16s} {k:6s} {blob*100:9.2f}% {tot*100:9.2f}% {br:7.1f} {ps:7.2f} {ss:7.4f}")


def dcb_psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def dcb_ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    k = (11, 11)
    mu1 = cv2.GaussianBlur(a, k, 1.5)
    mu2 = cv2.GaussianBlur(b, k, 1.5)
    s1 = cv2.GaussianBlur(a * a, k, 1.5) - mu1 * mu1
    s2 = cv2.GaussianBlur(b * b, k, 1.5) - mu2 * mu2
    s12 = cv2.GaussianBlur(a * b, k, 1.5) - mu1 * mu2
    smap = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
    return float(smap.mean())


if __name__ == "__main__":
    main()

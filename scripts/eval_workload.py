"""量化对比：原版 Zero-DCE 基线 vs 去马赛克最终版（两种工作点）。

用法：
  python scripts\\eval_workload.py [-o OUT]

输出：
  OUT\\metrics.csv              每张图 × 每个模式的全指标
  OUT\\summary.md              按数据集汇总的表（可直接进论文/报告）
  OUT\\fig_workload.png         汇总柱状图

数据集（有 GT 的只有 eval15）：
  eval15  : lol_test\\eval15\\low + high 15 对
  LIME    : 低照图库\\02_LIME 10 张
  DICM    : 低照图库\\03_DICM 取前 16 张
  极限暗图: 低照图库\\01_真实拍摄 的 126_3_ / 147_0_ / 101_0_
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import enhance as EN          # noqa: E402
import preprocess as pp       # noqa: E402
import preprocess_v4 as pv4   # noqa: E402

OUT = os.path.join(ROOT, "demo_mosaic", "workload")


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else 10.0 * np.log10(255.0 ** 2 / mse)


def ssim_rgb(a, b):
    """RGB 三通道 11x11 高斯 (σ=1.5) SSIM 的平均（与项目其它评测口径一致）。"""
    from math import exp
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    win = cv2.getGaussianKernel(11, 1.5)
    w = np.outer(win, win)
    vals = []
    for c in range(3):
        x = a[..., c].astype(np.float64)
        y = b[..., c].astype(np.float64)
        mx, my = cv2.filter2D(x, -1, w), cv2.filter2D(y, -1, w)
        sxx = cv2.filter2D(x * x, -1, w) - mx * mx
        syy = cv2.filter2D(y * y, -1, w) - my * my
        sxy = cv2.filter2D(x * y, -1, w) - mx * my
        s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        vals.append(float(np.mean(s)))
    return float(np.mean(vals))


def chroma_noise(u8):
    c = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    return float(np.mean(np.abs(c - cv2.GaussianBlur(c, (0, 0), 1.5))))


def luma_std8(u8):
    g = np.asarray(Image.fromarray(u8).convert("L"), dtype=np.float32)
    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    return float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))


def bright(u8):
    return float(np.asarray(Image.fromarray(u8).convert("L"), dtype=np.float32).mean())


def grad_cos(low, out):
    """梯度方向一致性（全局，1=完全同向）"""
    def grad(u8):
        g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return gx, gy
    ax, ay = grad(low)
    bx, by = grad(out)
    num = float(np.sum(ax * bx + ay * by))
    den = float(np.sqrt(np.sum(ax ** 2 + ay ** 2) * np.sum(bx ** 2 + by ** 2))) + 1e-9
    return num / den


def run_mode(net, low, mode):
    t0 = time.time()
    if mode == "base":
        out, _ = EN.enhance_image(net, Image.fromarray(low))
        u8 = np.asarray(out)
    elif mode in ("default", "clean"):
        s = 0.20 if mode == "default" else 0.45
        info = pp.auto_preprocess(low, strength=s)
        out, _ = EN.enhance_image(net, Image.fromarray(info["img"]))
        u8 = np.asarray(out)
        if info["mean"] < 5.0:
            u8 = pp.chroma_guided_smooth(u8, strength=0.5 * s)
            u8 = pp.quantize8(pp.limit_dark_gain(u8, np.asarray(info["img"]), l0=3.0, l1=12.0,
                                                 gamma=0.55, strength=0.80 * s,
                                                 smooth_radius=8, dither=0.0), dither=0.4)
        elif mode == "clean":
            u8 = pp.quantize8(np.asarray(pp.chroma_denoise_flat(
                u8, low, info["noise"], strength=min(1.2, max(0.0, info["k"] * 0.45 * s)),
                target_amp=None), dtype=np.float32), dither=0.4)
        else:
            u8 = pp.quantize8(u8.astype(np.float32), dither=0.4)
        # 阴影补偿（与 enhance.py 默认一致）
        if info["mean"] < 15.0:
            u8 = pp.shadow_lift(u8, lift=4.0)
    else:
        raise SystemExit(mode)
    return u8, time.time() - t0


def collect_sets():
    sets = []
    lo = os.path.join(ROOT, "lol_test", "eval15", "low")
    hi = os.path.join(ROOT, "lol_test", "eval15", "high")
    if os.path.isdir(lo):
        pairs = []
        for f in sorted(os.listdir(lo)):
            if f.lower().endswith(".png") and os.path.exists(os.path.join(hi, f)):
                pairs.append((os.path.join(lo, f), os.path.join(hi, f)))
        if pairs:
            sets.append(("eval15", pairs))
    def single(d, n=None, pattern=("*.png", "*.jpg", "*.bmp")):
        out = []
        if not os.path.isdir(d):
            return out
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(tuple(p[1:] for p in pattern)) or f.lower().endswith((".png", ".jpg", ".bmp")):
                out.append(os.path.join(d, f))
        return out[:n] if n else out
    lime = single(os.path.join(ROOT, "低照图库", "02_LIME"))
    if lime:
        sets.append(("LIME", [(p, None) for p in lime]))
    dicm = single(os.path.join(ROOT, "低照图库", "03_DICM"), 16)
    if dicm:
        sets.append(("DICM16", [(p, None) for p in dicm]))
    real = os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp")
    ext = [os.path.join(real, f) for f in ("126_3_.png", "147_0_.png", "101_0_.png")
           if os.path.exists(os.path.join(real, f))]
    if ext:
        sets.append(("extreme3", [(p, None) for p in ext]))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=OUT)
    req = ap.parse_args()
    os.makedirs(req.out, exist_ok=True)
    net = EN.get_net(EN.MODELS["sice"])
    rows = []
    for setname, pairs in collect_sets():
        for p, gt in pairs:
            im = Image.open(p).convert("RGB")
            low = np.asarray(im)
            if low.shape[0] * low.shape[1] > 6_000_000:
                im = im.resize((low.shape[1] // 2, low.shape[0] // 2), Image.LANCZOS)
                low = np.asarray(im)
            gtl = None
            if gt and os.path.exists(gt):
                g = np.asarray(Image.open(gt).convert("RGB"))
                if g.shape[:2] != low.shape[:2]:
                    g = np.asarray(Image.fromarray(g).resize((low.shape[1], low.shape[0]), Image.LANCZOS))
                gtl = g
            for mode in ("base", "default", "clean"):
                u8, dt = run_mode(net, low, mode)
                r = dict(set=setname, file=os.path.basename(p), mode=mode,
                         ms=dt * 1000.0, bright=bright(u8), chroma=chroma_noise(u8),
                         std8=luma_std8(u8), edges=pv4.artificial_edges(u8),
                         cos=grad_cos(low, u8),
                         psnr_gt=psnr(u8, gtl) if gtl is not None else "",
                         ssim_gt=ssim_rgb(u8, gtl) if gtl is not None else "")
                rows.append(r)
                print(f"{setname:<9}{os.path.basename(p):<18}{mode:<8} 亮度 {r['bright']:6.1f} "
                      f"色噪 {r['chroma']:5.2f} 细节 {r['std8']:5.2f} 边界 {r['edges']:5.1f} "
                      f"cos {r['cos']:.3f} " + (f"PSNR {r['psnr_gt']:.2f} SSIM {r['ssim_gt']:.4f}" if gtl is not None else ""))
    with open(os.path.join(req.out, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # 汇总
    lines = ["# 原版基线 vs 去马赛克最终版（两种工作点）", "",
             "- **base**：原版 Zero-DCE（无任何预处理）",
             "- **default**：去马赛克最终版默认（s=0.20 + 不碰色度 + 阴影补偿）",
             "- **clean**：干净档（s=0.45 + 局部色度平滑 + 阴影补偿）", "",
             "指标：亮度(0-255)、色噪(色度高频残差)、细节(3×3 局部 std8)、人造边界(每千像素)、"
             "梯度方向 cos、PSNR/SSIM（仅 eval15 有 GT）", ""]
    sets = sorted({r["set"] for r in rows})
    for s in sets:
        sub = [r for r in rows if r["set"] == s]
        lines += [f"## {s}（{len(sub)//3} 张）", "",
                  "| 指标 | base | default | clean |", "| --- | --- | --- | --- |"]
        for key, name, fmt in (("bright", "平均亮度", "{:.1f}"), ("chroma", "色噪", "{:.2f}"),
                               ("std8", "细节 std8", "{:.2f}"), ("edges", "人造边界/千像素", "{:.1f}"),
                               ("cos", "梯度方向 cos", "{:.3f}"), ("ms", "单张耗时 ms", "{:.0f}")):
            vs = []
            for m in ("base", "default", "clean"):
                vals = [r[key] for r in sub if r["mode"] == m]
                vs.append(fmt.format(sum(vals) / len(vals)))
            lines.append(f"| {name} | " + " | ".join(vs) + " |")
        if sub[0]["psnr_gt"] != "":
            for key, name, fmt in (("psnr_gt", "PSNR vs GT (dB)", "{:.2f}"), ("ssim_gt", "SSIM vs GT", "{:.4f}")):
                vs = []
                for m in ("base", "default", "clean"):
                    vals = [r[key] for r in sub if r["mode"] == m and r[key] != ""]
                    vs.append(fmt.format(sum(vals) / len(vals)) if vals else "—")
                lines.append(f"| {name} | " + " | ".join(vs) + " |")
        lines.append("")
    with open(os.path.join(req.out, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))
    print("→", req.out)


if __name__ == "__main__":
    main()

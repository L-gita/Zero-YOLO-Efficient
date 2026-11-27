#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""马赛克预处理对照实验（E1~E5）：验证“先降噪再增强”确实能减轻马赛克。

管线定义
--------
E0 低照原图（不增强，参考）
E1 原图              → Zero-DCE                          （基线）
E2 色度优先降噪      → Zero-DCE                          （先降噪）
E3 原图              → Zero-DCE → 色度优先降噪            （后降噪）
E4 弱降噪            → Zero-DCE → 保边平滑                （轻量组合）
E5 去块+降噪+轻抬    → Zero-DCE → 保边平滑                （方案B 全套 = 本文方案）

评估指标（全部在“输入暗区掩码”内统计，保证可比）
------------------------------------------------
亮度        ：输出灰度均值（看提亮程度）
色度噪声    ：Cb/Cr 高频残差 std（马赛克花斑的主要来源，越小越好）
块效应比    ：8x8 网格边界跳变 / 非边界跳变（>1 且有 8 网格 = 有块状结构，越小越好）
M 放大      ：平均梯度幅值 输出/输入（论文 4.11；越小说明噪声/对比放大越少）
PSNR / SSIM ：有正常光 GT 时（eval15），RGB 11x11 高斯 σ=1.5

用法
----
    python scripts\\mosaic_ablation.py                 # 全部：eval15 + LIME + 极端暗图
    python scripts\\mosaic_ablation.py --sets eval15    # 只跑某几组
"""
import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import model  # noqa: E402
import preprocess as pp  # noqa: E402

CKPT = os.path.join(CODE, "snapshots_sice", "Epoch50.pth")
OUT = os.path.join(ROOT, "demo_mosaic", "ablation")
FONT = r"C:\Windows\Fonts\msyh.ttc"


# ---------------------------------------------------------------- 数据
def collect(sets):
    """返回 [(group, name, low_path, gt_path_or_None), ...]"""
    out = []
    if "eval15" in sets:
        low_dir = os.path.join(ROOT, "lol_test", "eval15", "low")
        high_dir = os.path.join(ROOT, "lol_test", "eval15", "high")
        for f in sorted(os.listdir(low_dir)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue
            gt = os.path.join(high_dir, f)
            out.append(("eval15", os.path.splitext(f)[0], os.path.join(low_dir, f),
                        gt if os.path.exists(gt) else None))
    if "lime" in sets:
        d = os.path.join(ROOT, "低照图库", "02_LIME")
        for f in sorted(os.listdir(d)):
            if f.lower().endswith((".bmp", ".jpg", ".jpeg", ".png")):
                out.append(("LIME", os.path.splitext(f)[0], os.path.join(d, f), None))
    if "extreme" in sets:
        d = os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp")
        for n in ("126_3_.png", "101_0_.png", "147_0_.png"):
            p = os.path.join(d, n)
            if os.path.exists(p):
                out.append(("extreme", os.path.splitext(n)[0], p, None))
    if "dicm" in sets:
        d = os.path.join(ROOT, "低照图库", "03_DICM")
        files = [f for f in sorted(os.listdir(d)) if f.lower().endswith(".jpg")][:8]
        for f in files:
            out.append(("DICM", os.path.splitext(f)[0], os.path.join(d, f), None))
    return out


# ---------------------------------------------------------------- 模型
_NET = None


def get_net():
    global _NET
    if _NET is None:
        net = model.enhance_net_nopool()
        net.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
        net.eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _NET = net
    return _NET


def _local_std(ch, k):
    m = cv2.boxFilter(ch, -1, (k, k))
    m2 = cv2.boxFilter(ch * ch, -1, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def chroma_plateau(rgb, mask, step=6, amp_thr=8.0):
    """最大“有色”同色块面积占比（只统计明显有色的像素，排除中性平坦暗区）。

    为什么只算有色：近黑区域本来就没有颜色信息，抹成中性灰是"干净"而不是"色块"；
    真正难看的是带颜色的斑块（棕/绿/紫），所以要按饱和度筛。
    """
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    amp = np.sqrt((cr - 128.0) ** 2 + (cb - 128.0) ** 2)
    sat = (amp >= amp_thr) & mask
    key = (np.round(cr / step) * 200 + np.round(cb / step)).astype(np.int32)
    best = 0
    for v in np.unique(key[sat])[:200]:
        m = ((key == v) & sat).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            best = max(best, int(stats[1:, cv2.CC_STAT_AREA].max()))
    return best / (rgb.shape[0] * rgb.shape[1])


def luma_std8(rgb, mask):
    y = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32)
    return float(_local_std(y, 8)[mask].mean())

def enhance(rgb_u8):
    net = get_net()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(rgb_u8.astype(np.float32) / 255.0).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        _, out, _ = net(x)
    y = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.clip(y * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 指标
def sobel_M(rgb_u8):
    f = rgb_u8.astype(np.float32) / 255.0
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    M = []
    for c in range(3):
        ix = cv2.filter2D(f[:, :, c], -1, kx, borderType=cv2.BORDER_REPLICATE)
        iy = cv2.filter2D(f[:, :, c], -1, ky, borderType=cv2.BORDER_REPLICATE)
        M.append(np.abs(ix) + np.abs(iy))
    return np.mean(M, axis=0)


def chroma_noise(rgb_u8, mask):
    ycc = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    vals = []
    for c in (1, 2):
        ch = ycc[:, :, c]
        resid = (ch - cv2.blur(ch, (5, 5)))[mask]
        vals.append(float(resid.std()) if resid.size else 0.0)
    return sum(vals)


def blockiness(gray_u8):
    g = gray_u8.astype(np.float32)
    dv = np.abs(np.diff(g, axis=1))
    dh = np.abs(np.diff(g, axis=0))
    cols = np.arange(dv.shape[1])
    rows = np.arange(dh.shape[0])
    num = dv[:, (cols % 8) == 7].mean() + dh[(rows % 8) == 7, :].mean()
    den = dv[:, (cols % 8) != 7].mean() + dh[(rows % 8) != 7, :].mean() + 1e-9
    return float(num / den)


def psnr(a, b):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-12 else float(10.0 * np.log10(255.0 ** 2 / mse))


def ssim_rgb(a, b):
    """RGB 3 通道分别算 SSIM 再平均（11x11 高斯, σ=1.5），与项目其它脚本口径一致。"""
    def _ssim(x, y):
        x = x.astype(np.float64)
        y = y.astype(np.float64)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        mu1 = cv2.GaussianBlur(x, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(y, (11, 11), 1.5)
        s1 = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu1 ** 2
        s2 = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu2 ** 2
        s12 = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2)
        return float(np.mean(num / den))
    return float(np.mean([_ssim(a[:, :, c], b[:, :, c]) for c in range(3)]))


# ---------------------------------------------------------------- 管线
def run_pipeline(name, rgb):
    """返回该管线的输出（uint8 RGB）；E0 原样返回。"""
    if name == "E0":
        return rgb
    if name == "E1":
        return enhance(rgb)
    if name == "E2":
        return enhance(pp.denoise_chroma(rgb, 1.0))
    if name == "E3":
        return pp.denoise_chroma(enhance(rgb), 1.0)
    if name == "E4":
        pre = pp.denoise_chroma(rgb, 0.6)
        return pp.post_smooth(enhance(pre), 0.8)
    if name == "E5":
        # 旧方案：色度 NLM + 双边 —— 色噪最低但会把近黑区抹成大片色斑（见 e5_check 诊断）
        pre = pp.preprocess(rgb, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)
        out = enhance(pre)
        out = pp.denoise_chroma(out, 0.6)
        return pp.post_smooth(out, 0.6)
    if name == "E6":
        # 新方案：预处理轻一些 + 后处理改成“亮度引导的色度平滑”（只动色度、不动亮度）
        pre = pp.preprocess(rgb, denoise=0.6, deblock_strength=0.5, prelift_amount=0.15)
        out = enhance(pre)
        return pp.chroma_guided_smooth(out, luma_guide=out, radius=2, eps=1e-3, strength=0.7)
    if name == "E7":
        # E6 + 极暗无信号区抑制（保守参数：只处理真正没有信号的区域）
        out = run_pipeline("E6", rgb)
        return pp.suppress_low_snr(out, rgb, dark_thr=25, var_thr=5.0, strength=0.7,
                                   feather=10, max_gain=1.6)
    if name == "E8":
        # 本文推荐：E6 + 近黑区限增益（中性抬亮替代彩色放大，消除彩色大斑块）
        out = run_pipeline("E6", rgb)
        return pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.40, strength=0.85)
    raise ValueError(name)


PIPELINES = ["E0", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
DESC = {
    "E0": "低照原图（参考）",
    "E1": "原图→Zero-DCE（基线）",
    "E2": "色度降噪→Zero-DCE",
    "E3": "Zero-DCE→色度降噪",
    "E4": "弱降噪→Zero-DCE→保边平滑",
    "E5": "旧方案：重降噪+双边（会抹出大色块）",
    "E6": "本文方案：轻预处理+亮度引导色度平滑",
    "E7": "E6 + 极暗无信号区抑制",
    "E8": "E6 + 近黑区限增益（本文推荐）",
}


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="eval15,lime,extreme,dicm")
    ap.add_argument("--limit", type=int, default=0, help="每组最多处理几张（0=全部）")
    args = ap.parse_args()
    sets = [s.strip().lower() for s in args.sets.split(",") if s.strip()]

    os.makedirs(OUT, exist_ok=True)
    items = collect(sets)
    if not items:
        raise SystemExit("没有找到图片，请检查 --sets")
    print(f"[数据] {len(items)} 张：", ", ".join(f"{g} {sum(1 for it in items if it[0]==g)}" for g in dict.fromkeys(it[0] for it in items)))

    rows = []
    t0 = time.time()
    for i, (group, name, low_path, gt_path) in enumerate(items, 1):
        rgb = np.asarray(Image.open(low_path).convert("RGB"))
        gt = np.asarray(Image.open(gt_path).convert("RGB")) if gt_path else None
        if gt is not None and gt.shape != rgb.shape:
            gt = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < 40
        if mask.sum() < 0.02 * mask.size:          # 暗区太少就放宽，避免指标不稳
            mask = gray < 80
        m_in = float(sobel_M(rgb)[mask].mean())
        blk_in = blockiness(gray)
        bright_in = float(gray.mean())
        for pl in PIPELINES:
            out = run_pipeline(pl, rgb)
            g_out = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
            m_out = float(sobel_M(out)[mask].mean())
            row = dict(
                group=group, name=name, pipeline=pl,
                bright=float(g_out.mean()),
                chroma_noise=chroma_noise(out, mask),
                blockiness=blockiness(g_out),
                M_gain=float(m_out / max(m_in, 1e-6)),
                psnr=psnr(out, gt) if gt is not None else "",
                ssim=ssim_rgb(out, gt) if gt is not None else "",
                dark_frac=float(mask.mean()), bright_in=bright_in,
                chroma_in=chroma_noise(rgb, mask), blockiness_in=blk_in,
                plateau=chroma_plateau(out, mask), luma_std8=luma_std8(out, mask),
            )
            rows.append(row)
        print(f"[{i}/{len(items)}] {group}/{name}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"色噪 E1={rows[-5]['chroma_noise']:.2f} → E5={rows[-1]['chroma_noise']:.2f}  "
              f"块效应 E1={rows[-5]['blockiness']:.2f} → E5={rows[-1]['blockiness']:.2f}  ({time.time()-t0:.0f}s)")

    # ---- CSV
    cols = ["group", "name", "pipeline", "bright", "bright_in", "chroma_noise", "chroma_in",
            "blockiness", "blockiness_in", "M_gain", "psnr", "ssim", "dark_frac",
            "plateau", "luma_std8"]
    csv_path = os.path.join(OUT, "metrics.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]) for k in cols})

    # ---- 分组均值
    groups = list(dict.fromkeys(r["group"] for r in rows))
    summary = ["# 马赛克预处理对照实验结果（E0~E7）", ""]
    summary += ["> 指标都在“输入暗区掩码”内统计；色噪/块效应/M放大 越小越好；PSNR/SSIM 只在有 GT 的 eval15 上算。", ""]
    for g in groups:
        summary.append(f"## {g}")
        summary.append("")
        has_gt = any(r["psnr"] != "" for r in rows if r["group"] == g)
        head = "| 管线 | 说明 | 亮度 | 色噪 | 块效应比 | M放大 | 彩色块占比 | 亮度细节(std8) |"
        sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"
        if has_gt:
            head += " PSNR | SSIM |"
            sep += " --- | --- |"
        summary += [head, sep]
        for pl in PIPELINES:
            rs = [r for r in rows if r["group"] == g and r["pipeline"] == pl]
            if not rs:
                continue
            n = len(rs)
            line = (f"| {pl} | {DESC[pl]} | {sum(r['bright'] for r in rs)/n:.1f} | "
                    f"{sum(r['chroma_noise'] for r in rs)/n:.3f} | "
                    f"{sum(r['blockiness'] for r in rs)/n:.3f} | "
                    f"{sum(r['M_gain'] for r in rs)/n:.2f}× | "
                    f"{sum(r['plateau'] for r in rs)/n:.4f} | "
                    f"{sum(r['luma_std8'] for r in rs)/n:.2f} |")
            if has_gt:
                line += (f" {sum(float(r['psnr']) for r in rs)/n:.2f} | "
                         f"{sum(float(r['ssim']) for r in rs)/n:.4f} |")
            summary.append(line)
        summary.append("")
    summary += ["## 怎么读", "",
                "- **E1 → E2**：把降噪放到增强之前 → 色噪和块效应应该明显下降（核心结论）",
                "- **E1 → E3**：降噪放到增强之后 → 也有改善，但不如 E2（噪声已被放大成结构化色块）",
                "- **E5**：旧方案（重降噪+双边）→ 色噪最低，但会把近黑区抹成大片色斑（彩色块占比升高、亮度细节下降）",
                "- **E6**：本文方案（轻预处理 + 亮度引导色度平滑）→ 色噪仍低，同时不糊亮度、不产生大色块",
                "- **E7**：E6 + 极暗无信号区抑制 → 过强，整体变暗、PSNR 掉（不推荐）",
                "- **E8**：E6 + 近黑区限增益（中性抬亮替代彩色放大）→ 彩色大斑块基本消失，本文推荐",
                "- 代价：降噪越强，细节（尤其暗部纹理）越容易被抹平，需在“干净”和“细节”之间取平衡",
                ""]
    md_path = os.path.join(OUT, "summary.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(summary))

    print(f"\n[完成] CSV: {csv_path}\n[完成] 汇总: {md_path}")


if __name__ == "__main__":
    main()

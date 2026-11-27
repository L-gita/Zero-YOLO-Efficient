# -*- coding: utf-8 -*-
"""诊断：E5（本文方案）是否把"小马赛克"变成了"大色块/更模糊"？

对比 E1（基线）与 E5（我们的预处理+后处理）：
1. 多尺度色度方差：窗口 3/8/16/32 内 CbCr 的标准差 → 看平滑是否把色度抹成了大片均匀区
2. 多尺度亮度方差：同上（判断整体是否更模糊）
3. 色度-亮度边缘对齐度：色度梯度与亮度梯度的余弦相似度 → 判断色块是否"糊到不该糊的地方"
4. 色度平台尺寸：把色度量化后统计连通域，看最大色块面积
5. 生成 6× 放大对比图 + 色度通道可视化（把 CbCr 反变换成伪彩色图）
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import torch  # noqa: E402
import model  # noqa: E402
import preprocess as pp  # noqa: E402

OUT = os.path.join(ROOT, "demo_mosaic", "e5_check")
os.makedirs(OUT, exist_ok=True)
FONT = r"C:\Windows\Fonts\msyh.ttc"

CASES = [
    ("extreme_126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
    ("eval15_748", os.path.join(ROOT, "lol_test", "eval15", "low", "748.png")),
    ("LIME_1", os.path.join(ROOT, "低照图库", "02_LIME", "1.bmp")),
]


def load_model():
    net = model.enhance_net_nopool().cuda()
    sd = torch.load(os.path.join(CODE, "snapshots_sice", "Epoch50.pth"), map_location="cuda")
    net.load_state_dict(sd)
    net.eval()
    return net


def run(net, rgb):
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(x)
    o = out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return (o * 255).round().astype(np.uint8)


def pipeline_e1(net, rgb):
    return run(net, rgb)


def pipeline_e5(net, rgb):
    pre = pp.preprocess(rgb, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)
    out = run(net, pre)
    out = pp.denoise_chroma(out, 0.6)
    out = pp.post_smooth(out, 0.6)
    return out


def pipeline_e5b(net, rgb):
    """候选改进：亮度引导的色度平滑（不糊亮度），色度降噪更轻，后处理只做引导平滑"""
    pre = pp.preprocess(rgb, denoise=0.6, deblock_strength=1.0, prelift_amount=0.15)
    out = run(net, pre)
    out = pp.chroma_guided_smooth(out, luma_guide=out, radius=2, eps=1e-3, strength=0.8)
    return out


def chroma_decompose(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    return y.astype(np.float32), cr.astype(np.float32), cb.astype(np.float32)


def local_std(ch, k):
    m = cv2.boxFilter(ch, -1, (k, k))
    m2 = cv2.boxFilter(ch * ch, -1, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0))


def multiscale(rgb, mask):
    y, cr, cb = chroma_decompose(rgb)
    chroma = np.sqrt((cr - 128) ** 2 + (cb - 128) ** 2)
    out = {}
    for k in (3, 8, 16, 32):
        cy = local_std(chroma, k)[mask].mean()
        ly = local_std(y, k)[mask].mean()
        out[f"chroma_std_{k}"] = float(cy)
        out[f"luma_std_{k}"] = float(ly)
    return out


def chroma_luma_alignment(rgb, mask):
    y, cr, cb = chroma_decompose(rgb)
    gx_c = cv2.Sobel(cr, cv2.CV_32F, 1, 0, ksize=3)
    gy_c = cv2.Sobel(cr, cv2.CV_32F, 0, 1, ksize=3)
    gx_l = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy_l = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    num = gx_c * gx_l + gy_c * gy_l
    den = np.sqrt(gx_c ** 2 + gy_c ** 2) * np.sqrt(gx_l ** 2 + gy_l ** 2) + 1e-9
    sim = num / den
    return float(sim[mask].mean())


def chroma_plateau(rgb, mask, step=4):
    """把 Cb/Cr 量化到 step 级，统计最大同色连通块面积（归一化到图像面积）"""
    y, cr, cb = chroma_decompose(rgb)
    key = ((np.round(cr / step) * 100 + np.round(cb / step)).astype(np.int32))
    best = 0
    for v in np.unique(key[mask])[:60]:
        m = (key == v).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            best = max(best, int(stats[1:, cv2.CC_STAT_AREA].max()))
    return best / (rgb.shape[0] * rgb.shape[1])


def crop_zoom(img, box, scale):
    x0, y0, x1, y1 = box
    patch = img[y0:y1, x0:x1]
    return cv2.resize(patch, ((x1 - x0) * scale, (y1 - y0) * scale), interpolation=cv2.INTER_NEAREST)


def label(img, text, bar_h=22):
    h, w = img.shape[:2]
    canvas = np.full((h + bar_h, w, 3), 18, np.uint8)
    canvas[bar_h:] = img
    pil = Image.fromarray(canvas)
    d = ImageDraw.Draw(pil)
    try:
        f = ImageFont.truetype(FONT, 15)
    except Exception:
        f = ImageFont.load_default()
    d.text((4, 3), text, fill=(235, 235, 235), font=f)
    return np.asarray(pil)


def main():
    net = load_model()
    rows = []
    for name, path in CASES:
        if not os.path.exists(path):
            print("[skip]", path)
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < 60
        if mask.mean() < 0.02:
            mask = gray < 90
        outs = {"E1": pipeline_e1(net, rgb), "E5": pipeline_e5(net, rgb), "E5b": pipeline_e5b(net, rgb)}
        for k, v in outs.items():
            m = multiscale(v, mask)
            row = {"case": name, "pipe": k}
            row.update(m)
            row["chroma_luma_align"] = chroma_luma_alignment(v, mask)
            row["chroma_plateau"] = chroma_plateau(v, mask)
            rows.append(row)
            print(name, k, f"chroma_std32={m['chroma_std_32']:.2f} luma_std8={m['luma_std_8']:.2f} "
                  f"align={row['chroma_luma_align']:.3f} plateau={row['chroma_plateau']:.4f}")

        # 视觉：6× 放大（选暗区里色噪最强的窗口）
        dark = mask.astype(np.float32)
        cs = local_std(cv2.cvtColor(outs["E1"], cv2.COLOR_RGB2YCrCb)[:, :, 1].astype(np.float32), 16)
        score = cv2.boxFilter(cs * dark, -1, (64, 64))
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        h, w = rgb.shape[:2]
        half = 70
        box = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
        scale = 6
        panels = [label(crop_zoom(outs[k], box, scale), k) for k in ["E1", "E5", "E5b"]]
        # 色度伪彩图：只保留 CbCr（Y=128），放大看色块
        chroma_panels = []
        for k in ["E1", "E5", "E5b"]:
            ycc = cv2.cvtColor(outs[k], cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycc)
            fake = cv2.cvtColor(cv2.merge([np.full_like(y, 128), cr, cb]), cv2.COLOR_YCrCb2RGB)
            chroma_panels.append(label(crop_zoom(fake, box, scale), f"{k} 色度(CbCr)通道"))
        sheet = np.vstack([np.hstack(panels), np.hstack(chroma_panels)])
        Image.fromarray(sheet).save(os.path.join(OUT, f"zoom6x_{name}.png"))
        print("[ok]", os.path.join(OUT, f"zoom6x_{name}.png"))

    import csv
    with open(os.path.join(OUT, "e5_check.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[ok] e5_check.csv")


if __name__ == "__main__":
    main()

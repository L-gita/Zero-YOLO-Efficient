# -*- coding: utf-8 -*-
"""E6 方案试验：解决 E5 "过度平滑 → 更大色块 + 更模糊" 的问题。

E5 的问题：色度 NLM(h=12) + 双边(sigmaColor=12) 在近黑区域把噪声抹成大片均匀色斑，
           同时亮度也被平滑（luma_std 下降），所以"块更大、图更糊"。

E6 的设计：
  增强前：去块(0.5) + 轻度色度降噪(0.6) + 暗部轻抬(0.15)
  增强后：亮度引导的色度平滑（guided filter，radius=2，只动色度、不动亮度）
  最后  ：极暗无信号区抑制（去饱和 + 压回亮度，避免彩色大斑块）

对比：E1 基线 / E5 现方案 / E6a（只换后处理）/ E6b（E6a + 无信号区抑制）
"""
import os
import sys
import csv
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
import diag_e5_blocks as dg  # noqa: E402

OUT = os.path.join(ROOT, "demo_mosaic", "e6_try")
os.makedirs(OUT, exist_ok=True)
FONT = r"C:\Windows\Fonts\msyh.ttc"

CASES = [
    ("extreme_126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
    ("extreme_147_0", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "147_0_.png")),
    ("eval15_748", os.path.join(ROOT, "lol_test", "eval15", "low", "748.png")),
    ("eval15_748_gt", os.path.join(ROOT, "lol_test", "eval15", "high", "748.png")),
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


def pipe_e1(net, rgb):
    return run(net, rgb)


def pipe_e5(net, rgb):
    pre = pp.preprocess(rgb, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)
    out = run(net, pre)
    out = pp.denoise_chroma(out, 0.6)
    return pp.post_smooth(out, 0.6)


def pipe_e6a(net, rgb):
    pre = pp.preprocess(rgb, denoise=0.6, deblock_strength=0.5, prelift_amount=0.15)
    out = run(net, pre)
    return pp.chroma_guided_smooth(out, luma_guide=out, radius=2, eps=1e-3, strength=0.7)


def pipe_e6b(net, rgb):
    out = pipe_e6a(net, rgb)
    return pp.suppress_low_snr(out, rgb, dark_thr=45, var_thr=10.0, strength=1.0,
                               feather=14, max_gain=2.0)


PIPES = {"E1": pipe_e1, "E5": pipe_e5, "E6a": pipe_e6a, "E6b": pipe_e6b}


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
        if "gt" in name or not os.path.exists(path):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        mask = gray < 60
        if mask.mean() < 0.02:
            mask = gray < 90
        outs = {k: fn(net, rgb) for k, fn in PIPES.items()}
        for k, v in outs.items():
            m = dg.multiscale(v, mask)
            row = {"case": name, "pipe": k}
            row.update(m)
            row["chroma_luma_align"] = dg.chroma_luma_alignment(v, mask)
            row["chroma_plateau"] = dg.chroma_plateau(v, mask)
            rows.append(row)
            print(f"{name:16s} {k:4s} chroma_std32={m['chroma_std_32']:6.2f} "
                  f"luma_std8={m['luma_std_8']:6.2f} align={row['chroma_luma_align']:+.3f} "
                  f"plateau={row['chroma_plateau']:.4f}")

        # 6× 放大：挑暗区里 E1 色噪最强的窗口
        ycc1 = cv2.cvtColor(outs["E1"], cv2.COLOR_RGB2YCrCb)
        cs = dg.local_std(ycc1[:, :, 1].astype(np.float32), 16)
        score = cv2.boxFilter(cs * mask.astype(np.float32), -1, (64, 64))
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        h, w = rgb.shape[:2]
        half = 70
        box = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))
        scale = 6
        keys = ["E1", "E5", "E6a", "E6b"]
        p1 = [label(dg.crop_zoom(outs[k], box, scale), k) for k in keys]
        p2 = [label(dg.crop_zoom(outs[k], box, scale), k) for k in keys]

        def chroma_view(img_rgb):
            ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycc)
            return cv2.cvtColor(cv2.merge([np.full_like(y, 128), cr, cb]), cv2.COLOR_YCrCb2RGB)

        p3 = [label(dg.crop_zoom(chroma_view(outs[k]), box, scale), f"{k} 色度通道") for k in keys]
        sheet = np.vstack([np.hstack(p1), np.hstack(p2), np.hstack(p3)])
        Image.fromarray(sheet).save(os.path.join(OUT, f"e6_zoom6x_{name}.png"))
        print("[ok]", os.path.join(OUT, f"e6_zoom6x_{name}.png"))

    with open(os.path.join(OUT, "e6_metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("[ok] e6_metrics.csv")

    # 分组均值
    cases = sorted({r["case"] for r in rows})
    print("\n=== 均值 ===")
    for c in cases:
        for k in PIPES:
            vals = [r for r in rows if r["case"] == c and r["pipe"] == k]
            if not vals:
                continue
            v = vals[0]
            print(f"{c:16s} {k:4s} chroma={v['chroma_std_32']:6.2f} luma8={v['luma_std_8']:6.2f} "
                  f"align={v['chroma_luma_align']:+.3f} plateau={v['chroma_plateau']:.4f}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""专门量化“彩色大斑块”：只统计明显有色（色度幅度>阈值）的连通块面积。

原因：上一轮的 chroma_plateau 把“中性平坦暗区”也算进去了，而输入暗区天然是中性平坦的，
      所以那个指标不能直接说明“变丑”。这里只看“有色”的斑块。
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

OUT = os.path.join(ROOT, "demo_mosaic", "color_blobs")
os.makedirs(OUT, exist_ok=True)
FONT = r"C:\Windows\Fonts\msyh.ttc"

CASES = [
    ("extreme_126_3", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "126_3_.png")),
    ("extreme_147_0", os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp", "147_0_.png")),
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
    return (out[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def pipe_e1(net, rgb):
    return run(net, rgb)


def pipe_e5(net, rgb):
    pre = pp.preprocess(rgb, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)
    out = run(net, pre)
    out = pp.denoise_chroma(out, 0.6)
    return pp.post_smooth(out, 0.6)


def pipe_e6(net, rgb):
    pre = pp.preprocess(rgb, denoise=0.6, deblock_strength=0.5, prelift_amount=0.15)
    out = run(net, pre)
    return pp.chroma_guided_smooth(out, luma_guide=out, radius=2, eps=1e-3, strength=0.7)


def pipe_e8(net, rgb):
    out = pipe_e6(net, rgb)
    return pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.40, strength=0.85)


def chroma_amp(rgb):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr = ycc[:, :, 1] - 128.0
    cb = ycc[:, :, 2] - 128.0
    return np.sqrt(cr * cr + cb * cb)


def colored_blob(rgb, amp_thr=8.0, step=6):
    """明显有色区域里，最大同色连通块占全图的比例"""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cr, cb = ycc[:, :, 1], ycc[:, :, 2]
    amp = chroma_amp(rgb)
    sat = amp >= amp_thr
    key = (np.round(cr / step) * 200 + np.round(cb / step)).astype(np.int32)
    best, total = 0, 0
    vals = np.unique(key[sat])
    if len(vals) > 400:
        vals = vals[:400]
    for v in vals:
        m = ((key == v) & sat).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n > 1:
            a = int(stats[1:, cv2.CC_STAT_AREA].max())
            best = max(best, a)
            total += int(stats[1:, cv2.CC_STAT_AREA].sum())
    area = rgb.shape[0] * rgb.shape[1]
    return best / area, total / area


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
    pipes = {"E1": pipe_e1, "E5": pipe_e5, "E6": pipe_e6, "E8": pipe_e8}
    print(f"{'case':16s} {'pipe':4s} {'最大有色块占比':>12s} {'有色总面积占比':>12s}")
    for name, path in CASES:
        if not os.path.exists(path):
            continue
        rgb = np.asarray(Image.open(path).convert("RGB"))
        outs = {k: fn(net, rgb) for k, fn in pipes.items()}
        for k, v in outs.items():
            blob, tot = colored_blob(v)
            print(f"{name:16s} {k:4s} {blob*100:11.2f}% {tot*100:11.2f}%")
        # 色度通道 6× 放大图（选 E1 有色最强的窗口）
        amp = chroma_amp(outs["E1"])
        score = cv2.boxFilter(amp, -1, (48, 48))
        cy, cx = np.unravel_index(np.argmax(score), score.shape)
        h, w = rgb.shape[:2]
        half = 70
        box = (max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half))

        def chroma_view(img_rgb):
            ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycc)
            return cv2.cvtColor(cv2.merge([np.full_like(y, 128), cr, cb]), cv2.COLOR_YCrCb2RGB)

        panels = []
        for k in ["E1", "E5", "E6", "E8"]:
            p = chroma_view(outs[k])[box[1]:box[3], box[0]:box[2]]
            p = cv2.resize(p, ((box[2] - box[0]) * 6, (box[3] - box[1]) * 6), interpolation=cv2.INTER_NEAREST)
            panels.append(label(p, f"{k} 色度通道"))
        Image.fromarray(np.hstack(panels)).save(os.path.join(OUT, f"chroma6x_{name}.png"))
        print("[ok]", os.path.join(OUT, f"chroma6x_{name}.png"))


if __name__ == "__main__":
    main()

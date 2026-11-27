"""基线 vs 去马赛克(auto / E9) 三列对比：输入 | 基线增强 | 去马赛克增强
用法：
    python scripts\\compare_preprocess.py 图片或文件夹... [-o 输出目录] [--mode auto|mosaic] [--zoom 3]
会为每张图输出 xxx_对比.png（上排三列 + 下排 3× 局部放大）与 metrics.csv / summary.md
"""
import os
import sys
import argparse
import csv
import numpy as np
import cv2
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import model  # noqa: E402
import preprocess as pp  # noqa: E402

FONT = r"C:\Windows\Fonts\msyh.ttc"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CKPT = os.path.join(CODE, "snapshots_sice", "Epoch50.pth")


def load_net():
    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))
    net.eval()
    return net


def enhance(net, rgb):
    t = torch.from_numpy(rgb.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, o, _ = net(t)
    return np.clip(o[0].permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)


def chroma_noise(rgb):
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    return float(np.mean([np.std(ycc[..., ci] - cv2.GaussianBlur(ycc[..., ci], (0, 0), 1.5)) for ci in (1, 2)]))


def luma_std8(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = 8
    h, w = g.shape[0] // k * k, g.shape[1] // k * k
    b = g[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    return float(b.std(axis=1).mean())


def band_jump(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = g < 40
    if m.mean() < 0.02:
        m = g < 80
    v = g[m]
    if v.size < 10:
        return 0.0
    h = np.bincount(v, minlength=256).astype(np.float64) / v.size
    big = [i for i, p in enumerate(h) if p > 0.01]
    return float(max(np.diff(big))) if len(big) > 1 else 0.0


def zoom_box(rgb):
    """选"最暗且纹理最多"的区域，放大后最容易看出马赛克"""
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    H, W = g.shape
    bw, bh = min(140, W // 2), min(100, H // 2)
    score = cv2.blur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (bw, bh)) - 0.5 * cv2.blur(g, (bw, bh))
    _, _, _, (mx, my) = cv2.minMaxLoc(score)
    return max(0, min(W - bw, mx - bw // 2)), max(0, min(H - bh, my - bh // 2)), bw, bh


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.lower().endswith(IMG_EXT):
                    out.append(os.path.join(p, f))
        elif os.path.isfile(p):
            out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--mode", choices=["auto", "mosaic", "light", "dark"], default="auto")
    ap.add_argument("--zoom", type=float, default=3.0)
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        raise SystemExit("没有找到图片")
    out_dir = args.out or os.path.join(os.path.dirname(files[0]), "对比结果_基线vs去马赛克")
    os.makedirs(out_dir, exist_ok=True)

    net = load_net()
    rows = []
    for i, f in enumerate(files, 1):
        try:
            rgb = np.asarray(Image.open(f).convert("RGB"))
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] 跳过 {os.path.basename(f)}: {e}")
            continue
        base = enhance(net, rgb)
        info = pp.auto_preprocess(rgb)
        enh2 = enhance(net, info["img"])
        if args.mode == "auto":
            if info["mean"] < 5.0:
                enh2 = pp.chroma_guided_smooth(enh2, luma_guide=enh2, radius=2, eps=1e-3, strength=0.5)
                enh2 = pp.quantize8(pp.limit_dark_gain(enh2, rgb, l0=3.0, l1=12.0, gamma=0.55,
                                                      strength=0.80, smooth_radius=8, dither=0.0), dither=0.4)
            else:
                amp_ref = pp.chroma_amp(enh2)
                enh2 = pp.chroma_denoise_flat(enh2, rgb, info["noise"],
                                              strength=min(1.0, max(0.35, info["k"] * 0.7)),
                                              target_amp=amp_ref * 0.98)
                enh2 = pp.quantize8(enh2, dither=0.4)
            tag = "auto(E10)"
        else:
            pre = pp.preprocess(rgb, denoise=0.5, deblock_strength=0.5, prelift_amount=0.10)
            e = enhance(net, pre)
            e = pp.chroma_guided_smooth(e, luma_guide=e, radius=2, eps=1e-3, strength=0.5)
            e = pp.limit_dark_gain(e, rgb, l0=3.0, l1=12.0, gamma=0.55, strength=0.80,
                                   smooth_radius=8, dither=0.6)
            enh2 = pp.quantize8(e, dither=0.6)
            tag = "E9"

        stem = os.path.splitext(os.path.basename(f))[0]
        x0, y0, bw, bh = zoom_box(rgb)
        panels = []
        for t, img in ((f"低照输入  亮度{rgb.mean():.1f}", rgb),
                       (f"基线 Zero-DCE  亮度{base.mean():.1f}", base),
                       (f"去马赛克 {tag}  亮度{enh2.mean():.1f}", enh2)):
            h, w = img.shape[:2]
            tw = 460
            r = Image.fromarray(img).resize((tw, max(50, int(tw * h / w))), Image.LANCZOS)
            c = Image.new("RGB", (r.width, r.height + 34), (26, 26, 26))
            c.paste(r, (0, 34))
            ImageDraw.Draw(c).text((6, 6), t, font=ImageFont.truetype(FONT, 18), fill=(240, 240, 240))
            panels.append(np.asarray(c))
        H = max(a.shape[0] for a in panels)
        top = np.full((H, sum(a.shape[1] for a in panels), 3), 26, np.uint8)
        x = 0
        for a in panels:
            top[:a.shape[0], x:x + a.shape[1]] = a
            x += a.shape[1]

        zs = []
        for t, img in (("输入", rgb), ("基线", base), ("去马赛克", enh2)):
            crop = img[y0:y0 + bh, x0:x0 + bw]
            z = cv2.resize(crop, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_NEAREST)
            c = Image.new("RGB", (z.shape[1], z.shape[0] + 30), (26, 26, 26))
            c.paste(Image.fromarray(z), (0, 30))
            ImageDraw.Draw(c).text((6, 5), f"{t} 放大{args.zoom:g}×", font=ImageFont.truetype(FONT, 18), fill=(240, 240, 240))
            zs.append(np.asarray(c))
        Hz = max(a.shape[0] for a in zs)
        bot = np.full((Hz, sum(a.shape[1] for a in zs), 3), 26, np.uint8)
        x = 0
        for a in zs:
            bot[:a.shape[0], x:x + a.shape[1]] = a
            x += a.shape[1]

        W = max(top.shape[1], bot.shape[1])
        sheet = np.full((top.shape[0] + bot.shape[0], W, 3), 26, np.uint8)
        sheet[:top.shape[0], :top.shape[1]] = top
        sheet[top.shape[0]:, :bot.shape[1]] = bot
        Image.fromarray(sheet).save(os.path.join(out_dir, f"{stem}_对比.png"))

        rows.append(dict(file=os.path.basename(f), mode=tag,
                         bright_in=round(float(rgb.mean()), 2),
                         bright_base=round(float(base.mean()), 2),
                         bright_out=round(float(enh2.mean()), 2),
                         chroma_base=round(chroma_noise(base), 3), chroma_out=round(chroma_noise(enh2), 3),
                         std8_base=round(luma_std8(base), 2), std8_out=round(luma_std8(enh2), 2),
                         band_base=band_jump(base), band_out=band_jump(enh2)))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  "
              f"色噪 {rows[-1]['chroma_base']}→{rows[-1]['chroma_out']}  "
              f"细节 {rows[-1]['std8_base']}→{rows[-1]['std8_out']}  "
              f"条带 {rows[-1]['band_base']}→{rows[-1]['band_out']}")

    if rows:
        with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        def avg(k):
            return float(np.mean([r[k] for r in rows]))
        lines = [
            "# 基线 vs 去马赛克 对比",
            "",
            f"共 {len(rows)} 张，模式 `{rows[0]['mode']}`。逐图看 `*_对比.png`（上排全图、下排 3× 局部）。",
            "",
            "| 指标 | 基线 | 去马赛克 |",
            "| --- | --- | --- |",
            f"| 色度噪声 std | {avg('chroma_base'):.2f} | **{avg('chroma_out'):.2f}** |",
            f"| 亮度细节 std8 | {avg('std8_base'):.2f} | {avg('std8_out'):.2f} |",
            f"| 暗区条带跳变 | {avg('band_base'):.1f} | **{avg('band_out'):.1f}** |",
            f"| 平均亮度 | {avg('bright_base'):.1f} | {avg('bright_out'):.1f} |",
            "",
            "读法：色噪越低越干净；std8 越接近基线说明细节保留得越好；条带跳变越小说明灰块/色阶越少。",
        ]
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"\n[完成] {len(rows)} 张 → {out_dir}")
        print(f"  汇总：{os.path.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()

r"""在官方 LOL eval15 测试集（15 对真实低照/正常光参考图）上做增强并生成对比图。

用法（在 Zero-DCE_code 目录下）：
    python ..\..\scripts\demo_pairs.py
输出：C:\Users\12983\Documents\repro\demo_pairs\
"""
import os
import sys
import csv
import glob
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))          # C:\...\repro
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
import model  # noqa: E402  (Zero-DCE 官方的 DCE-Net 定义)

LOW = os.path.join(ROOT, "lol_test", "eval15", "low")
HIGH = os.path.join(ROOT, "lol_test", "eval15", "high")
OUT = os.path.join(ROOT, "demo_pairs")
os.makedirs(OUT, exist_ok=True)

CKPTS = [
    ("官方权重 (Epoch99)", os.path.join(CODE, "snapshots", "Epoch99.pth")),
    ("复现模型 (SICE 50ep)", os.path.join(CODE, "snapshots_sice", "Epoch50.pth")),
    ("对照: LOL 训练 (200ep)", os.path.join(CODE, "snapshots_run", "Epoch199.pth")),
]

FONT = r"C:\Windows\Fonts\msyh.ttc"
PANEL_H = 400          # 每格显示高度（宽按 3:2 等比）


def load_net(ckpt):
    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(ckpt, map_location="cuda"))
    net.eval()
    return net


def enhance(net, pil):
    with torch.no_grad():
        x = np.asarray(pil, dtype=np.float32) / 255.0
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).cuda()
        _, out, _ = net(t)
        out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return 99.0 if mse <= 1e-12 else 10 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    g = lambda z: cv2.GaussianBlur(z, (11, 11), 1.5)
    mu_a, mu_b = g(a), g(b)
    sa, sb = g(a * a) - mu_a ** 2, g(b * b) - mu_b ** 2
    sab = g(a * b) - mu_a * mu_b
    return float(np.mean(((2 * mu_a * mu_b + C1) * (2 * sab + C2)) /
                         ((mu_a ** 2 + mu_b ** 2 + C1) * (sa + sb + C2))))


def strip(panels, labels, subs, path):
    """panels: list of PIL.Image（同尺寸），labels/subs：文字"""
    w, h = panels[0].size
    head = 34
    canvas = Image.new("RGB", (w * len(panels), h + head + 22), (18, 18, 18))
    d = ImageDraw.Draw(canvas)
    f1 = ImageFont.truetype(FONT, 20)
    f2 = ImageFont.truetype(FONT, 17)
    for i, (im, lb, sb) in enumerate(zip(panels, labels, subs)):
        canvas.paste(im, (i * w, head))
        d.text((i * w + 10, 8), lb, font=f1, fill=(255, 255, 255))
        if sb:
            d.text((i * w + 10, head + h + 1), sb, font=f2, fill=(150, 220, 150))
    canvas.save(path)
    return canvas


def fit(im, h=PANEL_H):
    w = int(round(im.width * h / im.height))
    return im.resize((w, h), Image.LANCZOS)


def main():
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LOW, "*.png")))
    print(f"eval15 图像: {len(names)} 张")
    nets = []
    for label, ckpt in CKPTS:
        if os.path.exists(ckpt):
            nets.append((label, load_net(ckpt)))
            print("loaded", label, os.path.basename(ckpt))
        else:
            print("missing", ckpt)

    rows, strips = [], []
    for name in names:
        low = Image.open(os.path.join(LOW, name)).convert("RGB")
        high = Image.open(os.path.join(HIGH, name)).convert("RGB")
        lo_np, hi_np = np.asarray(low), np.asarray(high)
        panels = [fit(low), fit(high)]
        labels = ["低照输入", "正常光参考 (GT)"]
        subs = [f"mean {lo_np.mean():.1f}", ""]
        row = {"image": name, "in_psnr": round(psnr(lo_np, hi_np), 3),
               "in_ssim": round(ssim(lo_np, hi_np), 4)}
        for label, net in nets:
            out = enhance(net, low)
            o_np = np.asarray(out)
            panels.append(fit(out))
            labels.append(label)
            p, s = psnr(o_np, hi_np), ssim(o_np, hi_np)
            subs.append(f"PSNR {p:.2f}  SSIM {s:.3f}  mean {o_np.mean():.1f}")
            key = label.split(" ")[0]
            row[f"{key}_psnr"], row[f"{key}_ssim"] = round(p, 3), round(s, 4)
        rows.append(row)
        strips.append(strip(panels, labels, subs, os.path.join(OUT, f"pair_{name}")))

    with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n%-8s %-22s %-22s %-22s" % ("image", "官方权重", "复现模型", "LOL训练对照"))
    for r in rows:
        def c(k): return f"{r[k+'_psnr']:.2f}/{r[k+'_ssim']:.3f}"
        print("%-8s %-22s %-22s %-22s" % (r["image"], c("官方权重"), c("复现模型"), c("对照:")))
    print("\n均值:")
    for k in ["官方权重", "复现模型", "对照:"]:
        ps = [r[k + "_psnr"] for r in rows]; ss = [r[k + "_ssim"] for r in rows]
        print(f"  {k:8s} PSNR {np.mean(ps):.3f}  SSIM {np.mean(ss):.4f}")
    print("  低照输入 PSNR %.3f  SSIM %.4f" % (np.mean([r['in_psnr'] for r in rows]),
                                              np.mean([r['in_ssim'] for r in rows])))


if __name__ == "__main__":
    main()

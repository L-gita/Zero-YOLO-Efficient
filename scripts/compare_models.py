# -*- coding: utf-8 -*-
"""把「基线模型（无边缘损失）」和「+Sobel 边缘损失模型」的增强结果并排输出，便于观察差异。

用法（在 C:\\Users\\12983\\Documents\\repro 下）：
    python scripts\\compare_models.py                      # 用默认的两个权重跑全部数据集
    python scripts\\compare_models.py --base snapshots_sice\\Epoch49.pth --edge snapshots_sice_edge\\Epoch49.pth
    python scripts\\compare_models.py --sets eval15 --tag ep50

输出：
    demo_edge\\<tag>\\*_compare.png   每张图的并排长图（低照输入 | 正常光GT | 基线 | 加边缘损失）
    demo_edge\\<tag>\\overview_*.png   各数据集的缩略总览
    demo_edge\\<tag>\\metrics.csv      逐图指标
    demo_edge\\<tag>\\summary.md       汇总表（可直接贴进报告）
"""
import argparse
import csv
import glob
import os
import sys
import warnings

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

if sys.platform == "win32":
    try:
        os.system("chcp 65001 >nul")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

REPRO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE = os.path.join(REPRO, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
warnings.filterwarnings("ignore")
import cv2  # noqa: E402
import model  # noqa: E402

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
LABEL_H = 30
BG = (24, 24, 24)
FG = (240, 240, 240)


# ----------------------------------------------------------------------------- 指标
def psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0:
        return 99.0
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ga = cv2.GaussianBlur(a, (11, 11), 1.5)
    gb = cv2.GaussianBlur(b, (11, 11), 1.5)
    gaa = cv2.GaussianBlur(a * a, (11, 11), 1.5)
    gbb = cv2.GaussianBlur(b * b, (11, 11), 1.5)
    gab = cv2.GaussianBlur(a * b, (11, 11), 1.5)
    va, vb, cab = gaa - ga * ga, gbb - gb * gb, gab - ga * gb
    m = ((2 * ga * gb + C1) * (2 * cab + C2)) / ((ga ** 2 + gb ** 2 + C1) * (va + vb + C2))
    return float(m.mean())


def sobel_mag_np(rgb_u8):
    """论文 (4.11) 的逐通道 |Ix|+|Iy|，返回 HxWx3（0~1 尺度）。"""
    x = rgb_u8.astype(np.float32) / 255.0
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], np.float32)
    out = np.zeros_like(x)
    for c in range(3):
        ix = cv2.filter2D(x[:, :, c], cv2.CV_32F, kx, borderType=cv2.BORDER_REPLICATE)
        iy = cv2.filter2D(x[:, :, c], cv2.CV_32F, ky, borderType=cv2.BORDER_REPLICATE)
        out[:, :, c] = np.abs(ix) + np.abs(iy)
    return out


def ledge_np(enh_u8, low_u8):
    """论文 (4.12)：1/(3N) ΣΣ |M(low) - M(enh)|。"""
    return float(np.abs(sobel_mag_np(low_u8) - sobel_mag_np(enh_u8)).mean())


# ----------------------------------------------------------------------------- 模型
class Net:
    def __init__(self, path, name):
        self.name = name
        self.path = path
        if not os.path.exists(path):
            raise SystemExit(f"找不到权重：{path}")
        net = model.enhance_net_nopool()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        self.net = net.cuda() if torch.cuda.is_available() else net
        print(f"[权重] {name}: {path}")

    @torch.no_grad()
    def __call__(self, rgb):
        t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        t = t.cuda() if torch.cuda.is_available() else t
        _, out, _ = self.net(t)
        out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (out * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------- 拼图
_FONT = None


def font():
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype(FONT_PATH, 20)
        except Exception:  # noqa: BLE001
            _FONT = ImageFont.load_default()
    return _FONT


def strip(panels, labels, max_w=None):
    """panels: [PIL RGB]；每块上方一条标签栏。"""
    w, h = panels[0].size
    canvas = Image.new("RGB", (w * len(panels) + 6 * (len(panels) - 1), h + LABEL_H), BG)
    d = ImageDraw.Draw(canvas)
    x = 0
    for im, lab in zip(panels, labels):
        canvas.paste(im, (x, LABEL_H))
        d.text((x + 8, 5), lab, font=font(), fill=FG)
        x += w + 6
    if max_w and canvas.width > max_w:
        r = max_w / canvas.width
        canvas = canvas.resize((max_w, round(canvas.height * r)), Image.LANCZOS)
    return canvas


def grid(strips, cols=2, max_w=1500):
    w = min(s.width for s in strips)
    scaled = []
    for s in strips:
        r = w / s.width
        scaled.append(s.resize((w, round(s.height * r)), Image.LANCZOS))
    rows = (len(scaled) + cols - 1) // cols
    rh = max(s.height for s in scaled)
    g = Image.new("RGB", (w * cols + 6 * (cols - 1), rh * rows + 6 * (rows - 1)), BG)
    for i, s in enumerate(scaled):
        g.paste(s, ((i % cols) * (w + 6), (i // cols) * (rh + 6)))
    if g.width > max_w:
        r = max_w / g.width
        g = g.resize((max_w, round(g.height * r)), Image.LANCZOS)
    return g


# ----------------------------------------------------------------------------- 数据集
def load_im(path):
    im = Image.open(path)
    try:
        from PIL import ImageOps
        im = ImageOps.exif_transpose(im)
    except Exception:  # noqa: BLE001
        pass
    return im.convert("RGB")


def build_sets(sets):
    data = []
    if "eval15" in sets:
        low = sorted(glob.glob(os.path.join(REPRO, "lol_test", "eval15", "low", "*.png")))
        for p in low:
            gt = os.path.join(os.path.dirname(os.path.dirname(p)), "high", os.path.basename(p))
            data.append(("eval15", p, gt if os.path.exists(gt) else None))
    if "lime" in sets:
        for p in sorted(glob.glob(os.path.join(CODE, "data", "test_data", "LIME", "*"))):
            data.append(("LIME", p, None))
    if "dicm" in sets:
        for p in sorted(glob.glob(os.path.join(CODE, "data", "test_data", "DICM", "*"))):
            data.append(("DICM", p, None))
    if "sice" in sets:
        for p in sorted(glob.glob(os.path.join(CODE, "data", "train_data_sice", "*.jpg")))[:12]:
            data.append(("SICE", p, None))
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(CODE, "snapshots_sice", "Epoch49.pth"),
                    help="基线权重（无边缘损失）")
    ap.add_argument("--edge", default=os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth"),
                    help="加边缘损失的权重")
    ap.add_argument("--base-name", default="基线(sice)")
    ap.add_argument("--edge-name", default="+边缘损失(4.12)")
    ap.add_argument("--out", default=os.path.join(REPRO, "demo_edge"))
    ap.add_argument("--tag", default="ep49_edge022")
    ap.add_argument("--sets", default="eval15,lime,dicm")
    ap.add_argument("--limit-dicm", type=int, default=16)
    args = ap.parse_args()

    sets = [s.strip() for s in args.sets.split(",") if s.strip()]
    data = build_sets(sets)
    if args.limit_dicm and "dicm" in sets:
        d = [x for x in data if x[0] == "DICM"][: args.limit_dicm]
        data = [x for x in data if x[0] != "DICM"] + d
    if not data:
        raise SystemExit("没有找到任何图片")

    outdir = os.path.join(args.out, args.tag)
    os.makedirs(outdir, exist_ok=True)
    base = Net(args.base, args.base_name)
    edge = Net(args.edge, args.edge_name)

    rows = []
    strips_by_set = {}
    for idx, (name, p, gt) in enumerate(data, 1):
        pil = load_im(p)
        low = np.asarray(pil)
        b = base(low)
        e = edge(low)
        gt_arr = np.asarray(load_im(gt)) if gt else None

        rec = {
            "set": name,
            "file": os.path.basename(p),
            "size": f"{pil.width}x{pil.height}",
            "bright_in": round(float(low.mean()), 2),
            "bright_base": round(float(b.mean()), 2),
            "bright_edge": round(float(e.mean()), 2),
            "M_in": round(float(sobel_mag_np(low).mean()), 4),
            "M_base": round(float(sobel_mag_np(b).mean()), 4),
            "M_edge": round(float(sobel_mag_np(e).mean()), 4),
            "Ledge_base": round(ledge_np(b, low), 4),
            "Ledge_edge": round(ledge_np(e, low), 4),
        }
        if gt_arr is not None:
            h = min(gt_arr.shape[0], low.shape[0])
            w = min(gt_arr.shape[1], low.shape[1])
            g0, l0, b0, e0 = gt_arr[:h, :w], low[:h, :w], b[:h, :w], e[:h, :w]
            rec["psnr_in"] = round(psnr(l0, g0), 2)
            rec["psnr_base"] = round(psnr(b0, g0), 2)
            rec["psnr_edge"] = round(psnr(e0, g0), 2)
            # SSIM 按 3 通道(RGB)算，与 demo_pairs.py / 论文对比口径一致
            rec["ssim_in"] = round(ssim(l0, g0), 4)
            rec["ssim_base"] = round(ssim(b0, g0), 4)
            rec["ssim_edge"] = round(ssim(e0, g0), 4)
        rows.append(rec)

        panels = [pil]
        labels = ["低照输入"]
        if gt_arr is not None:
            panels.append(load_im(gt))
            labels.append("正常光参考(GT)")
        panels += [Image.fromarray(b), Image.fromarray(e)]
        labels += [args.base_name, args.edge_name]
        s = strip(panels, labels)
        base_name = os.path.splitext(os.path.basename(p))[0]
        s.save(os.path.join(outdir, f"{name}_{base_name}_compare.png"))
        strips_by_set.setdefault(name, []).append((s, base_name))
        print(f"[{idx}/{len(data)}] {name}/{os.path.basename(p)}  "
              f"亮度 {rec['bright_in']}→{rec['bright_base']}/{rec['bright_edge']}  "
              f"Ledge {rec['Ledge_base']}/{rec['Ledge_edge']}"
              + (f"  PSNR {rec['psnr_base']}/{rec['psnr_edge']}  SSIM {rec['ssim_base']}/{rec['ssim_edge']}"
                 if gt_arr is not None else ""))

    # ---- 总览
    for name, items in strips_by_set.items():
        cols = 2 if len(items) > 1 else 1
        grid([s for s, _ in items], cols=cols).save(
            os.path.join(outdir, f"overview_{name}.png"))

    # ---- CSV
    keys = ["set", "file", "size", "bright_in", "bright_base", "bright_edge",
            "M_in", "M_base", "M_edge", "Ledge_base", "Ledge_edge",
            "psnr_in", "psnr_base", "psnr_edge", "ssim_in", "ssim_base", "ssim_edge"]
    with open(os.path.join(outdir, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})

    # ---- 汇总
    def avg(k, sel):
        v = [r[k] for r in rows if sel(r) and k in r and r[k] != ""]
        return float(np.mean(v)) if v else float("nan")

    lines = ["# 边缘损失对比（自动生成）", "",
             f"- 基线：`{args.base_name}` — `{args.base}`",
             f"- 加边缘损失：`{args.edge_name}` — `{args.edge}`",
             f"- 指标：PSNR/SSIM 用 GT 计算（仅 eval15 有 GT）；M=平均 Sobel 梯度幅值；Ledge=论文(4.12) 值与低照输入之差", ""]
    for name in strips_by_set:
        sel = lambda r, n=name: r["set"] == n  # noqa: E731
        lines += [f"## {name}（{sum(1 for r in rows if r['set'] == name)} 张）", "",
                  "| 指标 | 低照输入 | 基线 | +边缘损失 |", "|---|---|---|---|",
                  f"| 平均亮度(0-255) | {avg('bright_in', sel):.2f} | {avg('bright_base', sel):.2f} | {avg('bright_edge', sel):.2f} |",
                  f"| 平均梯度幅值 M | {avg('M_in', sel):.4f} | {avg('M_base', sel):.4f} | {avg('M_edge', sel):.4f} |",
                  f"| L_edge vs 低照 | — | {avg('Ledge_base', sel):.4f} | {avg('Ledge_edge', sel):.4f} |"]
        if name == "eval15":
            lines += [f"| PSNR(dB) | {avg('psnr_in', sel):.2f} | {avg('psnr_base', sel):.2f} | {avg('psnr_edge', sel):.2f} |",
                      f"| SSIM | {avg('ssim_in', sel):.4f} | {avg('ssim_base', sel):.4f} | {avg('ssim_edge', sel):.4f} |"]
        lines.append("")
    with open(os.path.join(outdir, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n===== 汇总（平均）=====")
    for name in strips_by_set:
        sel = lambda r, n=name: r["set"] == n  # noqa: E731
        print(f"{name}: 亮度 {avg('bright_in', sel):.1f}→{avg('bright_base', sel):.1f}/{avg('bright_edge', sel):.1f} | "
              f"M {avg('M_in', sel):.4f}→{avg('M_base', sel):.4f}/{avg('M_edge', sel):.4f} | "
              f"Ledge {avg('Ledge_base', sel):.4f}/{avg('Ledge_edge', sel):.4f}"
              + (f" | PSNR {avg('psnr_base', sel):.2f}/{avg('psnr_edge', sel):.2f}"
                 f" | SSIM {avg('ssim_base', sel):.4f}/{avg('ssim_edge', sel):.4f}" if name == "eval15" else ""))
    print(f"\n输出目录：{outdir}")


if __name__ == "__main__":
    main()

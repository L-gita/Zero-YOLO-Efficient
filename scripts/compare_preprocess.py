"""原版 vs 去噪版 三列对比：原图 | 原版 Zero-DCE | 去噪版（auto 预处理）
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
# 顺序很重要：先把 ROOT 放最前（repro\enhance.py 在这里），再是 scripts，最后是代码目录
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, ROOT)
import model  # noqa: E402
import enhance as EN  # noqa: E402  # 复用它的 OOM 降级（fp16 → 分块）
import preprocess as pp  # noqa: E402

FONT = r"C:\Windows\Fonts\msyh.ttc"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
CKPT = os.path.join(CODE, "snapshots_sice", "Epoch50.pth")


def load_net():
    # 复用 enhance.py 的加载逻辑（CUDA 不可用时自动退化）
    return EN.get_net(CKPT)


def enhance(net, rgb):
    """整图推理；显存不够时 enhance.py 内部会自动降级 fp16 → 分块，避免 CUDA OOM 崩溃"""
    out, _mode = EN.enhance_image(net, Image.fromarray(rgb))
    return np.ascontiguousarray(np.asarray(out))


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


def _run_enhance_subprocess(img_path, mode="auto"):
    """调用 enhance.py（= 用户双击的 bat 用的同一份代码）得到最终版图像。"""
    import subprocess, tempfile
    tmp = tempfile.mkdtemp(prefix="cmp_")
    cmd = [sys.executable, os.path.join(ROOT, "enhance.py"), img_path,
           "-o", tmp, "--preprocess", mode]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    stem = os.path.splitext(os.path.basename(img_path))[0]
    # 注意：jpg 输入时 enhance.py 会存成 *_enhanced.jpg（保持扩展名），所以两种都找
    cand = [os.path.join(tmp, stem + "_enhanced.png"), os.path.join(tmp, stem + "_enhanced.jpg")]
    out = next((p for p in cand if os.path.exists(p)), None)
    if r.returncode != 0 or out is None:
        raise RuntimeError("enhance.py 失败：" + (r.stderr or r.stdout)[-800:])
    return np.asarray(Image.open(out).convert("RGB"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--keep-enhanced", action="store_true", help="把最终版单图也保存到输出目录 _enhanced/ 下（便于与 enhance.py 逐像素核对）")
    ap.add_argument("--mode", choices=["auto", "mosaic", "light", "dark"], default="auto")
    ap.add_argument("--zoom", type=float, default=3.0)
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        raise SystemExit("没有找到图片")
    out_dir = args.out or os.path.join(os.path.dirname(files[0]), "对比_原版vs去噪")
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
        dec = pp.auto_decision(rgb) if args.mode == "auto" else None
        # 最终版必须与用户双击的 bat 完全一致：直接调用 enhance.py（同一份代码、同一个入口）
        enh2 = _run_enhance_subprocess(f, args.mode)
        _stem = os.path.splitext(os.path.basename(f))[0]
        if getattr(args, "keep_enhanced", False):
            _kd = os.path.join(os.path.abspath(args.out), "_enhanced")
            os.makedirs(_kd, exist_ok=True)
            Image.fromarray(np.asarray(enh2, dtype=np.uint8)).save(os.path.join(_kd, _stem + "_final.png"))
        if args.mode == "auto" and dec["mode"] == "skip":
            tag = "去噪版(判定=跳过,等同原版)"
        elif args.mode == "auto":
            tag = f"去噪版({dec['mode']})"
        else:
            tag = f"去噪版({args.mode})"

        stem = os.path.splitext(os.path.basename(f))[0]
        x0, y0, bw, bh = zoom_box(rgb)
        panels = []
        for t, img in ((f"低照输入  亮度{rgb.mean():.1f}", rgb),
                       (f"原版 Zero-DCE  亮度{base.mean():.1f}", base),
                       (f"{tag}  亮度{enh2.mean():.1f}", enh2)):
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
        for t, img in (("原图", rgb), ("原版", base), ("去噪", enh2)):
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
            "# 原版 vs 去噪版 对比",
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
            "读法：色噪越低越干净；std8 越接近原版说明细节保留得越好；条带跳变越小说明灰块/色阶越少。",
        ]
        with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        print(f"\n[完成] {len(rows)} 张 → {out_dir}")
        print(f"  汇总：{os.path.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()

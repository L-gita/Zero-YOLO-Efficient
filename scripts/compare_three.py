"""三版本对比：输入 / 原版 Zero-DCE / 最终版(auto)

用法：
  python scripts\\compare_four.py 图片或文件夹... [-o 输出目录] [--per-sheet 4]

输出（默认 <第一个输入旁的>对比_四版本）：
  *_三版本.png     每张图一行：输入 | 原版 | 最终版
  metrics.csv      每张图的亮度 / 色度噪声 / 细节 std8 / 暗区条带
  summary.md       均值表 + 判定分支说明

说明：三条增强结果都由 enhance.py 子进程真实生成（与双击 bat 同一条代码路径），
      不使用任何内部复刻管线，保证"图 = 程序真实输出"。
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import preprocess as pp  # noqa: E402

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def metrics(rgb):
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    chroma = float(np.mean([np.std(ycc[..., ci] - cv2.GaussianBlur(ycc[..., ci], (0, 0), 1.5))
                            for ci in (1, 2)]))
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = 8
    h, w = g.shape[0] // k * k, g.shape[1] // k * k
    blocks = g[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    std8 = float(blocks.std(axis=1).mean())
    gi = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = gi < 40
    if m.mean() < 0.02:
        m = gi < 80
    v = gi[m]
    if v.size < 10:
        jump = 0.0
    else:
        hist = np.bincount(v, minlength=256).astype(np.float64) / v.size
        big = [i for i, p in enumerate(hist) if p > 0.01]
        jump = float(max(np.diff(big))) if len(big) > 1 else 0.0
    return dict(bright=float(g.mean()), chroma=round(chroma, 3),
                std8=round(std8, 2), jump=round(jump, 1))


def run_enhance(img_path, mode_args, tag, cache_dir=None):
    stem0 = os.path.splitext(os.path.basename(img_path))[0]
    if cache_dir:
        cached = os.path.join(cache_dir, tag, stem0 + ".png")
        if os.path.exists(cached):
            with Image.open(cached) as im:
                return np.asarray(im.convert("RGB"))
    tmp = tempfile.mkdtemp(prefix="four_")
    cmd = [sys.executable, os.path.join(ROOT, "enhance.py"), img_path, "-o", tmp] + mode_args
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cand = [os.path.join(tmp, stem + "_enhanced" + e) for e in (".png", ".jpg", ".jpeg", ".bmp")]
    out = next((p for p in cand if os.path.exists(p)), None)
    if r.returncode != 0 or out is None:
        raise RuntimeError(f"{tag} 失败：{(r.stderr or r.stdout)[-400:]}")
    with Image.open(out) as im:
        arr = np.asarray(im.convert("RGB"))
    if cache_dir:
        os.makedirs(os.path.join(cache_dir, tag), exist_ok=True)
        Image.fromarray(arr).save(os.path.join(cache_dir, tag, stem0 + ".png"))
    return arr


def collect(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for dirpath, _, names in os.walk(p):
                for n in sorted(names):
                    if os.path.splitext(n)[1].lower() in EXTS:
                        files.append(os.path.join(dirpath, n))
        elif os.path.splitext(p)[1].lower() in EXTS:
            files.append(p)
    return files


def tile(arr, h=300):
    return np.asarray(Image.fromarray(arr).resize((int(h * arr.shape[1] / arr.shape[0]), h),
                                                 Image.LANCZOS))


def label(arr, line1, line2=""):
    f = ImageFont.truetype(FONT_PATH, 15)
    c = Image.new("RGB", (arr.shape[1], 300 + 42), (24, 24, 24))
    c.paste(Image.fromarray(arr), (0, 42))
    d = ImageDraw.Draw(c)
    d.text((5, 3), line1, font=f, fill=(245, 245, 245))
    d.text((5, 22), line2, font=f, fill=(150, 205, 255))
    return np.asarray(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--per-sheet", type=int, default=4)
    args = ap.parse_args()

    files = collect(args.inputs)
    if not files:
        print("没有找到图片")
        return 1
    out = args.out or os.path.join(os.path.dirname(files[0]), "对比_三版本")
    os.makedirs(out, exist_ok=True)

    rows, tiles = [], []
    for i, f in enumerate(files, 1):
        with Image.open(f) as im:
            low = np.asarray(im.convert("RGB"))
        dec = pp.auto_decision(low)
        m_in = metrics(low)
        cache = os.path.join(out, "_raw")
        base = run_enhance(f, ["-m", "sice"], "base", cache)
        final = run_enhance(f, ["--preprocess", "auto"], "final", cache)
        m_b, m_f = metrics(base), metrics(final)
        rows.append(dict(file=os.path.basename(f), mode=dec["mode"], chroma_noise=round(dec["chroma_noise"], 3),
                         bright_in=m_in["bright"], bright_base=m_b["bright"], bright_final=m_f["bright"],
                         chroma_in=m_in["chroma"], chroma_base=m_b["chroma"], chroma_final=m_f["chroma"],
                         std8_in=m_in["std8"], std8_base=m_b["std8"], std8_final=m_f["std8"],
                         jump_in=m_in["jump"], jump_base=m_b["jump"], jump_final=m_f["jump"]))
        mode_cn = {"near_black": "近黑", "skip": "仅色度", "normal": "常规"}.get(dec["mode"], dec["mode"])
        tiles.append(np.hstack([
            label(tile(low), f"{os.path.basename(f)}", f"输入·{mode_cn}·{m_in['chroma']}"),
            label(tile(base), "原版", f"{m_b['chroma']}"),
            label(tile(final), "最终版", f"{m_f['chroma']}"),
        ]))
        print(f"[{i}/{len(files)}] {os.path.basename(f)}  {mode_cn}  色噪 {m_b['chroma']}→{m_f['chroma']}")

    with open(os.path.join(out, "metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def avg(key, sub):
        vals = [r[f"{key}_{sub}"] for r in rows]
        return sum(vals) / len(vals)

    lines = ["# 三版本对比（输入 / 原版 / 最终版 / 单图分区混合）", "",
             "| 指标 | 输入 | 原版 | 最终版 auto |", "|---|---|---|---|"]
    for key, name in [("bright", "平均亮度"), ("chroma", "色度噪声"), ("std8", "细节 std8"), ("jump", "暗区条带跳变")]:
        lines.append(f"| {name} | {avg(key,'in'):.3f} | {avg(key,'base'):.3f} | {avg(key,'final'):.3f} |")
    lines += ["", "分支说明：近黑分支（均值<5，中性抬亮）/ 仅色度（σc<1.5，只做色度轻平滑、亮度不动）/ 常规暗图（σc≥1.5）。"]
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    per = args.per_sheet
    for gi in range(0, len(tiles), per):
        grp = tiles[gi:gi + per]
        wmax = max(t.shape[1] for t in grp)
        grp = [np.pad(t, ((0, 0), (0, wmax - t.shape[1]), (0, 0)), constant_values=24) for t in grp]
        fnt = ImageFont.truetype(FONT_PATH, 16)
        head = Image.new("RGB", (wmax, 30), (32, 32, 32))
        ImageDraw.Draw(head).text((6, 6), "列：输入 → 原版 Zero-DCE → 最终版(auto)   ｜   每格第二行=色度噪声（越小越干净）",
                                   font=fnt, fill=(235, 235, 235))
        p = os.path.join(out, f"sheet_{gi // per + 1}.png")
        Image.fromarray(np.vstack([np.asarray(head)] + grp)).save(p)
        print("[ok]", p)
    print(f"\n完成 {len(files)} 张 → {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

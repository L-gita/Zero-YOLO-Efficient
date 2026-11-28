"""批量对比：原版 Zero-DCE vs 去马赛克最终版（多数据集、按分支统计）

用法：
    python scripts/batch_branch_compare.py                 # 默认跑 低照图库 全部 153 张
    python scripts/batch_branch_compare.py --sets 01,02     # 只跑指定子集

产物（demo_branch_compare/）：
    base/<set>/*.png      原版输出
    final/<set>/*.png     最终版输出
    metrics.csv           每张图的指标 + 判定分支
    summary.md            按数据集/按分支的汇总
    sheet_*.png           三联图（输入 | 原版 | 最终版），左栏标注分支与指标
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import preprocess as pp  # noqa: E402

LIB = os.path.join(ROOT, "低照图库")
OUT = os.path.join(ROOT, "demo_branch_compare")
FONT = r"C:\Windows\Fonts\msyh.ttc"

SETS = [
    ("01_真实拍摄", os.path.join(LIB, "01_真实拍摄_ZeroDCEpp")),
    ("02_LIME", os.path.join(LIB, "02_LIME")),
    ("03_DICM", os.path.join(LIB, "03_DICM")),
    ("04_LOL评估集", os.path.join(LIB, "04_LOL评估集")),
]

MODE_CN = {"near_black": "近黑分支", "normal": "常规（仅色度轻平滑）"}


def imread_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def metrics(low, base, final):
    c_low = pp.estimate_chroma_noise(low)
    c_base = pp.estimate_chroma_noise(base)
    c_final = pp.estimate_chroma_noise(final)

    def std8(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        k = np.ones((8, 8), np.float32) / 64.0
        var = cv2.filter2D(g * g, -1, k) - cv2.filter2D(g, -1, k) ** 2
        return float(np.sqrt(np.maximum(var, 0.0)).mean())

    def band(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = g < 40
        if mask.mean() < 0.02:
            mask = g < 80
        vals, counts = np.unique(g[mask], return_counts=True)
        keep = vals[counts / mask.sum() > 0.01]
        if keep.size < 2:
            return 0.0, int(keep.size)
        return float(np.max(np.diff(keep))), int(keep.size)

    bj_base, nl_base = band(base)
    bj_final, nl_final = band(final)
    return dict(
        bright_low=float(low.mean()), bright_base=float(base.mean()), bright_final=float(final.mean()),
        chroma_low=c_low, chroma_base=c_base, chroma_final=c_final,
        std8_base=std8(base), std8_final=std8(final),
        band_base=bj_base, band_final=bj_final,
        levels_base=nl_base, levels_final=nl_final,
    )


def run_enhance(files, outdir, extra):
    os.makedirs(outdir, exist_ok=True)
    py = sys.executable
    ef = os.path.join(ROOT, "enhance.py")
    cmd = [py, ef] + files + ["-o", outdir] + extra
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        print(r.stderr[-3000:])
        raise RuntimeError("enhance 失败")
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="", help="只跑指定子集前缀，逗号分隔，如 01,02")
    args = ap.parse_args()
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]

    os.makedirs(OUT, exist_ok=True)
    rows = []
    n_total, n_skipped = 0, 0
    skipped_files = []
    for name, d in SETS:
        if wanted and not any(name.startswith(w) for w in wanted):
            continue
        if not os.path.isdir(d):
            print(f"[跳过] 不存在 {d}")
            continue
        files = sorted(
            os.path.join(d, f) for f in os.listdir(d)
            if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
        )
        if not files:
            continue
        print(f"[{name}] {len(files)} 张")
        bdir = os.path.join(OUT, "base", name)
        fdir = os.path.join(OUT, "final", name)
        run_enhance(files, bdir, ["-m", "sice"])
        run_enhance(files, fdir, ["--preprocess", "auto"])
        n_total += len(files)
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            low = imread_rgb(f)
            dec = pp.auto_decision(low)
            # enhance.py 输出统一加 _enhanced 后缀，且保留原扩展名（.bmp/.jpg/.png）
            exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"]
            bp = next((os.path.join(bdir, stem + "_enhanced" + e) for e in exts
                       if os.path.exists(os.path.join(bdir, stem + "_enhanced" + e))), None)
            fp = next((os.path.join(fdir, stem + "_enhanced" + e) for e in exts
                       if os.path.exists(os.path.join(fdir, stem + "_enhanced" + e))), None)
            if bp is None or fp is None:
                n_skipped += 1
                skipped_files.append(os.path.basename(f))
                print("  [跳过] 缺少输出:", stem)
                continue
            b_arr, f_arr = imread_rgb(bp), imread_rgb(fp)
            m = metrics(low, b_arr, f_arr)
            identical = bool(b_arr.shape == f_arr.shape and np.array_equal(b_arr, f_arr))
            rows.append(dict(set=name, file=os.path.basename(f),
                             base_file=os.path.basename(bp), final_file=os.path.basename(fp),
                             mode=dec["mode"], identical=identical,
                             mean=round(dec["mean"], 2), chroma_in=round(dec["chroma_noise"], 3), **{
                                 k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}))
        print(f"   完成 {len(files) - n_skipped} 张（累计跳过 {n_skipped}）")

    if not rows:
        raise SystemExit("没有结果")

    # CSV
    csv_path = os.path.join(OUT, "metrics.csv")
    keys = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("[ok]", csv_path)

    # 汇总
    def agg(rs, key):
        if not rs:
            return None
        return float(np.mean([r[key] for r in rs]))

    sets = sorted({r["set"] for r in rows})
    modes = ["near_black", "normal"]
    lines = ["# 原版 Zero-DCE vs 去马赛克最终版（批量对比）", "",
             f"共 **{len(rows)}** 张图。每张图先按算法判定分支，再分别跑原版与最终版。", "",
             "## 按数据集", "",
             "| 数据集 | 张数 | 分支分布 | 亮度(原→终) | 色噪(原→终) | 细节 std8(原→终) | 条带(原→终) |",
             "|---|---|---|---|---|---|---|"]
    for s in sets:
        rs = [r for r in rows if r["set"] == s]
        dist = ", ".join(f"{MODE_CN.get(m, m)}×{sum(1 for r in rs if r['mode'] == m)}"
                         for m in modes if any(r["mode"] == m for r in rs))
        lines.append(f"| {s} | {len(rs)} | {dist} | {agg(rs,'bright_base'):.1f}→{agg(rs,'bright_final'):.1f} "
                     f"| {agg(rs,'chroma_base'):.2f}→{agg(rs,'chroma_final'):.2f} "
                     f"| {agg(rs,'std8_base'):.2f}→{agg(rs,'std8_final'):.2f} "
                     f"| {agg(rs,'band_base'):.1f}→{agg(rs,'band_final'):.1f} |")
    lines += ["", "## 按判定分支", "",
              "| 分支 | 张数 | 亮度(原→终) | 色噪(原→终) | 变化 | 细节 std8(原→终) |",
              "|---|---|---|---|---|---|"]
    for m in modes:
        rs = [r for r in rows if r["mode"] == m]
        if not rs:
            continue
        cb, cf = agg(rs, "chroma_base"), agg(rs, "chroma_final")
        delta = (cf - cb) / cb * 100 if cb else 0.0
        lines.append(f"| {MODE_CN.get(m, m)} | {len(rs)} | {agg(rs,'bright_base'):.1f}→{agg(rs,'bright_final'):.1f} "
                     f"| {cb:.2f}→{cf:.2f} | {delta:+.1f}% | {agg(rs,'std8_base'):.2f}→{agg(rs,'std8_final'):.2f} |")
    same = sum(1 for r in rows if r.get("identical"))
    n_skip = sum(1 for r in rows if r["mode"] == "normal")
    lines += ["", "## 结论口径", "",
              f"- 「常规」分支共 {n_skip} 张：**完全不做亮度侧处理**（不做亮度降噪/不做输入预处理），只按输入色度噪声做轻度色度平滑；",
              f"  其中 **{same} 张（σc<0.3）输出与原版逐像素一致**。",
              "- 「近黑分支」是唯一会改变亮度的分支（中性抬亮 + 暗部补偿），色噪/彩色块下降最明显。",
              "",]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("[ok]", os.path.join(OUT, "summary.md"))

    # 三联图
    H = 260
    sheets, cur = [], []
    fnt = ImageFont.truetype(FONT, 15)
    for i, r in enumerate(rows, 1):
        d = dict(SETS)[r["set"]]
        src = os.path.join(d, r["file"])
        stem = os.path.splitext(r["file"])[0]
        low = imread_rgb(src)
        base = imread_rgb(os.path.join(OUT, "base", r["set"], r["base_file"]))
        final = imread_rgb(os.path.join(OUT, "final", r["set"], r["final_file"]))

        def fit(a):
            w = int(H * a.shape[1] / a.shape[0])
            return np.asarray(Image.fromarray(a).resize((w, H), Image.LANCZOS))

        def lab(a, t1, t2=""):
            c = Image.new("RGB", (a.shape[1], H + 38), (24, 24, 24))
            c.paste(Image.fromarray(a), (0, 38))
            dd = ImageDraw.Draw(c)
            dd.text((5, 2), t1, font=fnt, fill=(245, 245, 245))
            if t2:
                dd.text((5, 19), t2, font=fnt, fill=(150, 205, 255))
            return np.asarray(c)

        tile = np.hstack([
            lab(fit(low), r["file"], r["set"]),
            lab(fit(base), "原版", f"色噪 {r['chroma_base']:.2f}｜细节 {r['std8_base']:.2f}"),
            lab(fit(final), "最终版", f"{MODE_CN[r['mode']]}｜色噪 {r['chroma_final']:.2f}｜细节 {r['std8_final']:.2f}"),
        ])
        cur.append(tile)
        if len(cur) == 8 or i == len(rows):
            w = max(t.shape[1] for t in cur)
            cur = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0)), constant_values=24) for t in cur]
            p = os.path.join(OUT, f"sheet_{len(sheets)+1}.png")
            Image.fromarray(np.vstack(cur)).save(p)
            print("[ok]", p)
            sheets.append(p)
            cur = []


if __name__ == "__main__":
    main()

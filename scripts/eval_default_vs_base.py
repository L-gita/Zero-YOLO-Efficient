"""颗粒版（默认 auto）vs 原版基线：可复现的指标评测脚本。

为什么单独写一个脚本：
    之前汇报里的"平坦区亮度噪声 −19% / 8×8 块效应 −4.5%"等数字，是用临时脚本算的，
    不利于答辩时复现。这里把口径固定下来，一条命令重算全部数字。

用法：
    python scripts\\eval_default_vs_base.py                      # 默认 4 个集合
    python scripts\\eval_default_vs_base.py --sets eval15 LIME   # 只跑指定集合
    python scripts\\eval_default_vs_base.py --limit-dicm 16      # 限制 DICM 张数

产出（默认写到 demo_mosaic\\default_vs_base\\）：
    <set>/<图名>/base/*_enhanced.*    原版基线输出（-m sice）
    <set>/<图名>/def/*_enhanced.*     颗粒版输出（--preprocess auto，默认参数）
    metrics.csv                       逐图逐指标
    summary.md                        按集合汇总 + 指标口径说明

指标口径（全部在 0~255 灰度域计算）：
    bright              输出平均亮度
    luma_noise_flat     平坦区亮度高频残差：mean|Y - Gauss(Y,σ=1.5)|（平坦区 = 输入 |Sobel| < 20）
    chroma_noise_flat   平坦区色度高频残差：max over Cb,Cr 的同口径残差
    blockiness          8×8 块效应比 = 每 8 列/行边界处 |ΔY| 均值 ÷ 非边界 |ΔY| 均值（1=无块效应）
    detail_std8         8×8 块内局部 3×3 亮度标准差均值（含噪声与纹理，不能单独当纹理指标）
    cos_struct          与输入图的梯度方向余弦（1=结构完全一致）
    artificial_edges    每千像素"色度跳变≥3 且亮度跳变<1"的边界数（注意：压亮度噪声后该值会虚高）
    shadow_mean         输入灰度 <25 的像素，在输出里的平均灰度（看阴影是否被压暗）
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
ENH = os.path.join(ROOT, "enhance.py")

SETS = {
    "eval15": (os.path.join(ROOT, "lol_test", "eval15", "low"), None),
    "LIME": (os.path.join(ROOT, "低照图库", "02_LIME"), None),
    "DICM": (os.path.join(ROOT, "低照图库", "03_DICM"), 16),
    "extreme": (os.path.join(ROOT, "低照图库", "01_真实拍摄_ZeroDCEpp"), 6),
}


# ---------------------------------------------------------------- 指标
def _gray(u8: np.ndarray) -> np.ndarray:
    return np.asarray(Image.fromarray(u8).convert("L"), dtype=np.float32)


def _flat_mask(low_u8: np.ndarray, thr: float = 20.0) -> np.ndarray:
    g = _gray(low_u8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mask = (np.abs(gx) + np.abs(gy)) < thr
    if mask.sum() < 100:  # 整幅都是纹理时退化为全图
        mask = np.ones_like(g, dtype=bool)
    return mask


def _hf(x: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    return np.abs(x - cv2.GaussianBlur(x, (0, 0), sigma))


def measure(u8: np.ndarray, low_u8: np.ndarray) -> dict:
    g = _gray(u8)
    ycc = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    cbcr = ycc[..., 1:]
    flat = _flat_mask(low_u8)

    luma_hf = _hf(g)
    chroma_hf = np.max([_hf(cbcr[..., i]) for i in range(2)], axis=0)

    # 8×8 块效应：边界 vs 非边界
    d = np.abs(np.diff(g, axis=1))
    at = np.zeros_like(d, dtype=bool)
    at[:, 7::8] = True
    blockiness = float((d[at].mean() + 1e-6) / (d[~at].mean() + 1e-6))

    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    detail_std8 = float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))

    # 结构一致性：梯度方向余弦
    gl = _gray(low_u8)
    ax, ay = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    bx, by = cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gl, cv2.CV_32F, 0, 1, ksize=3)
    num = float((ax * bx + ay * by).sum())
    den = float(np.sqrt((ax * ax + ay * ay).sum() * (bx * bx + by * by).sum()) + 1e-9)
    cos_struct = num / den

    # 人造边界：色度跳变≥3 且亮度跳变<1
    cj = np.abs(np.diff(cbcr, axis=1)).max(axis=2)
    yj = np.abs(np.diff(g, axis=1))
    cnt = int(((cj >= 3.0) & (yj < 1.0)).sum())
    artificial_edges = cnt * 1000.0 / (g.shape[0] * g.shape[1])

    shadow = g[gl < 25]
    return {
        "bright": float(g.mean()),
        "luma_noise_flat": float(luma_hf[flat].mean()),
        "chroma_noise_flat": float(chroma_hf[flat].mean()),
        "blockiness": blockiness,
        "detail_std8": detail_std8,
        "cos_struct": float(cos_struct),
        "artificial_edges": float(artificial_edges),
        "shadow_mean": float(shadow.mean()) if shadow.size > 200 else float("nan"),
    }


# ---------------------------------------------------------------- 跑模型
def run_enhance(files: list[str], out_dir: str, extra: list[str]) -> None:
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            try:
                os.remove(os.path.join(out_dir, name))
            except OSError:
                pass
    os.makedirs(out_dir, exist_ok=True)
    cmd = [PY, ENH, *files, "-o", out_dir, *extra]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout[-800:], p.stderr[-800:], sep="\n")
        raise SystemExit(f"enhance.py 失败: {extra}")


def main() -> None:
    ap = argparse.ArgumentParser(description="颗粒版 vs 原版基线：可复现指标评测")
    ap.add_argument("--sets", nargs="*", default=list(SETS), choices=list(SETS))
    ap.add_argument("--limit-dicm", type=int, default=16)
    ap.add_argument("--out", default=os.path.join(ROOT, "demo_mosaic", "default_vs_base"))
    args = ap.parse_args()

    rows: list[dict] = []
    for name in args.sets:
        folder, limit = SETS[name]
        if name == "DICM" and limit is None:
            limit = args.limit_dicm
        files = sorted(glob.glob(os.path.join(folder, "*.*")))
        files = [f for f in files if os.path.splitext(f)[1].lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")]
        if limit:
            files = files[:limit]
        if not files:
            print(f"[skip] {name}: 目录为空 {folder}")
            continue
        print(f"[{name}] {len(files)} 张 …", flush=True)

        # 一次性批量跑（避免每张图重启一次 Python）
        tmp = os.path.join(args.out, name, "_tmp")
        run_enhance(files, os.path.join(tmp, "base"), ["-m", "sice"])
        run_enhance(files, os.path.join(tmp, "def"), ["--preprocess", "auto"])

        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            outs = {}
            for mode in ("base", "def"):
                cand = glob.glob(os.path.join(tmp, mode, stem + "_enhanced.*"))
                if not cand:
                    print(f"  [warn] 缺输出: {name}/{stem} {mode}")
                    continue
                outs[mode] = cand[0]
            if len(outs) < 2:
                continue
            low = np.asarray(Image.open(f).convert("RGB"))
            base = np.asarray(Image.open(outs["base"]).convert("RGB"))
            df = np.asarray(Image.open(outs["def"]).convert("RGB"))
            if base.shape != low.shape:
                base = cv2.resize(base, (low.shape[1], low.shape[0]))
            if df.shape != low.shape:
                df = cv2.resize(df, (low.shape[1], low.shape[0]))
            mb, md = measure(base, low), measure(df, low)
            rows.append({"set": name, "file": os.path.basename(f),
                         **{f"base_{k}": v for k, v in mb.items()},
                         **{f"def_{k}": v for k, v in md.items()}})
        print(f"  完成 {name}: {len(rows)} 行累计", flush=True)

    if not rows:
        raise SystemExit("没有产出任何数据")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    keys = [k for k in rows[0] if k.startswith("base_")]
    lines = ["# 颗粒版（默认 auto）vs 原版基线", "",
             f"数据：{csv_path}；图例：base=原版 Zero-DCE，def=颗粒版默认（s=0.20、不碰色度、自动阴影补偿）", "",
             "## 按集合汇总（均值）", "",
             "| 集合 | n | 指标 | 原版 | 颗粒版 | 变化 |", "| --- | --- | --- | --- | --- | --- |"]
    for name in args.sets:
        rs = [r for r in rows if r["set"] == name]
        if not rs:
            continue
        for k in keys:
            b = float(np.mean([r[k] for r in rs]))
            d = float(np.mean([r["def_" + k[5:]] for r in rs]))
            delta = "—" if b == 0 else f"{(d - b) / b * 100:+.1f}%"
            lines.append(f"| {name} | {len(rs)} | {k[5:]} | {b:.4f} | {d:.4f} | {delta} |")
    lines += ["", "> 注意：`artificial_edges` 在压亮度噪声后会虚高（原先被亮度噪声掩盖的色度跳变被计入），"
                  "`detail_std8` 含噪声成分、压噪后必然下降；这两个指标不能单独当结论。"]
    md_path = os.path.join(args.out, "summary.md")
    with open(md_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")

    print(f"\n完成 {len(rows)} 张 → {args.out}")
    print(f"  明细：{csv_path}\n  汇总：{md_path}")


if __name__ == "__main__":
    main()

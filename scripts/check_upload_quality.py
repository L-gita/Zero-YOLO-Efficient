"""检查一张图"是否被压缩/重编码过"，以及它的噪声/台阶/块效应到底有多重。

用途：手机拍完传到电脑，想知道微信/QQ/网盘有没有偷偷压缩画质时跑一下。

用法：
    python scripts\\check_upload_quality.py 照片.jpg
    python scripts\\check_upload_quality.py 目录\\            # 批量

输出（终端 + 可选 CSV）：
    分辨率 / 文件大小 / 每像素字节数 / 噪声 σ / 8×8 块效应比 /
    暗区灰度级数 / 最大台阶跳变 / 判定结论

判定依据（阈值来自本项目实测 + 文献常用经验值）：
    * 块效应比 > 1.05         → 有明显 8×8 JPEG 块（被重编码过）
    * 每像素字节 < 0.05       → 压缩比很高（1080p 通常 <100KB 就是被压了）
    * 暗区级数 < 40 且均值低   → 8-bit 台阶严重（放大后必出马赛克）
    * 噪声 σ > 6             → 噪点重（不是压缩问题，是拍摄/感光问题）
"""
from __future__ import annotations

import argparse
import csv
import glob
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def blockiness(gray: np.ndarray, flat_thr: float = 20.0) -> float:
    """8×8 网格边界处 |ΔY| 均值 ÷ 非边界 |ΔY| 均值（1 = 无块效应）。

    只在"平坦区"统计（输入梯度 < flat_thr），否则真实竖直边缘会污染比值、
    让无损 PNG 也判成 1.2+（实测踩过这个坑）。行、列两个方向取较大者。
    """
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    flat = np.abs(gx) + np.abs(gy) < flat_thr

    def ratio(d: np.ndarray, mask: np.ndarray, axis: int) -> float:
        # d 与 mask 对齐：mask 取较小一侧
        m = mask[:, :-1] if axis == 1 else mask[:-1, :]
        if m.sum() < 500:
            return 1.0
        at = np.zeros_like(d, dtype=bool)
        if axis == 1:
            at[:, 7::8] = True
        else:
            at[7::8, :] = True
        dd = d[m]
        aa = at[m]
        if not aa.any() or (~aa).sum() < 100:
            return 1.0
        return float((dd[aa].mean() + 1e-6) / (dd[~aa].mean() + 1e-6))

    col = ratio(np.abs(np.diff(gray, axis=1)), flat, 1)
    row = ratio(np.abs(np.diff(gray, axis=0)), flat, 0)
    return float(max(col, row))


def noise_sigma(gray: np.ndarray) -> float:
    """暗区拉普拉斯残差 MAD/0.6745（稳健噪声估计，0~255 单位）。"""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    m = gray < 40
    if m.mean() < 0.05:
        m = gray < 80
    v = lap[m] if m.any() else lap.ravel()
    return float(np.median(np.abs(v)) / 0.6745)


def dark_levels(gray: np.ndarray) -> tuple[int, float]:
    """暗区（<40）里"占面积>0.5% 的灰度级"个数，以及相邻大级之间的最大跳变。"""
    m = gray < 40
    if m.sum() < 1000:
        return 0, 0.0
    vals = gray[m].astype(np.uint8)
    hist = np.bincount(vals, minlength=256).astype(np.float64) / vals.size
    big = [i for i in range(40) if hist[i] > 0.005]
    jump = max((big[i + 1] - big[i] for i in range(len(big) - 1)), default=0)
    return len(big), float(jump)


def inspect(path: str) -> dict:
    im = Image.open(path)
    w, h = im.size
    rgb = np.asarray(im.convert("RGB"))
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    size = os.path.getsize(path)
    blk = blockiness(gray)
    sigma = noise_sigma(gray)
    nlev, jump = dark_levels(gray)
    per_px = size / float(w * h)

    flags = []
    thr_blk = 1.05 if gray.mean() >= 10 else 2.2   # 极暗图块效应判据不稳，阈值放宽
    if blk > thr_blk:
        flags.append("疑似被重编码(8×8块)")
    if per_px < 0.05:
        flags.append("压缩比高")
    if gray.mean() < 60 and nlev and nlev < 40:
        flags.append(f"暗区仅{nlev}级灰度(台阶风险)")
    if sigma > 6:
        flags.append("噪点重")
    if gray.mean() < 10:
        flags.append("极暗(块效应判据仅参考)")
    verdict = "；".join(flags) if flags else "看起来是原图/无损"

    return {
        "file": os.path.basename(path),
        "size": f"{w}x{h}",
        "bytes": size,
        "KB": round(size / 1024, 1),
        "B_per_px": round(per_px, 3),
        "mean": round(float(gray.mean()), 1),
        "noise_sigma": round(sigma, 2),
        "blockiness": round(blk, 3),
        "dark_levels": nlev,
        "max_dark_jump": jump,
        "verdict": verdict,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="检查图片是否被压缩/噪点/台阶严重")
    ap.add_argument("paths", nargs="+", help="图片文件或目录")
    ap.add_argument("--csv", default="", help="可选：把结果写到 CSV")
    args = ap.parse_args()

    files: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            files += sorted(f for f in glob.glob(os.path.join(p, "*")) if f.lower().endswith(EXTS))
        else:
            files.append(p)
    if not files:
        raise SystemExit("没找到图片")

    rows = [inspect(f) for f in files]
    w = [16, 12, 9, 10, 9, 7, 8, 11, 7, 12, 44]
    head = ["文件", "分辨率", "KB", "B/像素", "亮度", "噪声σ", "块效应", "暗区级数", "最大跳变", "", "判定"]
    line = "  ".join(str(h).ljust(x) for h, x in zip(head, w))
    print(line)
    print("-" * len(line))
    for r in rows:
        cells = [str(r["file"])[:16], r["size"], r["KB"], r["B_per_px"], r["mean"],
                 r["noise_sigma"], r["blockiness"], r["dark_levels"], r["max_dark_jump"], "", r["verdict"]]
        print("  ".join(str(c).ljust(x) for c, x in zip(cells, w)))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as fp:
            wr = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        print(f"\n已写 {args.csv}")


if __name__ == "__main__":
    main()

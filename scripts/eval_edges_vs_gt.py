"""边缘质量评测：原版基线 vs 颗粒版（默认）——直接服务于"识别/抓取是否更友好"。

为什么做这个：抓取/识别最怕的不是颗粒噪声，而是**假边缘**（平坦区冒出来的边界）
和**轮廓断裂**。所以用 Canny 边缘与正常光 GT 的边缘做匹配，量化：
    边缘精度 precision = 输出边缘里"确实落在 GT 边缘附近"的比例（越高=假边缘越少）
    边缘召回 recall    = GT 边缘里"输出也检到"的比例（越高=真轮廓越完整）
    边缘像素数        = 绝对数量（噪声多会虚高）

用法：
    python scripts\\eval_edges_vs_gt.py
    python scripts\\eval_edges_vs_gt.py --tol 3 --canny 50 120

前提：先用 scripts\\eval_default_vs_base.py 跑出两套增强结果（默认路径
      demo_mosaic\\default_vs_base2\\<set>\\_tmp\\{base,def}）。
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def canny(img_rgb: np.ndarray, lo: int, hi: int) -> np.ndarray:
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(g, lo, hi) > 0


def dilate(mask: np.ndarray, k: int) -> np.ndarray:
    return cv2.dilate(mask.astype(np.uint8), np.ones((k, k), np.uint8)) > 0


def main() -> None:
    ap = argparse.ArgumentParser(description="边缘精度/召回：基线 vs 颗粒版（需要 GT）")
    ap.add_argument("--out-root", default=os.path.join(ROOT, "demo_mosaic", "default_vs_base2"))
    ap.add_argument("--gt-dir", default=os.path.join(ROOT, "lol_test", "eval15", "high"))
    ap.add_argument("--low-dir", default=os.path.join(ROOT, "lol_test", "eval15", "low"))
    ap.add_argument("--tol", type=int, default=3, help="边缘匹配容差（像素）")
    ap.add_argument("--canny", type=int, nargs=2, default=[50, 120], metavar=("LO", "HI"))
    args = ap.parse_args()

    tmp = os.path.join(args.out_root, "eval15", "_tmp")
    if not os.path.isdir(tmp):
        raise SystemExit(f"找不到增强结果目录 {tmp}\n请先运行: python scripts\\eval_default_vs_base.py")

    stats: dict[str, list[tuple[float, float, float]]] = {"base": [], "def": []}
    used = 0
    for low_path in sorted(glob.glob(os.path.join(args.low_dir, "*.*"))):
        stem = os.path.splitext(os.path.basename(low_path))[0]
        gts = glob.glob(os.path.join(args.gt_dir, stem + ".*"))
        bp = glob.glob(os.path.join(tmp, "base", stem + "_enhanced.*"))
        dp = glob.glob(os.path.join(tmp, "def", stem + "_enhanced.*"))
        if not (gts and bp and dp):
            continue
        gt = np.asarray(Image.open(gts[0]).convert("RGB"))
        imgs = {"base": np.asarray(Image.open(bp[0]).convert("RGB")),
                "def": np.asarray(Image.open(dp[0]).convert("RGB"))}
        eg = canny(gt, *args.canny)
        near_gt = dilate(eg, args.tol)
        for tag, img in imgs.items():
            if img.shape != gt.shape:
                img = cv2.resize(img, (gt.shape[1], gt.shape[0]))
            e = canny(img, *args.canny)
            near_e = dilate(e, args.tol)
            prec = float((e & near_gt).sum() / max(int(e.sum()), 1))
            rec = float((eg & near_e).sum() / max(int(eg.sum()), 1))
            stats[tag].append((prec, rec, float(e.sum())))
        used += 1

    if not used:
        raise SystemExit("没有可用样本（检查 GT / 输出目录）")

    print(f"样本数 {used}（容差 {args.tol}px，Canny {args.canny[0]}/{args.canny[1]}）")
    print(f"{'版本':<6}{'边缘精度':>10}{'边缘召回':>10}{'边缘像素数':>12}")
    res = {}
    for tag, name in [("base", "原版基线"), ("def", "颗粒版")]:
        a = np.array(stats[tag], dtype=float)
        res[tag] = (a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean())
        print(f"{name:<6}{res[tag][0]:>10.3f}{res[tag][1]:>10.3f}{res[tag][2]:>12.0f}")
    dp = res["def"][0] - res["base"][0]
    dr = res["def"][1] - res["base"][1]
    print(f"\n变化：精度 {dp:+.3f}（{dp / max(res['base'][0], 1e-9) * 100:+.1f}%），"
          f"召回 {dr:+.3f}（{dr / max(res['base'][1], 1e-9) * 100:+.1f}%）；"
          f"边缘像素数 {res['base'][2]:.0f} → {res['def'][2]:.0f}")


if __name__ == "__main__":
    main()

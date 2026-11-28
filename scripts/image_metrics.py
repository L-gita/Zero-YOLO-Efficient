#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""单图指标计算器：输入图像 → 原图指标 + 最终版增强后指标（可选：原版基线 / 参考图对比）。

用法：
    python scripts/image_metrics.py 图片或文件夹 ... [-o 输出目录] [--with-base] [--gt 参考图目录]
    python scripts/image_metrics.py --self-test        # 自检：指标数学 + 管线一致性

输出（默认输出到第一张图旁边的 metrics_report_时间戳/）：
    metrics.csv   每张图一行：in_*（输入）/ out_*（最终版增强）/ base_*（可选）/ ref_*（有参考图时）
    report.md     公式说明 + 逐图表格 + 每张图的判定日志 + 口径退化标记
    figures/      每张图的 输入|最终版 并排图（--no-fig 可关）

诚实性约定（对应"不要碰到什么都直接通过"）：
  * 指标全部来自 metrics_lib（真实像素统计），该库自带 --self-test 的解析用例；
  * 增强走 `enhance.py --preprocess auto` 子进程 —— 与双击 bat 完全同一份代码；
  * 任何一张图失败/跳过都会写入 report.md 的「失败清单」并以退出码 1 结束，不会静默当成功；
  * 平坦区像素太少（掩码退化）、参考图尺寸不一致等情况会在报告里显式标注。
"""
import argparse
import csv
import datetime as dt
import hashlib
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import metrics_lib as M  # noqa: E402

ENHANCE = os.path.join(ROOT, "enhance.py")
FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"

METRIC_DOC = [
    ("brightness", "灰度均值（0~255）"),
    ("contrast", "灰度标准差"),
    ("saturation", "HSV 中 S 通道均值（0~255）"),
    ("luma_noise_flat", "平坦区 |Y−Gauss(Y,σ=1.5)| 均值；平坦区=|Sobel|<20，像素<100 时退化为全图并在报告标注"),
    ("chroma_noise_flat", "平坦区 max(|Cb−hf|,|Cr−hf|) 均值"),
    ("chroma_noise_global", "全图 Cb/Cr 高频残差 std 均值（不依赖掩码，便于跨图比较）"),
    ("detail_std8", "灰度 8×8 块内 std 的均值（与本项目报告里的“细节 std8”同口径）"),
    ("detail_std3", "3×3 邻域 std 均值（另一套口径，勿与 std8 混用）"),
    ("blockiness", "8 列/行网格边界 |ΔY| ÷ 非边界 |ΔY|；1.0=无块效应，>1.03 视为有 JPEG 块"),
    ("band_jump", "暗区（gray<40；占比<2% 放宽 <80）内“面积>1% 的灰度级”之间的最大间隔"),
    ("n_levels", "暗区内“面积>1% 的灰度级个数”（越多越细腻）"),
    ("artificial_edges_lo", "色度跳变≥3 且亮度跳变<1 的边界数 ×1000/(H·W)（宽松口径）"),
    ("artificial_edges_hi", "色度跳变≥6 且亮度跳变<3（严格口径，抗噪）"),
    ("colored_blob_share", "色度幅值>10 的最大连通块面积占比（%）"),
    ("cos_struct", "与输入图梯度方向的全局余弦（1=结构完全同向）"),
    ("psnr_vs_ref / ssim_vs_ref", "有参考图时：RGB PSNR / SSIM（11×11 高斯窗 σ=1.5）"),
    ("edge_precision / edge_recall", "有参考图时：Canny(50/120) 边缘 3px 容差匹配的精确率/召回率"),
]


def imread_rgb(path):
    """统一读成 RGB uint8（16-bit 输入按 255 缩放；灰度/带 alpha 转 3 通道）。"""
    arr = cv2_imread(path)
    if arr is None:
        raise RuntimeError("无法读取图像（可能损坏或格式不支持）")
    if arr.dtype == np.uint16:
        arr = (arr.astype(np.float32) / 65535.0 * 255.0 + 0.5).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=2)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    return np.ascontiguousarray(arr)


def cv2_imread(path):
    """用 Python 文件 API 读字节再 imdecode —— 规避 OpenCV 在 Windows 上打不开中文路径的问题。"""
    import cv2
    try:
        with open(path, "rb") as f:
            buf = f.read()
    except OSError as e:
        raise RuntimeError(f"无法打开文件：{e}")
    arr = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_UNCHANGED)
    return arr


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            for dirpath, _, files in os.walk(p):
                for f in sorted(files):
                    if f.lower().endswith(M.IMG_EXT):
                        out.append(os.path.join(dirpath, f))
        elif os.path.isfile(p) and p.lower().endswith(M.IMG_EXT):
            out.append(p)
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_enhance(img, out_dir, preset="auto", extra=()):
    """调 enhance.py 子进程（与 bat 同一份代码），返回 (输出图路径, 标准输出+错误)。"""
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, ENHANCE, img, "-o", out_dir]
    if preset:
        cmd += ["--preprocess", preset]
    cmd += list(extra)
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    log = (r.stdout or "") + (r.stderr or "")
    if r.returncode != 0:
        raise RuntimeError(f"enhance.py 失败（exit {r.returncode}）：\n{log.strip()}")
    stem = os.path.splitext(os.path.basename(img))[0]
    cands = [c for c in os.listdir(out_dir)
             if c.startswith(stem + "_enhanced") and "_compare" not in c and "_enhanced16" not in c]
    if not cands:
        raise RuntimeError(f"enhance.py 未产出增强图（目录：{out_dir}，内容：{os.listdir(out_dir)}）")
    return os.path.join(out_dir, sorted(cands)[0]), log


def dec_lines(log):
    keep = []
    for line in log.splitlines():
        s = line.strip()
        if s.startswith("[") or "亮度" in s:
            keep.append(s)
    return keep


def fig_pair(low, final, path, title):
    H = 420
    def fit(a):
        return np.asarray(Image.fromarray(a).resize((int(H * a.shape[1] / a.shape[0]), H), Image.LANCZOS))
    try:
        font = ImageFont.truetype(FONT_PATH, 16)
    except OSError:
        font = ImageFont.load_default()
    tiles = []
    for name, im in (("输入", low), ("最终版", final)):
        a = fit(im)
        c = Image.new("RGB", (a.shape[1], H + 26), (24, 24, 24))
        c.paste(Image.fromarray(a), (0, 26))
        ImageDraw.Draw(c).text((5, 4), name, font=font, fill=(240, 240, 240))
        tiles.append(np.asarray(c))
    gap = np.full((tiles[0].shape[0], 8, 3), 24, np.uint8)
    Image.fromarray(np.hstack([tiles[0], gap, tiles[1]])).save(path)


def fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        return f"{v:.3f}"
    return str(v)


def process(images, out_dir, with_base, gt_map, make_fig):
    rows, logs, fails = [], {}, []
    tmp = tempfile.mkdtemp(prefix="metrics_")
    fig_dir = os.path.join(out_dir, "figures")
    if make_fig:
        os.makedirs(fig_dir, exist_ok=True)
    for i, img in enumerate(images, 1):
        stem = os.path.splitext(os.path.basename(img))[0]
        print(f"[{i}/{len(images)}] {os.path.basename(img)}")
        try:
            low = imread_rgb(img)
            # 1) 最终版（与 bat 同一份代码）
            fin_path, log = run_enhance(img, os.path.join(tmp, "final"))
            logs[img] = dec_lines(log)
            final = imread_rgb(fin_path)
            row = {"file": os.path.basename(img), "gt": "", "figure": ""}
            m_in = M.measure_all(low, low)
            m_out = M.measure_all(final, low, ref_u8=None)
            for k, v in m_in.items():
                row["in_" + k] = v
            for k, v in m_out.items():
                row["out_" + k] = v
            # 2) 可选：原版基线（-m sice）
            if with_base:
                base_path, _ = run_enhance(img, os.path.join(tmp, "base"), preset=None, extra=("-m", "sice"))
                base = imread_rgb(base_path)
                for k, v in M.measure_all(base, low).items():
                    row["base_" + k] = v
                row["same_as_base"] = int(md5(fin_path) == md5(base_path))
            # 3) 可选：参考图（正常光 GT）
            gt = gt_map.get(stem)
            if gt:
                ref = imread_rgb(gt)
                if ref.shape[:2] != low.shape[:2]:
                    row["gt"] = f"{os.path.basename(gt)}（尺寸不一致 {ref.shape[1]}x{ref.shape[0]}，已缩放）"
                    ref = np.asarray(Image.fromarray(ref).resize((low.shape[1], low.shape[0]), Image.LANCZOS))
                else:
                    row["gt"] = os.path.basename(gt)
                for k, v in M.measure_all(low, low, ref_u8=ref).items():
                    if k.endswith("_vs_ref") or k.startswith("edge_") or k.endswith("_px"):
                        row["in_" + k] = v
                for k, v in M.measure_all(final, low, ref_u8=ref).items():
                    if k.endswith("_vs_ref") or k.startswith("edge_") or k.endswith("_px"):
                        row["out_" + k] = v
            # 4) 并排图
            if make_fig:
                fp = os.path.join(fig_dir, f"{stem}_compare.png")
                fig_pair(low, final, fp, stem)
                row["figure"] = os.path.relpath(fp, out_dir)
            rows.append(row)
        except Exception as e:  # noqa: BLE001 —— 失败要显式记录，不能当成功
            fails.append(f"{os.path.basename(img)}: {e}")
            print(f"    [FAIL] {e}")
    return rows, logs, fails, tmp


def write_report(out_dir, images, rows, logs, fails, with_base, gt_map, command):
    # CSV
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    csv_path = os.path.join(out_dir, "metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        if keys:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    # Markdown
    lines = ["# 单图指标报告", "",
             f"- 生成时间：{dt.datetime.now():%Y-%m-%d %H:%M:%S}",
             f"- 图片数：{len(images)}，成功 {len(rows)}，失败 {len(fails)}",
             f"- 增强方式：`enhance.py --preprocess auto`（与 `3_去马赛克最终版.bat` 同一份代码）",
             f"- 复现命令：`{command}`", ""]
    if fails:
        lines += ["## ⚠ 失败清单（不计入成功）", ""] + [f"- {x}" for x in fails] + [""]
    lines += ["## 指标定义", "", "| 指标 | 定义 |", "| --- | --- |"]
    lines += [f"| `{k}` | {d} |" for k, d in METRIC_DOC]
    lines += ["", "> 说明：`detail_std8` 是 8×8 块口径（本项目主口径）；`detail_std3` 是另一套口径，仅供交叉参考。",
              "> `flat_mask_degraded=1` 表示该图平坦像素不足 100，噪声类指标退化为全图统计（口径变化，已在表中标注）。", ""]

    def table(title, in_key, out_key, base_key=None):
        rows_t = []
        for r in rows:
            a = r.get(in_key)
            if not isinstance(a, (int, float)) or a != a:
                continue
            lin = f"| {r['file']} | {fmt(a)} | {fmt(r.get(out_key))} |"
            if base_key:
                lin += f" {fmt(r.get(base_key))} |"
            rows_t.append(lin)
        if not rows_t:
            return
        lines.append(f"## {title}")
        lines.append("")
        head = "| 文件 | 输入 | 最终版 |" + (" 原版基线 |" if base_key else "")
        sep = "| --- | --- | --- |" + (" --- |" if base_key else "")
        lines.extend([head, sep] + rows_t + [""])

    table("亮度（0~255，越高越亮）", "in_brightness", "out_brightness", "base_brightness" if with_base else None)
    table("色度噪声（平坦区，越低越干净）", "in_chroma_noise_flat", "out_chroma_noise_flat",
          "base_chroma_noise_flat" if with_base else None)
    table("色度噪声（全图，越低越干净）", "in_chroma_noise_global", "out_chroma_noise_global",
          "base_chroma_noise_global" if with_base else None)
    table("平坦区亮度噪声（越低越干净）", "in_luma_noise_flat", "out_luma_noise_flat",
          "base_luma_noise_flat" if with_base else None)
    table("细节 std8（越高纹理越多）", "in_detail_std8", "out_detail_std8", "base_detail_std8" if with_base else None)
    table("8×8 块效应比（1.0=无块）", "in_blockiness", "out_blockiness", "base_blockiness" if with_base else None)
    table("暗区条带跳变（越小越好）", "in_band_jump", "out_band_jump", "base_band_jump" if with_base else None)
    table("伪造边缘-宽松口径（每千像素）", "in_artificial_edges_lo", "out_artificial_edges_lo",
          "base_artificial_edges_lo" if with_base else None)
    table("彩色大块占比（%）", "in_colored_blob_share", "out_colored_blob_share",
          "base_colored_blob_share" if with_base else None)
    table("饱和度（0~255）", "in_saturation", "out_saturation", "base_saturation" if with_base else None)

    if gt_map:
        table("PSNR vs 参考图（dB，越高越好）", "in_psnr_vs_ref", "out_psnr_vs_ref")
        table("SSIM vs 参考图（越高越好）", "in_ssim_vs_ref", "out_ssim_vs_ref")
        table("边缘精确率 vs 参考图（越高越好）", "in_edge_precision", "out_edge_precision")
        table("边缘召回率 vs 参考图（越高越好）", "in_edge_recall", "out_edge_recall")

    lines += ["## 每张图的判定日志（来自 enhance.py 真实输出）", ""]
    for r in rows:
        img_key = None
        for k in logs:
            if os.path.basename(k) == r["file"]:
                img_key = k
                break
        lines.append(f"### {r['file']}" + (f"（参考图：{r['gt']}）" if r.get("gt") else ""))
        lines.append("")
        if img_key:
            lines += ["```"] + logs[img_key] + ["```"]
        else:
            lines.append("（无日志）")
        if r.get("figure"):
            lines.append(f"\n并排图：`{r['figure']}`")
        lines.append("")
    lines += ["## 口径说明与已知限制", "",
              "1. 所有指标都是对真实像素做卷积/统计（实现见 `scripts/metrics_lib.py`，自带 20 项解析自检 `--self-test`）。",
              "2. 平坦区掩码退化、参考图尺寸不一致都会在表中/行内显式标注，不会静默处理。",
              "3. `cos_struct` 与输入图比较；颜色类指标不评价“美观”，只看统计量。",
              "4. 本报告不判断“好不好看”，只给可复核数字；结论请结合并排图与判定日志。"]
    md_path = os.path.join(out_dir, "report.md")
    with open(md_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    return csv_path, md_path


def self_test():
    print("== 1) 指标库解析自检 ==")
    ok, fails = M.self_test(verbose=True)
    print("== 2) 管线一致性自检 ==")
    tmp = tempfile.mkdtemp(prefix="st_")
    # 近黑图：应当 near_black，且输出 != 输入
    dark = np.full((96, 160, 3), 2, np.uint8)
    dark[20:40, 20:60] = 7      # 小块稍亮，整体均值仍 < 5（才会进入近黑分支）
    dark[60:76, 100:130] = 4
    dpath = os.path.join(tmp, "dark.png")
    Image.fromarray(dark).save(dpath)
    fp, log = run_enhance(dpath, os.path.join(tmp, "d"))
    fin = imread_rgb(fp)
    mean_dark = float(np.mean(dark.astype(np.float32)))
    near = ("near_black" in log) if mean_dark < 5.0 else ("normal" in log or "skip" in log)
    print(f"      （测试图均值 {mean_dark:.2f}，期望分支 {'near_black' if mean_dark < 5.0 else 'normal/skip'}）")
    changed = md5(fp) != md5(dpath)
    print(f"  [{'PASS' if near else 'FAIL'}] 近黑图走 near_black 分支")
    print(f"  [{'PASS' if changed else 'FAIL'}] 近黑图输出与输入不同（确实做了处理）")
    if not near:
        fails.append("近黑图未走 near_black")
    if not changed:
        fails.append("近黑图输出与输入相同")
    # 亮而平坦的图：应当 skip（色度噪声极低），且输出与 -m sice 逐像素一致
    bright = np.full((96, 160, 3), 128, np.uint8)   # 完全干净（色度噪声≈0）才会走 skip
    bpath = os.path.join(tmp, "bright.png")
    Image.fromarray(bright).save(bpath)
    fp_auto, log2 = run_enhance(bpath, os.path.join(tmp, "ba"))
    fp_base, _ = run_enhance(bpath, os.path.join(tmp, "bb"), preset=None, extra=("-m", "sice"))
    same = md5(fp_auto) == md5(fp_base)
    print(f"  [{'PASS' if 'normal' in log2 or 'skip' in log2 else 'FAIL'}] 亮图判定为常规/跳过分支")
    print(f"  [{'PASS' if same else 'FAIL'}] 色度干净时最终版 == 原版基线（md5 一致）")
    if not same:
        fails.append("色度干净图的最终版与原版基线不一致")
    print("== 结果 ==")
    if fails:
        print("  [FAIL] 自检失败：", "; ".join(fails))
        return False
    print("  [PASS] 指标数学自检 + 管线一致性自检全部通过")
    return True


def main():
    ap = argparse.ArgumentParser(description="单图指标计算器（输入图指标 + 最终版增强后指标）")
    ap.add_argument("inputs", nargs="*", help="图片或文件夹（可多个）")
    ap.add_argument("-o", "--out", default=None, help="输出目录（默认在第一张图旁新建 metrics_report_时间戳）")
    ap.add_argument("--with-base", action="store_true", help="同时计算原版 Zero-DCE（-m sice）基线指标")
    ap.add_argument("--gt", nargs="*", default=None, help="参考图（正常光 GT）文件或文件夹，按文件名 stem 匹配")
    ap.add_argument("--no-fig", action="store_true", help="不生成并排图")
    ap.add_argument("--self-test", action="store_true", help="只做自检（指标数学 + 管线一致性）")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if not args.inputs:
        ap.error("请提供至少一张图片或一个文件夹")
    images = collect(args.inputs)
    if not images:
        ap.error("没有找到可处理的图片（支持 " + ", ".join(M.IMG_EXT) + "）")

    gt_map = {}
    if args.gt:
        for g in collect(args.gt):
            gt_map[os.path.splitext(os.path.basename(g))[0]] = g

    out = args.out
    if not out:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(os.path.dirname(os.path.abspath(images[0])), f"metrics_report_{stamp}")
    os.makedirs(out, exist_ok=True)
    command = "python scripts/image_metrics.py " + " ".join(args.inputs) + f" -o {out}"
    if args.with_base:
        command += " --with-base"

    rows, logs, fails, tmp = process(images, out, args.with_base, gt_map, not args.no_fig)
    csv_path, md_path = write_report(out, images, rows, logs, fails, args.with_base, gt_map, command)

    print("")
    print(f"完成：成功 {len(rows)} / {len(images)} 张")
    print(f"  指标表：{csv_path}")
    print(f"  报告：  {md_path}")
    if fails:
        print("  [FAIL] 以下图片未成功处理（已计入失败，不会算作成功）：")
        for x in fails:
            print("   -", x)
        print(f"  临时目录保留以便排查：{tmp}")
        sys.exit(1)


if __name__ == "__main__":
    main()

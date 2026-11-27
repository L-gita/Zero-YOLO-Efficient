# -*- coding: utf-8 -*-
"""
诊断"边缘损失"到底改变了什么（消除亮度混淆）
================================================

问题：+边缘损失 的模型输出整体更暗（eval15 97.4 -> 87.3）。
亮度一降，Sobel 幅值 M 自然跟着降，于是 L_edge 也降——
这到底是"边缘保住了"，还是"图变暗了"？本脚本用三个亮度无关的指标判定。

指标（每个模型 × 每个数据集）：
1. cos(g_low, g_enh)  —— 梯度向量余弦相似度（尺度无关，仅结构/方向）
                         1.0 = 梯度场方向完全一致，越低 = 结构被改得越多
2. corr(M_low, M_enh) —— 梯度幅值图的 Pearson 相关（也基本尺度无关）
3. 亮度归一后的 L_edge_norm —— 先把增强图按均值亮度缩放到与低照输入同亮度，
                              再算论文 (4.12) 的边缘差；这样"变暗"不再能取巧
参考：同样指标对 低照输入 vs 正常光 GT 也算一遍（下限参考）。

用法：
    python scripts/diag_edge_effect.py
"""
import os
import sys
import csv
import numpy as np
import cv2
import torch
import torch.nn.functional as F

REPRO = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(REPRO, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)

from model import enhance_net_nopool  # noqa: E402

BASE_CKPT = os.path.join(CODE, "snapshots_sice", "Epoch49.pth")
EDGE_CKPT = os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth")

EVAL15_LOW = os.path.join(REPRO, "lol_test", "eval15", "low")
EVAL15_HIGH = os.path.join(REPRO, "lol_test", "eval15", "high")
LIME = os.path.join(CODE, "data", "test_data", "LIME")
DICM = os.path.join(CODE, "data", "test_data", "DICM")

OUT = os.path.join(REPRO, "demo_edge", "diag_edge_effect")


def load_model(ckpt):
    net = enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(ckpt, map_location="cuda"))
    net.eval()
    return net


def enhance(net, bgr):
    """BGR uint8 -> 增强后 BGR float32 [0,1]（原分辨率整图推理）"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(t)
    out = out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def sobel_fields(img_bgr):
    """返回 (gx, gy, M)，逐通道：raw Sobel（不除 4），M=|Ix|+|Iy|（论文 4.11）"""
    x = torch.from_numpy(img_bgr.transpose(2, 0, 1)).float().unsqueeze(0)  # [1,3,H,W]
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(xp, kx, groups=3)
    gy = F.conv2d(xp, ky, groups=3)
    return gx, gy, gx.abs() + gy.abs()


def cosine_grad(gx1, gy1, gx2, gy2):
    num = (gx1 * gx2 + gy1 * gy2).sum().item()
    den = torch.sqrt((gx1 ** 2 + gy1 ** 2).sum() * (gx2 ** 2 + gy2 ** 2).sum()).item()
    return num / (den + 1e-12)


def corr(a, b):
    a = a.flatten().double()
    b = b.flatten().double()
    a = a - a.mean()
    b = b - b.mean()
    return float((a * b).mean() / (a.std() * b.std() + 1e-12))


def ledge_norm(M_low, M_enh):
    """论文 (4.12)：1/(3N) Σ_c Σ_i |M_low - M_enh|"""
    return float((M_low - M_enh).abs().mean().item())


def stats(list_of_pairs):
    keys = ["cos_low", "corr_low", "ledge_raw", "ledge_norm", "m_low", "m_enh", "bright_low", "bright_enh"]
    out = {k: [] for k in keys}
    for d in list_of_pairs:
        for k in keys:
            out[k].append(d[k])
    return {k: float(np.mean(v)) for k, v in out.items()}


def run_set(nets, files, has_gt=False):
    rows = []
    for f in files:
        bgr = cv2.imread(f, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        low = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gx_l, gy_l, M_l = sobel_fields(low)
        bright_low = float(low.mean())
        row = {"file": os.path.basename(f), "bright_low": bright_low * 255, "m_low": float(M_l.mean())}
        if has_gt:
            gt = cv2.imread(os.path.join(EVAL15_HIGH, os.path.basename(f)), cv2.IMREAD_COLOR)
            gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gx_g, gy_g, M_g = sobel_fields(gt)
            row["cos_low"] = cosine_grad(gx_l, gy_l, gx_g, gy_g)
            row["corr_low"] = corr(M_l, M_g)
            row["ledge_raw"] = ledge_norm(M_l, M_g)
            row["ledge_norm"] = ledge_norm(M_l, M_g)
            row["m_enh"] = float(M_g.mean())
            row["bright_enh"] = float(gt.mean()) * 255
            rows.append(dict(row, model="GT"))
        for name, net in nets.items():
            out = enhance(net, bgr)
            gx_e, gy_e, M_e = sobel_fields(out)
            bright_enh = float(out.mean())
            # 亮度归一：把增强图按均值缩放到与低照输入同亮度，消除"变暗"带来的幅值下降
            scale = bright_low / (bright_enh + 1e-12)
            out_scaled = np.clip(out * scale, 0, 1).astype(np.float32)
            _, _, M_s = sobel_fields(out_scaled)
            rows.append({
                "file": os.path.basename(f),
                "model": name,
                "bright_low": bright_low * 255,
                "bright_enh": bright_enh * 255,
                "m_low": float(M_l.mean()),
                "m_enh": float(M_e.mean()),
                "cos_low": cosine_grad(gx_l, gy_l, gx_e, gy_e),
                "corr_low": corr(M_l, M_e),
                "ledge_raw": ledge_norm(M_l, M_e),
                "ledge_norm": ledge_norm(M_l, M_s),
            })
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    nets = {"baseline": load_model(BASE_CKPT), "edge": load_model(EDGE_CKPT)}
    sets = []

    def files_of(d, exts):
        out = []
        for n in sorted(os.listdir(d)):
            if os.path.splitext(n)[1].lower() in exts:
                out.append(os.path.join(d, n))
        return out

    sets.append(("eval15", files_of(EVAL15_LOW, {".png", ".jpg"}), True))
    sets.append(("LIME", files_of(LIME, {".bmp", ".jpg", ".png"}), False))
    sets.append(("DICM", files_of(DICM, {".bmp", ".jpg", ".png"})[:16], False))

    all_rows = []
    summary = {}
    for name, files, has_gt in sets:
        rows = run_set(nets, files, has_gt)
        all_rows.extend(rows)
        for model in sorted({r["model"] for r in rows}):
            sub = [r for r in rows if r["model"] == model]
            summary[(name, model)] = stats(sub)
        print(f"[{name}] {len(files)} 张 完成")

    with open(os.path.join(OUT, "diag_metrics.csv"), "w", newline="", encoding="utf-8-sig") as fp:
        w = csv.DictWriter(fp, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    lines = ["# 边缘损失效果诊断（亮度无关指标）", ""]
    lines.append("说明：`cos_low` 是梯度向量余弦相似度（1.0=梯度方向完全一致）；`corr_low` 是梯度幅值图相关；")
    lines.append("`ledge_raw` 是论文 (4.12) 原值；`ledge_norm` 是**把增强图亮度缩放到与输入相同后**再算的 (4.12)，用来排除“变暗取巧”。")
    lines.append("")
    for name, _, has_gt in sets:
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| 模型 | 亮度(输入→输出) | M(输入→输出) | cos(g_low,g_enh) | corr(M_low,M_enh) | L_edge 原始 | L_edge 亮度归一 |")
        lines.append("|---|---|---|---|---|---|---|")
        order = ["GT", "baseline", "edge"] if has_gt else ["baseline", "edge"]
        for model in order:
            s = summary.get((name, model))
            if not s:
                continue
            lines.append(
                f"| {model} | {s['bright_low']:.1f}→{s['bright_enh']:.1f} "
                f"| {s['m_low']:.3f}→{s['m_enh']:.3f} "
                f"| {s['cos_low']:.4f} | {s['corr_low']:.4f} "
                f"| {s['ledge_raw']:.4f} | {s['ledge_norm']:.4f} |"
            )
        lines.append("")
    with open(os.path.join(OUT, "diag_report.md"), "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n输出目录：{OUT}")


if __name__ == "__main__":
    main()

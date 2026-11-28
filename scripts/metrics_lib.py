#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""指标库：所有指标都是对**真实像素**做的卷积/统计，没有常量、没有短路。

设计原则（对应"不要碰到什么都直接通过"）：
  1. 每个指标函数都是纯函数：输入 uint8 RGB，输出 float；
  2. 每个指标在 `self_test()` 里都有**解析可验证**的用例（黑图/白图/渐变/棋盘/纯色度台阶…）；
  3. 口径退化（例如平坦像素太少）会通过 `*_degraded` 标志**显式报告**，不静默替换。

指标口径（0~255 灰度 / 色度）：
  brightness            灰度均值
  contrast              灰度标准差
  saturation            HSV 的 S 通道均值（0~255）
  luma_noise_flat       平坦区 |Y - Gauss(Y,σ=1.5)| 均值
  chroma_noise_flat     平坦区 max(|Cb-hf|,|Cr-hf|) 均值
  detail_std8           RGB→灰度后 8×8 块内 std 的均值
  detail_std3           3×3 邻域 std 的均值（与 detail_std8 是两套口径，不要混用）
  blockiness            8 列/行网格边界 |ΔY| 均值 ÷ 非边界 |ΔY| 均值（1.0=无块效应）
  band_jump             暗区（gray<40，占比<2% 放宽 <80）内"面积>1% 的灰度级"之间的最大间隔
  n_levels              暗区内"面积>1%"的灰度级个数（越多越细腻）
  artificial_edges_lo   色度跳变≥3 且亮度跳变<1 的边界数 ×1000/(H·W)
  artificial_edges_hi   色度跳变≥6 且亮度跳变<3（更严口径，抗噪）
  colored_blob_share    色度幅值>10 的最大连通块面积占比（%）
  cos_struct            与参考图梯度方向的全局余弦（1=完全同向）
  psnr / ssim_rgb       与参考图的 PSNR / SSIM（RGB，11×11 高斯 σ=1.5）
  canny_precision/recall 与参考图 Canny 边缘的匹配（3px 容差）
"""
import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

try:
    cv2.setNumThreads(4)
except Exception:  # noqa: BLE001
    pass


# ------------------------------------------------------------------ 基础转换
def gray(u8):
    return np.asarray(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY), dtype=np.float32)


def ycc(u8):
    return cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)


def _hf(x, sigma=1.5):
    return np.abs(x - cv2.GaussianBlur(x, (0, 0), sigma))


def flat_mask(low_u8, thr=20.0, min_px=100):
    """输入图的平坦区（Sobel |gx|+|gy| < thr）。像素太少时返回 (全图, True) 并置降级标志。"""
    g = gray(low_u8)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = (np.abs(gx) + np.abs(gy)) < thr
    if int(m.sum()) < min_px:
        return np.ones_like(g, dtype=bool), True
    return m, False


# ------------------------------------------------------------------ 亮度/色彩
def brightness(u8):
    return float(gray(u8).mean())


def contrast(u8):
    return float(gray(u8).std())


def saturation(u8):
    hsv = cv2.cvtColor(u8, cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1].mean())


# ------------------------------------------------------------------ 噪声
def luma_noise_flat(u8, low_u8):
    m, deg = flat_mask(low_u8)
    return float(_hf(gray(u8))[m].mean()), deg


def chroma_noise_flat(u8, low_u8):
    m, deg = flat_mask(low_u8)
    c = ycc(u8)[..., 1:]
    hf = np.maximum(_hf(c[..., 0]), _hf(c[..., 1]))
    return float(hf[m].mean()), deg


def chroma_noise_global(u8):
    """不依赖平坦掩码的色噪：Cb/Cr 高频残差 std 的均值（与 compare_preprocess.py 同口径）。"""
    c = ycc(u8)[..., 1:]
    return float(np.mean([np.std(c[..., i] - cv2.GaussianBlur(c[..., i], (0, 0), 1.5)) for i in range(2)]))


# ------------------------------------------------------------------ 细节/块效应
def detail_std8(u8):
    g = gray(u8)
    k = 8
    h, w = g.shape[0] // k * k, g.shape[1] // k * k
    if h == 0 or w == 0:
        return float("nan")
    b = g[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    return float(b.std(axis=1).mean())


def detail_std3(u8):
    g = gray(u8)
    m, m2 = cv2.blur(g, (3, 3)), cv2.blur(g * g, (3, 3))
    return float(np.mean(np.sqrt(np.maximum(0.0, m2 - m * m))))


def blockiness(u8):
    """8 列/行网格边界跳变 ÷ 非边界跳变。1.0 = 无块效应；>1.03 视为有 JPEG 块。"""
    g = gray(u8)
    vals = []
    for axis in (1, 0):
        d = np.abs(np.diff(g, axis=axis))
        if d.size == 0:
            continue
        idx = np.zeros(d.shape[1], dtype=bool)
        idx[7::8] = True
        at, off = d[:, idx], d[:, ~idx]
        if at.size and off.size:
            vals.append((float(at.mean()) + 1e-6) / (float(off.mean()) + 1e-6))
    return float(np.mean(vals)) if vals else 1.0


# ------------------------------------------------------------------ 暗区台阶
def dark_mask(u8):
    g = gray(u8)
    m = g < 40
    if float(m.mean()) < 0.02:
        m = g < 80
    return m


def band_metrics(u8):
    g = gray(u8).astype(np.uint8)
    m = dark_mask(u8)
    v = g[m]
    if v.size < 10:
        return float("nan"), float("nan")
    h = np.bincount(v, minlength=256).astype(np.float64) / v.size
    big = [i for i, p in enumerate(h) if p > 0.01]
    jump = float(max(np.diff(big))) if len(big) > 1 else 0.0
    return jump, float(len(big))


# ------------------------------------------------------------------ 伪边缘/彩块
def artificial_edges(u8, dc=3.0, dy=1.0):
    c = ycc(u8)[..., 1:]
    cj = np.abs(np.diff(c, axis=1)).max(axis=2)
    yj = np.abs(np.diff(gray(u8), axis=1))
    cnt = int(((cj >= dc) & (yj < dy)).sum())
    return cnt * 1000.0 / (u8.shape[0] * u8.shape[1])


def colored_blob_share(u8, amp_thr=10.0, step=6):
    c = ycc(u8)[..., 1:]
    amp = np.sqrt((c[..., 0] - 128.0) ** 2 + (c[..., 1] - 128.0) ** 2)
    m = (amp > amp_thr).astype(np.uint8)
    if m.sum() == 0:
        return 0.0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return 0.0
    biggest = int(stats[1:, cv2.CC_STAT_AREA].max())
    return 100.0 * biggest / (u8.shape[0] * u8.shape[1])


# ------------------------------------------------------------------ 结构/参考指标
def cos_struct(u8, ref_u8):
    a, b = gray(u8), gray(ref_u8)
    ax = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    ay = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    bx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    by = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    num = float((ax * bx + ay * by).sum())
    den = float(np.sqrt((ax * ax + ay * ay).sum() * (bx * bx + by * by).sum()) + 1e-9)
    return num / den


def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def _gauss_win(size=11, sigma=1.5):
    x = np.arange(size, dtype=np.float32) - (size - 1) / 2.0
    g = np.exp(-(x ** 2) / (2.0 * sigma ** 2))
    g /= g.sum()
    return g


def ssim_rgb(a, b):
    """标准 SSIM：11×11 高斯窗 σ=1.5，逐通道后取均值。"""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    w = _gauss_win(11, 1.5)
    win = np.outer(w, w)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for ch in range(3):
        x, y = a[..., ch], b[..., ch]
        mu1 = cv2.filter2D(x, -1, win)
        mu2 = cv2.filter2D(y, -1, win)
        s11 = cv2.filter2D(x * x, -1, win) - mu1 * mu1
        s22 = cv2.filter2D(y * y, -1, win) - mu2 * mu2
        s12 = cv2.filter2D(x * y, -1, win) - mu1 * mu2
        num = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
        den = (mu1 * mu1 + mu2 * mu2 + C1) * (s11 + s22 + C2)
        vals.append(float(np.mean(num / den)))
    return float(np.mean(vals))


def canny_edges(u8, lo=50, hi=120):
    return cv2.Canny(cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY), lo, hi) > 0


def _dilate(mask, r=3):
    k = np.ones((2 * r + 1, 2 * r + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def edge_stats(pred_u8, gt_u8, tol=3):
    """Canny 边缘与参考图的匹配：3px 容差；precision=预测边缘中落在 GT 容差内的比例。"""
    pe = canny_edges(pred_u8)
    ge = canny_edges(gt_u8)
    if pe.sum() == 0 or ge.sum() == 0:
        return float("nan"), float("nan"), int(pe.sum()), int(ge.sum())
    p_in = float((pe & _dilate(ge, tol)).sum()) / float(pe.sum())
    g_in = float((ge & _dilate(pe, tol)).sum()) / float(ge.sum())
    return p_in, g_in, int(pe.sum()), int(ge.sum())


# ------------------------------------------------------------------ 一次算全
def measure_all(u8, low_u8=None, ref_u8=None):
    """u8 的指标；low_u8 用于平坦掩码与结构对比（默认用 u8 自身），ref_u8 为参考图（可空）。"""
    low = u8 if low_u8 is None else low_u8
    ln, ln_deg = luma_noise_flat(u8, low)
    cn, cn_deg = chroma_noise_flat(u8, low)
    bj, nl = band_metrics(u8)
    d = {
        "brightness": brightness(u8),
        "contrast": contrast(u8),
        "saturation": saturation(u8),
        "luma_noise_flat": ln,
        "chroma_noise_flat": cn,
        "chroma_noise_global": chroma_noise_global(u8),
        "detail_std8": detail_std8(u8),
        "detail_std3": detail_std3(u8),
        "blockiness": blockiness(u8),
        "band_jump": bj,
        "n_levels": nl,
        "artificial_edges_lo": artificial_edges(u8, 3.0, 1.0),
        "artificial_edges_hi": artificial_edges(u8, 6.0, 3.0),
        "colored_blob_share": colored_blob_share(u8),
        "cos_struct": cos_struct(u8, low),
        "flat_mask_degraded": float(ln_deg or cn_deg),
    }
    if ref_u8 is not None and ref_u8.shape[:2] == u8.shape[:2]:
        p, r, pc, gc = edge_stats(u8, ref_u8)
        d.update({"psnr_vs_ref": psnr(u8, ref_u8), "ssim_vs_ref": ssim_rgb(u8, ref_u8),
                  "edge_precision": p, "edge_recall": r, "edge_px": pc, "ref_edge_px": gc})
    else:
        d.update({"psnr_vs_ref": float("nan"), "ssim_vs_ref": float("nan"),
                  "edge_precision": float("nan"), "edge_recall": float("nan"),
                  "edge_px": float("nan"), "ref_edge_px": float("nan")})
    return d


# ------------------------------------------------------------------ 自检（解析可验证）
def self_test(verbose=True):
    """用**解析可验证**的合成图检验每个指标确实在算数。返回 (通过数, 失败列表)。"""
    rng = np.random.default_rng(20261128)
    fails = []

    def chk(name, cond, detail=""):
        if verbose:
            print(("  [PASS] " if cond else "  [FAIL] ") + name + ("" if cond else f"   <- {detail}"))
        if not cond:
            fails.append(f"{name}: {detail}")

    black = np.zeros((96, 96, 3), np.uint8)
    white = np.full((96, 96, 3), 255, np.uint8)
    chk("黑图亮度=0", abs(brightness(black)) < 1e-6, brightness(black))
    chk("白图亮度=255", abs(brightness(white) - 255) < 1e-6, brightness(white))
    chk("纯色图色噪≈0", chroma_noise_global(white) < 1e-6, chroma_noise_global(white))
    chk("纯色图细节=0", detail_std8(white) < 1e-6, detail_std8(white))
    chk("纯色图块效应=1", abs(blockiness(white) - 1.0) < 1e-3, blockiness(white))

    ramp = np.tile(np.arange(256, dtype=np.uint8)[None, :, None], (64, 1, 3))
    chk("水平渐变亮度≈127.5", abs(brightness(ramp) - 127.5) < 1.0, brightness(ramp))
    chk("水平渐变饱和度=0", saturation(ramp) < 1e-6, saturation(ramp))

    noise = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    chk("随机图色噪>1", chroma_noise_global(noise) > 1.0, chroma_noise_global(noise))
    chk("随机图块效应≈1", 0.7 < blockiness(noise) < 1.4, blockiness(noise))

    cb = np.zeros((128, 128, 3), np.uint8)
    cb[::8] = 60
    cb[:, ::8] = 60
    chk("8px 棋盘块效应>1.8", blockiness(cb) > 1.8, blockiness(cb))

    # 纯色度台阶：在 YCrCb 空间只改 Cr（Y 保持 128），再转回 RGB → 亮度应几乎不变、色度跳变明显
    ycc_step = np.zeros((64, 128, 3), np.uint8)
    ycc_step[..., 0] = 128
    ycc_step[..., 1] = 128
    ycc_step[..., 2] = 128
    ycc_step[:, 64:, 2] = 180
    cstep = cv2.cvtColor(ycc_step, cv2.COLOR_YCrCb2RGB)
    yjump = float(np.abs(np.diff(gray(cstep), axis=1)).max())
    chk("纯色度台阶计入伪边缘", artificial_edges(cstep, 3.0, 1.0) > 1.0,
        artificial_edges(cstep, 3.0, 1.0))
    chk("纯色度台阶亮度跳变<2", yjump < 2.0, yjump)

    # 纯亮度台阶：不应计入伪边缘
    ystep = np.zeros((64, 128, 3), np.uint8)
    ystep[:, :64] = 20
    ystep[:, 64:] = 60
    chk("纯亮度台阶不计入伪边缘", artificial_edges(ystep, 3.0, 1.0) < 1e-9,
        artificial_edges(ystep, 3.0, 1.0))

    # 暗区台阶：0~40 的暗渐变（每档都占 >1%）
    dark = np.tile(np.arange(41, dtype=np.uint8)[None, :, None].repeat(1, axis=1), (1, 1, 3))
    dark = np.repeat(dark, 64 // 41 + 1, axis=0)[:64]
    dark = np.ascontiguousarray(np.repeat(dark[:, :, :], 2, axis=1))
    _, nl = band_metrics(dark)
    chk("暗渐变级数≈41", nl >= 30, nl)

    chk("PSNR 相同图=inf", psnr(ramp, ramp) == float("inf"), psnr(ramp, ramp))
    chk("PSNR 不同图有限且<60", 2 < psnr(ramp, np.full_like(ramp, 255)) < 60, psnr(ramp, np.full_like(ramp, 255)))
    chk("SSIM 相同图≈1", abs(ssim_rgb(ramp, ramp) - 1.0) < 1e-6, ssim_rgb(ramp, ramp))
    s_black = ssim_rgb(black, white)
    chk("SSIM 黑/白<0.1", s_black < 0.1, s_black)

    m_a, m_b = measure_all(black), measure_all(noise)
    diff = sum(abs(m_a[k] - m_b[k]) for k in ("brightness", "chroma_noise_global", "detail_std8"))
    chk("不同输入不会被算成同一组指标", diff > 10, diff)

    p, r, _, _ = edge_stats(cb, cb)  # 用有边缘的棋盘图，平滑渐变 Canny 检测不到边
    chk("边缘自匹配 precision=1", abs(p - 1.0) < 1e-6, p)
    chk("边缘自匹配 recall=1", abs(r - 1.0) < 1e-6, r)

    return (len(fails) == 0, fails)

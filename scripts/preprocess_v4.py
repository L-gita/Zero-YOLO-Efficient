"""v4 原型：A（色度噪声判据）+ B（连续权重 + 引导滤波羽化）的色度去噪。

与 auto v3 的区别
-----------------
v3：用「亮度平坦」判据决定要不要处理色度 → 判据和问题错位（暗但彩色的布料/玩偶被漏掉或过度处理）。
v4：
  A. 判据换成「色度高频能量」减去「可由亮度结构解释的部分」：
     只在"色度在抖、但亮度这里没有对应结构"的地方处理（即真正的色度噪声），
     真实彩色边缘（色度与亮度同时跳变）保持不动。
  B. 权重连续化 + 用亮度做引导滤波羽化（半径 16 px）：权重沿真实轮廓走，
     不会再沿"色度等值线"切出一条人造分界线。
"""
import numpy as np
import cv2


def _box(x, r):
    return cv2.boxFilter(x, -1, (2 * r + 1, 2 * r + 1), normalize=True, borderType=cv2.BORDER_REFLECT)


def _guided_filter(guide, src, r=16, eps=1e-3):
    """He 等人快速引导滤波（guide/src 都是 float32，单通道）"""
    I = guide.astype(np.float32)
    p = src.astype(np.float32)
    mI = _box(I, r); mp = _box(p, r)
    corr_I = _box(I * I, r); corr_Ip = _box(I * p, r)
    varI = corr_I - mI * mI
    covIp = corr_Ip - mI * mp
    a = covIp / (varI + eps)
    b = mp - a * mI
    mean_a = _box(a, r); mean_b = _box(b, r)
    return mean_a * I + mean_b


def _sstep(x, lo=0.0, hi=1.0):
    t = np.clip((x - lo) / max(1e-6, (hi - lo)), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def chroma_denoise_v4(rgb_u8, strength=1.0, tau=5.0, beta=1.6,
                      guide_radius=16, smooth_radius=3, protect_real_edges=True):
    """A+B 版色度去噪。

    rgb_u8 : HxWx3 uint8
    strength : 总强度（0=不处理）
    tau    : 色度高频能量阈值（8-bit 单位，越大越保守）
    beta   : 亮度结构对"噪声判定"的扣除系数（越大越保护真实彩色边缘）
    guide_radius : 权重羽化半径（像素），越大过渡越柔和
    smooth_radius: 色度平滑半径（小半径即可，羽化由权重负责）
    """
    if strength <= 0:
        return rgb_u8
    x = rgb_u8.astype(np.float32)
    ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    Y, Cr, Cb = ycc[..., 0].copy(), ycc[..., 1].copy(), ycc[..., 2].copy()
    C = np.dstack([Cr, Cb])

    # ---- A：色度高频能量 & 亮度高频能量 ----
    C_lp = cv2.GaussianBlur(C, (0, 0), 1.5)
    hp_c = np.abs(C - C_lp).max(axis=2)                       # 色度抖动
    Y_hp = np.abs(Y - cv2.GaussianBlur(Y, (0, 0), 1.5))       # 亮度结构

    num = hp_c - beta * Y_hp if protect_real_edges else hp_c
    w_raw = _sstep(num / max(tau, 1e-3), 0.0, 1.0)            # 0~1 连续权重
    # ---- B：用亮度做引导羽化，让权重沿真实轮廓过渡 ----
    w = _guided_filter(Y, w_raw, r=guide_radius, eps=1e-2)
    w = np.clip(w, 0.0, 1.0) * float(strength)
    w3 = w[..., None]

    # 色度平滑（小半径盒式，避免大半径造成大色块；真正的"大范围一致"由羽化权重控制）
    C_sm = cv2.blur(C, (2 * smooth_radius + 1, 2 * smooth_radius + 1))
    C_out = C * (1.0 - w3) + C_sm * w3

    ycc2 = np.dstack([Y, C_out[..., 0], C_out[..., 1]])
    out = cv2.cvtColor(ycc2, cv2.COLOR_YCrCb2RGB)
    return np.clip(out, 0, 255).astype(np.uint8)


def seam_prominence(rgb_u8, flat_pct=30, pct=95):
    """人造分界线指标：亮度平坦区的色度梯度 p95（越低越少"色块边"）。"""
    y = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2YCrCb)[..., 1:].astype(np.float32)
    gy = np.abs(cv2.Sobel(y, cv2.CV_32F, 1, 0, 3)) + np.abs(cv2.Sobel(y, cv2.CV_32F, 0, 1, 3))
    gc = np.zeros_like(gy)
    for k in range(2):
        gc = np.maximum(gc, np.abs(cv2.Sobel(c[..., k], cv2.CV_32F, 1, 0, 3))
                        + np.abs(cv2.Sobel(c[..., k], cv2.CV_32F, 0, 1, 3)))
    flat = gy <= np.percentile(gy, flat_pct)
    return float(np.percentile(gc[flat], pct))

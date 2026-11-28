#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""马赛克预处理（方案 B）：色度优先降噪 / JPEG 去块 / 暗部轻抬 / 保边平滑 / 16-bit 输出。

为什么需要它
------------
Zero-DCE 的曲线在暗区斜率 >1（实测放大 3.6~5.7 倍），会把原本就藏在暗处的
三个缺陷一起放大：
  ① 8-bit 量化台阶   → 抬升后变成一块块色阶
  ② 暗光色度噪声     → 变成彩色花斑（"马赛克"最主要的观感来源）
  ③ JPEG 8x8 压缩缝  → 块边界被放大
本模块在“增强之前/之后”加轻量预处理，把这三类缺陷先压下去（顺序不能反：
先降噪再增强，否则噪声已经被放大成结构化色块，洗不掉）。

所有函数都吃/吐 uint8 RGB（HxWx3, np.uint8），便于和 enhance.py / 批量脚本共用。
参数取值范围见文件末尾 main() 里的示例。

用法（库）：
    from preprocess import denoise_chroma, deblock, prelift, post_smooth
    img = denoise_chroma(img, strength=1.0)
    img = deblock(img, strength=1.0)
    img = prelift(img, amount=0.15)
"""
import cv2
import numpy as np

# OpenCV 的 NLM 在多核上会互相抢线程，批量跑时限制一下更稳
try:
    cv2.setNumThreads(4)
except Exception:  # noqa: BLE001
    pass


# ----------------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------------
def _to_u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def _to_f32(img):
    return img.astype(np.float32)


# ----------------------------------------------------------------------------
# ① 色度优先降噪：颜色重降、亮度轻降（人眼对颜色细节不敏感）
# ----------------------------------------------------------------------------
def _chroma_amp(ycc):
    return float(np.sqrt((ycc[..., 1] - 128.0) ** 2 + (ycc[..., 2] - 128.0) ** 2).mean())


def _rescale_chroma(img_ycc, target_amp, max_gain=1.8):
    """把 img_ycc 的色度幅度缩放到 target_amp（保色调、只调饱和），避免滤波把颜色洗白。"""
    cur = _chroma_amp(img_ycc)
    if cur < 1e-3:
        return img_ycc
    k = float(np.clip(target_amp / cur, 1.0, max_gain))
    if abs(k - 1.0) < 1e-3:
        return img_ycc
    out = img_ycc.copy()
    out[..., 1] = 128.0 + (out[..., 1] - 128.0) * k
    out[..., 2] = 128.0 + (out[..., 2] - 128.0) * k
    return np.clip(out, 0, 255)


def denoise_chroma(img_rgb, strength=1.0):
    """YCrCb 空间中：Y（亮度）轻降噪，Cr/Cb（色度）重降噪。

    strength: 0=不降；1=推荐（色噪明显）；2=很强（极端暗图，可能糊掉彩色细节）
    返回 uint8 RGB。
    """
    if strength <= 0:
        return img_rgb
    s = float(strength)
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    amp_before = _chroma_amp(ycc)
    y, cr, cb = cv2.split(ycc)
    h_y = 1.5 + 2.5 * s          # 亮度：轻
    h_c = 2.0 + 6.0 * s          # 色度：重（但比 v1 的 4+8s 温和）
    y = cv2.fastNlMeansDenoising(y.astype(np.uint8), None, h=h_y,
                                 templateWindowSize=7, searchWindowSize=15).astype(np.float32)
    cr = cv2.fastNlMeansDenoising(cr.astype(np.uint8), None, h=h_c,
                                  templateWindowSize=7, searchWindowSize=15).astype(np.float32)
    cb = cv2.fastNlMeansDenoising(cb.astype(np.uint8), None, h=h_c,
                                  templateWindowSize=7, searchWindowSize=15).astype(np.float32)
    out = np.stack([y, cr, cb], axis=2)
    # 关键：NLM 会把色度往均值拉 → 颜色变淡；这里把色度幅度补回去（保色调、只调饱和）
    out = _rescale_chroma(out, amp_before)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


# ----------------------------------------------------------------------------
# ② JPEG 去块：只磨 8x8 网格上的“缝”，真边缘（跨缝差很大）不动
# ----------------------------------------------------------------------------
def deblock(img_rgb, strength=1.0, block=8, thr=12.0, passes=1):
    """在亮度通道上做方向性去块（类似 H.264 的 deblocking）。

    原理：JPEG 把图切成 8x8 小块分别 DCT，块与块交界处出现不连续。
    若交界两侧的差很小（< thr），就认为这是压缩缝，用 4 抽头平滑抹平；
    若差很大（> thr），说明这里是真边缘，保持不动。
    """
    if strength <= 0:
        return img_rgb
    s = float(np.clip(strength, 0.0, 1.0))
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[:, :, 0]
    h, w = y.shape

    for _ in range(max(1, int(passes))):
        # 竖直边界（列 x = block, 2*block, ...）
        for b in range(block, w, block):
            if b < 2 or b + 1 >= w:
                continue
            l2, l1, r1, r2 = y[:, b - 2], y[:, b - 1], y[:, b], y[:, b + 1]
            mask = np.abs(r1 - l1) < thr
            sm = (l2 + 3.0 * l1 + 3.0 * r1 + r2) / 8.0
            y[:, b - 1] = np.where(mask, l1 * (1.0 - s) + sm * s, l1)
            y[:, b] = np.where(mask, r1 * (1.0 - s) + sm * s, r1)
        # 水平边界（行 y = block, 2*block, ...）
        for b in range(block, h, block):
            if b < 2 or b + 1 >= h:
                continue
            u2, u1, d1, d2 = y[b - 2, :], y[b - 1, :], y[b, :], y[b + 1, :]
            mask = np.abs(d1 - u1) < thr
            sm = (u2 + 3.0 * u1 + 3.0 * d1 + d2) / 8.0
            y[b - 1, :] = np.where(mask, u1 * (1.0 - s) + sm * s, u1)
            y[b, :] = np.where(mask, d1 * (1.0 - s) + sm * s, d1)

    ycc[:, :, 0] = np.clip(y, 0, 255)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)


# ----------------------------------------------------------------------------
# ③ 暗部轻抬：增强前先“打个手电”，降低网络需要的增益
# ----------------------------------------------------------------------------
def prelift(img_rgb, amount=0.15):
    """在 LAB 的 L 通道做轻微 gamma 提升（只抬暗部，不动饱和区域）。

    amount: 0=不抬；0.15=推荐；0.3 以上容易把噪声一起抬起来 + 偏离训练分布
    """
    if amount <= 0:
        return img_rgb
    a = float(np.clip(amount, 0.0, 1.0))
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0
    gamma = 1.0 / (1.0 + 1.5 * a)          # a=0.15 -> gamma≈0.82
    l = np.power(np.clip(l, 0, 1), gamma)
    # 只作用于暗部：亮部（L>0.6）用原值，避免整体发灰
    w = np.clip((0.6 - l) / 0.6, 0.0, 1.0)
    l = l * w + (lab[:, :, 0].astype(np.float32) / 255.0) * (1 - w)
    lab[:, :, 0] = _to_u8(l * 255.0)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ----------------------------------------------------------------------------
# ④ 保边平滑（后处理）：最后轻扫一遍残余色噪
# ----------------------------------------------------------------------------
def post_smooth(img_rgb, strength=1.0):
    """双边滤波轻平滑：sigmaColor 小 → 只抹低对比差异（噪声），保留边缘。"""
    if strength <= 0:
        return img_rgb
    s = float(strength)
    d = 5 if s < 1.5 else 7
    return cv2.bilateralFilter(img_rgb, d, 12.0 * s, 5.0)


# ----------------------------------------------------------------------------
# ④b 亮度引导的色度平滑（推荐替代 post_smooth）
#     -- 用亮度 Y 当“导游”，只在亮度平坦处平滑色度；亮度有边缘时色度不动
#     好处：不会把颜色抹过大片平坦暗区形成“大色块”，也不会跨物体边界串色
# ----------------------------------------------------------------------------
def _box(x, radius):
    k = 2 * int(radius) + 1
    return cv2.boxFilter(x, -1, (k, k))


def guided_filter(guide, src, radius=2, eps=1e-3):
    """快速引导滤波（He et al. 的非近似版），guide/src 均为 float32 单通道。"""
    r = int(radius)
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    corr_i = _box(guide * guide, r)
    corr_ip = _box(guide * src, r)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = _box(a, r)
    mean_b = _box(b, r)
    return mean_a * guide + mean_b


def chroma_guided_smooth(img_rgb, luma_guide=None, radius=2, eps=1e-3, strength=1.0):
    """亮度引导的色度平滑：只动 Cb/Cr，不动 Y（不糊亮度细节）。

    img_rgb    : uint8 RGB
    luma_guide : 用来当导游的图（默认就是自己；若要“用增强前的亮度引导”，传原图）
    radius     : 局部窗口半径，2 表示 5×5；越大越平滑（>4 容易抹出大色块）
    eps        : 引导滤波正则项，越小越"跟着亮度走"（1e-3 ≈ 灰度 0.03 以内都算平）
    strength   : 0~1，平滑结果与原色度的插值比例
    """
    if strength <= 0:
        return img_rgb
    ycc = cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    guide_rgb = img_rgb if luma_guide is None else luma_guide
    g = cv2.cvtColor(np.ascontiguousarray(guide_rgb), cv2.COLOR_RGB2YCrCb)[:, :, 0].astype(np.float32) / 255.0
    out = ycc.copy()
    s = float(np.clip(strength, 0.0, 1.0))
    amp_before = _chroma_amp(ycc)
    for ci in (1, 2):
        src = ycc[:, :, ci] / 255.0
        sm = guided_filter(g, src, radius=radius, eps=eps) * 255.0
        out[:, :, ci] = ycc[:, :, ci] * (1.0 - s) + sm * s
    out = _rescale_chroma(out, amp_before)      # 平滑会把色度拉向均值 → 补回饱和度
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)



# ----------------------------------------------------------------------------
# ④c 极暗无信号区抑制（E6 方案的关键一步）
#     问题：近黑区域里"只有噪声、没有信号"，任何降噪都只能把它抹成一块均匀色斑，
#           增强后就是用户看到的"更大块马赛克 / 一片色块"。
#     做法：用输入低照图判断"有没有信号"，对没信号的区域
#           - 色度拉回中性（去色，避免彩色斑块）
#           - 亮度压回到 输入×max_gain 以内（不再死命提亮）
#           掩码羽化，避免出现新的硬边界。
# ----------------------------------------------------------------------------
def suppress_low_snr(enh_rgb, low_rgb, dark_thr=45, var_thr=10.0, strength=1.0,
                     feather=12, max_gain=2.0):
    """极暗无信号区抑制：该区域宁可压暗去饱和，也不要变成彩色大色块。

    enh_rgb   : 增强结果 uint8 RGB
    low_rgb   : 对应的输入低照图 uint8 RGB（用它判断有没有信号）
    dark_thr  : 灰度低于此值算"暗"（0~255）
    var_thr   : 5×5 局部标准差低于此值算"没纹理"（0~255）
    strength  : 抑制强度 0~1
    feather   : 掩码羽化半径（像素，越大过渡越柔和）
    max_gain  : 这些区域允许的亮度放大倍数上限
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0:
        return img_rgb if enh_rgb is None else enh_rgb
    low = np.ascontiguousarray(low_rgb)
    g = cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m1 = cv2.boxFilter(g, -1, (5, 5))
    m2 = cv2.boxFilter(g * g, -1, (5, 5))
    std = np.sqrt(np.maximum(m2 - m1 * m1, 0.0))
    mask = ((g < dark_thr) & (std < var_thr)).astype(np.float32)
    mask = cv2.GaussianBlur(mask, (0, 0), float(max(1.0, feather))) * s
    m = np.clip(mask, 0.0, 1.0)[..., None]

    e = cv2.cvtColor(np.ascontiguousarray(enh_rgb), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    l = cv2.cvtColor(low, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y_e, y_l = e[..., 0:1], l[..., 0:1]
    target_y = np.minimum(y_e, y_l * float(max_gain))
    e[..., 0:1] = y_e * (1.0 - m) + target_y * m
    e[..., 1:3] = e[..., 1:3] * (1.0 - m) + 128.0 * m
    return cv2.cvtColor(np.clip(e, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


# ----------------------------------------------------------------------------
# ④d 近黑区"限增益"（E8 关键一步）——解决彩色大斑块
#     近黑区域里没有可靠的颜色信息：模型把微小噪声放大后，不同通道被放大得不一样，
#     就会出现大片棕/绿色斑块（用户看到的“更大的马赛克”）。
#     做法：亮度 < l0 的地方，用“中性 gamma 抬亮”（三通道同一映射）替代模型输出；
#           l0~l1 之间线性过渡。这样既保留亮度提升，又不会产生彩色斑块。
# ----------------------------------------------------------------------------
def neutral_lift(low_rgb, gamma=0.40, gain=1.0):
    """按灰度整体 gamma 抬亮（三通道同一映射 → 绝对中性，不可能产生偏色）"""
    g = cv2.cvtColor(np.ascontiguousarray(low_rgb), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    lift = np.power(np.clip(g, 0.0, 1.0), float(gamma)) * 255.0 * float(gain)
    return np.repeat(np.clip(lift, 0, 255)[..., None], 3, axis=2)


def _smooth_luma(low, radius=8, passes=2):
    """大核多次 box 模糊近似高斯：专门用来把近黑区的 8bit 台阶/8x8 块抹成连续梯度。"""
    y = low.astype(np.float32).mean(axis=2) / 255.0
    k = int(radius) * 2 + 1
    for _ in range(max(1, passes)):
        y = cv2.blur(y, (k, k))
    return y


def limit_dark_gain(enh, low, l0=6.0, l1=20.0, gamma=0.40, strength=0.85,
                    smooth_radius=8, dither=0.6, seed=0):
    """近黑区限增益（v2）：近黑区不再直接用模型输出（会有彩色大斑），
    改用"平滑后亮度 + 中性 gamma 抬亮"；关键是先把亮度大模糊，否则
    输入只剩几档灰度 → 输出就是几大块硬边灰块（这正是 v1 的问题）。

    enh/low: HxWx3 uint8。返回 float32 HxWx3（0~255，未加抖动，抖动在 8bit 量化时加）。
    """
    enh_f = enh.astype(np.float32)
    low_f = low.astype(np.float32)
    y = low_f.mean(axis=2) / 255.0
    ys = _smooth_luma(low, radius=smooth_radius, passes=2)   # 连续化
    lifted = (np.power(np.clip(ys, 0.0, 1.0), float(gamma)) * 255.0)[..., None]
    neutral = np.repeat(lifted, 3, axis=2)
    # 羽化掩码：y < l0 全中性，l0~l1 过渡，>l1 用模型
    m = np.clip((l1 / 255.0 - y) / (max(l1 - l0, 1e-3) / 255.0), 0.0, 1.0) * float(strength)
    sigma = max(1.0, (l1 - l0) / 4.0)
    m = cv2.GaussianBlur(m, (0, 0), sigma)                   # 再羽化，去掉过渡带硬轮廓
    out = m[..., None] * neutral + (1.0 - m[..., None]) * enh_f
    return np.clip(out, 0.0, 255.0)


def quantize8(img_float, dither=0.0, seed=0):
    """float(0~255) → uint8，可选加抖动（抑制 8bit 量化条带）。"""
    x = np.asarray(img_float, np.float32)
    if dither and dither > 0:
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-dither, dither, x.shape)
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)

def save_16bit(img_float01, path):
    """把 [0,1] 的 float 图存成 16-bit PNG（避免再被压回 8-bit 丢精度）。

    注意：PIL 不能直接存 3 通道 16-bit，这里用 cv2 编码后自己写字节，
    同时避开 cv2.imwrite 对中文路径的编码问题。
    """
    arr = np.clip(np.asarray(img_float01, dtype=np.float32), 0.0, 1.0)
    arr16 = np.round(arr * 65535.0).astype(np.uint16)
    if arr16.ndim == 3 and arr16.shape[2] == 3:
        arr16 = cv2.cvtColor(arr16, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", arr16)
    if not ok:
        raise RuntimeError(f"16-bit PNG 编码失败: {path}")
    with open(path, "wb") as fp:
        fp.write(buf.tobytes())


if __name__ == "__main__":
    # 小自测：读一张图，逐步打印和原图的差异（需要命令行传图片路径）
    import sys
    from PIL import Image
    if len(sys.argv) < 2:
        print("用法: python preprocess.py 某张暗光图.png")
        raise SystemExit(0)
    src = np.asarray(Image.open(sys.argv[1]).convert("RGB"))
    steps = [
        ("原图", src),
        ("去块 1.0", deblock(src, 1.0)),
        ("色度降噪 1.0", denoise_chroma(src, 1.0)),
        ("暗部轻抬 0.15", prelift(src, 0.15)),
        ("全套", preprocess(src, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)),
    ]
    print(f"{'步骤':<16}{'亮度':>8}{'相对原图MAE':>12}")
    for name, im in steps:
        mae = float(np.abs(im.astype(np.float32) - src.astype(np.float32)).mean())
        print(f"{name:<16}{float(im.mean()):>8.2f}{mae:>12.2f}")

def preprocess(img_rgb, denoise=0.0, deblock_strength=0.0, prelift_amount=0.0):
    """预处理总入口（顺序：去块 → 色度优先降噪 → 暗部轻抬）。

    img_rgb: HxWx3 uint8。三项都为 0 时原样返回。
    """
    x = _to_u8(img_rgb)
    if deblock_strength and deblock_strength > 0:
        x = deblock(x, strength=float(deblock_strength))
    if denoise and denoise > 0:
        x = denoise_chroma(x, strength=float(denoise))
    if prelift_amount and prelift_amount > 0:
        x = prelift(x, amount=float(prelift_amount))
    return x


def chroma_boost(img_rgb, ref_rgb, strength=0.6, max_gain=1.8):
    """把（被降噪/引导平滑冲淡的）色度按参考图幅度补回来，避免“发白发灰”。

    做法：分别算 ref 与 img 在 YCrCb 里的平均色度幅度，比例 clamped 到 [1, max_gain]，
    再按 strength 混合：out_chroma = c + strength * (gain - 1) * (c - 128)。
    只动色度，不动亮度与色调（角度不变），所以不会引入偏色。
    """
    if strength <= 0:
        return _to_u8(img_rgb)
    ref = cv2.cvtColor(_to_u8(ref_rgb), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    img = cv2.cvtColor(_to_u8(img_rgb), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    amp_ref = np.sqrt((ref[..., 1] - 128.0) ** 2 + (ref[..., 2] - 128.0) ** 2).mean()
    amp_img = np.sqrt((img[..., 1] - 128.0) ** 2 + (img[..., 2] - 128.0) ** 2).mean()
    if amp_img < 1e-3:
        return _to_u8(img_rgb)
    gain = float(np.clip(amp_ref / amp_img, 1.0, max_gain))
    k = 1.0 + float(strength) * (gain - 1.0)
    img[..., 1] = 128.0 + (img[..., 1] - 128.0) * k
    img[..., 2] = 128.0 + (img[..., 2] - 128.0) * k
    img = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

# ----------------------------------------------------------------------------
# ⑤ E10：场景自适应预处理（2026-11-27 第二版改进）
#     旧版（E9）的问题：固定全局管线 —— 在正常夜景上先把图抬亮再送给模型（域偏移），
#     且色度大核平滑会把天空/墙面这类平滑渐变切成"色块"。
#     E10 的改法：一切按需 + 只动该动的：
#       · 先估计噪声和块效应，噪声重才降噪、检测到 8×8 块才去块
#       · 色度只在"高频残差"上去噪，低频颜色（天空渐变）原样保留 → 不再切色块
#       · 暗部轻抬只在极暗图（均值<8）启用
#       · 近黑限增益只碰真正近黑（<2.5），且只改亮度、色度轻微拉中性；再对近黑区做一次掩码色度降噪
# ----------------------------------------------------------------------------
def chroma_light_strength(chroma_noise):
    """常规分支的色度轻度平滑强度（分段，按输入色度噪声 σc）。

    σc < 0.3          → 0（跳过，输出等同原版）
    0.3 ≤ σc < 0.6    → 0.35·σc（下限 0.15；101/147/12 这类，保持现状）
    0.6 ≤ σc < 1.0    → 0.5·σc （493=0.74、665=0.94 → 0.37/0.47）
    σc ≥ 1.0          → 0.7·σc，上限 0.7（778=1.13 → 0.70）
    """
    cn = float(chroma_noise)
    if cn < 0.3:
        return 0.0
    if cn < 0.6:
        return float(max(0.15, 0.35 * cn))
    if cn < 1.0:
        return float(0.5 * cn)
    return float(min(0.7, 0.7 * cn))



def estimate_noise(rgb):
    """用暗区拉普拉斯残差的 MAD 估噪声 sigma（0~255 单位）"""
    g = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    m = g < 40
    if m.mean() < 0.05:
        m = g < 80
    v = lap[m] if m.any() else lap.ravel()
    return float(np.median(np.abs(v)) / 0.6745)


def estimate_blockiness(rgb):
    """8×8 网格边界跳变 / 非边界跳变；>1.03 说明可能有 JPEG/MJPG 块"""
    g = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=1))
    W = g.shape[1]
    bcols = [c for c in range(7, W - 1, 8) if 0 < c - 1 < d.shape[1]]
    if not bcols:
        return 1.0
    on = np.mean([d[:, c - 1].mean() for c in bcols])
    off = np.mean([d[:, c - 1].mean() for c in range(1, d.shape[1] - 1) if (c % 8) != 0])
    return float(on / (off + 1e-6))


def chroma_denoise_hf(rgb, strength=1.0, lf_sigma=8.0):
    """只在色度高频残差上去噪：低频颜色（天空渐变）原样保留，避免切出大色块"""
    ycc = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    amp0 = _chroma_amp(ycc)
    out = ycc.copy()
    for ci in (1, 2):
        c = ycc[..., ci]
        low = cv2.GaussianBlur(c, (0, 0), lf_sigma)
        res = c - low
        res_d = cv2.bilateralFilter(res, d=5, sigmaColor=6.0 * float(strength), sigmaSpace=3)
        out[..., ci] = low + res_d
    out = _rescale_chroma(out, amp0)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def masked_chroma_denoise(rgb, low, thr=6.0, strength=2.2, lf_sigma=8.0):
    """只在输入亮度 < thr 的近黑区做加强版色度残差降噪（掩码羽化）"""
    ycc = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ly = cv2.cvtColor(np.asarray(low, dtype=np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    ly_s = cv2.GaussianBlur(ly, (0, 0), 1.5)
    m = np.clip((thr + 3.0 - ly_s) / 3.0, 0, 1)
    m = cv2.GaussianBlur(m, (0, 0), 1.0)
    out = ycc.copy()
    for ci in (1, 2):
        c = ycc[..., ci]
        lowf = cv2.GaussianBlur(c, (0, 0), lf_sigma)
        res = c - lowf
        res_d = cv2.bilateralFilter(res, d=5, sigmaColor=6.0 * float(strength), sigmaSpace=3)
        c2 = lowf + res_d
        out[..., ci] = c * (1 - m) + c2 * m
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def limit_dark_luma(enh, low, thr=2.5, feather=3.0, gamma=0.60, strength=0.9, chroma_k=0.85):
    """近黑区限增益（只改亮度；色度按同一羽化掩码轻微拉向中性，避免彩色斑）"""
    e = cv2.cvtColor(np.asarray(enh, dtype=np.uint8), cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ly = cv2.cvtColor(np.asarray(low, dtype=np.uint8), cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    ly_s = cv2.GaussianBlur(ly, (0, 0), 2.0)
    neutral = 255.0 * np.power(np.clip(ly_s, 0, 255) / 255.0, float(gamma))
    m = np.clip((thr + feather - ly_s) / feather, 0, 1) * float(strength)
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    e[..., 0] = e[..., 0] * (1 - m) + neutral * m
    m2 = m * float(chroma_k)
    for ci in (1, 2):
        e[..., ci] = e[..., ci] * (1 - m2) + 128.0 * m2
    return cv2.cvtColor(np.clip(e, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def auto_preprocess(low, strength=1.0):
    """E10 场景自适应预处理（返回 dict：预处理图 + 决策信息）

    strength: 全局强度 0~1+（--auto-strength）。会同时作用在
              去块 / 亮度降噪 / 轻抬 上，0 = 完全不做预处理（等价原版）。
    """
    low = np.asarray(low, dtype=np.uint8)
    s = float(max(0.0, strength))
    gray_mean = float(cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).mean())
    noise = estimate_noise(low)
    blk = estimate_blockiness(low)
    k = float(np.clip(noise / 2.0, 0.35, 1.6))
    x = low
    did_deblock = False
    if blk > 1.03 and s > 0:
        x = deblock(x, strength=min(1.0, (blk - 1.0) * 6.0) * s)
        did_deblock = True
    # 只对亮度做轻度降噪：色度降噪留给"增强后、低信号平坦区"（在输入上动色度会造成偏色，
    # 因为 Zero-DCE 的曲线是逐通道独立的，输入色度一变，输出颜色就跟着偏）
    if s > 0:
        x = luma_denoise(x, h=1.5 + 0.5 * k, strength=min(1.0, s))
    did_prelift = False
    if gray_mean < 4.0 and s > 0:      # 只有"几乎全黑"才轻抬，避免正常暗图被抬亮后变淡
        x = prelift(x, amount=0.08 * s)
        did_prelift = True
    return dict(img=x, mean=gray_mean, noise=noise, blockiness=blk, k=k,
                deblock=did_deblock, prelift=did_prelift)


# ---------------------------------------------------------------------------
# E10 后处理（v2，保留细节版）：只在"低信号平坦区"做色度平滑，不动亮度、不整体去色
# ---------------------------------------------------------------------------
def flat_signal_mask(low, sigma, k_mean=2.0, k_std=0.7, feather=2.0):
    """低信号平坦区掩码：局部均值低于 k_mean*sigma 且局部标准差低于 k_std*sigma。

    这样"暗但有纹理"的区域（例如深色花纹布料）不会被误判成无信号区。
    """
    g = cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = cv2.blur(g, (8, 8))
    m2 = cv2.blur(g * g, (8, 8))
    s = np.sqrt(np.maximum(m2 - m * m, 0.0))
    mask = ((m < k_mean * max(sigma, 1e-3)) & (s < k_std * max(sigma, 1e-3))).astype(np.float32)
    return cv2.GaussianBlur(mask, (0, 0), feather)


def chroma_denoise_flat(img_rgb, low, sigma, strength=1.0, target_amp=None,
                        desat=0.0, radius=5, sigma_color=18):
    """只在"低信号平坦区"平滑色度（bilateral），并把整体色度幅度补回 target_amp。

    - 不动亮度 Y：避免"深色区变黑/变灰、细节被淹"
    - 平坦区才平滑：暗但有纹理的地方（布料花纹等）保持原样
    - target_amp：一般传"增强后未处理图"的色度幅度，防止颜色被洗淡
    """
    if strength <= 0 and desat <= 0:
        return img_rgb
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    m = flat_signal_mask(low, sigma) * float(np.clip(strength, 0.0, 1.5))
    if m.max() > 0:
        for ci in (1, 2):
            sm = cv2.bilateralFilter(ycc[..., ci].astype(np.uint8), radius,
                                     sigma_color, radius).astype(np.float32)
            ycc[..., ci] = ycc[..., ci] * (1.0 - m) + sm * m
    if desat > 0:
        for ci in (1, 2):
            ycc[..., ci] = ycc[..., ci] * (1.0 - desat * m) + 128.0 * (desat * m)
    if target_amp is not None:
        ycc = _rescale_chroma(ycc, target_amp, max_gain=1.15)
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def chroma_amp(img_rgb):
    """RGB(uint8) 的色度幅度均值（供外部在不 import cv2 时调用）。"""
    return _chroma_amp(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32))

def luma_denoise(img_rgb, h=2.0, template=7, search=15, strength=1.0):
    """只对亮度通道做非局部均值降噪（不动色度，避免偏色）。

    strength = 0~1 的"混合比例"：NLM 结果与原图按 strength 混合。
    注意：cv2.fastNlMeansDenoising 的 h 即使很小也会明显平滑（非线性悬崖），
    所以"强度控制"不能靠 h 缩放，必须靠这里的混合比例，否则 s=0.12 也会大幅吃掉纹理。
    """
    s = float(np.clip(strength, 0.0, 1.0))
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    if s <= 0:
        return img_rgb
    yd = cv2.fastNlMeansDenoising(ycc[..., 0].astype(np.uint8), None, h=float(h),
                                  templateWindowSize=template, searchWindowSize=search).astype(np.float32)
    out = ycc.copy()
    out[..., 0] = ycc[..., 0] * (1.0 - s) + yd * s
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

# ----------------------------------------------------------------------------
# ⑥ 阴影补偿：近黑图增强后暗部会略变暗（去噪去掉噪声正向偏置 + 曲线不稳定），
#    在 Y 通道暗部抬 1~3 级，把"阴影被加重"找回来。只作用于很暗的像素。
# ----------------------------------------------------------------------------
def shadow_lift(img_rgb, lift=2.0, lref=60.0):
    """在亮度通道对暗部抬 lift 个灰度级（越暗抬得越多，lref 以上不抬）。"""
    if lift <= 0:
        return img_rgb
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0]
    y = y + float(lift) * np.clip(1.0 - y / float(lref), 0.0, 1.0)
    ycc[..., 0] = np.clip(y, 0.0, 255.0)
    return cv2.cvtColor(ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)

def estimate_chroma_noise(img_rgb):
    """估计色度噪声强度（0~255 单位），与 scripts/compare_preprocess.py 的 chroma_noise 完全同口径：
    Cb/Cr 减高斯(sigma=1.5) 的高频残差，取两通道 std 的平均。
    经验划分（本库 17 张暗图实测，输入图上的口径）：
      < 0.8   → 色度本来较干净（512=0.10、132=0.17、101=0.38、147=0.40~0.47、12=0.55、493=0.74）
                → 再降噪只会把颗粒抹平，跳过改进，输出等同原版
      0.8~1.5 → 中等色噪（18=0.93、665=0.94）→ 轻度处理（s=0.10）
      > 1.5   → 明显色噪（778=1.13…）→ 正常处理（s=0.20）
    """
    ycc = cv2.cvtColor(np.asarray(img_rgb, dtype=np.float32), cv2.COLOR_RGB2YCrCb)
    vals = []
    for c in (1, 2):
        ch = ycc[..., c]
        vals.append(float(np.std(ch - cv2.GaussianBlur(ch, (0, 0), 1.5))))
    return float(np.mean(vals))


def auto_decision(img_rgb, mean=None, chroma_noise=None):
    """A 档：按图像内容决定"跳过改进 / 轻度 / 正常 / 近黑强处理"。

    返回 dict(mode, s_eff, mean, chroma_noise)。
      skip       : 色度噪声 < 1.5 且不是近黑 → 亮度侧不处理（保持原版纹理）；
                   色度侧仍会按 chroma_light 分段强度做轻度平滑（σc<0.3 时才完全不动）
      normal     : 色度噪声 ≥ 1.5 → 亮度侧轻度处理（s=0.20）
      near_black : 均值<5 的近黑图 → s=1.0（走中性抬亮）
    """
    arr = np.asarray(img_rgb)
    if mean is None:
        mean = float(np.mean(arr))
    if chroma_noise is None:
        chroma_noise = estimate_chroma_noise(arr)
    if mean < 5.0:
        mode, s_eff = "near_black", 1.0      # 近黑：预处理 + 中性抬亮
    elif chroma_noise < 1.5:
        mode, s_eff = "skip", 0.0            # 跳过"亮度侧"处理（色度侧仍按 chroma_light 轻平滑）
    else:
        mode, s_eff = "normal", 0.20         # 常规暗图：轻度亮度处理
    return dict(mode=mode, s_eff=s_eff, mean=float(mean), chroma_noise=float(chroma_noise))


def shadow_lift_float(img01, lift=4.0, lref=60.0):
    """暗部抬亮（浮点版，0~1 输入），给 --bit16 用，避免先压回 8bit 丢精度。"""
    x = np.asarray(img01, dtype=np.float32)
    ycc = cv2.cvtColor(x, cv2.COLOR_RGB2YCrCb)
    Y = ycc[..., 0] * 255.0
    w = np.clip((lref - Y) / max(lref, 1e-6), 0.0, 1.0)
    ycc[..., 0] = np.clip(Y + float(lift) * w, 0.0, 255.0) / 255.0
    return np.clip(cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB), 0.0, 1.0)

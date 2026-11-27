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
def denoise_chroma(img_rgb, strength=1.0):
    """YCrCb 空间中：Y（亮度）轻降噪，Cr/Cb（色度）重降噪。

    strength: 0=不降；1=推荐（色噪明显）；2=很强（极端暗图，可能糊掉彩色细节）
    返回 uint8 RGB。
    """
    if strength <= 0:
        return img_rgb
    s = float(strength)
    ycc = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycc)
    h_y = 2.0 + 2.0 * s          # 亮度：轻（h=4 在 s=1）
    h_c = 4.0 + 8.0 * s          # 色度：重（h=12 在 s=1）
    y = cv2.fastNlMeansDenoising(y, None, h=h_y, templateWindowSize=7, searchWindowSize=15)
    cr = cv2.fastNlMeansDenoising(cr, None, h=h_c, templateWindowSize=7, searchWindowSize=15)
    cb = cv2.fastNlMeansDenoising(cb, None, h=h_c, templateWindowSize=7, searchWindowSize=15)
    return cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)


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
# 组合管线
# ----------------------------------------------------------------------------
def preprocess(img_rgb, denoise=0.0, deblock_strength=0.0, prelift_amount=0.0):
    """增强前的预处理（顺序：先去块，再降噪，最后轻抬）。

    顺序理由：去块会引入轻微平滑，先做可避免把降噪结果重新弄脏；
    轻抬放最后，让降噪在“原始亮度分布”上工作更准。
    """
    out = img_rgb
    if deblock_strength > 0:
        out = deblock(out, deblock_strength)
    if denoise > 0:
        out = denoise_chroma(out, denoise)
    if prelift_amount > 0:
        out = prelift(out, prelift_amount)
    return out


# ----------------------------------------------------------------------------
# 16-bit 输出
# ----------------------------------------------------------------------------
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

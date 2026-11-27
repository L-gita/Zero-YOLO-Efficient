import io
import ast

P = r"C:\Users\12983\Documents\repro\scripts\preprocess.py"
t = io.open(P, encoding="utf-8").read()

add = '''

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


def auto_preprocess(low):
    """E10 场景自适应预处理（返回 dict：预处理图 + 决策信息）"""
    low = np.asarray(low, dtype=np.uint8)
    gray_mean = float(cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).mean())
    noise = estimate_noise(low)
    blk = estimate_blockiness(low)
    k = float(np.clip(noise / 2.0, 0.35, 1.6))
    x = low
    did_deblock = False
    if blk > 1.03:
        x = deblock(x, strength=min(1.0, (blk - 1.0) * 6.0))
        did_deblock = True
    x = chroma_denoise_hf(x, strength=1.0 * k, lf_sigma=8.0)
    did_prelift = False
    if gray_mean < 8.0:
        x = prelift(x, amount=0.10)
        did_prelift = True
    return dict(img=x, mean=gray_mean, noise=noise, blockiness=blk, k=k,
                deblock=did_deblock, prelift=did_prelift)
'''

t = t.rstrip() + add
io.open(P, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)

# ---------------------------------------------------------------- enhance.py
E = r"C:\Users\12983\Documents\repro\enhance.py"
et = io.open(E, encoding="utf-8").read()

# 1) 预设：新增 auto（E10），mosaic 保留为旧 E9
et = et.replace('"mosaic": dict(denoise=0.5, deblock=0.5, prelift=0.10, guided=0.5, limit_dark=0.80, dark_gamma=0.55, dark_smooth=8, dither=0.6),',
                '"auto": dict(denoise=-1.0, deblock=-1.0, prelift=-1.0, guided=0.0, limit_dark=-1.0, dark_gamma=0.60, dark_smooth=8, dither=0.4),\n    "mosaic": dict(denoise=0.5, deblock=0.5, prelift=0.10, guided=0.5, limit_dark=0.80, dark_gamma=0.55, dark_smooth=8, dither=0.6),')

old_pipeline = '''    # —— 预处理：去块 → 降噪 → 轻抬 ——
    if pp is not None and (args.denoise > 0 or args.deblock > 0 or args.prelift > 0):
        u8 = pp.preprocess(u8, args.denoise, args.deblock, args.prelift)'''
new_pipeline = '''    # —— 预处理：场景自适应（E10）或手动参数 ——
    auto_mode = (args.denoise < 0)
    if pp is not None and auto_mode:
        info = pp.auto_preprocess(u8)
        u8 = info["img"]
        print(f"    [auto] 均值 {info['mean']:5.1f} 噪声σ {info['noise']:4.2f} 块效应 {info['blockiness']:.3f} "
              f"→ 降噪×{info['k']:.2f} 去块={'开' if info['deblock'] else '关'} 轻抬={'开' if info['prelift'] else '关'}")
    elif pp is not None and (args.denoise > 0 or args.deblock > 0 or args.prelift > 0):
        u8 = pp.preprocess(u8, args.denoise, args.deblock, args.prelift)'''
assert old_pipeline in et
et = et.replace(old_pipeline, new_pipeline)

old_post = '''    # —— 后处理：亮度引导色度平滑 → 近黑区限增益 → 可选保边平滑 ——
    if pp is not None and args.guided > 0:
        u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3, strength=args.guided)
    if pp is not None and args.limit_dark > 0:
        u8 = pp.limit_dark_gain(u8, np.asarray(im), l0=3.0, l1=12.0, gamma=args.dark_gamma,
                                strength=args.limit_dark, smooth_radius=args.dark_smooth, dither=args.dither)'''
new_post = '''    # —— 后处理：E10 用「只动亮度的近黑限增益 + 掩码色度降噪」，旧档位保留原逻辑 ——
    if pp is not None and auto_mode:
        info = pp.auto_preprocess(np.asarray(im))
        if info["mean"] < 20.0:
            u8 = pp.limit_dark_luma(u8, np.asarray(im), thr=2.5, feather=3.0,
                                    gamma=args.dark_gamma, strength=0.9, chroma_k=0.85)
            u8 = pp.masked_chroma_denoise(u8, np.asarray(im), thr=6.0, strength=2.2)
    else:
        if pp is not None and args.guided > 0:
            u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3, strength=args.guided)
        if pp is not None and args.limit_dark > 0:
            u8 = pp.limit_dark_gain(u8, np.asarray(im), l0=3.0, l1=12.0, gamma=args.dark_gamma,
                                    strength=args.limit_dark, smooth_radius=args.dark_smooth, dither=args.dither)'''
assert old_post in et
et = et.replace(old_post, new_post)

et = et.replace('"--preprocess", choices=sorted(PRESETS), default=None',
                '"--preprocess", choices=sorted(PRESETS), default=None')
io.open(E, "w", encoding="utf-8", newline="\n").write(et)
ast.parse(et)
print("preprocess.py += E10 函数；enhance.py += auto 预设")

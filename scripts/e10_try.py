"""E10 试验：场景自适应 + 色度残差降噪 + 只动亮度的近黑限增益（不改 enhance.py，只做对比）"""
import os
import sys
import numpy as np
import cv2
import torch

ROOT = r"C:\Users\12983\Documents\repro"
sys.path.insert(0, os.path.join(ROOT, r"Zero-DCE\Zero-DCE_code"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import model  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CKPT = os.path.join(ROOT, r"Zero-DCE\Zero-DCE_code\snapshots_sice\Epoch50.pth")
FONT = r"C:\Windows\Fonts\msyh.ttc"
OUT = os.path.join(ROOT, r"demo_mosaic\e10_check")
os.makedirs(OUT, exist_ok=True)

net = model.enhance_net_nopool().cuda()
net.load_state_dict(torch.load(CKPT, map_location="cuda", weights_only=True))
net.eval()


# ---------------------------------------------------------------- 基础算子
def _amp(ycc):
    return float(np.sqrt((ycc[..., 1] - 128.) ** 2 + (ycc[..., 2] - 128.) ** 2).mean())


def _rescale(ycc, target, max_gain=1.8):
    cur = _amp(ycc)
    if cur < 1e-3 or target < 1e-3:
        return ycc
    k = float(np.clip(target / cur, 1.0, max_gain))
    out = ycc.copy()
    for ci in (1, 2):
        out[..., ci] = 128. + (ycc[..., ci] - 128.) * k
    return out


def noise_estimate(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    m = g < 40
    if m.mean() < 0.05:
        m = g < 80
    v = lap[m] if m.any() else lap.ravel()
    return float(np.median(np.abs(v)) / 0.6745)


def blockiness(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    d = np.abs(np.diff(g, axis=1))
    W = g.shape[1]
    bcols = [c for c in range(7, W - 1, 8) if 0 < c - 1 < d.shape[1]]
    if not bcols:
        return 1.0
    on = np.mean([d[:, c - 1].mean() for c in bcols])
    off = np.mean([d[:, c - 1].mean() for c in range(1, d.shape[1] - 1) if (c % 8) != 0])
    return float(on / (off + 1e-6))


def chroma_denoise_hf(rgb, strength=1.0, lf_sigma=8.0):
    """只在色度的高频残差上去噪，低频颜色（天空渐变）原样保留"""
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    amp0 = _amp(ycc)
    out = ycc.copy()
    for ci in (1, 2):
        c = ycc[..., ci]
        low = cv2.GaussianBlur(c, (0, 0), lf_sigma)
        res = c - low
        res_d = cv2.bilateralFilter(res, d=5, sigmaColor=6.0 * strength, sigmaSpace=3)
        out[..., ci] = low + res_d
    out = _rescale(out, amp0)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def deblock_light(rgb, strength=0.6):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    out = rgb.astype(np.float32).copy()
    for axis in (0, 1):
        for k in range(8, out.shape[axis], 8):
            if axis == 1:
                a, b = out[:, k - 1], out[:, k]
                diff = np.abs(b - a).max(axis=1)
                mask = diff < 10.0
                if not mask.any():
                    continue
                sm = (out[:, k - 3] + 2 * a + 2 * b + out[:, k + 2]) / 6.0
                out[mask, k - 1] = a[mask] * (1 - strength) + sm[mask] * strength
                out[mask, k] = b[mask] * (1 - strength) + sm[mask] * strength
    return np.clip(out, 0, 255).astype(np.uint8)


def prelift_light(rgb, amount=0.10):
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    y = ycc[..., 0] / 255.0
    gamma = 1.0 / (1.0 + 1.5 * amount)
    w = np.clip((0.6 - y) / 0.6, 0, 1) ** 1.5
    y2 = np.power(np.clip(y, 1e-6, 1.0), gamma)
    ycc[..., 0] = (y * (1 - w) + y2 * w) * 255.0
    return cv2.cvtColor(np.clip(ycc, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def limit_dark_v3(enh, low, thr=2.5, feather=3.0, gamma=0.60, strength=0.9, chroma_k=0.7):
    """只动亮度：近黑区换成中性 gamma 抬亮；色度按同一掩码轻微拉向中性（羽化很宽）"""
    e = cv2.cvtColor(enh, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    ly = cv2.cvtColor(low, cv2.COLOR_RGB2YCrCb)[..., 0].astype(np.float32)
    ly_s = cv2.GaussianBlur(ly, (0, 0), 2.0)
    neutral = 255.0 * np.power(np.clip(ly_s, 0, 255) / 255.0, gamma)
    m = np.clip((thr + feather - ly_s) / feather, 0, 1) * strength
    m = cv2.GaussianBlur(m, (0, 0), 1.2)
    e[..., 0] = e[..., 0] * (1 - m) + neutral * m
    m2 = m * chroma_k
    for ci in (1, 2):
        e[..., ci] = e[..., ci] * (1 - m2) + 128.0 * m2
    return cv2.cvtColor(np.clip(e, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)


def quantize_dither(rgb, dither=0.6, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rgb.astype(np.float32) + rng.uniform(-dither, dither, rgb.shape), 0, 255).astype(np.uint8)


def enhance(rgb):
    t = torch.from_numpy(rgb.astype(np.float32) / 255.).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, o, _ = net(t)
    return np.clip(o[0].permute(1, 2, 0).cpu().numpy() * 255., 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- 指标
def chroma_noise(rgb):
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    return float(np.mean([np.std(ycc[..., ci] - cv2.GaussianBlur(ycc[..., ci], (0, 0), 1.5)) for ci in (1, 2)]))


def luma_std8(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    k = 8
    H, W = g.shape
    h, w = H // k * k, W // k * k
    b = g[:h, :w].reshape(h // k, k, w // k, k).transpose(0, 2, 1, 3).reshape(-1, k * k)
    return float(b.std(axis=1).mean())


def band_jump(rgb):
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    m = g < 40
    if m.mean() < 0.02:
        m = g < 80
    vals = g[m]
    if vals.size < 10:
        return 0.0
    h = np.bincount(vals, minlength=256).astype(np.float64) / vals.size
    big = [i for i, p in enumerate(h) if p > 0.01]
    return float(max(np.diff(big))) if len(big) > 1 else 0.0


def smooth_chroma_step(rgb):
    """平滑区域（亮度梯度小）里，低频色度的台阶强度：越大 = 天空/墙面越容易出现色块"""
    ycc = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2YCrCb)
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gy = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)), (0, 0), 1.5)
    smooth = gy < np.percentile(gy, 60)
    vals = []
    for ci in (1, 2):
        low = cv2.GaussianBlur(ycc[..., ci], (0, 0), 4.0)
        gx = np.abs(cv2.Sobel(low, cv2.CV_32F, 1, 0, ksize=3))
        gyy = np.abs(cv2.Sobel(low, cv2.CV_32F, 0, 1, ksize=3))
        vals.append(np.maximum(gx, gyy)[smooth])
    v = np.concatenate(vals)
    return float(np.mean(v > 1.5))


# ---------------------------------------------------------------- 管线
def pipe_e1(low):
    return enhance(low)


def pipe_e9(low):
    """旧 E9（固定全局管线）"""
    from preprocess import preprocess, chroma_guided_smooth, limit_dark_gain, quantize8
    pre = preprocess(low, denoise=0.5, deblock_strength=0.5, prelift_amount=0.10)
    en = enhance(pre)
    g = chroma_guided_smooth(en, luma_guide=en, radius=2, eps=1e-3, strength=0.5)
    L = limit_dark_gain(g, np.asarray(pre), l0=3.0, l1=12.0, gamma=0.55, strength=0.80, smooth_radius=8, dither=0.6)
    return quantize8(L, dither=0.6)


def pipe_e10(low):
    """E10：场景自适应 + 色度残差降噪 + 只动亮度的近黑限增益"""
    mean = float(cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).mean())
    n = noise_estimate(low)
    k = float(np.clip(n / 2.0, 0.35, 1.6))
    x = low
    blk = blockiness(low)
    if blk > 1.03:
        x = deblock_light(x, strength=0.6)
    x = chroma_denoise_hf(x, strength=1.0 * k, lf_sigma=8.0)
    if mean < 8.0:                      # 只有极暗图才做暗部轻抬
        x = prelift_light(x, amount=0.10)
    en = enhance(x)
    out = en
    if mean < 20.0:                     # 只有暗图才需要近黑限增益
        out = limit_dark_v3(out, low, thr=2.5, feather=3.0, gamma=0.60, strength=0.9, chroma_k=0.7)
    return quantize_dither(out, dither=0.4)


CASES = [
    ("极暗 126_3_", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\126_3_.png")),
    ("极暗 147_0_", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\147_0_.png")),
    ("正常夜景 748", os.path.join(ROOT, r"lol_test\eval15\low\748.png")),
    ("威尼斯 1", os.path.join(ROOT, r"低照图库\02_LIME\1.bmp")),
    ("有天空 LIME/2", os.path.join(ROOT, r"低照图库\02_LIME\2.bmp")),
    ("黄昏天空 DICM/31", os.path.join(ROOT, r"低照图库\03_DICM\31.jpg")),
]

rows = []
for name, p in CASES:
    low = np.asarray(Image.open(p).convert("RGB"))
    outs = {"E1": pipe_e1(low), "E9": pipe_e9(low), "E10": pipe_e10(low)}
    for tag, o in outs.items():
        rows.append((name, tag, chroma_noise(o), luma_std8(o), band_jump(o), smooth_chroma_step(o), float(o.mean())))
        Image.fromarray(o).save(os.path.join(OUT, f"{os.path.splitext(os.path.basename(p))[0]}_{tag}.png"))

print(f"{'图':16s} {'方案':4s} {'色噪':>6s} {'亮度细节std8':>10s} {'暗区条带':>7s} {'平滑区色块':>8s} {'亮度':>6s}")
for r in rows:
    print(f"{r[0]:16s} {r[1]:4s} {r[2]:6.2f} {r[3]:10.2f} {r[4]:7.1f} {r[5]:8.3f} {r[6]:6.1f}")

# 拼接对比图（原图 / E1 / E9 / E10）
for name, p in CASES:
    stem = os.path.splitext(os.path.basename(p))[0]
    panels = []
    for tag, fn in (("低照输入", p), ("E1 基线", os.path.join(OUT, f"{stem}_E1.png")),
                    ("E9 旧", os.path.join(OUT, f"{stem}_E9.png")), ("E10 新", os.path.join(OUT, f"{stem}_E10.png"))):
        im = np.asarray(Image.open(fn).convert("RGB"))
        h, w = im.shape[:2]
        tw = 420
        r = Image.fromarray(im).resize((tw, max(60, int(tw * h / w))), Image.LANCZOS)
        c = Image.new("RGB", (r.width, r.height + 34), (26, 26, 26))
        c.paste(r, (0, 34))
        ImageDraw.Draw(c).text((6, 6), tag, font=ImageFont.truetype(FONT, 20), fill=(240, 240, 240))
        panels.append(np.asarray(c))
    H = max(a.shape[0] for a in panels)
    W = sum(a.shape[1] for a in panels)
    canvas = np.full((H, W, 3), 26, np.uint8)
    x = 0
    for a in panels:
        canvas[:a.shape[0], x:x + a.shape[1]] = a
        x += a.shape[1]
    Image.fromarray(canvas).save(os.path.join(OUT, f"cmp_{stem}.png"))
print("[ok]", OUT)

r"""用 Zero-DCE 增强自己的照片（可选：马赛克预处理 + 16-bit 输出）。

用法（Windows，命令行）：
    & C:\Users\12983\Documents\repro\.venv\Scripts\python.exe C:\Users\12983\Documents\repro\enhance.py <照片或文件夹> [更多...]

常用选项：
    -o / --out DIR        输出目录（默认：第一张照片旁边的 "enhanced" 文件夹）
    -m / --model NAME     official | sice | edge | lol | 直接给 .pth 路径
                          （默认 sice=论文配方复现；edge=额外加了论文 (4.12) Sobel 边缘损失）
    -c / --compare        额外生成"原图 | 增强"并排对比图
    -j / --jpg            输出 JPEG（默认：输入是 jpg 就存 jpg，其它存 png）
    -s / --scale N        先把长边缩到 N 再处理（默认 0=不缩放）
    --tile                强制分块处理（超大清不了显存时用）

马赛克预处理（默认全关，不写就和以前完全一样）：
    --denoise FLOAT       色度优先降噪强度（0=关；1=推荐；2=很强）
    --deblock FLOAT       JPEG 8x8 去块强度（0=关；1=推荐）
    --prelift FLOAT       增强前暗部轻抬（0=关；0.15=推荐）
    --post FLOAT          增强后保边平滑（0=关；1=推荐）
    --bit16               输出 16-bit PNG（避免再被压回 8-bit）
    --preprocess preset   一键预设：auto（推荐，E10 场景自适应）/ mosaic（旧版 E9）/ light / dark

例：
    python enhance.py D:\照片\夜景.jpg
    python enhance.py D:\照片\全家福 -o D:\增强结果 -c
    python enhance.py "D:\照片\夜景.jpg" -m official
    python enhance.py "D:\照片\夜景.jpg" --preprocess auto -c        # 治马赛克（推荐）
"""
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
import cv2
import torch
from PIL import Image, ImageOps

if sys.platform == "win32":          # 让控制台能正常显示中文
    try:
        os.system("chcp 65001 >nul")
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:                # noqa: BLE001
        pass

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
warnings.filterwarnings("ignore")

import model  # noqa: E402
import preprocess as pp
import preprocess_v4 as ppv4  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_FALLBACK_USED = {}


def _w(rel, backup):
    """权重路径：优先用 Zero-DCE_code 下的训练目录；若不存在（例如换机器只 clone 了仓库），
    回退到仓库里随代码一起提交的 weights/ 目录，保证开箱可用。
    注意：发生回退时会记入 _FALLBACK_USED，main() 启动时会打印 [warn]，避免"以为用了 A 权重其实用了 B"。"""
    p = os.path.join(CODE, rel)
    if os.path.exists(p):
        return p
    q = os.path.join(ROOT, "weights", backup)
    if os.path.exists(q):
        _FALLBACK_USED[rel] = backup
        return q
    return p


def warn_fallbacks():
    for rel, backup in sorted(_FALLBACK_USED.items()):
        print(f"[warn] 训练目录权重缺失：{rel} → 已回退到 weights/{backup}")
        print(f"       （要使用训练目录里的权重，请确认 {os.path.join(CODE, rel)} 存在）")


MODELS = {
    "official": _w(os.path.join("snapshots", "Epoch99.pth"), "official_Epoch99.pth"),
    "sice": _w(os.path.join("snapshots_sice", "Epoch50.pth"), "sice_Epoch50.pth"),
    "edge": _w(os.path.join("snapshots_sice_edge", "Epoch49.pth"), "edge_Epoch49.pth"),
    "lol": _w(os.path.join("snapshots_run", "Epoch199.pth"), "sice_Epoch49.pth"),
}
# 预处理预设：--preprocess 一键选
PRESETS = {
    # auto = E10 场景自适应：先估噪声/块效应，再决定要不要降噪/去块/轻抬；
    # 后处理只在暗图上做"只动亮度"的近黑限增益 + 近黑区掩码色度降噪
    "auto": dict(denoise=-1.0, deblock=-1.0, prelift=-1.0, guided=0.0, limit_dark=-1.0,
                 dark_gamma=0.60, dark_smooth=8),
    # mosaic（推荐）：轻度预处理 + 亮度引导色度平滑 + 近黑区限增益（消除彩色大斑块）
    "mosaic": dict(denoise=0.5, deblock=0.5, prelift=0.10, guided=0.5, limit_dark=0.80, dark_gamma=0.55, dark_smooth=8),
    # v4：保守预处理 + A+B 色度去噪（色度噪声判据 + 引导羽化），只动色度不动亮度结构
    "v4": dict(denoise=-1.0, deblock=-1.0, prelift=-1.0, guided=0.0, limit_dark=-1.0,
               dark_gamma=0.55, dark_smooth=8),
    # light（温和）：一般夜景，尽量少动细节
    "light": dict(denoise=0.4, deblock=0.0, prelift=0.06, guided=0.3, limit_dark=0.0, dark_gamma=0.70, dark_smooth=8),
    # dark（极暗）：更猛一点
    "dark": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.50, dark_smooth=10),
    # 兼容旧名字
    "strong": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.50, dark_smooth=10),
}
_net_cache = {}


def get_net(path):
    if path not in _net_cache:
        if not os.path.exists(path):
            raise SystemExit(f"找不到权重文件: {path}")
        net = model.enhance_net_nopool()
        net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        net.eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _net_cache[path] = net
        print(f"[模型] {path}")
    return _net_cache[path]


def _run_float(net, arr, half=False):
    """arr: HxWx3 float32 [0,1] -> 增强结果 float32 [0,1]"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        if half and device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                _, out, _ = net(t)
        else:
            _, out, _ = net(t)
    return out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()


def _run(net, arr, half=False):
    """arr: HxWx3 float32 [0,1] -> enhanced np.uint8"""
    out = _run_float(net, arr, half=half)
    return (out * 255.0 + 0.5).astype(np.uint8)


def enhance_image(net, pil, force_tile=False, return_float=False):
    """整图处理；显存不够时自动降级 fp16 → 分块。返回 (结果, 模式说明)"""
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    if force_tile:
        return _tiled(net, pil), "tile(force)"
    # 超大图（>6MP）直接分块：整图 fp32 会把中间特征图撑到 GB 级（显存 OOM 后本机内存也会被拖垮）
    W0, H0 = pil.size
    if W0 * H0 > 6_000_000:
        out = _tiled(net, pil)
        return (out if return_float else out), "tile(auto,big)"
    try:
        out = _run_float(net, arr)
        mode = "full/fp32"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            out = _run_float(net, arr, half=True)
            mode = "full/fp16"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return _tiled(net, pil), "tile"
    if return_float:
        return out, mode
    return Image.fromarray((out * 255.0 + 0.5).astype(np.uint8)), mode


def _tiled(net, pil, tile=1024, overlap=128):
    """分块（带重叠，只取每块内侧）处理超大图，避免显存爆掉。"""
    W, H = pil.size
    out = Image.new("RGB", (W, H))
    step = tile - overlap
    m = overlap // 2
    for y in range(0, H, step):
        for x in range(0, W, step):
            x2, y2 = min(x + tile, W), min(y + tile, H)
            crop = pil.crop((x, y, x2, y2))
            res = Image.fromarray(_run(net, np.asarray(crop, dtype=np.float32) / 255.0, half=True))
            ix1 = x + (0 if x == 0 else m)
            iy1 = y + (0 if y == 0 else m)
            ix2 = x2 - (0 if x2 == W else m)
            iy2 = y2 - (0 if y2 == H else m)
            out.paste(res.crop((ix1 - x, iy1 - y, ix2 - x, iy2 - y)), (ix1, iy1))
    return out


def collect_inputs(paths):
    files = []
    for p in paths:
        p = os.path.abspath(p.strip('"'))
        if os.path.isdir(p):
            for ext in EXTS:
                files += glob.glob(os.path.join(p, "**", "*" + ext), recursive=True)
                files += glob.glob(os.path.join(p, "**", "*" + ext.upper()), recursive=True)
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"[跳过] 找不到: {p}")
    return sorted(dict.fromkeys(files))


def main():
    ap = argparse.ArgumentParser(description="用 Zero-DCE 增强自己的照片（可选马赛克预处理）")
    ap.add_argument("inputs", nargs="+", help="照片文件或文件夹（可多个）")
    ap.add_argument("-o", "--out", default=None, help="输出目录")
    ap.add_argument("-m", "--model", default="sice", help="official|sice|edge|lol|.pth 路径")
    ap.add_argument("-c", "--compare", action="store_true", help="生成并排对比图")
    ap.add_argument("-j", "--jpg", action="store_true", help="输出 jpg")
    ap.add_argument("-s", "--scale", type=int, default=0, help="长边缩放到 N 像素（0=不缩放）")
    ap.add_argument("--tile", action="store_true", help="强制分块处理")
    # 马赛克预处理
    ap.add_argument("--preprocess", default=None, help="预设：auto（推荐，场景自适应）| light | dark | strong（旧预设 mosaic 仍可用）")
    ap.add_argument("--denoise", type=float, default=0.0, help="色度优先降噪强度（0=关, 1=推荐）")
    ap.add_argument("--deblock", type=float, default=0.0, help="JPEG 去块强度（0=关, 1=推荐）")
    ap.add_argument("--prelift", type=float, default=0.0, help="增强前暗部轻抬（0=关, 0.15=推荐）")
    ap.add_argument("--post", type=float, default=0.0, help="增强后双边平滑（旧参数，一般不用）")
    ap.add_argument("--guided", type=float, default=0.0, help="增强后亮度引导的色度平滑（推荐 0.5~0.8）")
    ap.add_argument("--limit-dark", type=float, default=0.0,
                    help="近黑区限增益：0=关，0.85=推荐（用中性抬亮替代彩色放大，消除大色块）")
    ap.add_argument("--dark-gamma", type=float, default=0.55,
                    help="近黑区中性抬亮的 gamma（越大越暗；0.55=默认，1.0=几乎不提亮）")
    ap.add_argument("--dark-smooth", type=int, default=8,
                    help="近黑区抬亮前对亮度做大模糊的半径（消除 8bit 台阶造成的硬灰块）")
    ap.add_argument("--auto-strength", type=float, default=-1.0,
                    help="auto 模式强度；-1=自动（A 档：非近黑图不做亮度预处理、仅按色度噪声轻平滑；近黑图强处理），1.0=强制全强度，0=完全等价原版，也可给 0~2")
    ap.add_argument("--chroma-mode", choices=["light", "smooth", "desat", "none"], default="light",
                    help="auto 模式下如何处理色度：light=只做轻度色度平滑(不动亮度/纹理，推荐想少噪点又想保颗粒时用)；smooth=局部平滑(更干净但可能出色块边界)；desat=全局等比去饱和；none=完全不碰色度")
    ap.add_argument("--v4-strength", type=float, default=0.9, help="v4 色度去噪强度（默认 0.9）")
    ap.add_argument("--v4-tau", type=float, default=5.0, help="v4 色度噪声阈值（越大越保守）")
    ap.add_argument("--v4-beta", type=float, default=1.6, help="v4 亮度结构保护系数（越大越保护真实彩边）")
    ap.add_argument("--v4-guide", type=int, default=16, help="v4 权重羽化半径（像素）")
    ap.add_argument("--shadow-lift", type=float, default=-1.0,
                    help="阴影补偿（Y 通道暗部抬几级）；-1=自动（启用预处理且输入均值<15 时抬 4 级；-m sice 原版不受影响），0=关闭")
    ap.add_argument("--bit16", action="store_true", help="输出 16-bit PNG")
    args = ap.parse_args()
    warn_fallbacks()
    # 新抖动模式下不再重复做旧抖动（避免双重抖动/混淆）

    # 预设 → 具体参数（显式给出的参数优先）
    if args.preprocess:
        key = args.preprocess.lower()
        if key not in PRESETS:
            raise SystemExit(f"未知预设 {key}，可选：{', '.join(PRESETS)}")
        p = PRESETS[key]
        if args.denoise == 0.0:
            args.denoise = p["denoise"]
        if args.deblock == 0.0:
            args.deblock = p["deblock"]
        if args.prelift == 0.0:
            args.prelift = p["prelift"]
        if args.post == 0.0:
            args.post = 0.0
        if args.guided == 0.0:
            args.guided = p["guided"]
        if args.limit_dark == 0.0:
            args.limit_dark = p["limit_dark"]

    _pp = (args.preprocess or "").lower()
    if _pp and _pp not in PRESETS:
        raise SystemExit(f"未知的 --preprocess 预设：{_pp}（可选：{', '.join(PRESETS)}）")
    v4_mode = _pp == "v4"
    auto_mode = _pp in ("auto", "v4") or args.denoise < 0
    use_pre = (args.denoise > 0 or args.deblock > 0 or args.prelift > 0)
    use_post = (args.post > 0 or args.guided > 0 or args.limit_dark > 0)
    _mode = None
    if auto_mode:
        print("[预处理] auto 场景自适应（E10）" + ("+v4（A+B 色度去噪）" if v4_mode else "") +
              "：先估噪声/块效应，再按需降噪/去块/轻抬；暗图才做近黑限增益")
    elif use_pre or use_post:
        print(f"[预处理] 去块={args.deblock:g}  色度降噪={args.denoise:g}  "
              f"暗部轻抬={args.prelift:g}  | 后处理: 引导平滑={args.guided:g}  "
              f"近黑限增益={args.limit_dark:g}  16bit={args.bit16}")

    ckpt = MODELS.get(args.model.lower(), args.model)
    net = get_net(ckpt)

    files = collect_inputs(args.inputs)
    if not files:
        raise SystemExit("没有找到任何图片（支持 jpg/jpeg/png/bmp/tif/webp）")

    first_dir = os.path.dirname(files[0]) if os.path.isfile(files[0]) else files[0]
    outdir = args.out or os.path.join(first_dir, "enhanced")
    os.makedirs(outdir, exist_ok=True)
    print(f"[输入] {len(files)} 张\n[输出] {outdir}\n")

    done = 0
    t_all = time.time()
    for i, f in enumerate(files, 1):
        try:
            im = ImageOps.exif_transpose(Image.open(f))
            im = im.convert("RGB")
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] 读取失败 {os.path.basename(f)}: {e}")
            continue
        if args.scale and max(im.size) > args.scale:
            r = args.scale / max(im.size)
            im = im.resize((round(im.width * r), round(im.height * r)), Image.LANCZOS)

        t0 = time.time()
        try:
            # ① 增强前的预处理：auto=场景自适应（E10），否则按手动/旧预设参数
            pre = im
            info = None
            _skipped = False
            if auto_mode:
                # A 档：按图像均值/色度噪声自动决定 常规（仅色度轻平滑）/ 近黑（强处理）（见 scripts/preprocess.py: auto_decision）
                if args.auto_strength < 0:
                    # A 档：按图像内容自动决定强度（跳过 / 轻度 / 常规 / 近黑）
                    _dec = pp.auto_decision(np.asarray(im))
                    s_eff = _dec["s_eff"]
                    print(f"    [判定] 均值 {_dec['mean']:5.1f}  色度噪声 {_dec['chroma_noise']:4.2f} "
                          f"→ {_dec['mode']}（强度 {s_eff:.2f}）")
                    _mode = _dec["mode"]
                else:
                    s_eff = args.auto_strength
                    _dec = None
                    _mode = None
                if s_eff <= 0.0:
                    # 常规暗图：不做输入侧预处理（色度噪声 <0.3 时连色度也不动，输出与 "-m sice" 逐像素一致）
                    pre = im
                    _skipped = True
                    print("    [常规] 不做亮度预处理（色度侧稍后按噪声强度轻平滑）")
                else:
                    info = pp.auto_preprocess(np.asarray(im), strength=s_eff)
                    pre = Image.fromarray(info["img"])
                    print(f"    [auto] 均值 {info['mean']:5.1f}  噪声σ {info['noise']:4.2f}  "
                          f"块效应 {info['blockiness']:.3f} → 降噪×{info['k']:.2f}  "
                          f"去块={'开' if info['deblock'] else '关'}  轻抬={'开' if info['prelift'] else '关'}")
            elif use_pre:
                pre = Image.fromarray(pp.preprocess(
                    np.asarray(im), denoise=args.denoise,
                    deblock_strength=args.deblock, prelift_amount=args.prelift))
            # ② Zero-DCE 增强（bit16 时保留 float 结果，避免再被压回 8-bit）
            _need_float = args.bit16
            raw, mode = enhance_image(net, pre, args.tile, return_float=_need_float)
            # ③ 增强后的处理：auto=近黑限增益（只动亮度）+ 近黑区掩码色度降噪；否则用旧逻辑
            if auto_mode and s_eff <= 0.0:
                # 常规暗图：亮度侧不处理；若 --chroma-mode light，则只对色度做轻度平滑（不动亮度/纹理）
                u8 = (np.clip(raw, 0, 1) * 255.0 + 0.5).astype(np.uint8) if (isinstance(raw, np.ndarray) and raw.dtype != np.uint8) else np.asarray(raw).astype(np.uint8)
                if args.chroma_mode == "light":
                    _cn = pp.estimate_chroma_noise(np.asarray(im))
                    # 色度本来就很干净（<0.3）就不动，避免"越降越差"
                    _s = float(pp.chroma_light_strength(_cn))
                    _noise = float(pp.estimate_noise(u8))
                    u8 = pp.chroma_denoise_flat(u8, np.asarray(im), _noise, strength=_s, target_amp=None)
                    u8 = pp.quantize8(np.asarray(u8, dtype=np.float32))
                    print(f"    [light] 仅色度轻度平滑：色度噪声 {_cn:.2f} → 强度 {_s:.2f}（亮度/纹理不动）")
                out_f = u8.astype(np.float32) / 255.0
                out = out_f if args.bit16 else Image.fromarray(u8)
            elif auto_mode:
                u8 = (np.clip(raw, 0, 1) * 255.0 + 0.5).astype(np.uint8) if (isinstance(raw, np.ndarray) and raw.dtype != np.uint8) else np.asarray(raw).astype(np.uint8)
                # E10 后处理 v2（分两档）：
                #   mean<5 的"几乎全黑"图：走中性抬亮（E9 路线，抑制彩色大斑）
                #   其余暗图：只在"低信号平坦区"平滑色度，不动亮度（保留深色区细节与颜色）
                if info["mean"] < 5.0:
                    if args.chroma_mode != "none" or v4_mode:
                        print("    [提示] 近黑图走中性抬亮分支，--chroma-mode / --preprocess v4 在此分支不生效")
                    # 几乎全黑：走 E9 路线（引导平滑 → 平滑亮度中性抬亮）
                    u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3, strength=0.5 * s_eff)
                    u8 = pp.quantize8(pp.limit_dark_gain(
                        u8, np.asarray(im), l0=3.0, l1=12.0, gamma=args.dark_gamma,
                        strength=0.80 * s_eff, smooth_radius=args.dark_smooth),)
                else:
                    if v4_mode:
                        # v4（A+B）：色度噪声判据 + 引导羽化，只动色度、不动亮度结构
                        _v4 = ppv4.chroma_denoise_v4(u8, strength=args.v4_strength,
                                                    tau=args.v4_tau, beta=args.v4_beta,
                                                    guide_radius=args.v4_guide)
                        u8 = pp.quantize8(_v4.astype(np.float32))
                    elif args.chroma_mode == "smooth":
                        u8 = pp.chroma_denoise_flat(u8, np.asarray(im), info["noise"],
                                                    strength=min(1.2, max(0.0, info["k"] * 0.45 * s_eff)),
                                                    target_amp=None)
                        u8 = pp.quantize8(u8)
                    elif args.chroma_mode == "desat":
                        # 全局、等比去饱和：不产生局部平台，因此不会新增"色块分界线"
                        _ycc = cv2.cvtColor(u8.astype(np.float32), cv2.COLOR_RGB2YCrCb)
                        _k = 1.0 - 0.15 * min(1.0, s_eff)
                        _ycc[..., 1] = 128 + (_ycc[..., 1] - 128) * _k
                        _ycc[..., 2] = 128 + (_ycc[..., 2] - 128) * _k
                        u8 = np.clip(cv2.cvtColor(_ycc, cv2.COLOR_YCrCb2RGB), 0, 255).astype(np.uint8)
                    else:
                        u8 = pp.quantize8(u8.astype(np.float32))
                out_f = u8.astype(np.float32) / 255.0
                out = out_f if args.bit16 else Image.fromarray(u8)
            elif use_post:
                if isinstance(raw, np.ndarray) and raw.dtype != np.uint8:
                    u8 = (np.clip(raw, 0, 1) * 255.0 + 0.5).astype(np.uint8)
                else:
                    u8 = np.asarray(raw).astype(np.uint8)
                if args.guided > 0:
                    u8 = pp.chroma_guided_smooth(u8, luma_guide=u8, radius=2, eps=1e-3,
                                                 strength=args.guided)
                if args.limit_dark > 0:
                    u8 = pp.quantize8(
                        pp.limit_dark_gain(u8, np.asarray(im), l0=3.0, l1=12.0,
                                           gamma=args.dark_gamma, strength=args.limit_dark,
                                           smooth_radius=args.dark_smooth),)
                if args.post > 0:
                    u8 = pp.post_smooth(u8, args.post)
                out_f = u8.astype(np.float32) / 255.0
                out = out_f if args.bit16 else Image.fromarray(u8)
            else:
                out_f = raw if isinstance(raw, np.ndarray) else np.asarray(raw, dtype=np.float32) / 255.0
                out = raw if (args.bit16 or isinstance(raw, Image.Image)) else Image.fromarray(raw)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] 处理失败 {os.path.basename(f)}: {e}")
            continue
        dt = time.time() - t0

        # 阴影补偿（只对很暗的图）：找回"阴影被加重"的 1~3 级
        _lift = args.shadow_lift
        if _lift < 0:
            # 只有"启用了预处理"时才自动加暗部补偿；-m sice（纯原版）必须保持原样
            _pre_on = auto_mode or (args.preprocess or "none").lower() != "none"
            _m = float(np.mean(np.asarray(im.convert("L"), dtype=np.float32)))
            _near_black = (_mode == "near_black") if auto_mode else False
            # 近黑分支已经做过"近黑限增益（含中性抬亮）"，再叠 +4 级会重复抬暗部 → 跳过
            _lift = 4.0 if (_m < 15.0 and _pre_on and not _near_black) else 0.0
        if _skipped:
            _lift = 0.0        # 跳过改進时必须与 "-m sice" 完全一致
        if _lift > 0:
            if args.bit16:
                out_f = pp.shadow_lift_float(out_f, lift=_lift)
                out = out_f
            else:
                _u8 = np.asarray(out if isinstance(out, Image.Image) else out, dtype=np.uint8)
                out = Image.fromarray(pp.shadow_lift(_u8, lift=_lift))
        base = os.path.splitext(os.path.basename(f))[0]
        save_jpg = args.jpg or os.path.splitext(f)[1].lower() in (".jpg", ".jpeg")
        if args.bit16:
            outp = os.path.join(outdir, base + "_enhanced16.png")
            pp.save_16bit(out_f, outp)
            out8 = Image.fromarray((np.clip(out_f, 0, 1) * 255.0 + 0.5).astype(np.uint8))
            ext = ".png"
        else:
            out8 = out if isinstance(out, Image.Image) else Image.fromarray(out)
            ext = ".jpg" if save_jpg else ".png"
            outp = os.path.join(outdir, base + "_enhanced" + ext)
            if save_jpg:
                out8.save(outp, quality=95, subsampling=1)
            else:
                out8.save(outp)

        if args.compare:
            w, h = im.size
            canvas = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
            canvas.paste(im, (0, 0))
            canvas.paste(out8, (w + 8, 0))
            cp = os.path.join(outdir, base + "_compare" + (".jpg" if save_jpg else ".png"))
            if save_jpg:
                canvas.save(cp, quality=92, subsampling=1)
            else:
                canvas.save(cp)

        before = float(np.asarray(im).mean())
        after = float(np.asarray(out8).mean())
        print(f"[{i}/{len(files)}] {os.path.basename(f):28s} {im.width}x{im.height}  "
              f"亮度 {before:5.1f} → {after:5.1f}   {dt:4.1f}s  ({mode})  -> {os.path.basename(outp)}")
        done += 1

    print(f"\n完成 {done}/{len(files)} 张，用时 {time.time() - t_all:.1f}s，输出目录：{outdir}")


if __name__ == "__main__":
    main()

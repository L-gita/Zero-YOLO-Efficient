r"""用自己的照片跑 Zero-DCE 增强。

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

例：
    python enhance.py D:\照片\夜景.jpg
    python enhance.py D:\照片\全家福 -o D:\增强结果 -c
    python enhance.py "D:\照片\夜景.jpg" -m official
"""
import argparse
import glob
import os
import sys
import time
import warnings

import numpy as np
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
warnings.filterwarnings("ignore")

import model  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
MODELS = {
    "official": os.path.join(CODE, "snapshots", "Epoch99.pth"),
    "sice": os.path.join(CODE, "snapshots_sice", "Epoch50.pth"),
    "edge": os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth"),   # 加论文 (4.12) Sobel 边缘损失训练的模型（50 轮）
    "lol": os.path.join(CODE, "snapshots_run", "Epoch199.pth"),
}
_net_cache = {}


def get_net(path):
    if path not in _net_cache:
        if not os.path.exists(path):
            raise SystemExit(f"找不到权重文件: {path}")
        net = model.enhance_net_nopool()
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        if torch.cuda.is_available():
            net = net.cuda()
        _net_cache[path] = net
        print(f"[模型] {path}")
    return _net_cache[path]


def _run(net, arr, half=False):
    """arr: HxWx3 float32 [0,1] -> enhanced np.uint8"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        if half and device == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                _, out, _ = net(t)
        else:
            _, out, _ = net(t)
        out = out.float().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (out * 255.0 + 0.5).astype(np.uint8)


def enhance_image(net, pil, force_tile=False):
    """整图处理；显存不够时自动降级 fp16 → 分块。返回 (PIL.Image, 模式说明)"""
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    if force_tile:
        return _tiled(net, pil), "tile(force)"
    try:
        return Image.fromarray(_run(net, arr)), "full/fp32"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        try:
            return Image.fromarray(_run(net, arr, half=True)), "full/fp16"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return _tiled(net, pil), "tile"


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
    ap = argparse.ArgumentParser(description="用 Zero-DCE 增强自己的照片")
    ap.add_argument("inputs", nargs="+", help="照片文件或文件夹（可多个）")
    ap.add_argument("-o", "--out", default=None, help="输出目录")
    ap.add_argument("-m", "--model", default="sice", help="official|sice|lol|.pth 路径")
    ap.add_argument("-c", "--compare", action="store_true", help="生成并排对比图")
    ap.add_argument("-j", "--jpg", action="store_true", help="输出 jpg")
    ap.add_argument("-s", "--scale", type=int, default=0, help="长边缩放到 N 像素（0=不缩放）")
    ap.add_argument("--tile", action="store_true", help="强制分块处理")
    args = ap.parse_args()

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
            out, mode = enhance_image(net, im, args.tile)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(files)}] 处理失败 {os.path.basename(f)}: {e}")
            continue
        dt = time.time() - t0

        base = os.path.splitext(os.path.basename(f))[0]
        save_jpg = args.jpg or os.path.splitext(f)[1].lower() in (".jpg", ".jpeg")
        ext = ".jpg" if save_jpg else ".png"
        outp = os.path.join(outdir, base + "_enhanced" + ext)
        if save_jpg:
            out.save(outp, quality=95, subsampling=1)
        else:
            out.save(outp)

        if args.compare:
            w, h = im.size
            canvas = Image.new("RGB", (w * 2 + 8, h), (20, 20, 20))
            canvas.paste(im, (0, 0))
            canvas.paste(out, (w + 8, 0))
            cp = os.path.join(outdir, base + "_compare" + (".jpg" if save_jpg else ".png"))
            if save_jpg:
                canvas.save(cp, quality=92, subsampling=1)
            else:
                canvas.save(cp)

        before = float(np.asarray(im).mean())
        after = float(np.asarray(out).mean())
        print(f"[{i}/{len(files)}] {os.path.basename(f):28s} {im.width}x{im.height}  "
              f"亮度 {before:5.1f} → {after:5.1f}   {dt:4.1f}s  ({mode})  -> {os.path.basename(outp)}")
        done += 1

    print(f"\n完成 {done}/{len(files)} 张，用时 {time.time() - t_all:.1f}s，输出目录：{outdir}")


if __name__ == "__main__":
    main()

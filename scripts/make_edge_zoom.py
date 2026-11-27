# -*- coding: utf-8 -*-
"""
边缘损失：局部细节放大 + 梯度幅值图对比
========================================
对若干细节丰富的图，裁一块区域放大 2 倍，输出两行：
  第 1 行：低照输入 | 基线(sice) | +边缘损失(4.12)
  第 2 行：对应的 Sobel 幅值图 M=|Ix|+|Iy|（论文 4.11），用于直观看梯度场差异
输出到 repro\demo_edge\ep49_edge022\zoom\ 。
"""
import os
import sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

REPRO = r"C:\Users\12983\Documents\repro"
CODE = os.path.join(REPRO, "Zero-DCE", "Zero-DCE_code")
sys.path.insert(0, CODE)
from model import enhance_net_nopool  # noqa: E402

BASE_CKPT = os.path.join(CODE, "snapshots_sice", "Epoch49.pth")
EDGE_CKPT = os.path.join(CODE, "snapshots_sice_edge", "Epoch49.pth")
OUT = os.path.join(REPRO, "demo_edge", "ep49_edge022", "zoom")
FONT = r"C:\Windows\Fonts\msyh.ttc"

# (显示名, 路径, 裁剪比例框 x0,y0,x1,y1)
CASES = [
    ("eval15/748 羽毛球场", os.path.join(REPRO, "lol_test", "eval15", "low", "748.png"), (0.30, 0.30, 0.95, 0.80)),
    ("eval15/493 玩偶+色卡", os.path.join(REPRO, "lol_test", "eval15", "low", "493.png"), (0.05, 0.05, 0.75, 0.75)),
    ("LIME/1 威尼斯夜景", os.path.join(CODE, "data", "test_data", "LIME", "1.bmp"), (0.50, 0.30, 0.99, 0.80)),
    ("DICM/01 埃菲尔铁塔", os.path.join(CODE, "data", "test_data", "DICM", "01.jpg"), (0.25, 0.05, 0.75, 0.60)),
]


def load(ckpt):
    net = enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(ckpt, map_location="cuda"))
    net.eval()
    return net


def enhance(net, rgb):
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(t)
    return out.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()


def sobel_M(rgb):
    x = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0)
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
    gx = torch.nn.functional.conv2d(xp, kx, groups=3)
    gy = torch.nn.functional.conv2d(xp, ky, groups=3)
    M = (gx.abs() + gy.abs()).mean(dim=1)[0].numpy()  # [H,W]
    return np.clip(M / 1.2, 0, 1)  # 归一化到 0-1 便于显示


def crop(arr, box):
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = box
    return arr[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def up(img, k=2):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * k, h * k), interpolation=cv2.INTER_CUBIC)


def label(img, text, size=26, pad=6):
    """在图片顶部加一条黑底白字标签"""
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        img = img.copy()
    f = ImageFont.truetype(FONT, size)
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + size + pad * 2), (0, 0, 0))
    canvas.paste(Image.fromarray(img), (0, size + pad * 2))
    d = ImageDraw.Draw(canvas)
    d.text((8, pad), text, font=f, fill=(255, 255, 255))
    return np.array(canvas)


def main():
    os.makedirs(OUT, exist_ok=True)
    base_net = load(BASE_CKPT)
    edge_net = load(EDGE_CKPT)
    for name, path, box in CASES:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out_b = enhance(base_net, rgb)
        out_e = enhance(edge_net, rgb)

        row1 = [label(up(crop(x, box)), t)
                for t, x in [("低照输入", rgb), ("基线 (SICE, 无边缘损失)", out_b), ("+边缘损失 (4.12)", out_e)]]
        row2 = [label(up(crop(m, box)), t)
                for t, m in [("M 输入", sobel_M(rgb)), ("M 基线", sobel_M(out_b)), ("M +边缘损失", sobel_M(out_e))]]

        sheet = np.vstack([np.hstack(row1), np.hstack(row2)])
        fn = os.path.join(OUT, name.replace("/", "_").replace(" ", "_") + "_zoom.png")
        # 注意：cv2.imwrite 在 Windows 上对含中文的路径会用 ANSI 编码打开文件，
        # 会把 UTF-8 字节当成 GBK → 文件名变成乱码。这里改用 PIL 保存（Unicode 安全）。
        Image.fromarray(sheet).save(fn, format="PNG")
        print(f"[ok] {fn}  {sheet.shape[1]}x{sheet.shape[0]}")


if __name__ == "__main__":
    main()

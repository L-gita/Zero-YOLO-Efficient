"""Side-by-side: low input | official Epoch99.pth | our own 200-epoch training."""
import os, sys
import numpy as np
import cv2
import torch

sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model  # noqa: E402

C = r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code"
OUT = r"C:\Users\12983\Documents\repro\compare"
os.makedirs(OUT, exist_ok=True)

ITEMS = [
    ("LIME_1", os.path.join(C, "data/test_data/LIME/1.bmp")),
    ("DICM_01", os.path.join(C, "data/test_data/DICM/01.jpg")),
    ("LOL_train_10", os.path.join(C, "data/train_data/10.jpg")),
    ("LOL_train_502", os.path.join(C, "data/train_data/502.jpg")),
]


def load(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def run(net, rgb):
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, out, _ = net(x)
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)


official = model.enhance_net_nopool().cuda()
official.load_state_dict(torch.load(os.path.join(C, os.environ.get("OFFICIALCKPT", "snapshots/Epoch99.pth"))))
official.eval()
our_ckpt = os.environ.get("OURSCKPT", "snapshots_run/Epoch199.pth")
ours = model.enhance_net_nopool().cuda()
ours.load_state_dict(torch.load(os.path.join(C, our_ckpt)))
ours.eval()
suffix = os.environ.get("SUFFIX", "")

H = 400
for name, path in ITEMS:
    low = load(path)
    a = run(official, low)
    b = run(ours, low)
    tiles = [low, a, b]
    tiles = [cv2.resize(t, (int(t.shape[1] * H / t.shape[0]), H), interpolation=cv2.INTER_LANCZOS4) for t in tiles]
    gap = 6
    canvas = np.full((H, sum(t.shape[1] for t in tiles) + gap * 2, 3), 255, np.uint8)
    x = 0
    for t in tiles:
        canvas[:, x:x + t.shape[1]] = t
        x += t.shape[1] + gap
    p = os.path.join(OUT, f"{name}_low_official_ours{suffix}.png")
    cv2.imwrite(p, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print("saved", p, "mean(low/official/ours):",
          round(float(low.mean()), 1), round(float(a.mean()), 1), round(float(b.mean()), 1))

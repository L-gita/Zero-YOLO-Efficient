"""Diagnostic: mean output brightness per checkpoint on a sample of LOL low images."""
import os, sys, glob
import numpy as np
import cv2
import torch

sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model  # noqa: E402

C = r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code"
files = sorted(glob.glob(os.path.join(C, "data/train_data/*.jpg")))[::24][:20]
print("images:", len(files))

ckpts = [("official_Epoch99", "snapshots/Epoch99.pth")] + [
    (f"ours_Epoch{e}", f"snapshots_run/Epoch{e}.pth") for e in (0, 49, 99, 149, 199)
]

for name, rel in ckpts:
    p = os.path.join(C, rel)
    if not os.path.exists(p):
        continue
    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(p, map_location="cuda"))
    net.eval()
    means, lows, ratios = [], [], []
    for f in files:
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            _, out, r = net(x)
        out = np.clip(out.squeeze(0).permute(1, 2, 0).cpu().numpy(), 0, 1)
        means.append(out.mean())
        lows.append(rgb.mean())
        ratios.append(r.mean().item())
    print(f"{name:18s} out_mean={np.mean(means):.3f}  in_mean={np.mean(lows):.3f}  curve_r_mean={np.mean(ratios):.3f}")

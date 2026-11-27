"""Compare the training loss of the official released weights vs our from-scratch model
on the same batch of LOL images.  If the official weights were trained on this exact data
with the loss in Myloss.py, their loss should be similar (or lower) than ours."""
import os, sys, glob
import numpy as np
import cv2
import torch

sys.path.insert(0, r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code")
import model, Myloss  # noqa: E402

C = r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code"
files = sorted(glob.glob(os.path.join(C, "data/train_data/*.jpg")))[::50][:10]

L_color, L_spa = Myloss.L_color(), Myloss.L_spa()
L_exp, L_TV = Myloss.L_exp(16, 0.6), Myloss.L_TV()


def batch_loss(net):
    tot = {"tv": 0.0, "spa": 0.0, "col": 0.0, "exp": 0.0}
    outs = []
    for f in files:
        rgb = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            _, enh, A = net(x)
        tv = 200 * L_TV(A)
        spa = torch.mean(L_spa(enh, x))
        col = 5 * torch.mean(L_color(enh))
        exp = 10 * torch.mean(L_exp(enh))
        tot["tv"] += tv.item(); tot["spa"] += spa.item(); tot["col"] += col.item(); tot["exp"] += exp.item()
        outs.append(enh.mean().item())
    n = len(files)
    return {k: v / n for k, v in tot.items()}, float(np.mean(outs))


for name, rel in [("official_Epoch99", "snapshots/Epoch99.pth"), ("ours_Epoch199", "snapshots_run/Epoch199.pth")]:
    net = model.enhance_net_nopool().cuda()
    net.load_state_dict(torch.load(os.path.join(C, rel), map_location="cuda"))
    net.eval()
    d, om = batch_loss(net)
    print(f"{name:16s} TV={d['tv']:.4f} SPA={d['spa']:.4f} COL={d['col']:.4f} EXP={d['exp']:.4f} "
          f"TOTAL={sum(d.values()):.4f}  out_mean={om:.3f}")

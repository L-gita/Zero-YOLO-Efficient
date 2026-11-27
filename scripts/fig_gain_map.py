import os, sys, numpy as np, torch, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image
FP = FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
ROOT = r"C:\Users\12983\Documents\repro"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"Zero-DCE","Zero-DCE_code")); sys.path.insert(0, os.path.join(ROOT,"scripts"))
import enhance as EN
net = EN.get_net(EN.MODELS["sice"])
def gain_map(path):
    low = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)/255.0
    t = torch.from_numpy(low).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad(): _, _, r = net(t)
    r = r.squeeze(0).cpu().numpy()
    alphas = np.stack([r[3*k:3*k+3] for k in range(8)], 0)
    x = low.transpose(2,0,1); xk = x.copy(); gain = np.ones_like(x)
    for k in range(8):
        a = alphas[k]; gain *= (1.0 + a*(2.0*xk - 1.0)); xk = xk + a*(xk*xk - xk)
    return low, gain.mean(axis=0), xk.transpose(1,2,0)
cases = [("126 极暗", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\126_3_.png")),
         ("493 常规暗", os.path.join(ROOT, r"lol_test\eval15\low\493.png")),
         ("132 同档暗", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\132_2_.png"))]
fig, axes = plt.subplots(len(cases), 3, figsize=(15, 4.2*len(cases)), dpi=110)
for i, (tag, path) in enumerate(cases):
    low, gm, y = gain_map(path)
    luma = low.mean(axis=2)
    ax = axes[i][0]; ax.imshow(low); ax.set_title(f"{tag}：输入（均值 {luma.mean()*255:.1f}）", fontproperties=FP, fontsize=13); ax.axis("off")
    ax = axes[i][1]
    im = ax.imshow(gm, cmap="inferno", vmin=1, vmax=max(14, np.percentile(gm,99)))
    ax.set_title(f"端到端总增益 d(out)/d(in)：暗区中位 {np.median(gm[luma<10/255]):.1f}×", fontproperties=FP, fontsize=13); ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    ax = axes[i][2]
    ax.hist(gm[luma<10/255].ravel(), bins=60, color="#e6194b", alpha=0.85)
    ax.set_title("极暗区(luma<10) 增益分布", fontproperties=FP, fontsize=13)
    ax.set_xlabel("总增益（倍）", fontproperties=FP); ax.set_ylabel("像素数", fontproperties=FP)
    ax.grid(alpha=0.3)
fig.suptitle("DCE-Net 的端到端增益图：1 级输入台阶 → 增益倍数的输出台阶（这就是马赛克的来源）", fontproperties=FP, fontsize=15)
fig.tight_layout()
out = os.path.join(ROOT, r"demo_mosaic\gain_map.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("[ok]", out)
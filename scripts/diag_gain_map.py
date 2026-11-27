import os, sys, numpy as np, torch, cv2
from PIL import Image
ROOT = r"C:\Users\12983\Documents\repro"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"Zero-DCE","Zero-DCE_code")); sys.path.insert(0, os.path.join(ROOT,"scripts"))
import enhance as EN
net = EN.get_net(EN.MODELS["sice"])

def gain_map(path):
    low = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)/255.0
    t = torch.from_numpy(low).permute(2,0,1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, _, r = net(t)
    r = r.squeeze(0).cpu().numpy()            # [24,H,W] = 8×3
    alphas = np.stack([r[3*k:3*k+3] for k in range(8)], 0)   # [8,3,H,W]
    x = low.transpose(2,0,1)                  # [3,H,W]
    xk = x.copy(); gain = np.ones_like(x)
    for k in range(8):
        a = alphas[k]
        gain *= (1.0 + a*(2.0*xk - 1.0))      # dLE/dx
        xk = xk + a*(xk*xk - xk)
    y = xk
    luma_in = low.mean(axis=2)
    return gain, y.transpose(1,2,0), luma_in

for tag, path in [("126 极暗", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\126_3_.png")),
                  ("493 常规暗", os.path.join(ROOT, r"lol_test\eval15\low\493.png")),
                  ("132 同档暗", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\132_2_.png"))]:
    g, y, luma = gain_map(path)
    gm = g.mean(axis=0)                        # 三通道平均总增益
    dark = luma < (10/255.0)
    mid  = (luma >= 10/255.0) & (luma < 40/255.0)
    br   = luma >= 40/255.0
    def st(m, name):
        if m.sum() < 50: return
        v = gm[m]
        print(f"    {name}: 像素 {100*m.mean():5.1f}%  总增益 中位 {np.median(v):5.2f}  均值 {v.mean():5.2f}  P95 {np.percentile(v,95):5.2f}  最大 {v.max():6.2f}")
    print(f"[{tag}]  输出/输入 亮度比 = {y.mean()/max(luma.mean(),1e-6):5.2f}×   (输入均值 {luma.mean()*255:.1f})")
    st(dark, "极暗区(luma<10)")
    st(mid,  "暗区(10~40)")
    st(br,   "亮区(>=40)")
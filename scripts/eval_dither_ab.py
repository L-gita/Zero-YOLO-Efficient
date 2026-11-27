import os, glob, numpy as np, cv2
from PIL import Image
ROOT = r"C:\Users\12983\Documents\repro"
AB = os.path.join(ROOT, r"demo_mosaic\dither_ab")
GT = os.path.join(ROOT, r"lol_test\eval15\high")
LOW = os.path.join(ROOT, r"lol_test\eval15\low")
V = ["V0_none", "V1_default", "V2_out09", "V3_in1"]
def load(p):
    return np.asarray(Image.open(p).convert("RGB"))
def psnr(a, b):
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse < 1e-9 else 10 * np.log10(255.0 ** 2 / mse)
def ssim(a, b):
    a = a.astype(np.float32); b = b.astype(np.float32)
    C1, C2 = (0.01*255)**2, (0.03*255)**2
    g = cv2.GaussianBlur(a, (11,11), 1.5); h = cv2.GaussianBlur(b, (11,11), 1.5)
    mu1, mu2 = g, h
    s1 = cv2.GaussianBlur(a*a, (11,11), 1.5) - mu1*mu1
    s2 = cv2.GaussianBlur(b*b, (11,11), 1.5) - mu2*mu2
    s12 = cv2.GaussianBlur(a*b, (11,11), 1.5) - mu1*mu2
    m = ((2*mu1*mu2 + C1)*(2*s12 + C2)) / ((mu1*mu1 + mu2*mu2 + C1)*(s1 + s2 + C2))
    return float(m.mean())
def canny(u8):
    g = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(cv2.GaussianBlur(g, (5,5), 1.0), 50, 120)
def edge_stats(pred, gt, tol=3):
    k = np.ones((2*tol+1, 2*tol+1), np.uint8)
    gtd = cv2.dilate(gt, k) > 0
    pd = pred > 0
    tp = int((pd & gtd).sum()); n_pred = int(pd.sum()); n_gt = int((gt > 0).sum())
    prec = tp / max(1, n_pred); rec = int((pred > 0).sum() and ((pred > 0) & (gt > 0)).sum()) / max(1, n_gt)
    # recall with tolerance: predicted edges near GT edges
    gtd2 = cv2.dilate(gt, k) > 0
    rec2 = float((pd & gtd2).sum()) / max(1, n_gt)
    return prec, rec2, n_pred, n_gt
files = sorted(glob.glob(os.path.join(LOW, "*.png")))
print("eval15 共", len(files), "对")
res = {}
for v in V:
    P = S = 0.0; prec = rec = 0.0; npred = ngt = 0
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        out = os.path.join(AB, v, stem + "_enhanced.png")
        if not os.path.exists(out): continue
        e = load(out); g = load(os.path.join(GT, stem + ".png"))
        P += psnr(e, g); S += ssim(e, g)
        pr, rc, n1, n2 = edge_stats(canny(e), canny(g))
        prec += pr; rec += rc; npred += n1; ngt += n2
    n = len(files)
    res[v] = dict(psnr=P/n, ssim=S/n, prec=prec/n, rec=rec/n, npred=npred, ngt=ngt)
    print(f"{v:<12} PSNR(vsGT) {P/n:6.2f}  SSIM {S/n:.4f}  边缘精度 {prec/n:.3f}  召回 {rec/n:.3f}  边缘像素 {npred}")
print("\n=== 与 V0（真·无抖动）的两两比较 ===")
for v in V[1:]:
    ps = []; sp = []; ch = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        a = os.path.join(AB, "V0_none", stem + "_enhanced.png"); b = os.path.join(AB, v, stem + "_enhanced.png")
        if not (os.path.exists(a) and os.path.exists(b)): continue
        A = load(a); B = load(b)
        ps.append(psnr(A, B)); sp.append(ssim(A, B)); ch.append(100.0*np.mean(np.any(A != B, axis=2)))
    print(f"{v:<12} vs V0: 图像级 PSNR {np.mean(ps):6.2f} dB  SSIM {np.mean(sp):.4f}  改变像素 {np.mean(ch):5.1f}%")
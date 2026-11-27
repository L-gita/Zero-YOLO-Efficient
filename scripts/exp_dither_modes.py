import os, sys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
ROOT = r"C:\Users\12983\Documents\repro"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT,"Zero-DCE","Zero-DCE_code")); sys.path.insert(0, os.path.join(ROOT,"scripts"))
import enhance as EN
rng = np.random.default_rng(0)
net = EN.get_net(EN.MODELS["sice"])
def enh_float(arr_float):
    a = np.clip(arr_float,0,255).astype(np.float32)/255.0
    return np.clip(EN._run_float(net, a),0,1)*255.0
def to_u8(f): return np.clip(np.rint(f),0,255).astype(np.uint8)
def band_metrics(u8, low_u8):
    y = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY); gl = cv2.cvtColor(low_u8, cv2.COLOR_RGB2GRAY)
    mask = gl < 40
    if mask.sum() < 0.02*mask.size: mask = gl < 80
    vals, counts = np.unique(y[mask], return_counts=True)
    keep = counts > 0.01*counts.sum(); lv = vals[keep]
    return len(lv), (int(np.max(np.diff(lv))) if len(lv)>1 else 0)
items = [("493 常规暗图", os.path.join(ROOT, r"lol_test\eval15\low\493.png")),
         ("126 极端暗光", os.path.join(ROOT, r"低照图库\01_真实拍摄_ZeroDCEpp\126_3_.png"))]
print(f"{'图':<12}{'方案':<28}{'亮度':>7}{'暗区级数':>8}{'最大跳变':>8}{'平坦颗粒':>9}")
vis={}
for title, path in items:
    low = np.asarray(Image.open(path).convert("RGB"))
    f0 = enh_float(low.astype(np.float32)); base = to_u8(f0)
    gl = cv2.cvtColor(low, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gy = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gain = np.clip((gy+1.0)/np.maximum(gl+1.0,1.0), 1.0, 12.0)
    variants = [("a 基线", base), ("b 输出端抖动 ±0.9", to_u8(f0 + rng.uniform(-0.9,0.9,f0.shape)))]
    for t in (0.5, 1.0):
        u = np.clip(t/gain, 0.02, 0.5)
        lowf = low.astype(np.float32) + rng.uniform(-1,1,low.shape)*u[...,None]
        variants.append((f"c 输入域抖动 t={t}", to_u8(enh_float(lowf))))
    cur = low.copy()
    for k in range(1,4):
        curf = cur.astype(np.float32) + rng.uniform(-0.4,0.4,cur.shape)
        cur = to_u8(enh_float(curf)); variants.append((f"d 循环 {k} 次", cur.copy()))
    for name, im in variants:
        nlev, jump = band_metrics(im, low)
        y = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = np.abs(y - cv2.GaussianBlur(y,(0,0),1.2))
        fx = cv2.Sobel(cv2.cvtColor(low,cv2.COLOR_RGB2GRAY),cv2.CV_32F,1,0,ksize=3)
        fy = cv2.Sobel(cv2.cvtColor(low,cv2.COLOR_RGB2GRAY),cv2.CV_32F,0,1,ksize=3)
        fmask = (np.abs(fx)+np.abs(fy)) < 20
        if fmask.sum() < 500: fmask = np.ones_like(fmask, bool)
        print(f"{title:<12}{name:<28}{y.mean():>7.1f}{nlev:>8d}{jump:>8d}{hp[fmask].mean():>9.3f}")
    vis[title] = [("低照输入", low), ("a 基线", base),
                  ("b 输出抖动 ±0.9", to_u8(f0 + rng.uniform(-0.9,0.9,f0.shape))),
                  ("c 输入域抖动 t=1.0", to_u8(enh_float(low.astype(np.float32)+rng.uniform(-1,1,low.shape)*np.clip(1.0/gain,0.02,0.5)[...,None]))),
                  ("d 循环 3 次", cur)]
FONT = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", 18)
OUT = os.path.join(ROOT, r"demo_mosaic\dither_test"); os.makedirs(OUT, exist_ok=True)
for title, ims in vis.items():
    panels=[]
    for name, im in ims:
        h,w = im.shape[:2]; sc = min(1.0, 560.0/max(h,w))
        z = cv2.resize(im, (int(w*sc), int(h*sc)), interpolation=cv2.INTER_AREA)
        c = Image.new("RGB",(z.shape[1], z.shape[0]+26),(26,26,26)); c.paste(Image.fromarray(z),(0,26))
        ImageDraw.Draw(c).text((6,4), name, font=FONT, fill=(240,240,240)); panels.append(np.asarray(c))
    Image.fromarray(np.hstack(panels)).save(os.path.join(OUT, f"dither_{title.split()[0]}.png"))
print("[ok]", OUT)
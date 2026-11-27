import os, sys, subprocess, hashlib, time, numpy as np
ROOT = r"C:\Users\12983\Documents\repro"
PY = sys.executable
ENH = os.path.join(ROOT, "enhance.py")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import preprocess as pp
from PIL import Image

def run(img, out, extra):
    args = [PY, ENH, img, "-o", out] + extra
    r = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r

def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()

fails = []
def ok(name, cond, info=""):
    print(("  [PASS] " if cond else "  [FAIL] ") + name + ("" if cond else f"  <-- {info}"))
    if not cond:
        fails.append(name)

print("=== 1) 常规分支：σc<0.3 必须逐像素等于 -m sice；σc≥0.3 必须变色噪下降、细节不变 ===")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from compare_preprocess import chroma_noise, luma_std8

for n in ["132_2_.png", "512_0_.png", "493.png", "101_0_.png"]:
    src = os.path.join(ROOT, "demo_final_vs_base", "input", n)
    o1 = os.path.join(ROOT, "demo_audit", "base"); o2 = os.path.join(ROOT, "demo_audit", "auto")
    run(src, o1, ["-m", "sice"]); run(src, o2, ["--preprocess", "auto"])
    f1 = os.path.join(o1, os.path.splitext(n)[0] + "_enhanced.png")
    f2 = os.path.join(o2, os.path.splitext(n)[0] + "_enhanced.png")
    if not (os.path.exists(f1) and os.path.exists(f2)):
        ok(f"{n} 输出文件存在", False, "缺文件")
        continue
    low = np.asarray(Image.open(src).convert("RGB"))
    s = pp.chroma_light_strength(pp.estimate_chroma_noise(low))
    if s <= 0:
        same = md5(f1) == md5(f2)
        ok(f"{n} σc={pp.estimate_chroma_noise(low):.2f} 跳过 → 与 -m sice 逐像素一致", same, "md5 不同")
    else:
        a = np.asarray(Image.open(f1).convert("RGB"))
        b = np.asarray(Image.open(f2).convert("RGB"))
        d_cn = chroma_noise(b) - chroma_noise(a)
        d_det = abs(luma_std8(b) - luma_std8(a))
        ok(f"{n} σc={pp.estimate_chroma_noise(low):.2f} s={s:.2f} → 色噪下降({d_cn:+.2f})且细节不变({d_det:.2f})",
           d_cn <= 0.01 and d_det < 0.6, f"d_cn={d_cn:+.2f} d_det={d_det:.2f}")

print("=== 1b) 对比工具输出必须与 enhance.py 逐像素一致（历史上出过“两套管线”bug）===")
src = os.path.join(ROOT, "demo_final_vs_base", "input", "493.png")
d1 = os.path.join(ROOT, "demo_audit", "cons_enh"); d2 = os.path.join(ROOT, "demo_audit", "cons_cmp")
run(src, d1, ["--preprocess", "auto"])
r = subprocess.run([PY, os.path.join(ROOT, "scripts", "compare_preprocess.py"), src, "-o", d2,
                    "--mode", "auto", "--keep-enhanced"],
                   capture_output=True, text=True, encoding="utf-8", errors="replace")
f1 = os.path.join(d1, "493_enhanced.png")
f2 = os.path.join(d2, "_enhanced", "493_final.png")
ok("compare 与 enhance 输出一致", os.path.exists(f1) and os.path.exists(f2) and md5(f1) == md5(f2),
   f"md5 不同或文件缺失：{os.path.exists(f1)}/{os.path.exists(f2)}")

print("=== 2) 近黑图（126）必须走 near_black 且不得报错 ===")
r = run(os.path.join(ROOT, "demo_final_vs_base", "input", "126_3_.png"), os.path.join(ROOT, "demo_audit", "nb"), ["--preprocess", "auto"])
ok("126 near_black 跑通", r.returncode == 0 and "near_black" in (r.stdout or ""), (r.stdout or "")[-200:] + (r.stderr or "")[-200:])

print("=== 3) bit16 输出：应存在 *_enhanced16.png 且为 16bit ===")
src = os.path.join(ROOT, "demo_final_vs_base", "input", "126_3_.png")
r = run(src, os.path.join(ROOT, "demo_audit", "b16"), ["--preprocess", "auto", "--bit16"])
f = os.path.join(ROOT, "demo_audit", "b16", "126_3__enhanced16.png")
import cv2 as _cv2
cond = (_cv2.imread(f, _cv2.IMREAD_UNCHANGED) is not None and _cv2.imread(f, _cv2.IMREAD_UNCHANGED).dtype == np.uint16)
ok(f"16bit 文件正常（dtype=uint16）", cond)

print("=== 4) jpg 输入应输出 *_enhanced.jpg，不得崩 ===")
r = run(os.path.join(ROOT, "demo_final_vs_base", "input", "12.jpg"), os.path.join(ROOT, "demo_audit", "jpg"), ["--preprocess", "auto"])
f = os.path.join(ROOT, "demo_audit", "jpg", "12_enhanced.jpg")
ok("jpg 输出存在", os.path.exists(f), (r.stdout or "")[-150:] + (r.stderr or "")[-150:])

print("=== 5) 大图（12MP）应自动分块不 OOM ===")
big = os.path.join(ROOT, "demo_audit", "big.jpg")
if not os.path.exists(big):
    a = np.asarray(Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", "493.png")).convert("RGB"))
    a = np.tile(a, (5, 5, 1))[:3000, :4000]
    Image.fromarray(a).save(big, quality=95)
t0 = time.time(); r = run(big, os.path.join(ROOT, "demo_audit", "big_out"), ["--preprocess", "auto", "-c"]); dt = time.time() - t0
ok(f"12MP 跑通（{dt:.1f}s）", r.returncode == 0, (r.stdout or "")[-300:] + (r.stderr or "")[-300:])

print("=== 6) 灰度 / RGBA / P 模式输入不得崩 ===")
for name, im in [("gray", Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", "147_0_.png")).convert("L")),
                 ("rgba", Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", "147_0_.png")).convert("RGBA")),
                 ("palette", Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", "147_0_.png")).convert("P"))]:
    p = os.path.join(ROOT, "demo_audit", f"{name}.png"); im.save(p)
    r = run(p, os.path.join(ROOT, "demo_audit", f"{name}_out"), ["--preprocess", "auto"])
    ok(f"{name} 输入正常", r.returncode == 0, (r.stdout or "")[-150:] + (r.stderr or "")[-150:])

print("=== 7) 文件夹递归 + 混合格式（含 jpg/png） ===")
d = os.path.join(ROOT, "demo_audit", "mixed")
os.makedirs(d, exist_ok=True)
for n in ["493.png", "12.jpg", "147_0_.png"]:
    Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", n)).convert("RGB").save(os.path.join(d, n))
r = run(d, os.path.join(ROOT, "demo_audit", "mixed_out"), ["--preprocess", "auto"])
outs = [f for f in os.listdir(os.path.join(ROOT, "demo_audit", "mixed_out")) if f.endswith("_enhanced.png") or f.endswith("_enhanced.jpg")]
ok(f"文件夹 3 张全出（{len(outs)}）", len(outs) == 3, str(outs))

print("=== 8) 损坏文件不应中断整批 ===")
bad = os.path.join(ROOT, "demo_audit", "bad", "broken.png"); os.makedirs(os.path.dirname(bad), exist_ok=True)
open(bad, "wb").write(b"not an image")
Image.open(os.path.join(ROOT, "demo_final_vs_base", "input", "493.png")).convert("RGB").save(os.path.join(ROOT, "demo_audit", "bad", "good.png"))
r = run(os.path.join(ROOT, "demo_audit", "bad"), os.path.join(ROOT, "demo_audit", "bad_out"), ["--preprocess", "auto"])
ok("坏文件被跳过、好文件仍出图", r.returncode == 0 and os.path.exists(os.path.join(ROOT, "demo_audit", "bad_out", "good_enhanced.png")), (r.stdout or "")[-300:])

print()
print("失败项：", fails if fails else "无")
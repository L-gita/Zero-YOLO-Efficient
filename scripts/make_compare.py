import os
from PIL import Image

C = r"C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code"
OUT = r"C:\Users\12983\Documents\repro\compare"
os.makedirs(OUT, exist_ok=True)

pairs = [
    ("data/test_data/LIME/1.bmp", "data/result/LIME/1.bmp", "LIME_1"),
    ("data/test_data/LIME/5.bmp", "data/result/LIME/5.bmp", "LIME_5"),
    ("data/test_data/DICM/01.jpg", "data/result/DICM/01.jpg", "DICM_01"),
]

for src, res, name in pairs:
    a = Image.open(os.path.join(C, src)).convert("RGB")
    b = Image.open(os.path.join(C, res)).convert("RGB")
    h = 512
    aw = int(a.width * h / a.height)
    bw = int(b.width * h / b.height)
    a = a.resize((aw, h), Image.LANCZOS)
    b = b.resize((bw, h), Image.LANCZOS)
    canvas = Image.new("RGB", (aw + bw + 8, h), (255, 255, 255))
    canvas.paste(a, (0, 0))
    canvas.paste(b, (aw + 8, 0))
    p = os.path.join(OUT, name + "_in_vs_out.png")
    canvas.save(p)
    print("saved", p, f"in_mean={sum(a.convert('L').getdata())/ (aw*h):.1f} out_mean={sum(b.convert('L').getdata())/(bw*h):.1f}")

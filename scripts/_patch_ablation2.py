import io, re, ast

F = r"C:\Users\12983\Documents\repro\scripts\mosaic_ablation.py"
t = io.open(F, encoding="utf-8").read()

# --- 1) 插入 band_jump / n_levels
if "def band_jump" not in t:
    funcs = '''def band_jump(gray_u8, mask):
    """暗区内占面积>1% 的灰度级之间最大的跳变（越大越容易出现可见条带/硬灰块）。"""
    v = gray_u8[mask].astype(np.int32)
    if v.size < 100:
        return 0
    hist = np.bincount(np.clip(v, 0, 255), minlength=256).astype(np.float64)
    hist /= hist.sum()
    big = np.where(hist > 0.01)[0]
    jumps = [int(b - a) for a, b in zip(big[:-1], big[1:]) if b - a > 1]
    return max(jumps) if jumps else 0


def n_levels(gray_u8, mask):
    """暗区内占面积>1% 的灰度级个数（越多越细腻）。"""
    v = gray_u8[mask].astype(np.int32)
    if v.size < 100:
        return 0
    hist = np.bincount(np.clip(v, 0, 255), minlength=256).astype(np.float64)
    hist /= hist.sum()
    return int((hist > 0.01).sum())


def luma_std8(rgb, mask):'''
    assert "def luma_std8(rgb, mask):" in t
    t = t.replace("def luma_std8(rgb, mask):", funcs, 1)
    print("banding metrics inserted")

# --- 2) 替换 E8 分支 + 增加 E9
old_e8 = '''    if name == "E8":
        # 本文推荐：E6 + 近黑区限增益（中性抬亮替代彩色放大，消除彩色大斑块）
        out = run_pipeline("E6", rgb)
        return pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.40, strength=0.85)'''
new_e8 = '''    if name == "E8":
        # v1（已证伪）：直接用 8bit 量化后的亮度做中性抬亮 → 只映射出少数几档，形成大片硬边灰块
        out = run_pipeline("E6", rgb)
        return pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.40, strength=0.85,
                                  smooth_radius=0, dither=0.0)
    if name == "E9":
        # v2（推荐）：抬亮前先大模糊亮度（连续化）+ 羽化掩码 + 8bit 抖动，消除硬边灰块
        out = run_pipeline("E6", rgb)
        return pp.quantize8(
            pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.70, strength=0.85,
                               smooth_radius=8, dither=0.0),
            dither=0.6)'''
assert old_e8 in t, "E8 branch not found"
t = t.replace(old_e8, new_e8)

io.open(F, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(io.open(F, encoding="utf-8").read())
print("ok")

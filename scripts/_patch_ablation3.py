import io, ast

F = r"C:\Users\12983\Documents\repro\scripts\mosaic_ablation.py"
t = io.open(F, encoding="utf-8").read()

# 1) 条带指标改为“在输出自身的暗区上统计”，并给函数默认 mask=None
t = t.replace('''def band_jump(gray_u8, mask):
    """暗区内占面积>1% 的灰度级之间最大的跳变（越大越容易出现可见条带/硬灰块）。"""
    v = gray_u8[mask].astype(np.int32)''',
'''def _out_dark_mask(gray_u8):
    m = gray_u8 < 40
    if m.mean() < 0.02:
        m = gray_u8 < 80
    return m


def band_jump(gray_u8, mask=None):
    """输出图暗区内“面积>1% 的灰度级”之间的最大跳变（越大越容易出现可见条带/硬灰块）。"""
    mask = _out_dark_mask(gray_u8) if mask is None else mask
    v = gray_u8[mask].astype(np.int32)''')

t = t.replace('''def n_levels(gray_u8, mask):
    """暗区内占面积>1% 的灰度级个数（越多越细腻）。"""
    v = gray_u8[mask].astype(np.int32)''',
'''def n_levels(gray_u8, mask=None):
    """输出图暗区内占面积>1% 的灰度级个数（越多越细腻）。"""
    mask = _out_dark_mask(gray_u8) if mask is None else mask
    v = gray_u8[mask].astype(np.int32)''')

# 2) 表格里调用改为不传 mask（用输出暗区口径）
t = t.replace("band_jump=band_jump(g_out, mask), n_levels=n_levels(g_out, mask),",
              "band_jump=band_jump(g_out), n_levels=n_levels(g_out),")

# 3) E9 参数收紧：只处理真正近黑（l0=3, l1=12），gamma 0.55
old = """            pp.limit_dark_gain(out, rgb, l0=6.0, l1=20.0, gamma=0.70, strength=0.85,
                               smooth_radius=8, dither=0.0),
            dither=0.6)"""
new = """            pp.limit_dark_gain(out, rgb, l0=3.0, l1=12.0, gamma=0.55, strength=0.80,
                               smooth_radius=8, dither=0.0),
            dither=0.6)"""
assert old in t, "E9 params not found"
t = t.replace(old, new)

io.open(F, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)
print("patched: banding metric on output-dark mask; E9 l0=3,l1=12,gamma=0.55,strength=0.80")

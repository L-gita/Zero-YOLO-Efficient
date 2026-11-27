import io, ast

F = r"C:\Users\12983\Documents\repro\enhance.py"
t = io.open(F, encoding="utf-8").read()

# 预设：E9 参数（l0=3, l1=12, gamma=0.55, strength=0.80）
t = t.replace('"mosaic": dict(denoise=0.6, deblock=0.5, prelift=0.15, guided=0.7, limit_dark=0.85, dark_gamma=0.70, dark_smooth=8, dither=0.6),',
              '"mosaic": dict(denoise=0.6, deblock=0.5, prelift=0.15, guided=0.7, limit_dark=0.80, dark_gamma=0.55, dark_smooth=8, dither=0.6),')
t = t.replace('"dark": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.55, dark_smooth=10, dither=0.6),',
              '"dark": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.50, dark_smooth=10, dither=0.6),')
t = t.replace('"strong": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.55, dark_smooth=10, dither=0.6),',
              '"strong": dict(denoise=1.0, deblock=1.0, prelift=0.20, guided=0.8, limit_dark=1.0, dark_gamma=0.50, dark_smooth=10, dither=0.6),')

# 调用处 l0/l1：6/20 → 3/12（只动真正近黑）
t = t.replace("pp.limit_dark_gain(u8, np.asarray(im), l0=6.0, l1=20.0,",
              "pp.limit_dark_gain(u8, np.asarray(im), l0=3.0, l1=12.0,")

# 默认参数注释与默认值
t = t.replace('ap.add_argument("--dark-gamma", type=float, default=0.70,',
              'ap.add_argument("--dark-gamma", type=float, default=0.55,')
t = t.replace('help="近黑区中性抬亮的 gamma（越大越暗；0.70=默认，1.0=几乎不提亮）")',
              'help="近黑区中性抬亮的 gamma（越大越暗；0.55=默认，1.0=几乎不提亮）")')

io.open(F, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)
print("enhance.py updated to E9 params")

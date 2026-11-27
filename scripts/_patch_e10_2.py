import io
import ast

E = r"C:\Users\12983\Documents\repro\enhance.py"
t = io.open(E, encoding="utf-8").read()

# 1) 预设加 auto（E10 场景自适应）
old = 'PRESETS = {\n'
assert old in t
t = t.replace(old, '''PRESETS = {
    # auto = E10 场景自适应：先估噪声/块效应，再决定要不要降噪/去块/轻抬；
    # 后处理只在暗图上做"只动亮度"的近黑限增益 + 近黑区掩码色度降噪
    "auto": dict(denoise=-1.0, deblock=-1.0, prelift=-1.0, guided=0.0, limit_dark=-1.0,
                 dark_gamma=0.60, dark_smooth=8, dither=0.4),
''', 1)

# 2) 帮助文本
t = t.replace('ap.add_argument("--preprocess", default=None, help="预设：mosaic|light|strong")',
              'ap.add_argument("--preprocess", default=None, help="预设：auto（推荐，场景自适应）| mosaic（旧版）| light | dark | strong")')
t = t.replace('    --preprocess preset   一键预设：mosaic（推荐整套）/ light / strong',
              '    --preprocess preset   一键预设：auto（推荐，E10 场景自适应）/ mosaic（旧版 E9）/ light / dark')
t = t.replace('python enhance.py "D:\\照片\\夜景.jpg" --preprocess mosaic -c      # 治马赛克',
              'python enhance.py "D:\\照片\\夜景.jpg" --preprocess auto -c        # 治马赛克（推荐）')

# 3) 预处理分支
old_pre = '''            # ① 增强前的预处理（去块 → 降噪 → 轻抬）
            pre = im
            if use_pre:
                pre = Image.fromarray(pp.preprocess(
                    np.asarray(im), denoise=args.denoise,
                    deblock_strength=args.deblock, prelift_amount=args.prelift))'''
new_pre = '''            # ① 增强前的预处理：auto=场景自适应（E10），否则按手动/旧预设参数
            pre = im
            if auto_mode:
                info = pp.auto_preprocess(np.asarray(im))
                pre = Image.fromarray(info["img"])
                print(f"    [auto] 均值 {info['mean']:5.1f}  噪声σ {info['noise']:4.2f}  "
                      f"块效应 {info['blockiness']:.3f} → 降噪×{info['k']:.2f}  "
                      f"去块={'开' if info['deblock'] else '关'}  轻抬={'开' if info['prelift'] else '关'}")
            elif use_pre:
                pre = Image.fromarray(pp.preprocess(
                    np.asarray(im), denoise=args.denoise,
                    deblock_strength=args.deblock, prelift_amount=args.prelift))'''
assert old_pre in t
t = t.replace(old_pre, new_pre)

# 4) 后处理分支
old_post = '''            # ③ 增强后的处理：引导平滑 → 近黑限增益（→ 可选双边）
            if use_post:'''
new_post = '''            # ③ 增强后的处理：auto=近黑限增益（只动亮度）+ 近黑区掩码色度降噪；否则用旧逻辑
            if auto_mode:
                u8 = (np.clip(raw, 0, 1) * 255.0 + 0.5).astype(np.uint8) if (isinstance(raw, np.ndarray) and raw.dtype != np.uint8) else np.asarray(raw).astype(np.uint8)
                info = pp.auto_preprocess(np.asarray(im))
                if info["mean"] < 20.0:      # 正常亮度的图不做后处理，避免发灰/发糊
                    u8 = pp.limit_dark_luma(u8, np.asarray(im), thr=2.5, feather=3.0,
                                            gamma=args.dark_gamma, strength=0.9, chroma_k=0.85)
                    u8 = pp.masked_chroma_denoise(u8, np.asarray(im), thr=6.0, strength=2.2)
                    u8 = pp.quantize8(u8, dither=args.dither)
                out_f = u8.astype(np.float32) / 255.0
                out = out_f if args.bit16 else Image.fromarray(u8)
            elif use_post:'''
assert old_post in t
t = t.replace(old_post, new_post)

# 5) auto_mode 定义（放在参数解析之后、use_pre 之前）
old_use = '    use_pre = (args.denoise > 0 or args.deblock > 0 or args.prelift > 0)'
new_use = '    auto_mode = (args.preprocess or "").lower() == "auto" or args.denoise < 0\n    use_pre = (args.denoise > 0 or args.deblock > 0 or args.prelift > 0)'
assert old_use in t
t = t.replace(old_use, new_use)

# 6) 打印信息里也要能显示 auto
t = t.replace('''    if use_pre or use_post:
        print(f"[预处理] 去块={args.deblock:g}  色度降噪={args.denoise:g}  "''',
              '''    if auto_mode:
        print("[预处理] auto 场景自适应（E10）：先估噪声/块效应，再按需降噪/去块/轻抬；"
              "暗图才做近黑限增益")
    elif use_pre or use_post:
        print(f"[预处理] 去块={args.deblock:g}  色度降噪={args.denoise:g}  "''')

io.open(E, "w", encoding="utf-8", newline="\n").write(t)
ast.parse(t)
print("enhance.py -> auto 预设接入完成")

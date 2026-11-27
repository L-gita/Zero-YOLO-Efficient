import io, os, re, sys

F = r"C:\Users\12983\Documents\repro\scripts\mosaic_ablation.py"
G = r"C:\Users\12983\Documents\repro\scripts\make_mosaic_figs.py"

t = io.open(F, encoding="utf-8").read()

t = t.replace(
    "                plateau=chroma_plateau(out, mask), luma_std8=luma_std8(out, mask),",
    "                plateau=chroma_plateau(out, mask), luma_std8=luma_std8(out, mask),\n"
    "                band_jump=band_jump(g_out, mask), n_levels=n_levels(g_out, mask),")

t = t.replace('            "plateau", "luma_std8"]',
              '            "plateau", "luma_std8", "band_jump", "n_levels"]')

t = t.replace('summary = ["# 马赛克预处理对照实验结果（E0~E7）", ""]',
              'summary = ["# 马赛克预处理对照实验结果（E0~E9）", ""]')

t = t.replace('head = "| 管线 | 说明 | 亮度 | 色噪 | 块效应比 | M放大 | 彩色块占比 | 亮度细节(std8) |"',
              'head = "| 管线 | 说明 | 亮度 | 色噪 | 块效应比 | M放大 | 彩色块占比 | 亮度细节(std8) | 条带跳变 | 暗区级数 |"')

t = t.replace('sep = "| --- | --- | --- | --- | --- | --- | --- | --- |"',
              'sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"')

old_line = '''                    f"{sum(r['luma_std8'] for r in rs)/n:.2f} |")'''
new_line = '''                    f"{sum(r['luma_std8'] for r in rs)/n:.2f} | "
                    f"{sum(r['band_jump'] for r in rs)/n:.1f} | "
                    f"{sum(r['n_levels'] for r in rs)/n:.1f} |")'''
assert old_line in t, "row line not found"
t = t.replace(old_line, new_line)

old_print = '''        print(f"[{i}/{len(items)}] {group}/{name}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"色噪 E1={rows[-5]['chroma_noise']:.2f} → E5={rows[-1]['chroma_noise']:.2f}  "
              f"块效应 E1={rows[-5]['blockiness']:.2f} → E5={rows[-1]['blockiness']:.2f}  ({time.time()-t0:.0f}s)")'''
new_print = '''        rr = {r["pipeline"]: r for r in rows[-len(PIPELINES):]}
        print(f"[{i}/{len(items)}] {group}/{name}  {rgb.shape[1]}x{rgb.shape[0]}  "
              f"色噪 {rr['E1']['chroma_noise']:.2f}->E9 {rr['E9']['chroma_noise']:.2f}  "
              f"条带跳变 {rr['E8']['band_jump']:.0f}->E9 {rr['E9']['band_jump']:.0f}  ({time.time()-t0:.0f}s)")'''
assert old_print in t, "print not found"
t = t.replace(old_print, new_print)

io.open(F, "w", encoding="utf-8", newline="\n").write(t)
print("ablation patched")

u = io.open(G, encoding="utf-8").read()
u = u.replace('PIPES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]',
              'PIPES = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9"]')
u = u.replace('COLORS = ["#6B8EAE", "#C4A35A", "#8FBC8F", "#D08C7A", "#9B7FBF", "#7FA6A0", "#B5747A", "#D9B44A"]',
              'COLORS = ["#6B8EAE", "#C4A35A", "#8FBC8F", "#D08C7A", "#9B7FBF", "#7FA6A0", "#B5747A", "#D9B44A", "#4F9D69"]')
u = u.replace('for p in ["E0", "E1", "E5", "E6", "E8"]:', 'for p in ["E0", "E1", "E5", "E8", "E9"]:')
u = re.sub(r"short = \{[^}]*\}", '''short = {"E0": "低照原图", "E1": "基线", "E2": "先降噪", "E3": "后降噪",
                 "E4": "弱降噪+平滑", "E5": "旧方案(重平滑)", "E6": "引导平滑",
                 "E7": "抑制无信号区", "E8": "限增益v1(硬灰块)", "E9": "限增益v2(本文)"}''', u)
io.open(G, "w", encoding="utf-8", newline="\n").write(u)
print("figs patched")

import ast
ast.parse(io.open(F, encoding="utf-8").read()); ast.parse(io.open(G, encoding="utf-8").read())
print("syntax ok")

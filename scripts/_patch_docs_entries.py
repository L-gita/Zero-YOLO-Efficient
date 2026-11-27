import io, os

base = r"C:\Users\12983\Documents\repro"

README_BLOCK = """

## 10. 三个一键入口（拖拽 / 双击都行，2026-11-27）

| 入口（双击运行） | 作用 | 输出目录（在第一张图旁边） | 每张产物 |
| --- | --- | --- | --- |
| `原版增强.bat` | **原版 Zero-DCE**，不做任何预处理 | `原版增强结果\\` | `xxx_enhanced.png` + `xxx_compare.png` |
| `去噪增强.bat` | **去噪版**：`--preprocess auto`（按噪声/块效应自适应去噪、去块、近黑处理） | `去噪增强结果\\` | `xxx_enhanced.png` + `xxx_compare.png` |
| `对比_原版vs去噪.bat` | **两者对比**：原图 / 原版 / 去噪版 三列 + 同区域 3× 放大 | `对比_原版vs去噪\\` | `xxx_对比.png` + `metrics.csv` + `summary.md` |

用法（三选一）：
1. **拖拽**：把照片（或整个文件夹）拖到对应的 `.bat` 图标上，松手即可；
2. **双击**：弹出选图窗口，按住 Ctrl 可多选；
3. **命令行**：`"C:\\...\\repro\\去噪增强.bat" "D:\\照片\\a.jpg" "D:\\照片\\b.jpg"`

说明：
- 三个入口都**不含**之前的边缘损失（`-m edge` / `--edge_weight`），是干净的原版 vs 去噪版；边缘损失以后要加随时能加。
- 大图（>6 MP）会自动分块，12 MP 约 11 s；统一实现在 `scripts\\three_entries.ps1`。
- 旧的 `去马赛克增强.bat`、`对比_基线vs去马赛克.bat`、`增强我的照片.bat` 仍保留（后者会同时出边缘损失那版），新三个入口是推荐用法。
"""

p = os.path.join(base, "README.md")
t = io.open(p, encoding="utf-8").read()
if "三个一键入口" not in t:
    io.open(p, "w", encoding="utf-8", newline="\n").write(t.rstrip() + README_BLOCK + "\n")
    print("[ok] README 追加 §10")
else:
    print("[skip] README 已有")

GUIDE_BLOCK = """
## 13. 三个一键入口（2026-11-27，推荐用法）

| 双击这个 bat | 做什么 | 输出在哪 |
| --- | --- | --- |
| `原版增强.bat` | 原版 Zero-DCE（无预处理） | 第一张图旁 `原版增强结果\\` |
| `去噪增强.bat` | 去噪版（`--preprocess auto`，自适应去噪） | 第一张图旁 `去噪增强结果\\` |
| `对比_原版vs去噪.bat` | 三列对比 + 3× 放大 + 指标表 | 第一张图旁 `对比_原版vs去噪\\` |

- 拖拽照片/文件夹到 bat 上即可；双击则弹窗选图（Ctrl 多选）。
- 三个入口都不含边缘损失；以后要加就用 `--edge_weight`（训练）/ `-m edge`（推理）。
- 大图自动分块（12 MP ≈ 11 s），不会再报 CUDA OOM。
"""
for gp in [r"C:\Users\12983\Documents\deepseek-harness\default-workspace\零-DCE_使用指南.md",
           os.path.join(base, r"docs\零-DCE_使用指南.md")]:
    if os.path.exists(gp):
        g = io.open(gp, encoding="utf-8").read()
        if "三个一键入口" not in g:
            io.open(gp, "w", encoding="utf-8", newline="\n").write(g.rstrip() + "\n" + GUIDE_BLOCK)
            print("[ok] 追加到", gp)
        else:
            print("[skip]", gp)

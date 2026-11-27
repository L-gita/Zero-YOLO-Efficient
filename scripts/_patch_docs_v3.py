import io, os

base = r"C:\Users\12983\Documents\repro"
BLOCK = """

## 9. 2026-11-27 修订：auto v3（修正“偏绿 / 噪点多”，并修好大图 OOM）

**问题**：用户反馈 493 那张玩偶图（eval15/493.png）用 `--preprocess auto` 后整体偏绿、噪点多；
另外用 `对比_基线vs去马赛克.bat` 处理手机大图时报 `torch.OutOfMemoryError: ... 12.14 GiB is allocated by PyTorch`。

**偏绿根因**（实测 Lab 色度）：
- 在**输入上做色度降噪**（`denoise_chroma`）会让输出 Lab `a*` 从 **+3.67 → −0.75**（绿移 4.4 个单位）。
  原因：Zero-DCE 的 LE 曲线是**逐通道独立**的，输入亮度/色度一变，R/G/B 三通道增益就不同 → 颜色漂移。
- `_rescale_chroma` 末尾的 `np.clip` 在放大色度时会不对称截断，再叠加一点偏色。

**修法（auto v3）**：
1. 预处理阶段**只对亮度降噪**（`luma_denoise`，NLM 只作用 Y，`h=0.8+0.15k`），不再动输入色度 → 消除偏绿主因；
2. 色度降噪挪到**增强之后**、只作用于“低信号平坦区”（局部均值 < 2σ 且局部 std < 0.7σ），强度 `0.45k`，**不做色度回补**；
3. 只有“几乎全黑”（灰度均值 < 4）才轻抬、< 5 才走 E9 中性抬亮，正常夜景不再被替换成灰；
4. **大图（>6 MP）直接分块推理**（1024 块 + 128 重叠），避免整图 fp32 把中间特征图撑到 GB 级；
   `scripts/compare_preprocess.py` 改为复用 `enhance.py` 的 `get_net/enhance_image`，因此同样有 fp16 → 分块的降级，不会再 OOM。

**新数字（`python scripts/compare_preprocess.py ...`，基线 → auto v3）**：

| 图 | 亮度 | 色度噪声 std | 亮度细节 std8 | 暗区条带跳变 |
| --- | --- | --- | --- | --- |
| eval15/493（玩偶+色卡） | 49.0 → 48.2 | 5.82 → **4.12** | 10.96 → 8.20 | 3 → 7 |
| eval15/748（羽毛球场） | 81.8 → 78.7 | 6.27 → **3.54** | 14.73 → 12.46 | 2 → 3 |
| real/126_3_（近黑，均值 2.9） | 16.4 → 18.8 | 1.36 → **0.69** | 4.02 → 3.12 | 12 → **3** |
| LIME/1（威尼斯夜景） | 95.4 → 94.9 | 3.35 → 3.31 | 11.78 → 10.60 | 3 → 3 |

**怎么调**：
- 还是偏绿/太重 → `--auto-strength 0.5`；噪点多想更强 → `--auto-strength 1.3`（默认 1.0）
- 只想要最原始的输出 → `--preprocess light` 或干脆不加 `--preprocess`
- 极暗图想更狠地压噪 → `--preprocess dark`
- 大图（>6 MP）会自动走分块，速度慢一些属正常（12 MP 约 11 s）

**复跑**：

```powershell
python scripts\\compare_preprocess.py 你的图.png -o 对比结果
python enhance.py 你的图.png --preprocess auto -c -o 增强结果
```
"""

p = os.path.join(base, "README.md")
t = io.open(p, encoding="utf-8").read()
if "auto v3" not in t:
    io.open(p, "w", encoding="utf-8", newline="\n").write(t.rstrip() + BLOCK + "\n")
    print("[ok] README 追加 §9")
else:
    print("[skip] README 已有 auto v3")

# 使用指南同步（default-workspace 与 repro\docs 两处）
guide_block = """
## 12. 调参速查（auto v3，2026-11-27）

- **偏绿 / 太重** → `--auto-strength 0.5`（再不行用 `--preprocess light`）
- **噪点多 / 想更强** → `--auto-strength 1.3`（更狠用 `--preprocess dark`）
- **大图（>6 MP）** 自动分块，12 MP 约 11 s；内存不够不会再报 CUDA OOM
- 预处理阶段只对**亮度**降噪（不碰输入色度），色度降噪只在增强后的“低信号平坦区”做 → 不会再整体偏绿
"""
for gp in [r"C:\Users\12983\Documents\deepseek-harness\default-workspace\零-DCE_使用指南.md",
           os.path.join(base, r"docs\零-DCE_使用指南.md")]:
    if os.path.exists(gp):
        g = io.open(gp, encoding="utf-8").read()
        if "调参速查（auto v3" not in g:
            io.open(gp, "w", encoding="utf-8", newline="\n").write(g.rstrip() + "\n" + guide_block)
            print("[ok] 追加调参速查:", gp)
        else:
            print("[skip] 已存在:", gp)

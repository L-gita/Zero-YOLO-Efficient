# Zero-DCE 马赛克预处理方案（方案 B）——实现、用法与对照实验

> 对应代码：`scripts/preprocess.py`（算子）+ `enhance.py`（一键开关）+ `scripts/mosaic_ablation.py`（E0~E5 对照实验）+ `scripts/make_mosaic_figs.py`（配图）
> 配图与数据：`demo_mosaic/ablation/`（`metrics.csv`、`summary.md`、`fig_A_pipeline_metrics.png`、`fig_B_reduction.png`、`strip_*.png`）

---

## 1. 问题是什么（一句话版）

Zero-DCE 的增强曲线在暗区斜率 >1（实测梯度放大 **3.6~6.1 倍**），会把暗光图里本来就存在的三个缺陷一起放大：

| 缺陷 | 表现 | 本次实测（极端暗图 / eval15） |
|---|---|---|
| ① 8-bit 量化台阶 | 色阶断层、一块块“色块” | 极端图暗区占 98.6%，可用灰度级仅个位到几十 |
| ② 色度噪声 | 彩色花斑（最主要的“马赛克”观感） | 色度高频残差 std：极端图 0.88 → **5.22**；eval15 2.22 → **12.18** |
| ③ JPEG 8×8 压缩缝 | 块边界台阶 | 块效应比：极端图 1.06 → **1.14**（>1 即存在 8 网格结构） |

> 结论：马赛克不是 Zero-DCE“制造的”，而是它**放大**了已经存在的缺陷。所以解法是“增强之前先把缺陷压下去”。

---

## 2. 我们的方案（方案 B）：四个预处理 + 主模型 + 一个后处理

```
输入低照图
  → ① 去块（JPEG 8×8 缝）        ← scripts/preprocess.py: deblock()
  → ② 色度优先降噪（色重、亮轻）   ← denoise_chroma()
  → ③ 暗部轻抬（先给网络打个手电）  ← prelift()
  → ④ Zero-DCE 增强（论文曲线）    ← enhance.py -m sice / edge
  → ⑤ 后置降噪 + 保边平滑          ← denoise_chroma(0.6) + post_smooth()
输出（可 16-bit PNG）
```

**为什么顺序不能反**：先降噪再增强（E2/E5），噪声从未被“放大成结构化色块”；先增强再降噪（E3）只能在最终图上抹，梯度放大依旧存在（实测 M 放大 3.85× vs E5 的 1.80×）。

### 各算子的做法与参数

| 算子 | 实现要点 | 默认参数 | 作用 |
|---|---|---|---|
| `deblock` | 只在 8×8 网格边界上做 4 抽头平滑，且仅当跨边界差 < `thr` 时（真边缘不动） | strength=1.0, block=8, thr=12 | 磨平压缩缝 |
| `denoise_chroma` | YCrCb 空间 NLM：Y 通道 h=4（轻），Cr/Cb h=12（重） | strength=1.0 | 压彩色花斑，保亮度细节 |
| `prelift` | LAB 的 L 通道 gamma 轻抬，且只作用于暗部（L<0.6 加权） | amount=0.15 | 降低网络需要的增益 → 少放大噪声 |
| `post_smooth` | 双边滤波，sigmaColor 小 → 只抹低对比噪声 | strength=1.0 | 收尾 |
| `save_16bit` | float→uint16 PNG，cv2 编码后写字节（避开中文路径坑） | — | 避免再被压回 8-bit |

### 三档预设（`enhance.py --preprocess`）

| 预设 | 去块 | 色度降噪 | 暗部轻抬 | 后处理 | 适用 |
|---|---|---|---|---|---|
| `light` | 0 | 0.6 | 0.10 | 0.6 | 一般夜景，怕抹细节 |
| `mosaic`（推荐） | 1.0 | 1.0 | 0.15 | 1.0 | 暗光/压缩重/色噪明显 |
| `strong` | 1.0 | 2.0 | 0.25 | 1.5 | 极端暗图（126 这类） |

---

## 3. 对照实验（E0~E5）

**设置**：同一个模型（`snapshots_sice/Epoch50.pth`，SICE 2002 张训练），只改预处理与顺序；指标全部在“输入暗区（灰度<40）掩码”内统计，避免亮度变化带来的偏差。

| 管线 | 说明 |
|---|---|
| E0 | 低照原图（参考，不增强） |
| E1 | 原图 → Zero-DCE（基线） |
| E2 | 色度降噪 → Zero-DCE（**先降噪**） |
| E3 | Zero-DCE → 色度降噪（后降噪） |
| E4 | 弱降噪 → Zero-DCE → 保边平滑 |
| E5 | 去块+降噪+轻抬 → Zero-DCE → 后置降噪+平滑（**本文方案**） |

### eval15（15 对，有正常光 GT）

| 管线 | 亮度 | 色噪 | 块效应比 | M 放大 | PSNR | SSIM |
|---|---|---|---|---|---|---|
| E0 | 15.2 | 2.218 | 1.006 | 1.00× | 7.77 | 0.1956 |
| E1 基线 | 80.5 | **12.177** | 1.004 | **5.47×** | 15.86 | 0.5584 |
| E2 | 81.2 | 2.692 | 1.021 | 2.10× | 15.94 | 0.6582 |
| E3 | 80.5 | 2.207 | 1.007 | 3.85× | 16.06 | **0.6860** |
| E4 | 81.1 | 2.482 | 1.025 | 2.19× | 15.97 | 0.6772 |
| **E5 本文** | 96.7 | **1.076** | **0.882** | **1.80×** | **16.08** | 0.6573 |

**相对基线 E1 的改善**：色噪 **−91.2%**、块效应比 −12.1%、梯度放大 **−67.1%**；PSNR +0.22 dB，SSIM +0.099。

### LIME（10 张）、极端暗图（3 张）、DICM（8 张）

| 数据集 | 色噪 E1→E5 | 块效应比 E1→E5 | M 放大 E1→E5 |
|---|---|---|---|
| LIME | 8.11 → **1.52**（−81.3%） | 1.022 → 0.788（−22.9%） | 3.88× → **2.05×**（−47.3%） |
| 极端暗图 | 5.22 → **1.34**（−74.4%） | 1.142 → 0.941（−17.6%） | 6.05× → **2.01×**（−66.7%） |
| DICM | 8.74 → **1.17**（−86.7%） | 1.000 → 0.813（−18.7%） | 4.23× → **2.21×**（−47.6%） |

![管线指标对比](C:/Users/12983/Documents/repro/demo_mosaic/ablation/fig_A_pipeline_metrics.png)
![相对基线的改善幅度](C:/Users/12983/Documents/repro/demo_mosaic/ablation/fig_B_reduction.png)

### 视觉对比（放大 3×）

![极端暗图 126：E0~E5](C:/Users/12983/Documents/repro/demo_mosaic/ablation/strip_extreme_126_3.png)
![eval15/748：E0~E5](C:/Users/12983/Documents/repro/demo_mosaic/ablation/strip_eval15_748.png)

### 三条结论（可直接写进论文）

1. **“先降噪再增强”是有效的**：E2 相对 E1 把色噪从 12.18 降到 2.69、梯度放大从 5.47× 压到 2.10×——说明噪声放大确实发生在增强环节，前置降噪能从根本上减少放大。
2. **只做后置降噪不够**：E3 的最终色噪更低（2.21），但**梯度放大仍 3.85×**——它只是把最终图抹平，结构层面的“噪声/对比被放大”没有解决；从视觉上 E3 的亮度噪点与块状感仍比 E5 明显。
3. **本文方案（E5）综合最优**：色噪最低（1.08）、块效应比最低（0.882）、梯度放大最低（1.80×≈ GT 的 3.75× 以下），PSNR 最高（16.08），SSIM 比 E1 高 0.099（略低于 E3 的 0.686，因为重度平滑会牺牲部分纹理——这是“干净”与“细节”的取舍，需在论文里说明）。

---

## 4. 怎么用（命令行 / 程序）

### 一键处理自己的照片

```powershell
cd C:\Users\12983\Documents\repro

# 推荐：整套预处理（去块+降噪+轻抬+后置平滑+16bit 输出）
python enhance.py "D:\照片\夜景.jpg" --preprocess mosaic -c --bit16

# 温和档（一般夜景）
python enhance.py "D:\照片" -o "D:\结果" --preprocess light -c

# 极端暗光（如 126 那种几乎全黑）
python enhance.py "D:\照片\极暗.jpg" --preprocess strong --bit16

# 也可以单独开关（不用预设）
python enhance.py 照片.jpg --denoise 1.0 --deblock 1.0 --prelift 0.15 --post 1.0

# 不开预处理 = 和以前完全一样
python enhance.py 照片.jpg -m sice -c
```

### 在 Python 里调用

```python
import sys
sys.path.insert(0, r"C:\Users\12983\Documents\repro\scripts")
import preprocess as pp
import numpy as np
from PIL import Image

img = np.asarray(Image.open("夜景.jpg").convert("RGB"))
clean = pp.preprocess(img, denoise=1.0, deblock_strength=1.0, prelift_amount=0.15)
Image.fromarray(clean).save("预处理后.png")
```

### 重跑对照实验 / 重画图

```powershell
python scripts\mosaic_ablation.py --sets eval15,lime,extreme,dicm
python scripts\make_mosaic_figs.py
```

---

## 5. 边界与注意事项（必须写进论文“局限”）

1. **预处理不是 ISP**：ISP 在相机内部、RAW 域完成；我们只能处理已经 8-bit + JPEG 压缩后的 sRGB 图，属于“终端侧预处理”。能改善观感，**无法恢复真正丢失的细节**。
2. **重度降噪会牺牲暗部纹理**：`strong` 档在极暗图上可能把细节抹成“塑料感”，需按场景选档。
3. **去块只对 JPEG 源有效**：PNG/RAW 源图块效应本来就接近 1，去块收益很小（实测 DICM 的 1.00→0.81 属于把自然 8 网格也压掉了，避免过头可把 `--deblock` 调 0.5）。
4. **色噪指标会“偏爱平滑”**：E3 的色噪最低但结构放大仍高，所以**必须与 M 放大/PSNR/SSIM 一起看**，不能只看单一指标。
5. **16-bit 输出需要软件支持**：Windows 照片查看器、微信会压回 8bit；用 Photoshop/GIMP/Python 查看才保留位深。

---

## 6. 产物清单

| 路径 | 内容 |
|---|---|
| `scripts/preprocess.py` | 四个算子 + 组合管线 + 16-bit 保存 |
| `enhance.py` | 新增 `--preprocess/--denoise/--deblock/--prelift/--post/--bit16` |
| `scripts/mosaic_ablation.py` | E0~E5 对照实验，输出 CSV + summary.md |
| `scripts/make_mosaic_figs.py` | 指标柱状图、改善幅度图、三组视觉条带 |
| `demo_mosaic/ablation/metrics.csv` | 36 张图 × 6 管线的全部指标（可进 Excel） |
| `demo_mosaic/ablation/summary.md` | 分组均值表 |
| `demo_mosaic/ablation/fig_A_*.png` / `fig_B_*.png` | 两张汇报图 |
| `demo_mosaic/ablation/strip_*.png` | 3 组放大 3× 视觉对比 |
| `demo_mosaic/mosaic_metrics.csv` | 早期马赛克成因诊断（4 张图） |
| `demo_mosaic/mosaic_01_extreme_126_3.png` | 马赛克成因示意图（输入 vs 增强） |

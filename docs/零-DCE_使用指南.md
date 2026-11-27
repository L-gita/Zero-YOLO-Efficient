# Zero-DCE 使用指南：拿自己的照片做增强

你手上已经有完整跑通的一套东西，用起来其实就是"喂图片、拿结果"。下面按从简单到灵活排三种用法。

## 0. 先搞清楚三样东西在哪

| 东西 | 位置 | 说明 |
|---|---|---|
| 环境 | `C:\Users\12983\Documents\repro\.venv` | Python + torch（已装好，不用管） |
| 代码 | `C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code` | 官方 Zero-DCE，已改成 Windows 能跑 |
| 权重 | 同上目录下 `snapshots\Epoch99.pth`（官方）、`snapshots_sice\Epoch50.pth`（我训的，推荐） | 真正决定效果的模型 |
| 一键脚本 | `C:\Users\12983\Documents\repro\enhance.py` | 我新写的，专门给你用自己的照片 |
| 三个版本入口 | `1_Zero原版.bat` / `2_边缘增强版.bat` / `3_去马赛克最终版.bat` | 见 §13，日常用这个 |
| 三个版本入口 | `1_Zero原版.bat` / `2_边缘增强版.bat` / `3_去马赛克最终版.bat` | 见 §13 |

## 1. 最省事：双击 → 选图 → 看结果（一次出两个版本）

1. 打开 `C:\Users\12983\Documents\repro\`，双击 **`3_去马赛克最终版.bat`**（或另外两个版本入口）；
2. 弹出选图窗口，按住 `Ctrl` 可以多选（一次几十张也行），也可以直接把照片/文件夹拖到 bat 上；
3. 点"打开"，黑窗口里会跑进度（每张 0.1~27 秒不等，见下文）；
4. 跑完会自动弹出结果文件夹 `增强结果_原版与edge\`，**每张图 4 个文件**：
   - `xxx_原版增强.png` —— **原版 Zero-DCE 的成片**（可以单独拿走用）
   - `xxx_edge增强.png` —— **加边缘损失后的成片**（改进版，可以单独拿走用）
   - `xxx_AB对比.png` —— 原图 | 原版 | edge 三列并排，下排自动选"梯度最强区域"放大 2×
   - `xxx_指标.png` —— 亮度 / 梯度幅值 M / L_edge / cos / corr 五项指标柱状对比
   另外还有 `overview_AB.png`（全部图总览）、`metrics.csv`（逐图数值）、`summary.md`（均值表）。
5. 结果默认存在**你选的第一张照片所在目录**下的 `增强结果_原版与edge` 文件夹里。

> 两个模型权重都是 50 轮训练（原版 `snapshots_sice\Epoch49.pth` vs edge `snapshots_sice_edge\Epoch49.pth`），
> **只差边缘损失这一项**，所以对比才公平。
>
> "双击选图"的弹窗分支我只做了语法检查 + 后台调用测试，没能自动帮你点那个弹窗；
> 如果双击后没反应，直接用第 2 或第 3 种方法（一样的）。

## 2. 拖拽：把照片拖到 bat 上

把一张、多张照片，或者**整个文件夹**拖到 `1_Zero原版.bat` / `2_边缘增强版.bat` / `3_去马赛克最终版.bat` 图标上松手，就会自动跑完并弹出结果目录，
**一次同时生成原版增强和 edge 版增强两张图**（外加 AB 对比图与指标图）。
（拖文件夹会递归处理里面所有 jpg/png/bmp/tif/webp。）

只要其中一个版本时，拖到对应的 bat：
- `1_Zero原版.bat` → 只出 `xxx_enhanced.png`（原版）+ `xxx_compare.png`
- `2_边缘增强版.bat` → 只出边缘增强版 + 左右对比图

### 2.1 想把 bat 放到桌面用（必须改绝对路径）

repro 里的 bat 用 `%~dp0` 找 `scripts\entry.ps1`，**单独复制到桌面就打不开**（要连同 repro 一起用）
（`%~dp0` 变成桌面，ps1 不在那儿，窗口闪一下就关）。桌面版要把路径写死：

```bat
@echo off
chcp 65001 >nul
setlocal
set "REPRO=C:\Users\12983\Documents\repro"
set "ZERODCE_MODE=base"
powershell -NoProfile -ExecutionPolicy Bypass -File "%REPRO%\scripts\entry.ps1" %*
```

`entry.ps1` 是用 `$PSScriptRoot` 反推 `.venv\Scripts\python.exe`
和 `enhance.py` 的，所以**它必须留在 repro 目录里**，桌面只放 bat。
`ZERODCE_MODE` 三档：`both`（默认，一次出原版+edge+AB+指标）、`base`（只原版）、`edge`（只 edge）。
repro 搬到别处时只改 `REPRO=` 一行。

## 3. 命令行 / VSCode（最灵活）

```powershell
$py = "C:\Users\12983\Documents\repro\.venv\Scripts\python.exe"
$enh = "C:\Users\12983\Documents\repro\enhance.py"

# 单张
& $py $enh "D:\照片\夜景.jpg"

# 整个文件夹，指定输出目录，带左右对比图
& $py $enh "D:\照片\全家福" -o "D:\增强结果" -c

# 用作者原始权重（更保守，不容易把本来够亮的地方提过头）
& $py $enh "D:\照片\夜景.jpg" -m official -c

# 手机大图先缩到长边 2000 再处理（省显存、快很多）
& $py $enh "D:\照片" -s 2000 -c
```

### 参数速查

| 参数 | 作用 |
|---|---|
| `-o DIR` / `--out` | 输出目录（默认：图片旁边的 `enhanced` 文件夹） |
| `-m NAME` / `--model` | `sice`（默认，推荐）/ `official`（作者原版权重）/ `edge`（加了论文 Sobel 边缘损失的版本）/ `lol`（过曝对照版，别用）或直接写 `.pth` 路径 |
| `-c` / `--compare` | 额外输出"原图 \| 增强"左右对比图 |
| `-j` / `--jpg` | 强制输出 JPEG（默认：输入是 jpg 就出 jpg，其它出 png） |
| `-s N` / `--scale` | 先把长边缩到 N 像素再处理，例如 `-s 2000` |
| `--tile` | 强制分块处理（显存不够时用；一般不用手动加，程序会自动降级） |

## 4. 三个权重怎么选

| 权重 | 特点 | 什么时候用 |
|---|---|---|
| `sice`（默认） | 我用论文真正的训练数据（SICE 多曝光 2002 张）从零训的；官方 LOL eval15 上 PSNR 15.81 / SSIM 0.557（官方是 14.80 / 0.556） | **日常首选**，暗部提得更到位 |
| `edge` | 在 `sice` 基础上加了论文 (4.9)~(4.12) 的 Sobel 边缘损失（λ=0.22，同样 50 轮） | 想看"保护边缘"版本的效果；对比细节请跑 `scripts\compare_models.py` |
| `official` | 作者发布，曲线更保守，输出更暗一些 | 照片本来就半亮（比如室内灯光片），不想提太过的时候 |
| `lol` | 只训过 LOL 485 张低照图，会把亮部推爆 | **只作反面教材**，不建议用 |

经验：**越暗的图越显效果**；白天/灯光充足的照片喂进去，两个模型都会偏亮，属于正常（这个网络没有"是否需要增强"的判断）。

## 5. 效果大概是什么样

实测（同一套权重，亮度是 0~255 的平均值）：

| 图片 | 原图亮度 | 增强后 |
|---|---|---|
| 24.9（书架，室内昏暗） | 24.9 | 94.5 |
| 9.3（几乎没有光的房间） | 9.3 | 63.5 |
| 12.6（游泳馆夜间） | 12.6 | 82.3 |
| LIME 1（威尼斯夜景） | 33.0 | 95.9 |
| DICM 01（埃菲尔铁塔夜景） | 20.3 | 51.6 |

![示例：夜间游泳馆 原图|增强](C:\Users\12983\Documents\repro\enhanced_test\748_compare.png)

## 6. 速度与显存

- 600×400 这种小图：**0.05 秒**
- 1200×900：约 0.3 秒
- **12MP 手机照片（4000×3000）：约 27 秒**（占用显存约 2~3 GB）
- 处理超大图时如果显存不够，程序会自动先试 fp16、再不行就自动分块，不会直接崩
- 机器上如果还开着游戏（会抢 GPU），速度可能慢好几倍；想快就先关掉

## 7. 常见问题

| 现象 | 原因 / 解决 |
|---|---|
| 结果偏亮、发白 | 照片本来不算暗；换 `-m official`，或少提亮（网络只能提亮，不能压暗） |
| 结果还是太暗 | 极暗图可用 `-m sice`，或者把输出的 `_enhanced` 再喂一次（会继续提亮，效果更激进） |
| 报"找不到权重文件" | 别移动 `Zero-DCE\Zero-DCE_code\snapshots*` 目录 |
| 中文路径/空格 | 没问题，脚本已处理 |
| RAW（.CR2/.NEF/.dng） | 不支持，先用相机软件或系统照片应用导出成 jpg/png |
| 手机 HEIC | 不支持，先在 Windows 照片应用里"另存为 jpg" |
| 想改回默认输出位置 | 加 `-o 你的目录` |

## 8. 在 VSCode 里跑

打开 `C:\Users\12983\Documents\repro\enhance.py`，解释器选
`C:\Users\12983\Documents\repro\.venv\Scripts\python.exe`（已配置好），
在"运行和调试"里给参数，例如 `"D:\照片" -c -o "D:\增强结果"`，就能断点调试。

---

## 9. 进阶：加入边缘损失的"改进版" Zero-DCE 怎么用

改进版和原版共用同一份推理代码，**只是多了一个权重文件**，用 `-m edge` 选它：

```powershell
# 改进版（加了论文 4.9~4.12 的 Sobel 边缘损失）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg -m edge -c

# 不加边缘损失的基线版（对比用）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg -m sice -c
```

- 改进版权重：`Zero-DCE_code\snapshots_sice_edge\Epoch49.pth`（50 轮、λ=0.22）
- 基线权重：`Zero-DCE_code\snapshots_sice\Epoch50.pth`
- 实测（eval15 十五张平均，数据在 `demo_edge\ep49_edge022\metrics.csv`）：基线亮度 97.4 → 改进版 **87.3**（−10.4%），PSNR 17.22→16.41 dB，SSIM 0.5509→0.5557；梯度方向一致性 +1.8%、幅值相关性 +2.9%。即改进版**更暗、更平**（不是更亮）。

**想自己重新训练（可随时开关）：**

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code

# 带边缘损失训练（λ=0.22）
python lowlight_train.py --lowlight_images_path data/train_data_sice/ `
    --num_workers 0 --num_epochs 50 --snapshots_folder snapshots_mine/ --edge_weight 0.22

# 完全关掉 = 原版 Zero-DCE（默认就是 0，行为/日志与原来一模一样）
python lowlight_train.py --lowlight_images_path data/train_data_sice/ `
    --num_workers 0 --num_epochs 50 --snapshots_folder snapshots_mine_base/
```

新权重可直接推理：`python enhance.py 照片.jpg -m C:\...\snapshots_mine\Epoch49.pth`

---

## 10. 进阶：给任意两张图出"指标 + 曲线图"

工具：`C:\Users\12983\Documents\repro\scripts\pair_compare.py`（输入两张图，输出一页卡片 + 梯度图 + CSV）

```powershell
# ① 有正常光参考图时：低照输入 vs 正常光 GT（会算 PSNR/SSIM）
python C:\Users\12983\Documents\repro\scripts\pair_compare.py `
    低照.png 正常光.png --label-a 低照输入 --label-b 正常光GT -o 对比结果\case1

# ② 评估一次增强：原图 vs 增强结果（无 GT，重点看梯度类指标）
python C:\Users\12983\Documents\repro\scripts\pair_compare.py `
    夜景.jpg enhanced\夜景_enhanced.png --label-a 低照原图 --label-b 增强结果

# ③ 两个模型对比：基线结果 vs 加边缘损失结果
python C:\Users\12983\Documents\repro\scripts\pair_compare.py `
    基线.png 加边缘.png --label-a 基线 --label-b 加边缘损失
```

输出目录里会有：

| 文件 | 内容 |
|---|---|
| `fig_metrics_card.png` | 一页卡片：两图并排 + 8 项指标 + 亮度/梯度柱状 + 边缘保持曲线 + \|ΔM\| 热力图 |
| `fig_gradient_maps.png` | 两图梯度幅值 M、差异热力图、\|ΔM\| 分布直方图 |
| `metrics.csv` | 全部数值（可进 Excel） |
| `summary.md` | 指标表 + 可直接贴报告的图注 |

指标口径：`L_edge` 是论文 (4.12) 的逐通道 Sobel 幅值差；`cos` 是梯度方向一致性（1=同向）；`放大倍数 M(B)/M(A)` 越接近 1 表示梯度尺度越接近。**两图尺寸不同会自动把 B 缩放到 A 并提示。**

---

## 11. 进阶：马赛克预处理（暗光 / 压缩重的图）

暗光图增强后容易出现彩色花斑或块状"马赛克"——这是把原图已有的 8-bit 台阶、色度噪声、JPEG 8×8 压缩缝一起放大了；
而且近黑区没有可靠信号，模型还可能把噪声放大成**棕/绿/紫色大斑块**。

现在的方案：先做 **E10 场景自适应（`--preprocess auto`，推荐）**，旧的 E9 固定管线保留在 `mosaic` 档。

旧方案（E9）：
**预处理（去块 → 色度优先降噪 → 暗部轻抬）→ Zero-DCE → 后处理（亮度引导色度平滑 → 近黑区限增益 v2）**。
限增益 v2 的三个关键点：**平滑亮度**（避免 8-bit 台阶变硬灰块）、**羽化掩码**、**输出抖动 ±0.6 级**（抑制条带）。

```powershell
# 推荐档（E10 场景自适应：自动判断要不要降噪/去块/轻抬，正常夜景不会被洗淡）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg --preprocess auto -c --bit16

# 旧版固定管线（E9：极暗图压得更狠，但正常夜景会发灰、天空可能有色块）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg --preprocess mosaic -c --bit16

# 温和档（怕抹细节、怕偏淡）/ 极暗档（几乎全黑）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg --preprocess light -c
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg --preprocess dark --bit16

# 单项手动控制（0 = 关闭）+ 细调
python C:\Users\12983\Documents\repro\enhance.py 照片.jpg --denoise 0.5 --deblock 0.5 --prelift 0.10 --guided 0.5 --limit-dark 0.80
python C:\Users\12983\Documents\repro\enhance.py 照片.jpg --preprocess mosaic --dark-gamma 0.50 --dark-smooth 10 --dither 0.8

# 不加参数 = 和以前完全一样（默认全关）
python C:\Users\12983\Documents\repro\enhance.py 照片.jpg -m sice -c
```

效果（eval15 15 对，相对"直接增强"基线）：

| 指标 | 基线 E1 | E8（旧，硬灰块） | **E9（推荐）** |
| --- | --- | --- | --- |
| 色度噪声 | 12.18 | 1.15 | **1.60** |
| 条带跳变（越小越好） | 2.5 | 9.0 | **2.8** |
| 极端暗图条带跳变 | 6.0 | 8.7 | **2.7** |
| 彩色大斑块（专项诊断） | — | 0.21%（126） | 0.23% |
| PSNR / SSIM | 15.86 / 0.5584 | 16.02 / 0.6978 | 16.02 / 0.6754 |

**注意取舍：** E9 在极暗图上收益最明显（彩色斑和硬灰块都没了），但正常夜景和基线差别不大、会**略偏淡**
（饱和度 748：0.051 → 0.034）；想保颜色就用 `--preprocess light` 或调小 `--prelift`。
早些版本的重降噪（E5）会把近黑区抹成大片彩色斑、图像发糊，**已被证伪，别再用于极暗图**。

> 全部对照（E0~E9）、专项诊断、参数说明见 `C:\Users\12983\Documents\repro\docs\零-DCE_马赛克预处理方案.md`；
> 对比图 `demo_mosaic/e9_check/final/compare_E9_vs_base.png`。

**注意：** `--bit16` 会另存一份 `_enhanced16.png`（16 位），但微信/照片查看器会压回 8 位；用 Photoshop/GIMP 才能看到位深差别。

**效果不对时怎么调（2026-11-27 更新，auto 后处理 v2）**

- 觉得"深色区变黑、细节/颜色被洗淡"→ `--auto-strength 0.5`（或 `0` 完全不做后处理）
- 觉得"还脏、马赛克还在"→ `--auto-strength 1.5`
- 想用最保守的固定管线（只轻微降噪）→ `--preprocess light`
- 想用旧版 E9（极端暗图很干净、但正常夜景会发灰）→ `--preprocess mosaic`
- 输出 16bit PNG 避免 8bit 条带 → 加 `--bit16`

## 12. 调参速查（auto v3，2026-11-27）

- **偏绿 / 太重** → `--auto-strength 0.5`（再不行用 `--preprocess light`）
- **噪点多 / 想更强** → `--auto-strength 1.3`（更狠用 `--preprocess dark`）
- **大图（>6 MP）** 自动分块，12 MP 约 11 s；内存不够不会再报 CUDA OOM
- 预处理阶段只对**亮度**降噪（不碰输入色度），色度降噪只在增强后的“低信号平坦区”做 → 不会再整体偏绿

## 13. 三个版本入口（2026-11-27 整理后：根目录只剩这 3 个 bat）

| 入口（双击 / 拖拽） | 是什么 | 输出目录（在第一张图旁边） | 每张产物 |
| --- | --- | --- | --- |
| `1_Zero原版.bat` | **Zero 原版**：原始 Zero-DCE，不加任何改进（`-m sice`） | `Zero原版结果\` | `xxx_enhanced.png` + `xxx_compare.png` |
| `2_边缘增强版.bat` | **只有边缘增强**：论文 (4.9)~(4.12) 的 Sobel 边缘损失权重（`-m edge`），不含马赛克预处理 | `边缘增强结果\` | `xxx_enhanced.png` + `xxx_compare.png` |
| `3_去马赛克最终版.bat` | **去马赛克最终版** = 之前说的 **auto v3**（`--preprocess auto`），不含边缘损失 | `去马赛克最终版结果\` | `xxx_enhanced.png` + `xxx_compare.png` |

> **名词对照（同一个东西的几种叫法）**：
> `去噪增强` = `去马赛克增强` = `--preprocess auto` = `auto v3` = 现在的 **`去马赛克最终版`**。
> 「边缘增强版」是另一条线：只加 Sobel 边缘损失、不做去马赛克预处理；两者互不包含。

用法（三选一）：
1. **拖拽**：把照片（或整个文件夹）拖到对应的 `.bat` 图标上松手；
2. **双击**：弹选图窗口，按住 Ctrl 可多选；
3. **命令行**：`"C:\Users\12983\Documents\repro\3_去马赛克最终版.bat" "D:\照片\a.jpg"`

文件夹整理（2026-11-27）：
- 根目录现在**只有这 3 个版本 bat**；
- `工具\`：`保存到Git.bat`、`推送到GitHub.bat`、`推送到GitHub_输入token.bat`、`open_vscode.bat`；
- `_archive\旧入口_20261127\`：旧版各种 bat（原版增强、去噪增强、对比、增强我的照片、旧 E9、回滚插件等），要用还能翻出来；
- 三个入口的统一实现：`scripts\entry.ps1`（旧的 `scripts\three_entries.ps1` 已归档）。

说明：
- 三个入口都不含「边缘损失 + 马赛克预处理」的组合版；需要更强去噪可用命令行 `--preprocess dark`，想调轻用 `--auto-strength 0.5`。
- 大图（>6 MP）自动分块推理，12 MP 约 11 s。
- 想对比两个版本：`python scripts\compare_preprocess.py 原图 改进图`，或 `enhance.py -c` 生成左右对比。

### 12.1 默认已是"保守档"（2026-11-27）

`--preprocess auto` 默认 `--auto-strength -1`：普通暗图自动用 0.45（保纹理、少出大片色块），
近黑图（均值<5）自动用 1.0（压彩色斑）。想手调：

```powershell
python enhance.py 图.jpg --preprocess auto --auto-strength 1.0   # 旧的全强度（更干净但更平滑）
python enhance.py 图.jpg --preprocess auto --auto-strength 0     # 完全不做预处理与后处理（等价原版）
python enhance.py 图.jpg --preprocess auto --auto-strength 0.25  # 更保守
```

### 12.2 v4 原型（A+B 色度去噪）

```powershell
python enhance.py 图.jpg --preprocess v4 -c                 # 默认强度 0.9
python enhance.py 图.jpg --preprocess v4 --v4-strength 1.3   # 更强（更干净，可能略糊）
```
思路：只在"色度在抖、亮度没有对应结构"的地方处理色度，并用亮度引导滤波把权重羽化，
所以不会沿色度等值线切出人造分界线。适合"结构/轮廓比绝对干净更重要"的抓取场景。

### 12.3 默认已改为"保颗粒"档（2026-11-27）

默认 `--auto-strength -1`：普通暗图用 **0.20** 且 **不碰色度**（`--chroma-mode none`），近黑图（均值<5）仍用 1.0。
- 优点：**不会出现大片色块和分界线**，适合识别/抓取；
- 代价：彩色颗粒噪声保留（比强降噪版脏一点）；
- 想要更干净（可能出色块）：`--auto-strength 0.45 --chroma-mode smooth`；
- 想看"全局去饱和而不造平台"：`--auto-strength 0.25 --chroma-mode desat`。

### 12.4 阴影补偿（`--shadow-lift`，2026-11-27）

很暗的图（输入均值 < 15）增强后暗部会略变暗（约 1~1.5 级）。默认已自动补偿 4 级（只抬 Y 通道暗部，`lref=60` 以上不抬）。
- 关掉：`--shadow-lift 0`
- 加钱/减钱：`--shadow-lift 2`（更轻）/ `--shadow-lift 6`（更亮）
- 亮图（均值 ≥ 15）自动不启用，不影响观感。

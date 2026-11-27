# Zero-DCE 复现环境（Windows 11 + RTX 4060 Laptop）

本目录是 Li-Chongyi/Zero-DCE 官方代码在本机跑通的完整工作区。

## 1. 目录结构

```
C:\Users\12983\Documents\repro\
├─ py311\python\python.exe        # 绿色版 CPython 3.11.17（python-build-standalone）
├─ .venv\                         # 虚拟环境：numpy 1.26.4 / pillow 11.1.0 /
│                                 #   opencv-python 4.10.0.84 / torch 2.5.1+cu124 /
│                                 #   torchvision 0.20.1+cu124
├─ PortableGit\                   # 便携版 git 2.56.0
├─ wheels\                        # 离线 torch 轮子（2.4 GB，留作备份）
├─ installers\                    # VSCode 安装包
├─ lol_repo\                      # LOL 数据集原始仓库（485 对训练 png）
├─ lol_test\                      # LOL 官方测试集 eval15（15 对真实低照/正常光参考图）
├─ sice_repo\                     # 含 2002 张 SICE 多曝光训练图的仓库（论文真正用的数据）
├─ demo_pairs\                    # eval15 增强对比图（pair_*.png / overview_*.png / metrics.csv）
├─ Zero-DCE\                      # 官方代码
│  └─ Zero-DCE_code\
│     ├─ data\train_data\         # 485 张低照度 jpg（LOL，仅作 Zero-DCE_Low 消融用）
│     ├─ data\train_high\         # 485 张对应正常光 jpg（仅用于算指标）
│     ├─ data\train_data_sice\    # 2002 张 512×512 多曝光图（论文配方，推荐训练用）
│     ├─ data\test_data\LIME\     # 10 张官方测试图（bmp）
│     ├─ data\test_data\DICM\     # 64 张官方测试图（jpg）
│     ├─ data\result\             # 推理输出（已跑过官方 Epoch99.pth）
│     ├─ snapshots\Epoch99.pth    # 官方预训练权重（未改动）
│     ├─ snapshots_run\           # LOL 数据训练 200 epoch 的权重
│     └─ snapshots_sice\          # SICE 数据训练 50 epoch 的权重（推荐，最接近论文）
└─ scripts\                       # prepare_data.py / dl_chunked.ps1 / eval_lol.py / demo_pairs.py / make_montage.py / make_compare3.py / diag_*.py
```

## 2. 激活环境

PowerShell 里：

```powershell
& C:\Users\12983\Documents\repro\.venv\Scripts\Activate.ps1
```

或在 VSCode 里打开 `C:\Users\12983\Documents\repro\Zero-DCE` 文件夹，
Python 解释器已预先写进 `.vscode\settings.json`。

## 2.1 系统版 Python 3.11.9 也能直接用（不用激活 venv）

机器上另装了一个**正常安装的** Python 3.11.9：

```
C:\Users\12983\AppData\Local\Programs\Python\Python311\python.exe
```

它已经装好了全套依赖（torch 2.5.1+cu124 / torchvision 0.20.1+cu124 / numpy 1.26.4 /
pillow 11.1.0 / opencv-python 4.10.0.84，`pip check` 无冲突），并且已排进 PATH 最前面。
所以**新开一个终端**（cmd / PowerShell / VSCode 集成终端都行），不需要激活任何环境：

```powershell
python --version                     # → Python 3.11.9
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
python lowlight_test.py              # 直接跑官方推理
python C:\Users\12983\Documents\repro\enhance.py "D:\照片" -o "D:\增强结果"
```

两个环境（旧 `.venv` 与系统 Python）依赖版本完全一致，随便用哪个都行；
`.venv` 保留不动，老命令照旧可用。

## 3. 跑推理（官方预训练权重）

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
python lowlight_test.py
```

输出在 `data\result\DICM\` 与 `data\result\LIME\`。

## 3.5 增强自己的照片（日常用法）

根目录只有三个版本入口，双击或把照片/文件夹拖上去即可（详见 §10）：

| bat | 版本 | 输出目录 |
| --- | --- | --- |
| `1_Zero原版.bat` | Zero 原版（不加任何改进） | `Zero原版结果\` |
| `2_边缘增强版.bat` | 只有边缘增强（Sobel 边缘损失） | `边缘增强结果\` |
| `3_去马赛克最终版.bat` | 去马赛克最终版（auto v3 预处理） | `去马赛克最终版结果\` |

每张图出两个文件：`xxx_enhanced.png`（成片）、`xxx_compare.png`（左右对比）。

命令行方式：

```powershell
$py = "C:\Users\12983\Documents\repro\.venv\Scripts\python.exe"
# Zero 原版
& $py C:\Users\12983\Documents\repro\enhance.py "D:\照片" -o "D:\结果" -c -m sice
# 边缘增强版
& $py C:\Users\12983\Documents\repro\enhance.py "D:\照片" -o "D:\结果" -c -m edge
# 去马赛克最终版（auto v3）
& $py C:\Users\12983\Documents\repro\enhance.py "D:\照片" -o "D:\结果" -c --preprocess auto
```

`-m` 可选 `sice`（默认）/ `edge` / `official`（作者权重，更保守）/ `lol`（过曝对照）。
详细说明见 `..\..\deepseek-harness\default-workspace\零-DCE_使用指南.md`。
12MP 手机照约 27 秒/张（自动分块），小图 0.05~0.6 秒。

## 4. 从零训练

**推荐：用论文真正的训练数据（SICE 多曝光，2002 张）**

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
python lowlight_train.py --lowlight_images_path "data/train_data_sice/" `
    --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_sice/
```

RTX 4060 Laptop 上约 55 秒/epoch（250 iters），100 epoch 约 1.5 小时。
本机已跑到 `snapshots_sice/Epoch50.pth`，实测在 LOL 485 对上 SSIM 0.6007 / PSNR 15.06，
与官方权重（0.5878 / 14.23）同一水平（详见 `..\..\deepseek-harness\default-workspace\Zero-DCE_复现报告.md`）。

**对照实验：只用 LOL 485 张低照图（= 论文 Fig.6 的 Zero-DCE_Low）**

```powershell
python lowlight_train.py --num_workers 0 --num_epochs 200 --snapshots_folder snapshots_run/
```

约 9 秒/epoch，全程约 30 分钟。结果会明显过曝（LIME 图亮度 33→161，官方只有 85）。

> ⚠️ `--num_workers` 必须是 **0**。用默认 4 在本机报
> `RuntimeError: Couldn't open shared file mapping: ..., error code: <1455>`
> （页面文件/共享内存不足），而且训练速度会掉到 1/10。

## 4.5 可选：论文 (4.9)~(4.12) 的 Sobel 边缘损失（默认关闭，可随时去掉）

`Myloss.py` 末尾新增了 `SobelMag` / `L_edge` 两个类，`lowlight_train.py` 新增两个参数：

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--edge_weight` | `0.0` | 边缘损失权重 λ；**0 = 完全关闭，训练行为与原版 Zero-DCE 一模一样** |
| `--edge_exact` | 关闭 | 用精确幅值 `sqrt(Ix²+Iy²)`（论文 4.10）；默认用近似 `|Ix|+|Iy|`（4.11） |

公式：`L_edge = 1/(3N) Σ_{c∈RGB} Σ_i | M(I_low^c)_i − M(I_enh^c)_i |`（零参考，不需要 GT）。
λ 的标定：在 SICE 256×256 上实测 `L_edge≈0.13~0.16`、原总 loss≈0.70，
所以 **λ=0.22 时边缘项约占 5%**（推荐起点 0.2~0.5，别填几十）。

**开启（λ=0.22，50 轮约 25 分钟）：**

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
python lowlight_train.py --lowlight_images_path "data/train_data_sice/" `
    --num_workers 0 --num_epochs 50 --snapshots_folder snapshots_sice_edge/ --edge_weight 0.22
```

日志会同时打印 base 与 edge 两部分：

```text
Loss at iteration 250 : 1.06155 | base 1.01853 | edge 0.04302
```

**关掉（恢复原版 Zero-DCE）：** 不写 `--edge_weight`（默认 0）或显式 `--edge_weight 0`。
此时 `L_edge` 不会被实例化、不参与计算，速度/显存/数值与原版完全一致。

**两种结果一起看：**

```powershell
cd C:\Users\12983\Documents\repro
python scripts\compare_models.py --tag ep49_edge022
```

输出在 `demo_edge\ep49_edge022\`：每张图并排 4 列（低照输入 | 正常光 GT | 基线 | +边缘损失），
外加 `metrics.csv`（PSNR/SSIM/亮度/平均梯度幅值/L_edge）、`summary.md` 和 `overview_*.png` 总览。

**实测结论（λ=0.22，SICE 2002 张 50 轮，`snapshots_sice\Epoch49` vs `snapshots_sice_edge\Epoch49`）：**
`L_edge` 41/41 张全部下降（平均 −9.4%），梯度方向 cos +1.8%、幅值 corr +2.9%；
代价是亮度 −10.4%、eval15 PSNR −0.81 dB、SSIM +0.0048 —— 是"结构正则项"而非锐化项。

**图表（汇报用，9 张）：**

```powershell
python scripts\make_edge_figures.py     # → demo_edge\figures\fig1...fig9 + summary_for_report.md
```

**任意两张图出"指标 + 曲线图"（通用小工具）：**

```powershell
# 低照输入 vs 正常光 GT / 原图 vs 增强结果 / 基线 vs 加边缘损失 —— 三种用法都行
python scripts\pair_compare.py 低照.png 正常光.png --label-a 低照输入 --label-b 正常光GT -o 对比结果\case1
```

输出 `fig_metrics_card.png`（一页卡片：两图 + 8 项指标 + 柱状 + 边缘保持曲线 + 热力图）、
`fig_gradient_maps.png`（梯度幅值图与差异分布）、`metrics.csv`、`summary.md`。

### 实测结果（λ=0.22，两边都训练 50 轮，SICE 2002 张）

对比用的是 `snapshots_sice/Epoch49.pth`（基线）与 `snapshots_sice_edge/Epoch49.pth`（+边缘损失），
完整数据与图见 `C:\Users\12983\Documents\deepseek-harness\default-workspace\零-DCE_边缘损失实验报告.md`，
图片在 `demo_edge\ep49_edge022\`。

| eval15（15 张，有 GT） | 低照输入 | 基线 | +边缘损失 |
| --- | --- | --- | --- |
| 平均亮度 | 15.15 | 97.40 | 87.30 |
| PSNR (dB) | 7.77 | 17.22 | 16.41 |
| SSIM | 0.1956 | 0.5509 | 0.5557 |
| 平均梯度幅值 M | 0.0595 | 0.3488 | 0.3217 |
| `L_edge`(4.12) | — | 0.2900 | **0.2627** |

- LIME 10 张 / DICM 16 张上 `L_edge` 同样下降 8.6% / 10.4%（0.3388→0.3096、0.2834→0.2539）。
- 剔除亮度影响后（把增强图缩放回输入亮度再算 4.12，并算梯度向量余弦相似度）结论仍成立：梯度场更贴近输入。
- 代价：整体变暗约 10%、eval15 PSNR −0.81 dB、SSIM +0.0048。λ 偏大时该冲突更明显，
  想保留效应又少掉亮度，建议试 `--edge_weight 0.05~0.10`。
- 训练日志里边缘项实测占总损失 4%~6%，与标定一致。

## 5. 相对官方仓库改了哪些文件

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `dataloader.py:51-52` | `Image.ANTIALIAS` → `getattr(Image,"ANTIALIAS",Image.LANCZOS)` | Pillow 10 起删除了 `ANTIALIAS` |
| `dataloader.py:48` | `Image.open(...)` → `.convert('RGB')` | 防止灰度图导致 `permute(2,0,1)` 报错 |
| `lowlight_test.py:22` | 同上 `.convert('RGB')` | 同上 |
| `lowlight_test.py:40-43` | 用 `os.path.dirname` 建输出目录 | Windows 下 glob 返回的路径混用 `/`、`\`，原写法建不出子目录 |
| `lowlight_train.py:32` | 加 `cudnn.benchmark = True` | 输入尺寸固定，让 cuDNN 选最快卷积算法 |
| `Myloss.py` 末尾 | 新增 `SobelMag` / `L_edge` 两个类 | 论文 (4.9)~(4.12) 的 Sobel 边缘损失 |
| `lowlight_train.py` | 新增 `--edge_weight` / `--edge_exact`、`use_edge` 分支、base+edge 日志 | 默认 0 = 行为与原版一致；改参数即可开关 |
| 新增 `snapshots_run/`、`snapshots_sice/`、`snapshots_sice_edge/` | 训练权重单独放 | 避免覆盖官方 `snapshots/Epoch99.pth` |

## 6. 数据来源

官方仓库的 `data/train_data` 并不在 git 里（README 给的是 Google Drive / 百度网盘，本机不可达）。

**关键：官方那份 train_data 不是 LOL 485 张，而是 SICE Part1 的多曝光子集**（论文 4.1 节：3022 张里取 2422 张，
512×512，刻意同时包含低照与过曝图像）。所以：

- LOL 485 对：`https://github.com/23225114YKW/Low-light-image-enhancement-with-LoL-dataset`
  （`traindata/low`、`traindata/high` 即 LOL 的 485 对）→ `data/train_data`、`data/train_high`
- SICE 多曝光 2002 张（社区镜像，512×512、亮度 0.01~0.97）：
  `https://github.com/Aiemu/Zero-DCE-improved` 的 `src/data/train_data` → `data/train_data_sice`
- `data/test_data/LIME`、`data/test_data/DICM` 与 `snapshots/Epoch99.pth` 由官方仓库自带

用 LOL 训练会得到论文 Fig.6 的过曝版本（Zero-DCE_Low），用 SICE 训练才等价于论文主模型。

## 7. 离线重装（网络再被墙时）

```powershell
& C:\Users\12983\Documents\repro\.venv\Scripts\pip.exe install --no-index --no-deps `
  C:\Users\12983\Documents\repro\wheels\torch-2.5.1+cu124-cp311-cp311-win_amd64.whl `
  C:\Users\12983\Documents\repro\wheels\torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl
```

普通小依赖走 `-i https://mirror.sjtu.edu.cn/pypi/web/simple`。

## 8. 马赛克预处理（方案 B，E9 推荐版）

Zero-DCE 的曲线在暗区斜率 >1，会把暗光图本来就有的 **8-bit 量化台阶、色度噪声、JPEG 8×8 压缩缝** 一起放大；
**更麻烦的是它自己会造出新伪影**：近黑区没有可靠信号，模型把噪声按通道放大后，不同通道增益不一致 → 出现棕/绿/紫色大斑块。

我们加了一套"终端侧预处理"（不是 ISP，但等价于 ISP 的降噪/色调映射环节），在增强前后夹住噪声放大：

```
输入 → ① 去块(按需) → ② 色度高频残差降噪(自适应) → ③ 暗部轻抬(仅均值<8)
     → ④ Zero-DCE 增强 → ⑤ 亮度引导色度平滑(0.5)
     → ⑥ 近黑区限增益 v2(0.80：平滑亮度 + 羽化 + 抖动) → 输出（可 16-bit）
```

**一句话使用（推荐档）：**

```powershell
cd C:\Users\12983\Documents\repro
python enhance.py "D:\照片\夜景.jpg" --preprocess mosaic -c --bit16
```


### 8.1 第二次改进（2026-11-27 晚）：`auto` 预设 = E10 场景自适应（推荐）

E9 被用户实测指出两个问题：① 748/威尼斯这类**正常夜景**被它先抬亮再送模型，输出发灰发糊；
② 色度大核平滑把**天空/墙面渐变切成色块**（威尼斯天空最明显）。

E10（`--preprocess auto`）的改法——**一切按需、只动该动的**：

| 步骤 | E9（旧） | E10（auto，新） |
| --- | --- | --- |
| 去块 | 固定 0.5 | 先测 8×8 块效应比，>1.03 才开（强度随块效应自适应） |
| 色度降噪 | NLM 全带 + 引导平滑 | **只去高频残差**，低频颜色（天空渐变）原样保留；强度随噪声σ自适应（σ/2，钳 0.35~1.6） |
| 暗部轻抬 | 固定 0.10 | 只有全图均值 <8 才开 |
| 近黑限增益 | 固定 l0=3 l1=12，改亮度+色度 | 只碰 <2.5 的近黑，**只改亮度**、色度按羽化掩码轻微拉中性；只对均值<20 的暗图执行 |
| 近黑色斑 | — | 新增：近黑区（输入<6）再做一次**掩码色度残差降噪** |
| 抖动 | 0.6 | 0.4 |

实测（同一权重 `snapshots_sice/Epoch50.pth`）：

| 图 | 指标 | E1 基线 | E9 旧 | **auto (E10)** |
| --- | --- | --- | --- | --- |
| 极暗 126 | 色噪 / 亮度细节 std8 / 暗区条带跳变 | 1.35 / 4.01 / 12 | 0.73 / 3.03 / 9 | **1.00 / 4.00 / 4** |
| 正常夜景 748 | 色噪 / 亮度细节 std8 / 亮度 | 6.26 / 14.73 / 81.8 | 1.66 / 8.24 / 93.6 | **3.00 / 14.66 / 78.6** |
| 威尼斯 1 | 色噪 / 亮度细节 std8 / 亮度 | 3.34 / 11.77 / 95.4 | 2.51 / 9.55 / 106.2 | **3.36 / 12.00 / 93.6** |
| 有天空 LIME/2 | 色噪 / 亮度细节 std8 / 亮度 | 3.19 / 18.77 / 118.0 | 2.13 / 13.93 / 123.5 | **2.99 / 18.22 / 116.3** |

一句话：**细节和亮度回到基线水平（std8 基本持平），色噪仍比基线低一半左右，而且天空不再切色块**。
代价是极暗图（126）的"抹平"程度不如 E9 激进——想更狠地压极暗噪声，可以用旧的 `--preprocess dark`。

```powershell
python enhance.py 夜景.jpg --preprocess auto -c --bit16    # 推荐（E10）
```

**旧版三档预设（E9，保留以便复现实验）：**

| 预设 | 去块 | 色度降噪 | 暗部轻抬 | 亮度引导平滑 | 近黑区限增益 | 适用 |
| --- | --- | --- | --- | --- | --- | --- |
| `light` | 0 | 0.4 | 0.06 | 0.3 | 0 | 一般夜景，怕抹细节、怕偏淡 |
| `mosaic`（推荐） | 0.5 | 0.5 | 0.10 | 0.5 | 0.80 | 常规暗光/色噪明显 |
| `dark`（`strong` 同义） | 1.0 | 1.0 | 0.20 | 0.8 | 1.0 | 极端暗图（几乎全黑） |

也可以单项手动开：`--denoise 0.5 --deblock 0.5 --prelift 0.10 --guided 0.5 --limit-dark 0.80`，
以及细调 `--dark-gamma 0.55 --dark-smooth 8 --dither 0.6`；
**不加任何参数 = 与原来完全一致**（默认全关）。

**E0~E9 对照实验结果（同一模型 `snapshots_sice/Epoch50.pth`，36 张图；暗区掩码统计）：**

| eval15（15 对） | 亮度 | 色度噪声 | 块效应比 | 梯度放大 | 彩色块占比 | 亮度细节 std8 | 条带跳变 | 暗区级数 | PSNR | SSIM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E1 基线 | 80.5 | 12.177 | 1.004 | 5.47× | 0.022 | 11.91 | **2.5** | 33.0 | 15.86 | 0.5584 |
| E5 旧方案（重降噪+双边） | 96.7 | **1.076** | 0.882 | 1.80× | 0.154 | 5.53 | 11.4 | 17.9 | 16.08 | 0.6573 |
| E6 轻预处理+引导平滑 | 96.5 | 2.328 | 0.917 | 2.28× | 0.145 | 6.71 | 10.9 | 16.3 | 16.12 | 0.6659 |
| E8 限增益 v1（硬灰块） | 88.0 | 1.154 | 0.959 | 2.54× | 0.096 | 7.15 | 9.0 | 22.4 | 16.02 | 0.6978 |
| **E9 限增益 v2（本文推荐）** | 88.6 | 1.598 | 0.934 | 1.94× | 0.128 | 5.93 | **2.8** | 24.1 | 16.02 | 0.6754 |

极端暗图（3 张，几乎全黑）：条带跳变 E1 6.0 / E5 17.7 / E8 8.7 / **E9 2.7**；色噪 E1 5.22 → E9 1.30。

**为什么有 E9：E8 的限增益 v1 会造出"硬灰块"。**
v1 直接拿 8-bit 量化后的亮度去做中性 gamma 映射，输入只有几档灰度 → 输出也只有几档，
实测 126_3_ 里 **41.9% 的像素落在灰度 0、31.6% 落在 24/25 两档**，大面积色阶最大跳变 **24 级**（基线只有 12 级）——
看起来就是"一整块灰板"。E9 的三处改动解决了它：① 先把亮度**大核模糊**成连续梯度再映射；
② 掩码用高斯羽化；③ 输出 8-bit 前加 **±0.6 级抖动**抑制条带。改后条带跳变 **2.8 ≈ 基线 2.5**，暗区级数 24.1。

**诚实结论：**
- E9 的收益集中在**极暗图**（去掉彩色大斑、去掉硬灰块、不再"一片糊"）；对 748/LIME 这类正常夜景，和基线的差别不大，**并且会略偏淡**（饱和度：748 基线 0.051 → E9 0.034；LIME/1 0.152 → 0.107）。想保住颜色就用 `--preprocess light`，或减小 `--prelift`。
- 近黑区（亮度<3）本来就没有可恢复的信号，E9 只是把它变成**中性灰**而不是"彩色马赛克"，谈不上恢复细节。
- 全部对照（E0~E9）、专项诊断、参数说明见 `docs/零-DCE_马赛克预处理方案.md`。

**产物：** `scripts/preprocess.py`（算子）、`scripts/mosaic_ablation.py`（E0~E9 实验）、`scripts/make_mosaic_figs.py`（画图）、
`scripts/diag_color_blobs.py`（彩色斑块专项诊断）、`scripts/diag_e5_blocks.py`（E5 多尺度诊断）、
`demo_mosaic/ablation/`、`demo_mosaic/color_blobs/`、`demo_mosaic/e5_check/`、`demo_mosaic/e9_check/final/compare_E9_vs_base.png`、
文档 `docs/零-DCE_马赛克预处理方案.md`。

### 8.2 E10 后处理 v2（2026-11-27 修正：不再"发黑/丢特征"）

上一版 auto 的后处理有两个毛病，实测在 `493.png`（深色花纹布料）上暴露：
`limit_dark_luma` 会把"暗但有纹理"的区域整体替换成中性灰（特征被淹），
`masked_chroma_denoise` 又在大范围暗区糊色度（颜色发灰、出现色块）。
**v2 改法**：不动亮度；只在"低信号平坦区"（局部均值 < 2σ 且局部标准差 < 0.7σ）平滑色度，
平滑后把色度幅度补回增强图原来的水平；只有"几乎全黑"（均值 < 5）的图仍走 E9 中性抬亮路线。

| 图 | 指标 | 基线 | v1 auto | **v2 auto** |
| --- | --- | --- | --- | --- |
| 493（mean 6.6） | 亮度 / 色噪 / 细节std8 / 饱和度 | 48.6 / 5.79 / 10.93 / 158 | 51.9 / 2.38 / 10.13 / 136 | **58.7 / 3.18 / 12.54 / 155** |
| 748（mean 13.2） | 亮度 / 色噪 / 细节std8 | 81.8 / 6.26 / 14.73 | 79.0 / 2.70 / 14.64 | **79.6 / 1.82 / 14.87** |
| 126（mean 3.5） | 亮度 / 色噪 / 细节std8 | 16.3 / 1.35 / 4.01 | 19.0 / 1.23 / 3.95 | **19.7 / 0.73 / 3.24** |
| LIME/1 | 亮度 / 色噪 / 细节std8 | 95.4 / 3.34 / 11.77 | 92.9 / 3.25 / 11.83 | **94.5 / 3.20 / 12.00** |

新增旋钮 `--auto-strength`（默认 1.0）：后处理总强度，0=不处理、0.5=保守、1.5=更强；
觉得"颜色被洗淡/细节被压"就调小，觉得"还脏"就调大。

> 马赛克这条线的完整时间线、每次被证伪的方案与全部数字，见 [`docs/马赛克改进记录.md`](docs/马赛克改进记录.md)

### 8.3 2026-11-27 修订：默认改为"保守档"（减轻大片色块/分界线）

用户反馈"去马赛克版很平滑，但出现明显分界线"。原因是色度被强降噪后大片落到同一个 8-bit 值，
区域交界就成了肉眼可见的线。现在 `--preprocess auto` 的默认强度改为**保守自动**：

- 图像均值 ≥ 5（普通暗景）：`--auto-strength 0.45`，保留纹理、色块更小；
- 图像均值 < 5（近黑图）：仍用 `1.0`，否则极暗区的彩色斑块会回来。

`--auto-strength` 现在同时控制**预处理（去块/亮度降噪/轻抬）与后处理（色度降噪/限增益）**：
`0` = 完全等价原版；`1.0` = 旧的全强度；也可以给 0~2 之间的任意值。

实测（493 / 748，原版基线 → 保守默认）：色噪 3.55→3.16 / 4.34→3.48，细节 std8 10.91→9.25 / 14.62→14.73。
5× 放大对比：`demo_mosaic\seam_check\seam_493_conservative.png`。

### 8.4 v4 原型：A+B（色度噪声判据 + 引导羽化色度去噪）

针对"保守档仍有色块分界线"，做了一版 v4 原型（`--preprocess v4`）：

- **A 判据**：只用"色度高频能量 − 亮度结构可解释部分"决定是否处理（真实彩色边缘不让它动）；
- **B 羽化**：权重 0/1 → 连续值，再用亮度引导滤波（半径 16 px）羽化，让过渡沿真实轮廓走，
  不再沿"色度等值线"切出人造线。

实测（原版基线 / 保守默认 0.45 / v4）：

| 图片 | 色噪 | 细节 std8 | 分界指标 |
| --- | --- | --- | --- |
| 493 玩偶 | 4.07 / 3.58 / **3.05** | 7.16 / 5.42 / **5.43** | 66 / 58 / **46** |
| 748 泳池 | 4.72 / 3.69 / **3.68** | 9.69 / 9.75 / **9.76** | 82 / 70 / **66** |
| 126 极暗 | 0.54 / 0.67 / **0.46** | 2.00 / 1.81 / 1.57 | — |
| LIME/1 | 2.00 / 1.96 / **1.78** | 7.14 / 6.02 / **6.03** | 46 / 40 / **36** |

命令：`python enhance.py 图.jpg --preprocess v4 -c`；参数 `--v4-strength/--v4-tau/--v4-beta/--v4-guide`。
对比图：`demo_mosaic\v4\<图名>_v4对比.png`、`demo_mosaic\v4\493_v4强档_vs_旧全强度.png`。

> **⚠ 2026-11-27 更正（重要）**：上面 §8.4 里"v4 降低分界线"的结论**不成立**。
> 当时用的指标（平坦区色度梯度 p95）会随"平滑"而下降，掩盖了"色块平台变大"这一事实。
> 用"每千像素色度跳变≥3 但亮度跳变<1 的边界数"重测：基线 26.1 → 保守档 48.3 → v4 45.7，
> **任何空间降噪都会把细颗粒噪声变成粗平台，平台边界更显眼**。v4 的 A+B 只让过渡柔和，不能消除平台。
> 另外发现 `luma_denoise` 的强度控制此前是"假的"（NLM 在 h 很小时也会大幅平滑），
> 已改为按比例混合，`--auto-strength` 现在真正线性可控。结论与选型见 `docs/马赛克改进记录.md` §7。

### 8.5 2026-11-27 最终默认：保颗粒、不造轮廓线（s=0.20 + 不碰色度）

结论汇总（为什么这样定）：

- 任何空间降噪都会把"细颗粒噪声"变成"粗平台"，平台边界就是看得到的"分界线"；
- 用户实测感受 + 指标（每千像素"色度跳变≥3 但亮度<1"的边界数）：基线 26 → 强处理 45~50；
- 因此默认改成**结构优先**：普通暗图 `--auto-strength 0.20`、`--chroma-mode none`（完全不碰色度），
  近黑图（均值<5）仍走中性抬亮（否则彩色大斑回来）。

实测（15 张与 493 同档暗图，均值 5~9）：亮度 50.2→48.6、色噪 2.25→2.63（略升=保留颗粒）、
细节 std8 6.32→5.25、边界 9.9→15.4、两者差异 PSNR 33.5 dB。

**较亮图（10 张，均值≥70）几乎无影响**：亮度 142.9→142.5、色噪 1.55→1.54、细节 10.69→10.39、
两者差异 **PSNR 44.4 dB**（≈像素级微小改动）。原因：不碰色度、亮度只 20% 混合、去块只在 8×8 阻塞>1.03 时轻触发。

想更干净（接受平台）用：`--auto-strength 0.45 --chroma-mode smooth`；
想看效果对比：`demo_mosaic\grain_default\sheet_dark.png` / `sheet_bright.png`。

### 8.6 2026-11-27 两张"另类"暗图的诊断（132 / 512）

**132_2_（均值 8.98，90% 像素 <25）看起来和基线差不多、马赛克还在**：
- 这不是方法"不泛用"，而是**输入信息量不足**：该图暗区只用到 **25 个灰度级**，增强把亮度放大 5.8 倍
  （8.98→52.5），每个灰度台阶被拉成约 6 级 → 变成"平台 + 台阶"，即视觉上的马赛克/色块。
- 任何预处理都不能"变出"信息：要么沿平台平滑（就是之前 E5/E9 那种"大色块"，你不喜欢），要么不动（≈基线）。
- 实测（基线 → 默认）：全图 52.52→51.83、色噪 0.64→0.69、细节 2.17→1.82、人造边界 3.6→4.2 —— 几乎不变，符合"不动"策略。
- 想更干净只能上：`--chroma-mode smooth --auto-strength 0.45`（换平台）、`--bit16`、或**采集端多帧平均**（治本）。

**512_0_ 阴影被加重（变暗）**：确认存在，阴影区（输入<25）平均亮度 47.38→46.03（−1.35 级）。
- 不是降噪强度导致的（s=0.10 → −2.98、s=0.05 → −1.63、s=0.20 → −1.35，非单调），
  而是近黑区 DCE 曲线本身不稳（输入微变→α 摆动），再叠加"噪声被去掉后暗部少了一点正向偏置"。
- 修法：新增 `--shadow-lift`（Y 通道暗部抬几级，`-1`=自动，仅输入均值<15 时抬 **4 级**；0=关闭）。
  实测补偿后阴影差：512 **+0.01**、132 **+0.67**、493 **−0.63**（基本找回）。

## 9. 2026-11-27 修订：auto v3（修正“偏绿 / 噪点多”，并修好大图 OOM）

**问题**：用户反馈 493 那张玩偶图（eval15/493.png）用 `--preprocess auto` 后整体偏绿、噪点多；
另外用旧的对比 bat 处理手机大图时报 `torch.OutOfMemoryError: ... 12.14 GiB is allocated by PyTorch`。

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
python scripts\compare_preprocess.py 你的图.png -o 对比结果
python enhance.py 你的图.png --preprocess auto -c -o 增强结果
```

## 10. 三个版本入口（2026-11-27 整理后：根目录只剩这 3 个 bat）

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

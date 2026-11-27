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

**双击** `C:\Users\12983\Documents\repro\增强我的照片.bat` → 选图 → 结果在图片旁的 `增强结果_原版与edge` 文件夹，
**一次同时生成原版增强与 edge 版增强两张图**（每张图 4 个文件）：
`xxx_原版增强.png`（原版 Zero-DCE）、`xxx_edge增强.png`（加边缘损失）、`xxx_AB对比.png`（原图|原版|edge + 局部放大）、
`xxx_指标.png`（亮度/M/L_edge/cos/corr 柱状）；另有 `overview_AB.png`、`metrics.csv`、`summary.md`。
也可以把照片/文件夹直接**拖到该 bat 上**。只要一个版本时用 `只要原版增强.bat` / `只要edge增强.bat`。

命令行方式：

```powershell
$py = "C:\Users\12983\Documents\repro\.venv\Scripts\python.exe"
# 两个版本一起出（公平对比：两个权重同为 50 轮）
& $py C:\Users\12983\Documents\repro\scripts\ab_enhance_compare.py "D:\照片" -o "D:\AB结果"
# 只要单个版本
& $py C:\Users\12983\Documents\repro\enhance.py "D:\照片" -o "D:\增强结果" -c -m sice
```

`-m` 可选 `sice`（默认，推荐）/ `edge`（加边缘损失）/ `official`（作者权重，更保守）/ `lol`（过曝对照）。
详细说明见 `..\..\deepseek-harness\default-workspace\零-DCE_使用指南.md`。
12MP 手机照约 27 秒/张，小图 0.05~0.3 秒；显存不够会自动降级 fp16 或分块。

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

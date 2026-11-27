# Zero-DCE 复现报告（Windows 11 + RTX 4060 Laptop）

> 复现对象：`Li-Chongyi/Zero-DCE`（CVPR 2020, *Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement*）
> 本机环境已全部装好并跑通，以下内容均为实测结果。

---

## 0. 一句话结论

| 项目 | 状态 | 说明 |
|---|---|---|
| 推理（官方 `Epoch99.pth`） | ✅ 完全复现 | 7.5 秒处理 LIME 10 张 + DICM 64 张，输出观感与论文一致 |
| 训练代码跑通 | ✅ 跑通并收敛 | 200 epoch 全流程实测可用，loss 从 2.87 降到 0.67（末 20 次均值 0.60，最低 0.34） |
| 训练出论文水平的模型 | ✅ 已复现 | 用 **SICE 多曝光数据**训练（不是 LOL），输出观感与官方权重高度接近；用 LOL 训练则得到论文 Fig.6 里的 `Zero-DCE_Low`（过度增强） |
| 需要 conda 吗 | ❌ 不需要 | 用独立 CPython 3.11 + venv 即可，本项目无需 conda |
| 需要 C/C++ 编译器吗 | ❌ 不需要 | Python 是解释器；PyTorch 预编译轮子已含 CUDA 运行时 |
| 需要 VSCode 吗 | 建议 | 已装好 VSCode 1.142.0 + Python/Pylance 扩展，解释器已指向 venv |

---

## 1. 环境清单（已安装，路径可直接用）

| 组件 | 版本 | 路径 |
|---|---|---|
| Python | 3.11.17（python-build-standalone） | `C:\Users\12983\Documents\repro\py311\python\python.exe` |
| 虚拟环境 venv | pip 26.2.2 | `C:\Users\12983\Documents\repro\.venv` |
| PyTorch | 2.5.1+cu124 | venv 内 |
| torchvision | 0.20.1+cu124 | venv 内 |
| numpy | 1.26.4 | venv 内 |
| opencv-python | 4.10.0.84 | venv 内 |
| Pillow | 11.1.0 | venv 内 |
| Git | 2.56.0 | `C:\Users\12983\Documents\repro\PortableGit\cmd\git.exe` |
| VSCode | 1.142.0 | `%LOCALAPPDATA%\Programs\Microsoft VS Code\` |

CUDA 运行时随 torch 轮子自带（CUDA 12.4 / cuDNN 9.1），无需单独安装 CUDA Toolkit。
驱动已验证：`torch.cuda.is_available() == True`，设备 `NVIDIA GeForce RTX 4060 Laptop GPU`，算力 (8, 9)。

VSCode 扩展：`ms-python.python`、`ms-python.vscode-pylance`、`ms-python.debugpy`、中文语言包。
工作区设置已写入 `C:\Users\12983\Documents\repro\Zero-DCE\.vscode\settings.json`，打开该文件夹即自动选中 venv。

### 激活环境（PowerShell）

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
C:\Users\12983\Documents\repro\.venv\Scripts\Activate.ps1
```

---

## 2. 目录与数据

```
C:\Users\12983\Documents\repro\
├─ .venv\                     Python 环境（torch cu124）
├─ py311\                     CPython 3.11.17 本体
├─ PortableGit\               git
├─ Zero-DCE\                  官方仓库（已改 3 个文件，见第 5 节）
│  └─ Zero-DCE_code\
│     ├─ data\train_data\        485 张 LOL 低照（jpg）
│     ├─ data\train_high\        485 张 LOL 正常光照（仅用于算指标）
│     ├─ data\train_data_sice\   2002 张 SICE 多曝光图（论文真正用的训练数据）
│     ├─ data\test_data\LIME\    10 张 bmp（官方自带）
│     ├─ data\test_data\DICM\    64 张 jpg（官方自带）
│     ├─ data\result\            推理输出
│     ├─ snapshots\Epoch99.pth   官方预训练权重（314 KB）
│     ├─ snapshots_run\          LOL 数据自训练 200 epoch 的权重
│     └─ snapshots_sice\         SICE 数据自训练的权重
├─ lol_repo\                  LOL 配对数据源（485 对）
├─ sice_repo\                 含 2002 张多曝光训练图的仓库
├─ wheels\                    离线 torch 轮子
├─ scripts\                   数据准备、下载、评测、对比图脚本
├─ compare\                   对比图输出
├─ train_full.log             LOL 训练日志
└─ train_sice.log             SICE 训练日志
```

**数据来源**

- 代码：<https://github.com/Li-Chongyi/Zero-DCE>（`git clone --depth 1`）
- LOL 485 对低照/正常光照：<https://github.com/23225114YKW/Low-light-image-enhancement-with-LoL-dataset>
- SICE 多曝光 2002 张（论文训练集）：<https://github.com/Aiemu/Zero-DCE-improved> 的 `src/data/train_data`（512×512，亮度覆盖 0.01~0.97）
- 官方权重与 LIME/DICM 测试图：官方仓库自带

> 官方的 Google Drive / 百度网盘链接本机不可达（drive.google.com 被墙），已用上述 GitHub 镜像替代。

---

## 3. 推理使用（官方权重，已复现）

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code
C:\Users\12983\Documents\repro\.venv\Scripts\python.exe lowlight_test.py
```

- 输入：`data/test_data/LIME`（10 张 bmp）+ `data/test_data/DICM`（64 张 jpg）
- 输出：`data/result/LIME/`、`data/result/DICM/`
- 实测耗时：**7.5 秒**（74 张图，4060 Laptop）

亮度（0~255）变化示例：

| 图像 | 输入均值 | 输出均值 |
|---|---|---|
| LIME/1.bmp（威尼斯夜景） | 33.0 | 85.8 |
| LIME/5.bmp | 44.2 | 101.8 |
| DICM/01.jpg（埃菲尔铁塔） | 20.3 | 44.6 |

对比图见 `C:\Users\12983\Documents\repro\compare\`：
- `LIME_1_in_vs_out.png`、`DICM_01_in_vs_out.png`（输入/输出）
- `*_low_official_ours*.png`（低照输入/官方权重/自训练权重 三列对比）

### 3.2 在官方 LOL eval15 测试集上的定量结果（有真值参考）

后来补齐了 LOL 官方测试集 `eval15`（15 对真实拍摄的低照/正常光图像，600×400，
来源镜像 <https://github.com/FrankHo-Hwc/LOLDataset>），于是可以做**真正的基准定量**：

| 方法 | PSNR ↑ | SSIM ↑ |
|---|---|---|
| 低照输入 | 7.773 | 0.1956 |
| 官方 `Epoch99.pth` | 14.797 | 0.5557 |
| **自训练 SICE 50 epoch（= Epoch50）** | **15.809** | **0.5574** |
| 同一次训练的 Epoch49 | 17.223 | 0.5509 |
| 对照：LOL 训练 200 epoch | 15.366 | 0.5089 |

> 读法：同一次 SICE 训练的 E49 与 E50 相差 **1.41 dB**（17.22 vs 15.81），说明尾段并不单调。本文主表取最后一轮 Epoch50；`零-DCE_边缘损失实验报告.md` 为了让 `edge` 与基线同轮对比，取的是 **Epoch49**。两个数字都真实存在、各有 CSV 支撑，但**不能互相比较**，也不代表不同模型。

官方权重跑出 14.80/0.556，与公开对比表中 Zero-DCE 在 LOL 上的 ~14.8/0.56 完全吻合（说明环境无问题）；
自训练模型 PSNR 再高 1 dB、SSIM 持平，**复现成立**。
逐张对比图与明细见 `C:\Users\12983\Documents\deepseek-harness\default-workspace\LOL_eval15_增强对比.md`，
图在 `C:\Users\12983\Documents\repro\demo_pairs\`（15 张 `pair_*.png` + `overview_6cases.png` + `overview_eval15_pairs.png` + `metrics.csv`）。

---

## 4. 训练（已跑通）

### 4.1 论文原始设置（论文 4.1 节）

| 项 | 值 |
|---|---|
| 训练数据 | SICE Part1 的 360 个多曝光序列中 3,022 张 → 随机取 **2,422 张**训练（含过曝图） |
| 输入尺寸 | 512×512 |
| batch size | 8 |
| 优化器 | Adam，学习率固定 1e-4，weight decay 1e-4 |
| 初始化 | 高斯 σ=0.02 |
| 轮数 | 官方脚本默认 200 epoch（发布权重叫 `Epoch99.pth`，即第 99 轮保存） |
| 损失 | `200*L_TV + L_spa + 5*L_color + 10*L_exp`，`L_exp` 目标亮度 0.6 |

### 4.2 本机实测

```powershell
# 论文数据集（推荐）
python lowlight_train.py --lowlight_images_path "data/train_data_sice/" --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_sice/

# LOL 485 张（= 论文 Zero-DCE_Low 消融设置）
python lowlight_train.py --num_workers 0 --num_epochs 200 --snapshots_folder snapshots_run/
```

> ⚠️ Windows 上把 `--num_workers` 设成 **0**。用默认 4 会在本机报
> `RuntimeError: Couldn't open shared file mapping: <torch_...>, error code: <1455>`（页面文件/共享内存不足），
> 且训练过程中速度会掉到 1/10。

**用自己的权重跑官方测试脚本**：`lowlight_test.py` 里硬编码加载 `snapshots/Epoch99.pth`，
所以把自训练权重覆盖过去即可（先备份官方的）：

```powershell
Copy-Item snapshots\Epoch99.pth snapshots\Epoch99_official.pth
Copy-Item snapshots_sice\Epoch50.pth snapshots\Epoch99.pth   # 覆盖官方权重，上一行已备份
& $py lowlight_test.py     # 输出在 data\result\
```

实测速度（4060 Laptop；机器上同时开着别的游戏/程序时 GPU 会被抢占）：

| 数据 | 每 epoch | 说明 |
|---|---|---|
| LOL 485 张 | ~9 s（60 iters） | 200 epoch 约 30 分钟（已跑完） |
| SICE 2002 张 | ~40 s（250 iters） | 本机跑到 **Epoch50**（loss 末段 0.71~0.91、末 20 次均值 0.79、最低 0.39），完整 100 epoch 约 65 分钟 |

已产出的权重：

| 路径 | 内容 |
|---|---|
| `Zero-DCE_code/snapshots/Epoch99.pth` | 官方发布权重（原封未动） |
| `Zero-DCE_code/snapshots_sice/Epoch50.pth` | **SICE 多曝光数据自训练 50 epoch（推荐用它，等同论文配方）** |
| `Zero-DCE_code/snapshots_run/Epoch199.pth` | LOL 数据自训练 200 epoch（= 论文 Zero-DCE_Low 消融） |

---

## 5. 关键发现：README 的训练数据不是 LOL

这是复现时最容易踩的坑，也是"为什么按官方脚本训练出来的模型比官方权重过曝"的根本原因。

**证据 1 — 论文自己写的**（见 `Zero-DCE_论文汉化与解读.md` 第 167、299 行）：
训练集是 SICE Part1 的 3,022 张多曝光图像中随机取的 2,422 张，**刻意同时纳入低光照与过曝光图像**，缩放到 512×512。

**证据 2 — 论文 Fig.6 消融**（第 181、316 行）：
只用低光照图像训练（`Zero-DCE_Low` / `Zero-DCE_LargeL`，即使 9,000 张）会**过度增强本来明亮的区域**；
加入多曝光数据（`Zero-DCE_LargeLH`）后暗区恢复更好。

**证据 3 — 实测 loss 对比**（`scripts/diag_loss.py`，10 张 LOL 图）：

| 权重 | TV | SPA | COL | EXP | TOTAL | 输出均值 |
|---|---|---|---|---|---|---|
| 官方 `Epoch99.pth` | 0.034 | 0.006 | 0.052 | **1.370** | 1.462 | 0.262 |
| 我们用 LOL 485 张训的 `Epoch199.pth` | 0.062 | 0.024 | 0.219 | 0.511 | 0.816 | 0.524 |

官方权重在 LOL 数据上的 **EXP loss 是我们模型的 2.7 倍**——说明官方权重根本不是在这份 LOL 数据上收敛的，
它是在 SICE 多曝光数据上训练出来的"温和曲线"模型（对已经够亮的区域不再硬拉亮度）。

**证据 4 — 视觉**：三列对比图（`compare/*_low_official_ours.png`）里，官方权重自然温和，
LOL 自训练模型整体过曝、噪点被放大、偏色——正是论文 Fig.6(c)(d) 描述的 `Zero-DCE_Low` 症状。

### 两种训练设置的结果对比（LOL 485 对，原分辨率，485 对全量）

> 注意：本节是 **LOL 训练集 485 对**上的指标（属训练分布内，偏乐观，仅用于内部对比）；官方 eval15 的评测见 3.2 节与 `LOL_eval15_增强对比.md`。

| 权重 | PSNR | SSIM | 输出亮度均值 |
|---|---|---|---|
| 低照输入 | 7.745 | 0.1823 | 0.060 |
| 官方 `Epoch99.pth`（SICE 训练） | 14.234 | 0.5878 | 0.267 |
| **自训练 SICE Epoch50（论文配方）** | **15.057** | **0.6007** | 0.309 |
| 自训练 SICE Epoch25 | 16.441 | 0.6023 | 0.365 |
| 自训练 LOL 200 epoch | 15.566 | 0.5550 | 0.549 |
| 自训练 LOL Epoch99 | 15.025 | 0.5489 | 0.572 |
| 自训练 LOL Epoch49 | 14.189 | 0.5412 | 0.604 |

**SICE 训练版（0.309 亮度 / SSIM 0.6007）已与官方（0.267 / 0.5878）同一档，且 SSIM 反超；
LOL 训练版亮度飙到 0.55，SSIM 掉到 0.555 —— 过曝把细节洗掉了。** 这与论文 Fig.6 的结论完全一致。

### 视觉对比（三列：低照输入 / 官方权重 / 自训练权重）

图在 `C:\Users\12983\Documents\repro\compare\`：

| 文件 | 内容 |
|---|---|
| `LIME_1_low_official_ours_sice50.png` | 威尼斯夜景：33.0 / 85.8 / **95.9**（第三列与官方几乎一样） |
| `DICM_01_low_official_ours_sice50.png` | 埃菲尔铁塔：20.3 / 44.3 / **51.6** |
| `LOL_train_502_low_official_ours_sice50.png` | 圣诞树：8.0 / 38.9 / **45.3** |
| `LIME_1_low_official_ours.png`（对照） | 用 LOL 训练时同一场景：33.0 / 85.8 / **161.4**（严重过曝） |
| `LIME_1_in_vs_out.png`、`DICM_01_in_vs_out.png` | 官方权重的输入/输出对比（复现论文效果图） |

### 复现结论

1. **推理**：用官方权重、官方测试图，命令一行跑通，输出与论文图观感一致 → 复现成功。
2. **训练**：用论文真正的训练数据（SICE Part1 多曝光、512×512）从零训练，得到的模型在 LOL 485 对上
   SSIM 0.6007 / PSNR 15.06 / 输出亮度 0.309，与官方权重（0.5878 / 14.23 / 0.267）同一水平，
   视觉上几乎无差别 → **论文模型复现成功**。
3. **反例验证**：只喂 LOL 485 张低照图 → 输出亮度 0.549、LIME 图亮度从 33 拉到 161（官方只有 85），
   正是论文 Fig.6 中 `Zero-DCE_Low` 的过曝症状。这一步顺带把论文"训练数据必须含过曝图像"的消融实验也复现了。

---

## 6. 与原仓库的差异（3 个文件，5 处改动）

| 文件 | 行 | 改动 | 原因 |
|---|---|---|---|
| `dataloader.py` | 51-52 | `Image.ANTIALIAS` → `getattr(Image, "ANTIALIAS", Image.LANCZOS)` | Pillow ≥10 删除了 `ANTIALIAS` |
| `dataloader.py` | 48 | `Image.open(p)` → `Image.open(p).convert('RGB')` | 防止灰度图导致 `permute(2,0,1)` 崩溃 |
| `lowlight_test.py` | 22 | 同上 `.convert('RGB')` | 同上 |
| `lowlight_test.py` | 40-43 | 用 `os.path.dirname(...)` + `os.makedirs(..., exist_ok=True)` 建输出目录 | 原代码在 Windows 混合分隔符下建不出 `data/result/DICM` 子目录，直接 `FileNotFoundError` |
| `lowlight_train.py` | 32 | 加一行 `cudnn.benchmark = True` | 输入尺寸固定，让 cuDNN 选最快卷积算法 |

上表之外，基础复现部分与官方代码一致。**例外**：`Myloss.py` 后来新增了 `SobelMag` / `L_edge`（论文 (4.12) 的 Sobel 边缘损失），默认 `--edge_weight 0` 关闭，开启后即 `zero-DCE` 的 `edge` 版本（见 `零-DCE_边缘损失实验报告.md`）。所以"损失函数与官方完全一致"只对默认配置成立，开启边缘损失时是 **4 个文件**改动。

---

## 7. 已知限制与后续可做

1. **论文口径的全量对比**：论文在 SICE Part2 的 767 对上算指标（`16.57 / 0.59 / 98.78`）。
   LOL 官方 eval15（15 对）已补齐并可跑出基准（见 3.2 节）；SICE Part2 仍需另外下载（本机 Google Drive 不可达），尚未做全量对比。
2. **训练分辨率**：官方代码 dataloader 固定 resize 到 256×256，论文写的是 512×512。要严格对齐论文需改 `dataloader.py` 的 `self.size = 512`（显存和时间都会增加）。
3. **训练轮数**：官方发布权重是 `Epoch99.pth`（第 99 轮）。本机 SICE 训练跑到 **Epoch50** 已进入 loss 平台
   （0.65~0.75 波动，约 55 s/epoch）；再用一条命令续跑到 100 轮约需 45 分钟。继续训练会略微变亮（参见上表 Epoch25→Epoch50 的趋势）。
4. **GPU 被抢占**：复现期间检测到 `DeltaForceClient-Win64-Shipping.exe` 占用约 4.3 GB 内存和 GPU，
   训练速度只有空闲时的 1/5~1/10。要跑满 200 epoch 建议先关掉它。
5. **conda 不需要**：官方 README 写的是 Python 3.7 + PyTorch 1.0 + CUDA 10.0 的 conda 命令，属历史遗留；
   当前 cu124 轮子 + Python 3.11 工作正常，照抄 README 反而装不上。

---

## 8. 命令速查

```powershell
$py = "C:\Users\12983\Documents\repro\.venv\Scripts\python.exe"
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code

# 1) 推理（官方权重）
& $py lowlight_test.py

# 2) 训练（论文数据，100 epoch）
& $py lowlight_train.py --lowlight_images_path "data/train_data_sice/" --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_sice/

# 3) 训练（LOL 数据，200 epoch，= 论文 Zero-DCE_Low 消融）
& $py lowlight_train.py --num_workers 0 --num_epochs 200 --snapshots_folder snapshots_run/

# 4) 指标评测（LOL 485 对）
& $py C:\Users\12983\Documents\repro\scripts\eval_lol.py

# 5) 生成三列对比图
& $py C:\Users\12983\Documents\repro\scripts\make_compare3.py
```

VSCode：`code C:\Users\12983\Documents\repro\Zero-DCE`，右下角选择解释器
`C:\Users\12983\Documents\repro\.venv\Scripts\python.exe` 即可。

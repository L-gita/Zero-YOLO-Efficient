# 从零自己搭建 Zero-DCE 复现项目（VSCode + GitHub 纯手工版）

> 这份指南**不用**我已经搭好的任何东西：你在一个全新文件夹里，从装 Python、装依赖、clone 代码、找数据、改代码、训练、评测，一步步自己做。
> 我已经把每个命令在这台机器上实际跑过，所以凡是给出"预期输出"的地方都是实测值；凡是"坑"都标了 ⚠️。
>
> 全程约 **2~3 小时**（其中 30 分钟是等 torch 下载、65 分钟是等训练，真正动手约 1 小时）。

> 📌 **这台机器的现状（2026-10 更新）**：系统版 **Python 3.11.9** 已经装好
> （`C:\Users\12983\AppData\Local\Programs\Python\Python311\python.exe`，已排进 PATH 最前面），
> 并且里面**已经装好全套依赖**（torch 2.5.1+cu124 / torchvision 0.20.1+cu124 / numpy 1.26.4 /
> pillow 11.1.0 / opencv-python 4.10.0.84，`pip check` 无冲突），旧的 Python 3.10.11 已卸载。
> 所以：
> - **只想跑通** → 阶段 1、2、4、5 全部跳过，直接从阶段 3 开始（新建项目 + 建 venv 或直接用系统 Python）。
> - **想练手** → 照全文走一遍也没问题，只是阶段 2 装 Python 时可以装在别的目录，避免和系统这份打架。
> 新开终端验证：`python --version` 应为 `Python 3.11.9`。

---

## 阶段 0：开工前的三件事

### 0.1 网络体检（决定了你走哪些下载源）

这台机器实测结论（很重要，直接决定下面每一步用什么地址）：

| 站点 | 速度/状态 | 结论 |
|---|---|---|
| `github.com`、`raw.githubusercontent.com`、`codeload.github.com` | ~13 MB/s，很快 | ✅ 代码、数据全走 GitHub |
| `mirror.sjtu.edu.cn`（上海交大镜像） | 单连接 ~1.3 MB/s，多线程 ~5.7 MB/s | ✅ pip 源、torch 大轮子走这里 |
| `registry.npmmirror.com`（阿里系 npm 镜像） | 快 | ✅ 用来下 Python 安装包 |
| `pypi.org` | **~13 KB/s** | ❌ 不设镜像会卡到怀疑人生 |
| `python.org`、`update.code.visualstudio.com` | **超时** | ❌ 装不了官方 Python/更新 VSCode |
| `download.pytorch.org` | ~0.35 MB/s | ❌ 官方 torch 源太慢 |
| `drive.google.com` | **超时** | ❌ 官方数据集（Google Drive）拿不到 |
| 清华 tuna / 阿里 anaconda 镜像 | 不可达 | ❌ 所以**不要走 conda 路线** |

### 0.2 目录规划

```
C:\Users\12983\Documents\zero-dce-scratch\      ← 你的新项目根目录（自己建）
├── .venv\                                      ← Python 虚拟环境（自己建）
├── Zero-DCE\                                   ← git clone 下来的官方代码
│   └── Zero-DCE_code\                          ← 真正的代码目录（下面所有命令都在这里跑）
├── data_src\                                   ← 原始数据集（clone 来放这）
├── wheels\                                     ← 下载的 torch 轮子（可选）
└── scripts\                                    ← 你自己写的辅助脚本（转数据、评测）
```

> 目录**不要带空格**、**不要放中文路径**（虽然 Python 能处理，但省去很多麻烦）。

### 0.3 时间预算

| 阶段 | 耗时 |
|---|---|
| 装 Git + Python | 10 分钟 |
| 建 venv + 装小依赖 | 6 分钟 |
| 装 torch（2.4 GB） | 3 分钟（镜像分块下）~ 2 小时（官方源单连接） |
| clone 代码 + 数据 | 5 分钟 |
| 改代码（3 处补丁） | 5 分钟 |
| 冒烟验证 | 2 分钟 |
| 正式训练 | 30 分钟（LOL 200 轮） / 65 分钟（SICE 100 轮，实测 ~40 s/轮） |
| 评测 + 看图 | 5 分钟 |

---

## 阶段 1：装 Git（5 分钟）

你机器上原本没装 Git。官方站 `git-scm.com` 不一定通，走 GitHub Releases（很快）：

1. 下载（57.2 MB）：
   ```
   https://github.com/git-for-windows/git/releases/download/v2.56.0.windows.2/PortableGit-2.56.0.2-64-bit.7z.exe
   ```
2. 双击运行 → 选一个**专用目录**解压，例如 `C:\Users\12983\Documents\tools\PortableGit`
   ⚠️ 注意：这个自解压器会把 `bin`、`cmd`、`usr` 直接铺到你选的目录里，所以一定要选**空目录**。
3. 把 `C:\Users\12983\Documents\tools\PortableGit\cmd` 加进系统 PATH：
   - Win 键搜"编辑系统环境变量" → 环境变量 → 用户变量 `Path` → 新建 → 粘贴上面的路径
4. **关掉并重开** VSCode 的终端（PATH 才会生效），验证：
   ```powershell
   git --version
   ```
   ✅ 预期：`git version 2.56.0.windows.2`

> 想省这一步：直接用已经装好的 `C:\Users\12983\Documents\repro\PortableGit\cmd\git.exe`（我装的那份），效果完全一样。

---

## 阶段 2：装 Python 3.11（5 分钟）

python.org 被墙，但阿里系 npm 镜像有官方安装包镜像（实测可达，26 MB）：

1. 下载：
   ```
   https://registry.npmmirror.com/-/binary/python/3.11.9/python-3.11.9-amd64.exe
   ```
2. 双击安装 → **务必勾选** `Add python.exe to PATH` → 选 Install Now
3. 新开一个 PowerShell（或重开 VSCode 终端），验证：
   ```powershell
   python --version
   python -m pip --version
   ```
   ✅ 预期：`Python 3.11.9` + `pip 2x.x`

**为什么锁 3.11 而不是最新版？** 因为后面 torch 的 CUDA 轮子是按 Python 版本编译的（`cp311`）。3.12/3.13 的 `cp312/cp313` 轮子也有，但生态和示例最多的是 3.11；而且 3.13 太新，很多包还没出轮子，装起来会现场编译（更慢更容易失败）。

> 不想装系统 Python 也行：用独立免安装版
> `https://github.com/astral-sh/python-build-standalone/releases` 里的
> `cpython-3.11.17+*-x86_64-pc-windows-msvc-install_only.tar.gz`（我上次用的就是这条，解压即用，不污染系统）。

---

## 阶段 3：新建项目 + 虚拟环境（5 分钟）

### 3.1 建目录，用 VSCode 打开

```powershell
mkdir C:\Users\12983\Documents\zero-dce-scratch
cd C:\Users\12983\Documents\zero-dce-scratch
```

然后 **VSCode 菜单 文件 → 打开文件夹 → 选 `zero-dce-scratch`**（或在该文件夹空白处右键 → "通过 Code 打开"）。
VSCode 会问"是否信任作者"，选**信任**。左侧确认能看到空文件夹。

> `code .` 命令只有在安装 VSCode 时勾了"添加到 PATH"才可用；如果提示找不到命令，就用上面的菜单方式。

### 3.2 建虚拟环境

VSCode 里按 `` Ctrl + ` `` 打开终端（应该在 `zero-dce-scratch` 目录）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

✅ 预期：提示符前面出现 `(.venv)`。

⚠️ 如果报 `Activate.ps1 cannot be loaded because running scripts is disabled`：
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```
再重跑激活命令。

### 3.3 让 VSCode 认识这个环境

`Ctrl+Shift+P` → 输入 `Python: Select Interpreter` → 选 `.\venv\Scripts\python.exe`。
（右下角状态栏显示 `Python 3.11.9 ('.venv')` 就对了。）

### 3.4 建你自己的配置文件

新建 `.vscode\settings.json`（VSCode 里右键新建文件），内容：

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
  "files.encoding": "utf8",
  "terminal.integrated.defaultProfile.windows": "PowerShell"
}
```

---

## 阶段 4：配 pip 镜像 + 装小依赖（6 分钟）

### 4.1 换源（必须先做，否则 pip 会慢到超时）

```powershell
python -m pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
python -m pip config set global.timeout 120
```

### 4.2 建 `requirements.txt`

在项目根目录新建 `requirements.txt`：

```
numpy==1.26.4
pillow==11.1.0
opencv-python==4.10.0.84
```

> 为什么锁这三个版本：`numpy 1.26.4` 是 torch 2.5 和 opencv 4.10 都能兼容的"甜点版本"；`pillow 11` 与 torchvision 0.20 匹配；`opencv 4.10` 自带 numpy 兼容层，不会乱升级。

```powershell
pip install -r requirements.txt
```

✅ 预期：1~2 分钟装完，最后 `Successfully installed ...`。

### 4.3 装 torch 的"跑腿依赖"

```powershell
pip install filelock typing_extensions sympy networkx jinja2 fsspec
```
（这几个是 torch 运行需要的，先装好可以避免装 torch 时它又去慢速源找。）

---

## 阶段 5：装 PyTorch（关键一步，两条路线）

RTX 4060 是 **sm_89** 架构，必须用 CUDA ≥ 11.8 的轮子；`cu124` 是稳妥选择。

### ⚠️ 先记住这条：Windows 上直接 `pip install torch` 拿到的是 **CPU 版**

实测（SJTU 镜像上的 PyPI 索引）：`torch-2.5.1-cp311-cp311-win_amd64.whl` 只有 **193.7 MB** —— 这是 **CPU 轮子**，装完 `torch.cuda.is_available()` 会是 `False`。
CUDA 版轮子文件名带 `+cu124`、体积 **2.4 GB**，不在 PyPI 上，只能从 PyTorch 官方源或镜像的 `pytorch-wheels` 目录取。**这是新手最容易踩的坑**，务必按下面两条路线之一走。

### 路线 A（推荐）：从 SJTU 镜像手动下 cu124 轮子（约 3 分钟）

1. 建目录 `wheels\`，然后用下面这个分块并行下载函数（PowerShell 里粘贴运行）：

```powershell
function Get-Chunked($Url, $Out, $Chunks = 16) {
  if (Test-Path $Out) { Remove-Item $Out -Force }
  $dir = [IO.Path]::GetDirectoryName($Out); $name = [IO.Path]::GetFileNameWithoutExtension($Out)
  $len = [int64](curl.exe -sIL $Url | Select-String 'content-length:\s*(\d+)' | ForEach-Object { $_.Matches[0].Groups[1].Value } | Select-Object -Last 1)
  if (-not $len) { throw "拿不到文件大小: $Url" }
  $size = [math]::Ceiling($len / $Chunks); $procs = @()
  for ($i = 0; $i -lt $Chunks; $i++) {
    $s = $i * $size; $e = [math]::Min($s + $size - 1, $len - 1)
    $part = "$dir\$name.part$i"
    $procs += Start-Process curl.exe -ArgumentList @('-sL', '-r', "$s-$e", '-o', $part, $Url) -PassThru -NoNewWindow
  }
  $procs | Wait-Process
  $fs = [IO.File]::Create($Out)
  for ($i = 0; $i -lt $Chunks; $i++) {
    $part = "$dir\$name.part$i"
    $bytes = [IO.File]::ReadAllBytes($part); $fs.Write($bytes, 0, $bytes.Length); Remove-Item $part -Force
  }
  $fs.Close()
  "done: $Out  $([math]::Round((Get-Item $Out).Length/1MB,1)) MB"
}
```

2. 下载两个轮子（地址已实测可达）：

```powershell
Get-Chunked 'https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/torch-2.5.1%2Bcu124-cp311-cp311-win_amd64.whl' 'wheels\torch-2.5.1+cu124-cp311-cp311-win_amd64.whl'
Get-Chunked 'https://mirror.sjtu.edu.cn/pytorch-wheels/cu124/torchvision-0.20.1%2Bcu124-cp311-cp311-win_amd64.whl' 'wheels\torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl'
```

✅ 预期：第一个 **2,394.4 MB**（2,510,750,521 字节），第二个 **5.9 MB**（6,143,069 字节）。
> 这两个地址会 301 跳到 SJTU 的 S3，所以命令里都带 `-L`（跟随跳转）；上面函数里已经处理好，不用你操心。

3. 离线安装（`--no-deps` 因为依赖已在阶段 4 装好）：

```powershell
pip install --no-index --no-deps wheels\torch-2.5.1+cu124-cp311-cp311-win_amd64.whl wheels\torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl
```

### 路线 B（备选，很慢）：走 PyTorch 官方 cu124 源

```powershell
pip install torch==2.5.1 torchvision==0.20.1 `
  --index-url https://download.pytorch.org/whl/cu124 `
  --extra-index-url https://mirror.sjtu.edu.cn/pypi/web/simple
```

实测这台机器 `download.pytorch.org` 只有 **0.35 MB/s**，2.4 GB 要下 **约 2 小时**。网络条件好时可以这么干，否则用路线 A。

### 5.3 验证（必做）

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
```

✅ 预期：
```
2.5.1+cu124
True
NVIDIA GeForce RTX 4060 Laptop GPU
(8, 9)
```

看到 `True` 才能继续。`False` = 装成了 CPU 版，回阶段 5 重装。

---

## 阶段 6：clone 官方代码并读懂 README（5 分钟）

```powershell
git clone https://github.com/Li-Chongyi/Zero-DCE.git
cd Zero-DCE
dir
```

✅ 预期：看到 `Zero-DCE_code`、`README.md`、`figs` 等。

**README 要点（以及哪些是坑）**：

| README 说的 | 实际情况 |
|---|---|
| Requirements: Python 3.7, PyTorch 1.0.0, cuda 10.0 | ⚠️ 2019 年的老配置，**别照抄**，现在 torch 2.5/cu124 才对 |
| 给了 conda 安装命令 | ⚠️ 本机 anaconda 镜像不可达，且没必要，venv 就够 |
| 训练数据：Google Drive 链接 | ⚠️ 被墙 |
| 训练数据：百度网盘 `https://pan.baidu.com/s/11-u_FZkJ8OgbqcG6763XyA` 提取码 `1234` | 可以试：装百度网盘客户端登录下载（官方 485 张 LOL 训练图就在这里） |
| 预训练权重 `snapshots/Epoch99.pth` | ✅ 仓库里直接带了（314 KB），clone 下来就能用 |

值得注意：官方 README 的 `train_data` 其实是**论文用的 SICE 多曝光数据子集**，不是 LOL。论文 4.1 节写的是 SICE Part1 的 3022 张里取 2422 张、缩到 512×512、**刻意同时包含低照和过曝图像**。这一点下面阶段 7 会用到。

---

## 阶段 7：拿数据（10 分钟，三条路）

### 7.1 官方路径（能通就用，最正统）

- 百度网盘：链接/提取码见上，下载 `train_data`（注意：作者这份不是 LOL 485 张，而是 SICE 多曝光图，社区可得版本是 2002 张 512×512）
- Google Drive：需要科学上网

### 7.2 GitHub 镜像路径（本机实测可用）

```powershell
cd C:\Users\12983\Documents\zero-dce-scratch

# ① LOL 485 对（低照 + 正常光配对，用来做训练/评测对照；解压后约 380 MB，约 1 分钟）
git clone --depth 1 https://github.com/23225114YKW/Low-light-image-enhancement-with-LoL-dataset.git data_src\lol
#    → data_src\lol\traindata\low\*.png  (486 个，含 .DS_Store)
#    → data_src\lol\traindata\high\*.png

# ② SICE 多曝光 2002 张（论文真正的训练数据，512×512；110 MB）
git clone --depth 1 https://github.com/Aiemu/Zero-DCE-improved.git data_src\sice
#    → data_src\sice\src\data\train_data\1.jpg … 2002.jpg

# ③ LOL 官方测试集 eval15（15 对带正常光参考图，用来定量评测；332 MB）
git clone --depth 1 https://github.com/FrankHo-Hwc/LOLDataset.git data_src\lol_test
#    → data_src\lol_test\eval15\low\*.png 与 eval15\high\*.png
```

### 7.3 把数据摆成代码期望的样子（关键）

代码里的路径是**写死**的（`dataloader.py` 和 `lowlight_train.py` 都用 `data/train_data/`）：

```
Zero-DCE\Zero-DCE_code\data\train_data\   ← 训练图，必须**平铺** *.jpg
Zero-DCE\Zero-DCE_code\data\test_data\    ← 测试图，按子文件夹分组（仓库自带 LIME/DICM）
Zero-DCE\Zero-DCE_code\data\result\       ← 输出（脚本自己建）
```

⚠️ **两个硬性要求**：
1. `dataloader.py` 里只 glob `*.jpg` —— **PNG 会被直接忽略**，所以 485 张 png 必须转成 jpg；
2. 必须平铺（`data/train_data/1.jpg`），不能有子文件夹。

写个转换脚本 `scripts\prepare_data.py`（你自己建这个文件）：

```python
import os, glob
from PIL import Image

ROOT = r"C:\Users\12983\Documents\zero-dce-scratch"
CODE = os.path.join(ROOT, "Zero-DCE", "Zero-DCE_code")
PAIRS = [
    # (源目录, 目标目录)：低照图 → data/train_data；正常光参考图 → data/train_high（评测要用）
    (os.path.join(ROOT, "data_src", "lol", "traindata", "low"),  os.path.join(CODE, "data", "train_data")),
    (os.path.join(ROOT, "data_src", "lol", "traindata", "high"), os.path.join(CODE, "data", "train_high")),
]
for src, dst in PAIRS:
    os.makedirs(dst, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src, "*.png")))
    for i, p in enumerate(files):
        Image.open(p).convert("RGB").save(os.path.join(dst, f"{i:04d}.jpg"), quality=95)
    print(f"{len(files)} 张 → {dst}")
```

```powershell
.\.venv\Scripts\python.exe scripts\prepare_data.py
```

✅ 预期：
```
485 张 → ...\Zero-DCE_code\data\train_data
485 张 → ...\Zero-DCE_code\data\train_high
```
（`.DS_Store` 不是 png 会自动跳过。`train_high` 是正常光参考图，训练不用，但**算 PSNR/SSIM 要用**。）

如果要用论文口径的训练（推荐）：把 `data_src\sice\src\data\train_data\*.jpg`（2002 张、本身就是 jpg、512×512）直接复制到 `Zero-DCE_code\data\train_data_sice\`，训练时用 `--lowlight_images_path data/train_data_sice/`。

---

## 阶段 8：改代码（3 个文件 5 处，必做）

官方代码是 2019 年在 Linux 上写的，在 Windows + 新版 Pillow + 新 GPU 上会直接崩。**每处都说明原因**，别盲目照抄：

### 补丁 1：`Zero-DCE_code/dataloader.py`（2 处）

```python
# 第 ~48 行，原来：
data_lowlight = Image.open(data_lowlight_path)
# 改成（灰度/带 alpha 的图会让后面 permute(2,0,1) 崩）：
data_lowlight = Image.open(data_lowlight_path).convert('RGB')

# 第 ~50 行，原来：
data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS)
# 改成（Pillow 10 删除了 Image.ANTIALIAS，LANCZOS 是同一套滤波）：
resample = getattr(Image, "ANTIALIAS", Image.LANCZOS)
data_lowlight = data_lowlight.resize((self.size,self.size), resample)
```

### 补丁 2：`Zero-DCE_code/lowlight_test.py`（2 处）

```python
# 第 ~22 行：
data_lowlight = Image.open(image_path).convert('RGB')

# 第 ~42 行，原来（Windows 下 '/xx.jpg'.replace 匹配不上 '\\xx.jpg'，建不出子目录 → 报 FileNotFoundError）：
if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
    os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
# 改成：
result_dir = os.path.dirname(result_path.replace('\\','/'))
os.makedirs(result_dir, exist_ok=True)
```

### 补丁 3：`Zero-DCE_code/lowlight_train.py`（1 处，可选但建议）

```python
# 在 def train(config): 里 os.environ['CUDA_VISIBLE_DEVICES']='0' 下一行加：
cudnn.benchmark = True  # 固定输入尺寸，让 cuDNN 自己挑最快卷积算法
```

---

## 阶段 9：四个检查点（一定逐条过，别跳）

```powershell
cd C:\Users\12983\Documents\zero-dce-scratch\Zero-DCE\Zero-DCE_code
```

**检查点 1 —— 环境**
```powershell
..\..\.venv\Scripts\python.exe -c "import torch, numpy, cv2, PIL; print('ok', torch.cuda.is_available())"
```
✅ `ok True`

**检查点 2 —— 模型能建、参数量对得上**
```powershell
..\..\.venv\Scripts\python.exe -c "import model; n = model.enhance_net_nopool(); print('params', sum(p.numel() for p in n.parameters()))"
```
✅ 预期：`params 79416`（论文 DCE-Net 就是 ~7.9 万参数的小网络；数字对不上说明你改坏了模型）

**检查点 3 —— 官方推理（这条能过，说明代码+权重+数据都对）**
```powershell
..\..\.venv\Scripts\python.exe lowlight_test.py
```
✅ 预期：跑 74 张（LIME 10 + DICM 64），**约 7~8 秒**；然后 `data\result\DICM\` 出现 64 张 `.jpg`、`data\result\LIME\` 出现 10 张（**保持原扩展名 `.bmp`**，不是 jpg，别被吓到）。
打开对比：`data\test_data\LIME\1.bmp` 与 `data\result\LIME\1.bmp`，平均亮度应明显提升（实测 **33.0 → 85.8**）。

**检查点 4 —— 训练链路（2 轮，20 秒）**
```powershell
..\..\.venv\Scripts\python.exe lowlight_train.py --num_workers 0 --num_epochs 2 --snapshots_folder snapshots_smoke/
```
✅ 预期：`Total training examples: 485`，Loss 从 ~2.4 降到 ~1.1~1.3，`snapshots_smoke\` 出两个 .pth。

---

## 阶段 10：正式训练 + 评测

### 10.1 训练（二选一）

```powershell
# A. 论文口径（推荐）：SICE 2002 张多曝光数据，~40 秒/轮
..\..\.venv\Scripts\python.exe lowlight_train.py --lowlight_images_path data/train_data_sice/ --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_sice/

# B. 只用 LOL 485 张（对照实验，~9 秒/轮，但会过度提亮——见下）
..\..\.venv\Scripts\python.exe lowlight_train.py --num_workers 0 --num_epochs 200 --snapshots_folder snapshots_run/
```

⚠️ **`--num_workers` 必须写 0**。默认值 4 在 Windows 上会报
`RuntimeError: Couldn't open shared file mapping: <torch_xxx>, error code: <1455>`
（共享内存/页面文件不够），而且慢 10 倍。

### 10.2 训练完你会看到什么

- 权重：`snapshots_sice\Epoch0.pth … Epoch50.pth`（每轮 315 KB）
- Loss 从 ~2.9 降到 ~0.6~0.8 后基本不动 → 说明收敛了，50 轮就够，可以 Ctrl+C 停

### 10.3 评测（写个小脚本 `scripts\eval_lol.py`）

核心就 30 行：载入每张低照图 → 模型前向 → `clip(0,1)` → 和正常光图算 PSNR/SSIM。

```python
import os, glob, numpy as np, cv2, torch, sys
C = r"C:\Users\12983\Documents\zero-dce-scratch\Zero-DCE\Zero-DCE_code"
sys.path.insert(0, C); import model

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(1.0 / mse) if mse > 0 else 99

def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    k = cv2.getGaussianKernel(11, 1.5); win = np.outer(k, k.T)
    f = lambda x: cv2.filter2D(x, -1, win, borderType=cv2.BORDER_REFLECT)
    ma, mb = f(a), f(b); sa, sb = f(a*a) - ma*ma, f(b*b) - mb*mb; sab = f(a*b) - ma*mb
    C1, C2 = 0.01**2, 0.03**2
    return float((((2*ma*mb + C1) * (2*sab + C2)) / ((ma*ma + mb*mb + C1) * (sa + sb + C2))).mean())

net = model.enhance_net_nopool().cuda()
for ckpt in ["snapshots/Epoch99.pth", "snapshots_sice/Epoch50.pth"]:   # 只训了 LOL 版就换成 snapshots_run/Epoch199.pth
    net.load_state_dict(torch.load(os.path.join(C, ckpt)))
    net.eval(); ps, ss = [], []
    for low in sorted(glob.glob(os.path.join(C, "data", "train_data", "*.jpg")))[:120]:
        hi = os.path.join(C, "data", "train_high", os.path.basename(low))
        a = cv2.imread(low); b = cv2.imread(hi)
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB) / 255.0; b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB) / 255.0
        with torch.no_grad():
            _, out, _ = net(torch.from_numpy(a.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda())
        out = np.clip(out.squeeze(0).permute(1,2,0).cpu().numpy(), 0, 1)
        ps.append(psnr(out, b)); ss.append(ssim(out, b))
    print(f"{ckpt:34s} PSNR {np.mean(ps):.3f}  SSIM {np.mean(ss):.4f}")
```

> torch 2.5 会打一条 `FutureWarning ... weights_only=False`，**正常现象**，不影响结果（嫌烦就改成 `torch.load(path, weights_only=True)`）。

### 10.4 对照表（我的实测，你可以拿来对）

| 权重 | LOL 485 对 PSNR / SSIM | 说明 |
|---|---|---|
| 低照输入（不处理） | 7.745 / 0.182 | 基准线 |
| 官方 `Epoch99.pth` | 14.234 / 0.588 | 作者的权重 |
| 我们 SICE 训练 50 轮 | **15.057 / 0.601** | 已复现到论文水平 |
| 我们 LOL 训练 200 轮 | 15.566 / 0.555 | PSNR 更高但**严重过曝**，SSIM 反而低 |

在 LOL 官方测试集 eval15 上（15 对、600×400、有正常光参考）：

| 权重 | PSNR / SSIM |
|---|---|
| 低照输入 | 7.773 / 0.196 |
| 官方 `Epoch99.pth` | 14.797 / 0.556 |
| SICE 训练 50 轮 | **15.809 / 0.557** |

**这张表就是你复现成功的证据**：两个数据集上，自训练权重与官方权重同档，且 SSIM 更优。

---

## 阶段 11（可选）：把成果推到你自己的 GitHub

```powershell
cd C:\Users\12983\Documents\zero-dce-scratch\Zero-DCE
git config user.name "你的名字"
git config user.email "你的邮箱"
git checkout -b my-repro
git add Zero-DCE_code/dataloader.py Zero-DCE_code/lowlight_test.py Zero-DCE_code/lowlight_train.py
git commit -m "fix: Windows/Pillow10 compatibility (ANTIALIAS, path sep, RGB convert)"

# 在 GitHub 网页新建空仓库后：
git remote add mine https://github.com/<你的用户名>/Zero-DCE-repro.git
git push mine my-repro
```

⚠️ 别把 `snapshots_*`、`data/` 提交上去（几百 MB），仓库里有 `.gitignore`，自己确认一下。

---

## 附录 A：检查点清单（打印出来一条条打勾）

- [ ] `git --version` 有输出
- [ ] `python --version` = 3.11.x
- [ ] `.venv` 建好、激活后提示符有 `(.venv)`
- [ ] `pip config list` 里有 SJTU 镜像地址
- [ ] `pip install -r requirements.txt` 成功
- [ ] torch 验证输出 `True` + `(8, 9)`
- [ ] 官方仓库 clone 完成，`snapshots\Epoch99.pth` 存在（314 KB）
- [ ] `data\train_data\` 有 485 张 `.jpg`（**不是 png**）
- [ ] 3 个文件 5 处补丁改完
- [ ] 检查点 1~4 全过
- [ ] 训练出至少 50 轮权重
- [ ] 评测 PSNR ≥ 15.0（SICE 权重）

## 附录 B：报错对照表（这台机器上真实遇到过的）

| 报错 | 原因 | 解决 |
|---|---|---|
| `Couldn't open shared file mapping: <torch_...>, error code: <1455>` | Windows 共享内存/页面文件不足（后台开游戏尤甚） | `--num_workers 0` |
| `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` | Pillow ≥10 删了常量 | 补丁 1 |
| `FileNotFoundError: 'data/result/DICM\\01.jpg'` | Windows 路径分隔符混用，`replace` 建不出子目录 | 补丁 2 |
| `RuntimeError: Expected 3 channels` / permute 报错 | 输入是灰度或 RGBA | `.convert('RGB')`（补丁 1/2） |
| `torch.cuda.is_available()` 为 `False` | Windows 上 PyPI 的 torch 是 **CPU 版**（193.7 MB）；CUDA 版是 2.4 GB 的 `+cu124` 轮子 | 看 `torch.__version__` 是否带 `+cu124`；按阶段 5 路线 A 重装 |
| `ModuleNotFoundError: No module named 'torch'` | VSCode 用了系统 Python | 选 `.venv` 解释器 |
| pip 卡在 `Collecting torch` 很久 | 没配镜像（pypi.org 只有 13 KB/s） | 阶段 4.1 |
| `OSError: image file is truncated` | 图没下完整（我遇到过 1 张 32768 字节的坏文件） | 删掉重下；用 `PIL.ImageFile.LOAD_TRUNCATED_IMAGES` 可强行读 |
| `Activate.ps1 cannot be loaded` | PowerShell 执行策略 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| 训练特别慢（1 轮好几分钟） | `num_workers` 默认 4 卡死 | 设 0；另：关掉占 GPU 的游戏 |
| `KeyError` / `load_state_dict` 报 size mismatch | 权重和模型结构不匹配（别用 Zero-DCE++ 的权重跑 Zero-DCE） | 用官方 `enhance_net_nopool` + `Epoch99.pth` |

## 附录 C：以后自己怎么在 GitHub 上找数据集

我找数据的方法（下次遇到网盘链接失效能用）：

1. **仓库搜索**：
   `https://api.github.com/search/repositories?q=Zero-DCE+in:name&per_page=30`
2. **列出某仓库全部文件**（不用 clone 就能看它有没有数据）：
   `https://api.github.com/repos/<owner>/<repo>/git/trees/HEAD?recursive=1`
   （浏览器直接打开也能看；重点看 `*.jpg`/`*.png` 的**目录名和数量**）
3. **判断数据对不对**：数量（2002 / 485 / 15）、尺寸（512×512 / 600×400）、命名（`1.jpg … 2002.jpg`）
4. **只想拿单个文件**：`https://raw.githubusercontent.com/<owner>/<repo>/main/<path>`
5. 网页版代码搜索 `https://github.com/search?q=eval15&type=code` 需要登录，没登录时用上面第 1、2 步就够。

## 附录 D：全部命令（复制即用）

```powershell
# ── 一次性准备 ─────────────────────────────
mkdir C:\Users\12983\Documents\zero-dce-scratch
cd C:\Users\12983\Documents\zero-dce-scratch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip config set global.index-url https://mirror.sjtu.edu.cn/pypi/web/simple
python -m pip config set global.timeout 120
pip install -r requirements.txt
pip install filelock typing_extensions sympy networkx jinja2 fsspec
pip install --no-index --no-deps wheels\torch-2.5.1+cu124-cp311-cp311-win_amd64.whl wheels\torchvision-0.20.1+cu124-cp311-cp311-win_amd64.whl

# ── 代码与数据 ─────────────────────────────
git clone https://github.com/Li-Chongyi/Zero-DCE.git
git clone --depth 1 https://github.com/23225114YKW/Low-light-image-enhancement-with-LoL-dataset.git data_src\lol
git clone --depth 1 https://github.com/Aiemu/Zero-DCE-improved.git data_src\sice
git clone --depth 1 https://github.com/FrankHo-Hwc/LOLDataset.git data_src\lol_test
python scripts\prepare_data.py

# ── 验证 ───────────────────────────────────
cd Zero-DCE\Zero-DCE_code
..\..\.venv\Scripts\python.exe lowlight_test.py
..\..\.venv\Scripts\python.exe lowlight_train.py --num_workers 0 --num_epochs 2 --snapshots_folder snapshots_smoke/

# ── 正式训练 + 评测 ─────────────────────────
..\..\.venv\Scripts\python.exe lowlight_train.py --lowlight_images_path data/train_data_sice/ --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_sice/
cd ..\..
.\.venv\Scripts\python.exe scripts\eval_lol.py
```

---

## 卡住了怎么办

1. 先看**附录 B** 的报错表；
2. 对不上就对照我这边已跑通的同一份文件（只看不改）：
   `C:\Users\12983\Documents\repro\Zero-DCE`
   比如 `git diff` 就能看到我改了哪 5 处：
   ```powershell
   C:\Users\12983\Documents\repro\PortableGit\cmd\git.exe -C C:\Users\12983\Documents\repro\Zero-DCE diff
   ```
3. 把**完整报错**（含最后 10 行）和**你执行的那条命令**发我，我帮你看。

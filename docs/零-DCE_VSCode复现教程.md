# 在 VSCode 里自己复现 Zero-DCE（小白版逐步操作）

> 目标：不用记命令，靠点菜单/按快捷键，把"推理 → 训练 → 评测 → 看结果"整套自己跑一遍。
> 预计耗时：跑通推理 5 分钟；训练冒烟 1 分钟；正式训练 45~60 分钟（可以挂着去干别的）。

---

## 0. 你要准备什么

什么都不用准备，机器上已经装好了：
- VSCode（中文界面）
- Python 环境 `C:\Users\12983\Documents\repro\.venv`
- 代码 `C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code`
- 数据（LOL 485 对 + SICE 2002 张 + 测试集 LIME/DICM）

我还在项目里配好了 **9 个任务**，你只要会"选任务"，等于会跑整套流程。

---

## 1. 第一步：用 VSCode 打开"项目文件夹"

**最省事的办法**：打开 `C:\Users\12983\Documents\repro\`，**双击 `open_vscode.bat`**，
VSCode 会自动弹出并打开正确的项目文件夹，直接跳到第 2 步。

手动方式：开始菜单搜 `Visual Studio Code` 打开 → 左上角菜单 **文件(File) → 打开文件夹(Open Folder)** → 选

```
C:\Users\12983\Documents\repro
```

→ 弹窗问"是否信任此文件夹的作者"，点 **是，我信任**。

**打开后你应该看到**：左侧一排文件/文件夹：`Zero-DCE`、`scripts`、`低照图库`、`enhance.py` 等。

> 关键：一定要打开 `repro` 这个**外层文件夹**，不是 `Zero-DCE`。
> 打开错了，下面说的任务列表会不存在。

---

## 2. 第二步：记住 3 个操作就够了

| 要做什么 | 怎么按 | 说明 |
|---|---|---|
| 打开/关闭**终端**（黑色命令行区） | `` Ctrl + ` ``（Tab 上面那个键） | 后面所有运行都在这里 |
| 打开**任务菜单**（自动跑预设命令） | 菜单 **终端(Terminal) → 运行任务(Run Task...)** | 会列出 0~8 号任务 |
| **直接跑默认任务**（任务 1：推理） | `Ctrl + Shift + B` | 最常用 |

第一次打开终端时，最左边应该显示 `(.venv)`，例如：

```
(.venv) PS C:\Users\12983\Documents\repro>
```

看到 `(.venv)` 就说明 Python 环境选对了。

---

## 3. 第三步：确认解释器（30 秒，只需做一次）

**看窗口右下角状态栏**，应该显示类似 `Python 3.11.17 ('.venv')`。

- 如果显示的是别的（比如 `3.12.x` 或 `3.13`），**点它** → 在弹出列表里选带 `.venv` 的那一个：
  `C:\Users\12983\Documents\repro\.venv\Scripts\python.exe`
- 列表里找不到就 **Ctrl+Shift+P** → 输入 `Python: Select Interpreter` → 回车 → 选上面那个路径。

---

## 4. 第四步：环境自检（任务 0）

菜单 **终端 → 运行任务** → 选 `0 环境自检（Python + torch + CUDA）`。

**预期输出**（大约 10 秒）：

```
Python 3.11.17
torch 2.5.1+cu124
CUDA available: True
GPU: NVIDIA GeForce RTX 4060 Laptop GPU
```

**看到 `CUDA available: True` 就说明 GPU 能用**，这是最关键的一句。
如果显示 `False`，先别往下走，把报错发我。

---

## 5. 第五步：跑推理——用官方权重增强 LIME + DICM（任务 1）

按 **`Ctrl + Shift + B`**（或 运行任务 → 选 `1 推理：官方权重跑 LIME+DICM`）。

**它在干什么**：加载作者发布的权重 `snapshots\Epoch99.pth`，把 `data\test_data\` 里 74 张暗图逐张提亮。

**预期**：约 **7~8 秒**跑完，终端最后没有红字。

**看结果**：左侧文件树展开
`Zero-DCE` → `Zero-DCE_code` → `data` → `result` → `DICM`（64 张）/ `LIME`（10 张）
→ 双击任意 `.jpg` 就能在 VSCode 里看到增强后的图。

对比原图：展开 `data` → `test_data` → `LIME`，双击同名原图，来回切标签页对比。

---

## 6. 第六步：训练冒烟测试（任务 3，约 20 秒）

运行任务 → `3 训练冒烟：2 轮`。

**它在干什么**：拿 485 对 LOL 图，只训练 2 个 epoch，纯粹确认"训练链路是通的"。

**预期输出**（节选）：

```
===> Epoch 0 of 2 ...
Epoch:0 iter:0 Loss=... 
Epoch:0 iter:5 Loss=...
```

**重点看 Loss 在下降**：第 0 轮大概 2.4 左右，结束时降到 1.1~1.3 就算正常。

> 这一步的价值：如果你以后手滑改坏了代码，先用它 20 秒验证，不用等 45 分钟才发现错了。

---

## 7. 第七步：正式训练（任务 4，45~60 分钟）

运行任务 → `4 正式训练：SICE 2002 张，100 轮`。

**它在干什么**：用论文真正的训练数据（SICE 多曝光 2002 张，512×512）从零训练，每 10 轮保存一个权重到
`Zero-DCE_code\snapshots_mine\`。

你会看到每轮刷一行损失，例如：

```
===> Epoch 10 of 100 ...
Epoch:10 iter:0 Loss=1.02
...
```

**怎么判断它在正常跑**：大约 **40 秒/轮**，Loss 从 2.x 逐步降到 0.6~0.8 后趋于平稳。

**训练期间**：
- 别关 VSCode，也别让电脑睡眠（设置 → 电源 → 屏幕和睡眠 → 改成"从不"）
- 想中途停：在终端里按 **Ctrl + C**
- 想后台跑（关掉窗口也继续）：用任务 4 跑起来后别管它就行，或者把本轮改成 `--num_epochs 200` 让它更久

**早停也行**：论文级效果 50 轮左右就到了，100 轮是给足余量。

---

## 8. 第八步：定量评测（任务 5，约 3~4 分钟）

运行任务 → `5 评测：485 对 LOL 图上算 PSNR/SSIM`。

**预期输出**（数值会有小幅随机波动）：

```
official_Epoch99   in PSNR 7.745 SSIM 0.1823 | out PSNR 14.234 SSIM 0.5878 | bright 0.060->0.267
ours_Epoch50       in PSNR 7.745 SSIM 0.1823 | out PSNR 15.057 SSIM 0.6007 | bright 0.060->0.309
```

**怎么读这行**：
- `in PSNR` = 原图和正常光图差多远（越低越暗）
- `out PSNR` = 增强后和正常光图差多远（**越高越好**）
- `bright` = 平均亮度变化

**判据**：我们训练的权重 out PSNR ≈ 15.0、SSIM ≈ 0.60，和官方 14.2 / 0.59 同档 → 复现成功。

---

## 9. 第九步：用自己训练的权重看效果（任务 2）

运行任务 → `2 推理：用我自己训练的 SICE 权重增强测试集`。

结果写到 `repro\enhanced_results\`，每张图旁边还多一张 `xxx_compare.png`（原图 | 增强 并排）。
（这一步不会覆盖官方的 `data\result\`，放心跑。）

---

## 10. 第十步：拿自己的照片试（任务 6）

运行任务 → `6 增强我自己的照片（弹窗选图）` → 选图 → 结果在图片旁的 `enhanced` 文件夹，会自动弹出。

---

## 11. 概念小抄（看不懂代码也能明白）

| 词 | 一句话解释 |
|---|---|
| **权重 / checkpoint (.pth)** | 模型学到的东西，就是"参数"，几百 KB。复现的核心产物 |
| **推理 / 测试** | 拿现成权重给新图提亮。快，8 秒 |
| **训练** | 拿成对（暗图 + 正常光图）让模型自己学怎么提亮。慢，按小时算 |
| **epoch** | 把训练集完整过一遍叫 1 个 epoch。100 epoch = 过 100 遍 |
| **iteration / iter** | 训练时每吃一个 batch 算一次，485 张 ÷ batch 8 ≈ 61 iter/epoch |
| **batch size** | 一次喂几张图。默认 8，调大会爆显存 |
| **lr（学习率）** | 每步学多少。默认 1e-4；改成 1e-3 很可能直接学崩 |
| **loss** | 损失，越小说明模型当前输出越接近目标。训练就是把它压下去 |
| **PSNR / SSIM** | 两个图像质量分数：越高越好。PSNR 看像素误差（dB），SSIM 看结构相似度（0~1） |

---

## 12. 想读懂代码：按这个顺序点开看

1. `Zero-DCE_code\model.py` —— **核心就 20 行**：7 层卷积 → 输出 24 通道 → 迭代 8 次 `x = x + r*(x² - x)`。这就是论文的 DCE-Net
2. `Zero-DCE_code\Myloss.py` —— 4 个无参考损失：TV（平滑）/ 空间一致性 / 颜色恒常 / 曝光控制
3. `Zero-DCE_code\dataloader.py` —— 怎么读图、缩放、转 tensor
4. `Zero-DCE_code\lowlight_train.py` —— 训练主循环：前向 → 4 个 loss 加权求和 → 反向传播 → 存权重
5. `Zero-DCE_code\lowlight_test.py` —— 测试脚本：读图 → 一次前向 → 存图

**想跟着代码走一遍**：打开 `lowlight_train.py`，在 `loss.backward()` 那一行左边点一下会出现红点（断点），按 **F5** 启动"调试：训练冒烟 2 轮"，程序就会停在那行，你能看到每个变量的值。按 F10 单步，F5 继续。

---

## 13. 常见报错对照表

| 你看到的报错 | 原因 | 怎么办 |
|---|---|---|
| `Couldn't open shared file mapping ... error code: <1455>` | Windows 共享内存不足（后台开着游戏/内存紧张） | **必须 `--num_workers 0`**，任务 3/4 已经设好了，别手动改 |
| `torch.cuda.OutOfMemoryError` | 显存不够 | 训练加 `--train_batch_size 4`；推理用 `-s 2000` 缩图 |
| `FileNotFoundError: ... data\result\DICM\01.jpg` | 老的 Windows 路径 bug | 已修好，别回滚 `lowlight_test.py` |
| `AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'` | Pillow ≥10 删了这个常量 | 已修好（`dataloader.py` 用 `getattr(Image,"ANTIALIAS",Image.LANCZOS)`） |
| `No module named torch` / `ModuleNotFoundError` | 解释器选错了，用到系统 Python | 回第 3 步选 `.venv` 的解释器 |
| 终端显示一堆红字但还是跑完了 | 是 `FutureWarning` 警告，不是错误 | 忽略即可 |
| 任务列表里只有 0~8 但没有 1 | 打开的文件夹不对 | 打开 `C:\Users\12983\Documents\repro` 而不是 Zero-DCE |

---

## 14. 命令速查（想敲命令行时用）

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code

# 推理（官方权重）
python lowlight_test.py

# 训练冒烟
python lowlight_train.py --num_workers 0 --num_epochs 2 --snapshots_folder snapshots_smoke/

# 正式训练
python lowlight_train.py --lowlight_images_path data/train_data_sice/ --num_workers 0 --num_epochs 100 --snapshots_folder snapshots_mine/

# 评测
cd C:\Users\12983\Documents\repro
$env:CKPTS="snapshots/Epoch99.pth;snapshots_sice/Epoch50.pth"
python scripts\eval_lol.py

# 增强自己的照片
python enhance.py "D:\我的照片" -c -o "D:\增强结果"
```

## 15. 产物在哪

| 产物 | 路径 |
|---|---|
| 官方权重 | `Zero-DCE_code\snapshots\Epoch99.pth` |
| 我已训好的权重 | `Zero-DCE_code\snapshots_sice\Epoch50.pth`、`snapshots_run\Epoch199.pth` |
| 你自己训练出的权重 | `Zero-DCE_code\snapshots_mine\Epoch0.pth … Epoch99.pth` |
| 官方测试结果 | `Zero-DCE_code\data\result\DICM|LIME\` |
| 你自己权重的增强结果 | `repro\enhanced_results\` |
| 增强私人照片 | 你图片旁的 `enhanced\` |
| 对比图（论文风格） | `repro\compare\`、`demo_pairs\` |

## 16. 时间与空间预算

- 推理 74 张：**8 秒**
- 训练冒烟 2 轮：**20 秒**
- 正式训练 100 轮（SICE 2002 张）：**45~60 分钟**
- 评测 485 对：**3~4 分钟**
- 磁盘：每轮权重 315 KB，100 轮约 32 MB；整个项目约 2.5 GB

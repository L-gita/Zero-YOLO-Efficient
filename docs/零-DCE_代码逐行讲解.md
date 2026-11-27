# Zero-DCE 代码逐行讲解（零基础版）

> 配套阅读：`零-DCE_论文汉化与解读.md`（论文原理）、`Zero-DCE_复现报告.md`（本机复现结果与指标）
>
> 代码位置：`C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code\`
> 一共 5 个 `.py` 文件、约 460 行，本文全部讲完。每个文件都是"代码 → 一行行解释 → 为什么这么写"。

---

## 0. 先看全景：这 5 个文件谁管什么

| 文件 | 行数 | 角色 | 一句话 |
|---|---|---|---|
| `model.py` | 59 | **网络结构** | 定义 DCE-Net：7 层小卷积 + 8 次亮度曲线迭代 |
| `dataloader.py` | 61 | **数据管道** | 把磁盘上的 jpg 读成网络要的张量（256×256，0~1） |
| `Myloss.py` | 157 | **损失函数** | 4 个"无参考"损失，告诉网络什么叫"好照片" |
| `lowlight_train.py` | 125 | **训练脚本** | 把上面三样拼起来，反向传播、存权重 |
| `lowlight_test.py` | 64 | **推理脚本** | 读一张图 → 过网络 → 存结果，不做训练 |

### 0.1 训练时数据怎么流

```
data/train_data/*.jpg
      │  dataloader.py：读图→RGB→resize 256×256→/255→CHW 张量
      ▼
[batch=8, 3, 256, 256]  ──►  model.py（DCE-Net）
      │                          │
      │                          ├─► enhanced_image（最终提亮结果）
      │                          ├─► enhanced_image_1（4 次迭代的中间结果，训练时不用）
      │                          └─► r（8×3=24 通道的曲线参数 A）
      │                                   │
      ▼                                   ▼
Myloss.py：4 个损失 ────────────────► loss = 200·L_TV(A) + L_spa + 5·L_color + 10·L_exp
                                          │
                                          ▼
                        loss.backward() → Adam 更新 79,416 个参数
```

### 0.2 推理时数据怎么流

```
data/test_data/LIME/1.bmp
      │ lowlight_test.py：读图→/255→CHW→[1,3,H,W]（H,W 是原图尺寸，不缩放！）
      ▼
DCE-Net（加载 snapshots/Epoch99.pth）
      ▼
取第 2 个返回值 enhanced_image  ──► data/result/LIME/1.bmp
```

**注意一个关键差别**：训练时图片被缩到 256×256；推理时**用原图分辨率**，因为网络是"逐像素 + 局部"的卷积，跟尺寸无关，而且它是全卷积（没有全连接层）。

---

## 1. `model.py`（59 行）—— 网络本体

### 1.1 网络长什么样

```
输入 x [b,3,H,W]
  │
  ├─ conv1 3→32 ─ relu ─► x1 ───────────────┐
  │                        conv2 32→32 ─relu► x2 ──┐
  │                                   conv3 ─► x3   │
  │                                   conv4 ─► x4   │
  │                                 cat(x3,x4)→64    │
  │                                   conv5 ─► x5    │
  │                                 cat(x2,x5)→64    │
  │                                   conv6 ─► x6    │
  │                                 cat(x1,x6)→64    │
  │                                   conv7 64→24 ─► tanh ─► x_r
  │                                                            │
  │                                          split 成 8 组 × 3 通道：r1…r8
  ▼                                                            ▼
  对 x 连续施加 8 次：x ← x + r_k·(x² − x)     ← 这就是"亮度增强曲线"的 8 次迭代
  │
  ├─► enhance_image_1（第 4 次迭代后的结果）
  └─► enhance_image（第 8 次迭代后的最终结果）
```

没有下采样（maxpool 被注释掉了），所以叫 "**no pool**"；但用了 3 次"跳连 concat"，类似 U-Net 的信息融合，只是不改变分辨率。

### 1.2 逐行

```python
1: import torch
2: import torch.nn as nn
3: import torch.nn.functional as F
4: import math
5: #import pytorch_colors as colors
6: import numpy as np
```
- 1–3 是 PyTorch 三件套：张量、网络层、函数式接口。
- 4、6 是历史遗留（`math`、`numpy` 在这个文件里其实没用到）。
- 5 被注释掉了，别管。

```python
8: class enhance_net_nopool(nn.Module):
```
- 继承 `nn.Module` 才能被 PyTorch 识别成"网络"，自动管理参数、`.cuda()`、`.train()`、`state_dict()`。
- 名字里的 `nopool` 是相对另一条分支（带池化的版本）而言的。

```python
13: self.relu = nn.ReLU(inplace=True)
```
- 激活函数，`inplace=True` 表示直接改写输入张量省显存（效果不变）。

```python
15: number_f = 32
16: self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True)
17: self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
18: self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
19: self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
20: self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
21: self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True)
22: self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True)
```
`nn.Conv2d(输入通道, 输出通道, 卷积核, 步长, padding)`：

| 层 | 输入→输出通道 | 作用 |
|---|---|---|
| conv1 | 3 → 32 | 从 RGB 提取第一层特征 |
| conv2–conv4 | 32 → 32 | 逐层扩大感受野（每层 3×3，4 层后能看到 9×9 邻域） |
| conv5 | **64** → 32 | 输入是 `cat(x3,x4)`，所以是 64 通道 |
| conv6 | **64** → 32 | 输入是 `cat(x2,x5)` |
| conv7 | **64** → **24** | 输入是 `cat(x1,x6)`，输出 24 = 8 组曲线 × RGB 3 通道 |

- `padding=1` 是为了 3×3 卷积后尺寸不变（same padding）。
- **为什么最后是 24？** 因为要预测 8 个迭代步、每步 RGB 三个通道各一个曲线参数：8 × 3 = 24。

```python
24: self.maxpool = nn.MaxPool2d(2, stride=2, ...)
25: self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
```
- 这两层**定义了但在 forward 里被全部注释掉**（见 32、34、36、40 行），所以没被用到，也不贡献参数。
- 历史原因：作者从"带池化的版本"改过来时保留了定义。真正把这些砍掉的是后来的 Zero-DCE++。

```python
29: def forward(self, x):
31: x1 = self.relu(self.e_conv1(x))
33: x2 = self.relu(self.e_conv2(x1))
35: x3 = self.relu(self.e_conv3(x2))
37: x4 = self.relu(self.e_conv4(x3))
39: x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
41: x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
43: x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
```
- 前 4 层是"顺次加深"；后 3 层每层都把**前面某一层的特征**拼进来（`torch.cat(...,1)` 在通道维拼接），这就是跳连，让曲线预测同时看到浅层的颜色细节和深层的语义。
- 43 行的 `tanh` 是**关键设计**：把卷积输出压到 **(-1, 1)**。这保证曲线参数 α 不会太大导致过曝或全黑，也让曲线在数学上"可迭代、有界"。

```python
44: r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
```
把 24 通道按每 3 个一组切成 8 组，每组是一个迭代步的 RGB 曲线参数。

```python
47: x = x + r1*(torch.pow(x,2)-x)
48: x = x + r2*(torch.pow(x,2)-x)
49: x = x + r3*(torch.pow(x,2)-x)
50: enhance_image_1 = x + r4*(torch.pow(x,2)-x)
51: x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)
52: x = x + r6*(torch.pow(x,2)-x)
53: x = x + r7*(torch.pow(x,2)-x)
54: enhance_image = x + r8*(torch.pow(x,2)-x)
```
这 8 行是整篇论文的核心——**逐像素二次曲线**：

$$LE(x) = x + \alpha \cdot (x^2 - x)$$

- 输入 x 是 0~1 的像素值，α 是网络预测的曲线参数（-1~1）。
- 因为 x ∈ [0,1] 时 `x² − x ≤ 0`（比如 x=0.3 时是 −0.21），所以：
  - **α < 0 → 像素变大（提亮）**
  - **α > 0 → 像素变小（压暗）**
  - **α = 0 → 输出等于输入（完全不变，网络初始状态）**
- 连续做 8 次，就能把很暗的图一步步拉亮，同时因为曲线是光滑的二次函数，不会把亮区截断成死白。

**例：x = 0.1（很暗的像素），α = −0.5**

| 迭代 | 计算 | 结果 |
|---|---|---|
| 1 | 0.1 + (−0.5)×(0.01−0.1) | 0.145 |
| 2 | 0.145 + (−0.5)×(0.021−0.145) | 0.207 |
| 3 | 0.207 + (−0.5)×(0.043−0.207) | 0.289 |
| 4 | … | 0.392 |
| 5 | … | 0.511 |
| 6 | … | 0.636 |
| 7 | … | 0.752 |
| 8 | … | **0.845** |

这就是"**一条曲线不够，就迭代 8 次**"的含义：迭代次数 = 表达能力。
注意后面几步涨得越来越快（0.392→0.845），因为像素越来越接近 0.5，而 `|x²−x|` 在 x=0.5 处取最大值 0.25 —— 曲线在"中间调"发力最猛，这也是为什么过曝问题总是先出现在中间调。

**实测三组权重的对比**（LIME/1.bmp，原图平均亮度 0.129，即 33/255）：

| 权重 | α 均值 | 1 次迭代 | 2 次 | 4 次 | 8 次（最终） |
|---|---|---|---|---|---|
| 官方 `snapshots/Epoch99.pth` | **−0.241** | 37.0 | 41.6 | 52.9 | **85.8** |
| 我们 SICE 训的 `Epoch50.pth` | −0.277 | 37.1 | 42.0 | 54.4 | **95.9** |
| 我们 LOL 训的 `Epoch199.pth`（过曝对照） | **−0.494** | 49.4 | 66.6 | 98.6 | **161.4** |

可以看到：α 越负，提亮越猛。LOL 那一版 α 直接是官方的两倍，所以 8 次迭代后亮度 161（严重过曝）。
在 LOL 图片上测的平均值也一致：官方 −0.284、我们 SICE −0.28 左右、LOL 版 −0.5。

> 这说明 **α 的均值就是"模型提亮力度"的度量**，甚至可以当成诊断工具：把权重换掉后 print 一下 `r.mean()`，就知道这个模型偏保守还是偏激进。

```python
56: return enhance_image_1, enhance_image, r
```
返回三个东西：
1. `enhance_image_1`：迭代 4 次的结果（训练时没用，是给可视化/扩展留的）
2. `enhance_image`：迭代 8 次的结果 = **真正要的输出**
3. `r`（即 A）：24 通道曲线参数，**只用于计算 L_TV 损失**（让它空间上平滑）

### 1.3 参数量 79,416 是怎么来的

| 层 | 权重 | 偏置 | 小计 |
|---|---|---|---|
| conv1 | 3×32×3×3 = 864 | 32 | 896 |
| conv2 | 32×32×3×3 = 9,216 | 32 | 9,248 |
| conv3 | 9,216 | 32 | 9,248 |
| conv4 | 9,216 | 32 | 9,248 |
| conv5 | 64×32×3×3 = 18,432 | 32 | 18,464 |
| conv6 | 18,432 | 32 | 18,464 |
| conv7 | 64×24×3×3 = 13,824 | 24 | 13,848 |
| **合计** | | | **79,416** |

只有 7.9 万个参数（对比 VGG16 一亿多），所以训练快、推理快、权重文件才 315 KB。

---

## 2. `dataloader.py`（61 行）—— 数据管道

```python
13: random.seed(1143)
```
固定打乱顺序的种子，保证每次跑实验数据顺序一样（可复现）。

```python
16: def populate_train_list(lowlight_images_path):
21:     image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
23:     train_list = image_list_lowlight
25:     random.shuffle(train_list)
27:     return train_list
```
- 扫描目录下**所有 .jpg** 组成列表并打乱。
- ⚠️ **坑 1**：只认 `*.jpg`。如果你放的是 png/bmp（比如 LIME 的 bmp），一张都读不到。这就是为什么当初我把 LOL 的 png 全部转成了 jpg。
- ⚠️ **坑 2**：路径必须**以 `/` 结尾**（默认 `data/train_data/`），否则拼出来是 `data/train_data1.jpg`。

```python
31: class lowlight_loader(data.Dataset):
```
自定义数据集类，必须实现 `__init__` / `__getitem__` / `__len__` 三个方法，PyTorch 的 DataLoader 才能用。

```python
33: def __init__(self, lowlight_images_path):
35:     self.train_list = populate_train_list(lowlight_images_path)
36:     self.size = 256
38:     self.data_list = self.train_list
39:     print("Total training examples:", len(self.train_list))
```
- `size = 256`：**训练分辨率**。论文用 512×512，官方代码训练时缩到 256×256，为了省显存；反正网络是全卷积的，测试时可以任意尺寸。
- 39 行会打印训练集数量（我们的 SICE 数据集打印 `Total training examples: 2002`）。

```python
44: def __getitem__(self, index):
46:     data_lowlight_path = self.data_list[index]
48:     data_lowlight = Image.open(data_lowlight_path).convert('RGB')   # ← 我们加的 .convert('RGB')
50:     resample = getattr(Image, "ANTIALIAS", Image.LANCZOS)           # ← 我们改的
52:     data_lowlight = data_lowlight.resize((self.size,self.size), resample)
54:     data_lowlight = (np.asarray(data_lowlight)/255.0)
55:     data_lowlight = torch.from_numpy(data_lowlight).float()
57:     return data_lowlight.permute(2,0,1)
```
逐行：
1. 取第 index 个文件路径。
2. PIL 打开。`.convert('RGB')` 是**我们加的补丁**：灰度图 / 带 alpha 通道 / CMYK 图会让后面 `permute(2,0,1)` 崩掉（只有 2 或 4 通道）。
3. 选重采样算法。原代码写 `Image.ANTIALIAS`，**Pillow 10 起这个常量被删了**，会直接 `AttributeError`。我们改成"有就用，没有就用 LANCZOS"（两者是同一个滤波器）。
4. 缩放到 256×256。
5. 像素值 /255 → 归一化到 **0~1**（不是 ImageNet 那种减均值除方差，因为这里网络输出也要 0~1 的图）。
6. numpy → torch 张量，转 float32。
7. `permute(2,0,1)`：把 PIL/numpy 的 **(高, 宽, 通道)** 变成 PyTorch 要的 **(通道, 高, 宽)**。

```python
59: def __len__(self):
60:     return len(self.data_list)
```
告诉 DataLoader 一共多少个样本。

---

## 3. `Myloss.py`（157 行）—— 4 个损失函数

先记住**总损失**（在 `lowlight_train.py` 里拼的）：

$$\mathcal{L} = 200\cdot L_{TV} + L_{spa} + 5\cdot L_{color} + 10\cdot L_{exp}$$

| 损失 | 类 | 管什么 | 直观含义 |
|---|---|---|---|
| `L_TV` | 90 行 | 曲线参数的空间平滑 | 相邻像素的提亮幅度不要突变（别出现"一块亮一块暗"） |
| `L_spa` | 29 行 | 空间结构一致 | 提亮后，相邻区域的**亮度差**要和原图保持一致（别把边缘抹平） |
| `L_color` | 9 行 | 颜色恒常 | 输出图 R/G/B 三个通道的平均值要相等（防偏色/发绿发紫） |
| `L_exp` | 74 行 | 曝光控制 | 平均亮度的目标是 **0.6**（太暗要加，太亮要减） |

**"无监督 / 零参考"是什么意思**：这 4 个损失全部只用"输入图 + 输出图自己"，**不需要正常光参考图（GT）**。所以训练数据只要一堆暗图就行，不需要配对——这正是 Zero-DCE 的卖点。

### 3.1 `L_color`（9–26 行）：颜色恒常性

```python
16: b,c,h,w = x.shape
18: mean_rgb = torch.mean(x,[2,3],keepdim=True)    # 对每个通道在整图上求平均 → [b,3,1,1]
19: mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
20: Drg = torch.pow(mr-mg,2)
21: Drb = torch.pow(mr-mb,2)
22: Dgb = torch.pow(mb-mg,2)
23: k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
26: return k
```
- 先算整图每个通道的平均值 mR、mG、mB。
- 再算两两差的平方，最后开根号 → 相当于 (mR−mG, mR−mB, mG−mB) 这个向量的长度。
- 如果三个通道平均亮度相等（灰世界假设），这个值为 0。
- **为什么需要**：暗图提亮时很容易把某个通道拉过头，出现整体偏绿/偏紫。这个损失把三通道平均亮度往"相等"拉。

### 3.2 `L_spa`（29–73 行）：空间一致性

```python
34-37: kernel_left/right/up/down = 4 个 3×3 差分核
38-41: 包成 nn.Parameter(requires_grad=False)  # 固定权重，不训练
42: self.pool = nn.AvgPool2d(4)
```
定义了 4 个方向的"求相邻像素差"的卷积核，比如 left 是 `[[0,0,0],[-1,1,0],[0,0,0]]`，卷积后就是"右邻居 − 自己"。

```python
46: org_mean = torch.mean(org,1,keepdim=True)       # 输入图转灰度
47: enhance_mean = torch.mean(enhance,1,keepdim=True)
49: org_pool = self.pool(org_mean)                  # 4×4 平均池化，降噪
50: enhance_pool = self.pool(enhance_mean)
...
56-64: 对两者分别做 4 个方向的差分
66-69: D_left = (原图左差 − 输出图左差)²  …同理右/上/下
70: E = D_left + D_right + D_up + D_down
73: return E
```
公式化就是论文里的：

$$L_{spa}=\frac{1}{K}\sum_{i}\sum_{j\in\{l,r,u,d\}}\left|\,(Y_i-Y_j)-(I_i-I_j)\,\right|^2$$

即：**输出图相邻两点之差 应该等于 输入图相邻两点之差**。这样提亮的时候，原本"这里暗、旁边亮"的结构不会被破坏（保护边缘和纹理对比）。

> 52–53 行算了 `weight_diff` 和 `E_1`，但 73 行 return 的是 `E`，所以那两行是**作者遗留的死代码**，不影响结果，别被它们绕晕。

### 3.3 `L_exp`（74–88 行）：曝光控制损失

```python
76: def __init__(self, patch_size, mean_val):
79:     self.pool = nn.AvgPool2d(patch_size)   # patch_size=16
80:     self.mean_val = mean_val               # mean_val=0.6
81: def forward(self, x):
84:     x = torch.mean(x,1,keepdim=True)       # 转灰度
85:     mean = self.pool(x)                    # 每 16×16 小块求平均亮度
87:     d = torch.mean(torch.pow(mean - 0.6, 2))  # 与 0.6 的平方差，再对全图求平均
88:     return d
```
- 把图切成 16×16 的小块，每块的平均亮度都希望接近 **0.6**（也就是 153/255，视觉上"中灰偏亮"）。
- **和我们的实测对得上**：训练好的模型输出平均亮度在 0.5~0.6 区间，就是这个损失在起作用。

### 3.4 `L_TV`（90–103 行）：曲线参数的总变差

```python
96-100: 计算水平和竖直方向的相邻像素差，并做归一化
101: h_tv = Σ (x[:,:,1:,:] - x[:,:,:h-1,:])²
102: w_tv = Σ (x[:,:,:,1:] - x[:,:,:,:w-1])²
103: return 2*(h_tv/count_h + w_tv/count_w)/batch_size
```
注意它作用的对象是 **A（24 通道曲线参数）**，不是图像本身（见训练脚本 63 行）。所以是让"提亮曲线在空间上平滑变化"，避免局部提亮忽强忽弱造成光晕/斑块。

### 3.5 没用到的两个类

- `Sa_Loss`（104–123）：基于颜色偏差的一个损失，训练脚本没实例化它。
- `perception_loss`（125–157）：VGG16 感知损失（需要下载预训练权重），同样没被用到。
- 顶部 `from torchvision.models.vgg import vgg16` 只是导入函数，**不会触发下载**；只要不实例化 `perception_loss` 就完全没影响。

---

## 4. `lowlight_train.py`（125 行）—— 训练脚本

```python
17: def weights_init(m):
19:     if classname.find('Conv') != -1:
20:         m.weight.data.normal_(0.0, 0.02)          # 正态分布 μ=0, σ=0.02
21:     elif classname.find('BatchNorm') != -1:
```
- 初始化策略：卷积权重用很小的随机数（σ=0.02），偏置保持 0。
- **为什么关键**：初始参数接近 0 → tanh 输出 α ≈ 0 → `x' = x + 0·(x²−x) = x` → **网络一开始的输出就等于输入**（恒等变换）。训练只是在这个基础上慢慢学会"该往哪调"，所以训练曲线很稳，不会一开头就把图炸掉。

```python
31: os.environ['CUDA_VISIBLE_DEVICES']='0'
32: cudnn.benchmark = True      # ← 我们加的（输入尺寸固定，让 cuDNN 自动选最快卷积算法）
34: DCE_net = model.enhance_net_nopool().cuda()
36: DCE_net.apply(weights_init)
37: if config.load_pretrain == True:
38:     DCE_net.load_state_dict(torch.load(config.pretrain_dir))
```
- 31 行写死用 0 号 GPU（多卡时才需要改）。
- 34 `.cuda()` 写死——**这就是为什么必须装 GPU 版 torch**，纯 CPU 环境会直接报错。
- 36 对每一层应用初始化。
- 37–38 是否加载预训练权重（默认 False）。

```python
39: train_dataset = dataloader.lowlight_loader(config.lowlight_images_path)
41: train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=config.num_workers, pin_memory=True)
```
- DataLoader 负责：每次取 8 张、随机打乱、开子进程读图。
- ⚠️ **坑**：`num_workers=4` 在这台机器上会崩（`Couldn't open shared file mapping ..., error code: <1455>`），必须 `--num_workers 0`。

```python
45: L_color = Myloss.L_color()
46: L_spa = Myloss.L_spa()
48: L_exp = Myloss.L_exp(16,0.6)     # 16×16 小块，目标亮度 0.6
49: L_TV = Myloss.L_TV()
52: optimizer = torch.optim.Adam(DCE_net.parameters(), lr=1e-4, weight_decay=1e-4)
```
- 实例化 4 个损失；优化器用 Adam，学习率 1e-4，权重衰减 1e-4（=L2 正则，防过拟合）。

```python
56: for epoch in range(config.num_epochs):
57:     for iteration, img_lowlight in enumerate(train_loader):
59:         img_lowlight = img_lowlight.cuda()
61:         enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
```
两层循环：epoch（把所有数据过一遍算一个 epoch）× iteration（一个 batch）。

```python
63: Loss_TV  = 200 * L_TV(A)                        # 曲线平滑，权重最大
65: loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
67: loss_col = 5 * torch.mean(L_color(enhanced_image))
69: loss_exp = 10 * torch.mean(L_exp(enhanced_image))
73: loss = Loss_TV + loss_spa + loss_col + loss_exp
```
这就是论文公式的代码版。注意 **L_spa 的权重是 1，但它内部已经对全图求了和**（不是平均），所以实际影响很大；**L_exp 权重 10** 决定整体亮度目标。

```python
77: optimizer.zero_grad()
78: loss.backward()
79: torch.nn.utils.clip_grad_norm(DCE_net.parameters(), 0.1)
80: optimizer.step()
```
标准三步：清梯度 → 反向传播求梯度 → **梯度裁剪**（把梯度总范数限制在 0.1，防止某一步梯度爆炸）→ 更新参数。
（小提示：`clip_grad_norm` 这个不带下划线的老名字在 torch 2.5 里会打一条 `FutureWarning`，提醒将来要改成 `clip_grad_norm_`，**不影响运行**，看到别慌。）

```python
82: if ((iteration+1) % config.display_iter) == 0:
83:     print("Loss at iteration", iteration+1, ":", loss.item())
84: if ((iteration+1) % config.snapshot_iter) == 0:
86:     torch.save(DCE_net.state_dict(), config.snapshots_folder + "Epoch" + str(epoch) + '.pth')
```
- `display_iter=10`：每 10 个 iteration 打印一次 loss。
- `snapshot_iter=10`：每 10 个 iteration 存一次权重。**坑**：文件名叫 `Epoch{当前epoch}.pth`，同一 epoch 内会被反复覆盖 25 次（2002 张 / batch 8 = 250 iter，250/10=25），所以最后每个 epoch 只留 1 个文件——我们用 SICE 跑 51 个 epoch 就正好得到 `Epoch0.pth … Epoch50.pth`。

```python
91: if __name__ == "__main__":
93:     parser = argparse.ArgumentParser()
96:     parser.add_argument('--lowlight_images_path', default="data/train_data/")
97:     ... --lr 0.0001 / --weight_decay 0.0001 / --grad_clip_norm 0.1
100:    ... --num_epochs 200 / --train_batch_size 8 / --num_workers 4
104:    ... --display_iter 10 / --snapshot_iter 10 / --snapshots_folder snapshots/
107:    ... --load_pretrain False / --pretrain_dir snapshots/Epoch99.pth
112:    if not os.path.exists(config.snapshots_folder):
113:        os.mkdir(config.snapshots_folder)
116:    train(config)
```
- `if __name__ == "__main__":`：只有直接 `python lowlight_train.py` 才执行，被 import 时不执行（所以 `num_workers>0` 理论上安全，但本机内存不够）。
- 112–113：输出目录不存在就创建。

---

## 5. `lowlight_test.py`（64 行）—— 推理脚本

```python
21: os.environ['CUDA_VISIBLE_DEVICES']='0'
22: data_lowlight = Image.open(image_path).convert('RGB')     # ← 我们加的 convert
26: data_lowlight = (np.asarray(data_lowlight)/255.0)
29: data_lowlight = torch.from_numpy(data_lowlight).float()
30: data_lowlight = data_lowlight.permute(2,0,1)
31: data_lowlight = data_lowlight.cuda().unsqueeze(0)          # 加 batch 维 → [1,3,H,W]
```
预处理和训练时一样（/255、HWC→CHW），**但少了 resize**——直接原图进网络。

```python
33: DCE_net = model.enhance_net_nopool().cuda()
34: DCE_net.load_state_dict(torch.load('snapshots/Epoch99.pth'))
```
**注意**：这两行写在 `lowlight()` 函数里，意味着**每处理一张图都会重新建一次网络、重新读一次权重**。74 张图就读了 74 次权重，非常低效但不会错（这也是为什么它慢）。工程上应该把模型建到函数外只建一次。

```python
35: start = time.time()
36: _, enhanced_image, _ = DCE_net(data_lowlight)
38: end_time = (time.time() - start)
39: print(end_time)
```
- 只取第 2 个返回值（最终结果），前两个 `_` 是丢弃的中间结果和曲线参数。
- 38–39 打印单张推理耗时（实测约 0.005 秒；这是纯 GPU 计算时间，不含读图写图）。

```python
40: image_path = image_path.replace('test_data','result')
41: result_path = image_path
```
把路径里的 `test_data` 替换成 `result`：
`data/test_data/LIME/1.bmp` → `data/result/LIME/1.bmp`

```python
44: result_dir = os.path.dirname(result_path.replace('\\','/'))
45: os.makedirs(result_dir, exist_ok=True)
```
**这是我们修的 bug**。原代码用 `image_path.replace('/'+image_path.split("/")[-1],'')` 取目录，在 Windows 上路径是 `data/test_data/LIME\1.bmp` 这种"混合分隔符"，那个技巧只能得到 `data/result`，**建不出 `LIME` 子目录**，于是保存时报
`FileNotFoundError: [Errno 2] No such file or directory: 'data/result/LIME\\1.bmp'`。
改成 `os.path.dirname` + `makedirs(exist_ok=True)` 后，任意层级目录都能自动建。

```python
47: torchvision.utils.save_image(enhanced_image, result_path)
```
- 把 [0,1] 的张量保存成图片，**扩展名跟着原图走**（bmp 存 bmp、jpg 存 jpg），输出会被 clamp 到 [0,1]。
- 小细节：存成 jpg 时是有损压缩（默认 quality 75），这也是为什么 DICM 的结果文件比原图小。

```python
49: if __name__ == '__main__':
51:     with torch.no_grad():
52:         filePath = 'data/test_data/'
54:         file_list = os.listdir(filePath)
56:         for file_name in file_list:
57:             test_list = glob.glob(filePath+file_name+"/*")
58:             for image in test_list:
60:                 print(image)
61:                 lowlight(image)
```
- `torch.no_grad()`：推理不需要梯度，关掉省显存、更快。
- 外层遍历 `data/test_data/` 下每个**子目录**（LIME、DICM…），内层遍历该目录下所有文件 → 一共 74 张。
- 加新测试集只要建 `data/test_data/我的图/` 丢进去即可。

---

## 6. 我们相对官方改了什么（5 处，都在复现报告里有记录）

| # | 文件:行 | 原文 | 改成 | 为什么 |
|---|---|---|---|---|
| 1 | `dataloader.py:51` | `Image.ANTIALIAS` | `getattr(Image,"ANTIALIAS",Image.LANCZOS)` | Pillow≥10 删了这个常量，必崩 |
| 2 | `dataloader.py:48` | `Image.open(p)` | `Image.open(p).convert('RGB')` | 灰度/带 alpha 图会让 `permute` 报错 |
| 3 | `lowlight_test.py:22` | 同上 | 同上 | 同上（测试图里有 bmp/灰度） |
| 4 | `lowlight_test.py:42-45` | 用字符串 replace 取目录 | `os.path.dirname` + `makedirs` | Windows 混合分隔符建不出子目录 |
| 5 | `lowlight_train.py:32` | 无 | `cudnn.benchmark = True` | 输入尺寸固定，让 cuDNN 选最快算法 |

---

## 7. 看着奇怪但别乱改的地方（新手 FAQ）

| 现象 | 解释 |
|---|---|
| `class enhance_net_nopool` 里有 `maxpool` 却不用 | 历史遗留，forward 里注释掉了；不影响结果 |
| 到处都是 `.cuda()` 写死 | 2019 年的代码风格，作者假设一定有 GPU。所以必须装 CUDA 版 torch |
| `torch.FloatTensor([...]).cuda()` 写法 | 老式建张量，torch 2.5 仍能跑，只是不优雅 |
| `Myloss.py` 导入了 vgg16 | 只导入函数不会下载权重，只要不实例化 `perception_loss` 就没事 |
| 训练集只读 `*.jpg` | 用 png 会"总样本数 = 0"然后除零报错 |
| `num_workers=4` 默认值 | 本机内存不足会崩，命令行传 `--num_workers 0` |
| 一个 epoch 只留一个 ckpt | `snapshot_iter=10` + 同名覆盖，正常现象 |
| 训练 256、测试原图 | 全卷积网络，尺寸无关 |
| 损失函数没有 GT | 这是"零参考"方法的本质：4 个损失都只看输入/输出自己 |

---

## 8. 建议的读码顺序（已经会了以后复习用）

1. `model.py` 全部（最短、最重要，理解 LE 曲线）
2. `lowlight_test.py` 的 `lowlight()` 函数（10 行看懂一次推理）
3. `dataloader.py` 的 `__getitem__`（看懂张量格式）
4. `lowlight_train.py` 的 56–80 行（训练循环）
5. `Myloss.py` 四个损失按 `L_exp → L_spa → L_color → L_TV` 顺序读（从直觉到抽象）

### 可以自己动手做的小实验

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code

# 1) 看网络的三个返回值形状和曲线参数范围（实测：a、b 是 [2,3,64,64]，r 是 [2,24,64,64]）
python -c "import torch,model; n=model.enhance_net_nopool().cuda(); x=torch.rand(2,3,64,64).cuda(); a,b,r=n(x); print(a.shape,b.shape,r.shape, round(float(r.min()),4), round(float(r.max()),4))"

# 2) 数参数（实测 79416）
python -c "import model; print(sum(p.numel() for p in model.enhance_net_nopool().parameters()))"

# 3) 验证"初始权重 ≈ 恒等变换"：随机初始化不训练，输出应几乎等于输入
#    实测 max|y-x| ≈ 0.057、mean|y-x| ≈ 0.017（输入的 1.7%，基本可忽略）
python -c "import torch,model; n=model.enhance_net_nopool().cuda(); n.apply(lambda m: m.weight.data.normal_(0,0.02) if m.__class__.__name__.find('Conv')>=0 else None); x=torch.rand(1,3,64,64).cuda(); _,y,_=n(x); print('max|y-x|', round(float((y-x).abs().max()),5), 'mean|y-x|', round(float((y-x).abs().mean()),6))"
```

（第 3 条的意义：验证了"初始化即恒等"这个设计——所以训练一开始 loss 就在 2~3 左右而不是爆炸，然后 α 慢慢被学成负值，图才开始变亮。）

---

## 9. 一句话总结

> 一个 7.9 万参数的小卷积网，**根据输入图给每个像素预测一条二次曲线的参数**，然后把这条曲线连续施加 8 次完成提亮；训练时不用任何正常光参考图，只靠 4 个"曝光合适、颜色均衡、结构不变、曲线平滑"的约束来学习。

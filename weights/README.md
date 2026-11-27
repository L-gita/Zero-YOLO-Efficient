# 权重文件说明（weights/）

仓库里只放**推理必需的 4 个权重**（每个约 300 KB），训练产生的全部 checkpoint 不进仓库（体积大、可由命令重新生成）。

| 文件 | 来源 | 用途 | 推荐度 |
|---|---|---|---|
| `official_Epoch99.pth` | 李重义官方仓库 `snapshots/Epoch99.pth` **原封未动** | 官方预训练权重，做「官方 vs 我们」对比 | ⭐⭐⭐ 对比基准 |
| `sice_Epoch49.pth` | 本项目训练（SICE 2002 张，50 轮，无边缘损失） | **原版 Zero-DCE 基线**，与改进版同轮数，A/B 对比必须用它 | ⭐⭐⭐ AB 对比用 |
| `sice_Epoch50.pth` | 本项目训练（同一轮次的第 50 轮） | 日常增强基线（`enhance.py -m sice`） | ⭐⭐ 日常增强用 |
| `edge_Epoch49.pth` | 本项目训练（SICE 2002 张，50 轮，**λ_edge = 0.22**） | 改进版（论文 4.9~4.12 的 Sobel 边缘损失） | ⭐⭐⭐ 改进版 |

## 用法

```powershell
# 用官方权重（原版行为）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg -m official -c

# 用改进版权重（+边缘损失）
python C:\Users\12983\Documents\repro\enhance.py 你的照片.jpg -m edge -c

# 一键出「原版增强 vs 改进版增强」对比（同轮数权重，默认就是 Epoch49 对 Epoch49）
python C:\Users\12983\Documents\repro\scripts\ab_enhance_compare.py 你的照片.jpg
```

也可以用绝对路径直接指定任意权重：

```powershell
python enhance.py 照片.jpg -m C:\Users\12983\Documents\repro\weights\edge_Epoch49.pth
```

## 重新训练得到同样的权重

```powershell
cd C:\Users\12983\Documents\repro\Zero-DCE\Zero-DCE_code

# 基线（不带边缘损失）
python lowlight_train.py --lowlight_images_path data/train_data_sice/ `
    --num_workers 0 --num_epochs 50 --snapshots_folder snapshots_sice/

# 改进版（带边缘损失 λ=0.22）
python lowlight_train.py --lowlight_images_path data/train_data_sice/ `
    --num_workers 0 --num_epochs 50 --snapshots_folder snapshots_sice_edge/ --edge_weight 0.22
```

> 训练 50 轮 ≈ 25 分钟（RTX 4060 Laptop，约 28 s/epoch）。数据集来自
> `https://github.com/Aiemu/Zero-DCE-improved` 的 `src/data/train_data`（2002 张 512×512 多曝光图）。

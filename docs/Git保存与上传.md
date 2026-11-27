# 用 Git 保存和上传本项目（傻瓜版）

> 这份说明写给「没怎么用过 git」的情况。照着敲就行，所有命令都在
> **`C:\Users\12983\Documents\repro`** 目录下执行。

---

## 0. 先认识三个词

| 词 | 大白话 |
|---|---|
| **仓库（repository）** | 一个被 git 管理的文件夹，会记录每一次改动 |
| **提交（commit）** | 给当前状态拍一张「快照」，附一句说明，以后能回到这张快照 |
| **远程（remote）/ 推送（push）** | 把本地快照上传到 GitHub，换电脑/给导师看 |

本项目**已经初始化成一个 git 仓库并完成了第一次提交**，你以后只需要「改完 → 提交 →（可选）推送」。

---

## 1. 每次改完代码，怎么存一张快照（提交）

打开 PowerShell，进入项目目录：

```powershell
cd C:\Users\12983\Documents\repro

# 用便携版 git（本项目自带，不用装到系统里）
$git = ".\PortableGit\cmd\git.exe"

# ① 看看改了什么
& $git status

# ② 把所有改动加入「待提交」
& $git add -A

# ③ 拍快照（说明写清楚这次改了什么）
& $git commit -m "加上了预处理：色度降噪 + 去块 + 暗部轻抬"

# ④ 看历史（每次快照都在这里）
& $git log --oneline --graph -10
```

> **建议习惯**：每完成一个小功能就提交一次，别攒一大坨。提交说明用中文完全没问题。

想省事的话，可以直接双击我准备的 **`保存到Git.bat`**（见第 5 节）。

---

## 2. 想回到之前某次快照

```powershell
& $git log --oneline            # 找到那条提交的编号，比如 a1b2c3d
& $git show a1b2c3d             # 先看看那次改了什么
& $git checkout a1b2c3d -- .    # 把文件恢复到那次快照（工作区改动会被覆盖，先提交！）
```

如果只是「撤销还没提交的改动」：

```powershell
& $git restore .                # 全部撤销
& $git restore 某个文件.py      # 只撤销一个文件
```

---

## 3. 上传到 GitHub（第一次）

1. 打开 https://github.com/new ，新建一个仓库：
   - 名字随便，比如 `zero-dce-graduation`
   - **不要**勾选 “Add a README / .gitignore / license”（本地已经有了，避免冲突）
   - 公开（Public）或私有（Private）都行
2. 回到 PowerShell，在项目目录执行（把 `<你的用户名>` 换成你的 GitHub 用户名）：

```powershell
cd C:\Users\12983\Documents\repro
$git = ".\PortableGit\cmd\git.exe"

# 告诉 git 你的名字和邮箱（第一次需要，写上 GitHub 的邮箱）
& $git config user.name  "你的名字"
& $git config user.email "你的邮箱@example.com"

# 添加远程仓库地址
& $git remote add origin https://github.com/<你的用户名>/zero-dce-graduation.git

# 推送（第一次会弹出登录窗口，用浏览器登录 GitHub 即可）
& $git push -u origin main
```

推送成功后，刷新 GitHub 页面就能看到全部代码、脚本、文档和 4 个权重文件。

> **上传会花一点时间**：仓库里大约 70~80 MB（主要是官方测试图 30 MB + 报告图表）。
> 网速慢的话第一次可能几分钟，之后就只传改动部分。

---

## 4. 以后改完怎么再推一次

```powershell
cd C:\Users\12983\Documents\repro
$git = ".\PortableGit\cmd\git.exe"
& $git add -A
& $git commit -m "这次改了什么"
& $git push
```

就三步。

---

## 5. 一键脚本：`保存到Git.bat`

项目根目录里有一个 **`保存到Git.bat`**，双击它会：

1. 显示当前改了什么（`git status`）；
2. 让你输入一句提交说明（直接回车就用默认说明）；
3. 自动 `add` + `commit`；
4. 如果已经配置过远程仓库，还会自动 `push`。

> ⚠️ 双击前请确认已经做过第 3 节的「添加远程仓库」；没配置远程时它只会本地提交，不会报错。

---

## 6. 常见问题

**Q：`git status` 里一大堆文件，看着乱？**
A：那些是被 `.gitignore` 排除的大文件（数据集、虚拟环境、训练权重、日志），本来就**不该**进仓库。仓库里只放代码/脚本/文档/关键权重。

**Q：不小心 `git add` 了一个大文件怎么办？**
A：先 `& $git reset 文件名` 把它从待提交里拿出来，再把它的路径写进 `.gitignore`。

**Q：`push` 报错 `Authentication failed` / 弹窗登录失败？**
A：GitHub 已经不支持账号密码推送。解决办法：用浏览器登录，或者去 GitHub → Settings → Developer settings → Personal access tokens 生成一个 token，密码栏填 token。

**Q：仓库里的中文文件名在 GitHub 上显示成 `\344\270\255...`？**
A：不影响文件内容。想正常显示可以在仓库里执行：
```powershell
& $git config core.quotepath false
```

**Q：想把数据集也传上去？**
A：不建议（几百 MB 到 GB 级，GitHub 单文件上限 100 MB）。数据集用脚本重新下载即可，见 `scripts\prepare_data.py` 和 `docs\零-DCE_从零搭建指引.md`。

---

## 7. 仓库里都放了什么

| 目录/文件 | 内容 |
|---|---|
| `Zero-DCE/` | 官方代码 + 我们的 5 处修改（`Myloss.py` 加了 `SobelMag`/`L_edge`；`lowlight_train.py` 加了 `--edge_weight` 开关；dataloader/test 的兼容性修复） |
| `scripts/` | 全部自研脚本：数据准备、训练冒烟、评测、对比、画图、批量 A/B 对比、马赛克诊断 |
| `docs/` | 全部报告与教程（复现报告、边缘损失实验报告、代码逐行讲解、使用指南、从零搭建指引……） |
| `weights/` | 4 个推理权重（官方 + 基线 + 改进版），见 `weights/README.md` |
| `demo_edge/figures/` | 汇报用的 9 张图表 |
| `enhance.py` | 一键增强照片的入口 |
| `pick_and_enhance.ps1` + 3 个 `.bat` | 双击/拖拽图形入口 |
| `README.md` | 项目总说明 |

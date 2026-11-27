# 零基础看懂 Git：它是什么、怎么保存、怎么查看、怎么调用

> 这份文档写给"第一次用 Git"的人。看完你只需要记住 4 个动作：**看状态、存快照、翻旧版、传网盘**。

---

## 0. 一句话理解 Git

**Git = 给你项目文件夹装了一台"时间机器 + 相册"。**

- 你每按一次"保存快照"（叫 **提交 commit**），Git 就把那一刻**所有文件的完整样子**记下来，并贴一张便签（说明你改了什么）。
- 以后随时可以翻到任何一张快照，看当时是什么样、改了什么、甚至整体退回那一刻。
- 快照只存在你电脑上的隐藏文件夹 `.git` 里，**不上网**；想放到网上（GitHub）是另一个动作（叫 push）。

类比：

| Git 概念 | 类比 |
|---|---|
| 仓库 repository | 一本相册（就是 `C:\Users\12983\Documents\repro` 这个文件夹） |
| 提交 commit | 相册里的一页照片（某一刻的完整状态 + 说明） |
| 工作区 working tree | 你现在桌上摊着的文件（可能还没拍进相册） |
| 暂存区 stage | "这一页我要收进相册的照片"暂存筐 |
| 分支 branch | 相册的分册（默认叫 `main`） |
| 远程 remote / GitHub | 把相册复印一份放到网盘上 |
| `origin` | 网盘那个地址的默认名字 |

**最重要的一句：Git 只记录你"主动提交"的内容；不提交的改动它不负责。**

---

## 1. 我们的仓库在哪、里面有什么

```
C:\Users\12983\Documents\repro        ← 仓库根目录（在这里打开终端 / 双击 bat）
├── .git\                             ← Git 的数据库（隐藏文件夹，别手动改）
├── Zero-DCE\Zero-DCE_code\           ← 官方代码 + 我们的 5 处修改 + 边缘损失
├── scripts\                          ← 我们写的所有脚本（preprocess / 评测 / 画图 / 预处理…）
├── weights\                          ← 4 个推理权重（小文件，方便带走）
├── docs\                             ← 所有报告与教程（md）
├── demo_edge\、demo_mosaic\          ← 实验图与指标 CSV
├── 保存到Git.bat                     ← 一键保存（推荐日常用这个）
└── README.md                         ← 总说明
```

**不进去的东西**（由 `.gitignore` 自动跳过，因为太大或可再生成）：`.venv`（虚拟环境）、`wheels`（2.4 GB 安装包）、`低照图库`/`lol_test` 等数据集、`snapshots_*`（训练出来的大量权重）、`__pycache__`、日志。

> 换句话说：仓库只存"代码 + 文档 + 关键图 + 小权重"，不存"能重新下载/重新生成的大文件"。

---

## 2. 日常只有三件事

### ① 看：我现在改了什么？

```powershell
cd C:\Users\12983\Documents\repro
& .\PortableGit\cmd\git.exe status
```

- 红色 = 改了但还没保存快照
- 绿色 = 已经放进"暂存筐"（下一步）
- 输出 `nothing to commit, working tree clean` = 什么都没改

想看**具体改了哪几行**：

```powershell
& .\PortableGit\cmd\git.exe diff            # 未保存的改动
& .\PortableGit\cmd\git.exe diff --stat     # 只看每个文件改了几行
```

### ② 存：把当前状态拍成一张快照

最省事：**双击 `保存到Git.bat`**（会先给你看改了哪些文件，然后让你输入一句说明，回车即可）。

命令行等价于：

```powershell
& .\PortableGit\cmd\git.exe add -A                              # 把所有改动放进暂存筐
& .\PortableGit\cmd\git.exe commit -m "说明：这次做了什么"      # 拍快照
```

说明写清楚一点，比如 `加了马赛克预处理 preprocess.py 和 E0~E5 对照实验`。

### ③ 翻：看历史 / 回到过去

```powershell
& .\PortableGit\cmd\git.exe log --oneline           # 列出所有快照（一行一条）
& .\PortableGit\cmd\git.exe show <快照号>           # 看某次快照改了什么
& .\PortableGit\cmd\git.exe show <快照号>:README.md # 看那次快照里某个文件的内容
& .\PortableGit\cmd\git.exe diff <快照1> <快照2>    # 两次快照之间的差异
```

**只把某个文件退回到某次快照**（最常用、最安全）：

```powershell
& .\PortableGit\cmd\git.exe restore --source=<快照号> -- Zero-DCE\Zero-DCE_code\Myloss.py
```

**整体退回某次快照**（会丢弃之后的改动，慎用）：

```powershell
& .\PortableGit\cmd\git.exe reset --hard <快照号>
```

快照号就是 `git log --oneline` 每行开头那串 7 位字符（如 `a206a34`）。

---

## 3. 传到 GitHub（"调用" = 在别处拿到它）

### 第一次：建一个空仓库并推上去

1. 打开 https://github.com/new ，仓库名随便（如 `zero-dce-repro`），**不要**勾选 "Add a README"，点 Create。
2. 在终端里执行（把地址换成你自己的）：

```powershell
cd C:\Users\12983\Documents\repro
& .\PortableGit\cmd\git.exe remote add origin https://github.com/<你的用户名>/zero-dce-repro.git
& .\PortableGit\cmd\git.exe push -u origin main
```

> 现在仓库里填的是占位身份 `12983 / 12983@users.noreply.github.com`，推之前建议改成你自己的：
> `& .\PortableGit\cmd\git.exe config user.email "你的邮箱"`

### 以后：改完就推

```powershell
& .\PortableGit\cmd\git.exe add -A
& .\PortableGit\cmd\git.exe commit -m "说明"
& .\PortableGit\cmd\git.exe push
```

（`保存到Git.bat` 在有 `origin` 时会自动帮你 push。）

### 在别的电脑上"调用"这份代码

```powershell
& .\PortableGit\cmd\git.exe clone https://github.com/<你的用户名>/zero-dce-repro.git
```

---

## 4. 常见疑问

**Q：我改了文件但没提交，会发生什么？**
Git 不管；关电脑也不丢（文件还在硬盘上），但**拍照册里没有**。下次提交时一起存。

**Q：提交后发现改错了怎么办？**
- 只丢某个文件的改动：`git restore -- 路径`
- 想退回上一个快照：`git reset --hard HEAD`（会丢弃所有未提交改动，慎用）
- 已提交但想撤销：`git revert <快照号>`（生成一次"反向提交"，最安全）

**Q：.venv、数据集为什么不进仓库？**
太大且可重建（`.venv` 重装很快、数据集能重新下载）。仓库要"轻"，才能秒开、好备份、能推到 GitHub。

**Q：Git 和 GitHub 是一回事吗？**
不是。Git 是**装在你电脑上的工具**（离线也能用）；GitHub 是**放仓库的网站**。类比：Git 是 Word，GitHub 是网盘。

**Q：我能用别的客户端吗？**
可以：VSCode 左侧"源代码管理"图标、GitHub Desktop、或命令行。它们做的是同一件事。

**Q：我记不住命令怎么办？**
日常只用双击 `保存到Git.bat`；要查看/回退时再看本文件第 2 节。命令可以复制粘贴。

---

## 5. 这个仓库现在有什么（快照记录）

| 快照 | 说明 |
|---|---|
| `a206a34` | 初始提交：Zero-DCE 复现 + 论文 (4.9~4.12) Sobel 边缘损失改进 + 预处理方案 + 全部报告与图表 |
| `33748b7` | 修正 `保存到Git.bat` 的中文控制台输出编码 |
| 下一次 | 马赛克预处理（`scripts/preprocess.py` + `enhance.py` 开关 + E0~E5 对照实验与图表 + 文档） |

用 `git log --oneline` 能看到完整列表。

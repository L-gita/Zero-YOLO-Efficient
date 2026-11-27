# 保存到 Git：查看改动 → 输入说明 → 提交 →（若已配置远程）推送
# 由 保存到Git.bat 调用。本文件必须以 UTF-8 with BOM 保存，否则 PS 5.1 读中文会乱码。

# 让控制台按 UTF-8 显示中文（否则双击运行时中文提示是乱码）
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
try { $OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$git = Join-Path $root 'PortableGit\cmd\git.exe'

if (-not (Test-Path $git)) {
    Write-Host "找不到 git：$git" -ForegroundColor Red
    Read-Host "按回车键退出"
    exit 1
}

Write-Host "===== 当前改动 =====" -ForegroundColor Cyan
& $git status --short
Write-Host ""

$msg = Read-Host "请输入本次提交说明（直接回车 = 用默认说明）"
if ([string]::IsNullOrWhiteSpace($msg)) { $msg = "更新：代码与文档" }

& $git add -A
& $git commit -m $msg
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "没有新改动需要提交（或提交失败），已停止。" -ForegroundColor Yellow
    Read-Host "按回车键退出"
    exit 0
}

$remotes = & $git remote
if ($remotes -contains 'origin') {
    Write-Host ""
    Write-Host "===== 推送到 GitHub =====" -ForegroundColor Cyan
    & $git push
} else {
    Write-Host ""
    Write-Host "还没有配置远程仓库，本次只做了本地提交。" -ForegroundColor Yellow
    Write-Host "上传到 GitHub 的方法见：docs\Git保存与上传.md 第 3 节" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "===== 最近提交 =====" -ForegroundColor Cyan
& $git log --oneline -5
Read-Host "按回车键退出"

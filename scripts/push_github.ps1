param(
    [Parameter(Position = 0)][string]$RepoUrl
)

# 把本地仓库推送到 GitHub（首次会要求登录：浏览器弹窗 / PAT）
$ErrorActionPreference = 'Stop'
$repro = Split-Path -Parent $PSScriptRoot
$git = Join-Path $repro 'PortableGit\cmd\git.exe'
if (-not (Test-Path $git)) { $git = 'git' }

if (-not $RepoUrl -or $RepoUrl.Trim() -eq '') {
    Write-Host '请输入你的 GitHub 仓库地址（形如 https://github.com/你的用户名/zero-dce-repro）：' -ForegroundColor Cyan
    $RepoUrl = Read-Host '仓库地址'
}
$RepoUrl = $RepoUrl.Trim().Trim('"').Trim("'")
if ($RepoUrl -notmatch '^(https://|git@)') {
    Write-Host "地址看起来不对：$RepoUrl" -ForegroundColor Red
    Write-Host '正确示例：https://github.com/abc/zero-dce-repro'; Read-Host '按回车关闭'; exit 1
}
if ($RepoUrl -match '^https://[^/]+/[^/]+$') { $RepoUrl = "$RepoUrl.git" }

Push-Location $repro
try {
    $remotes = & $git remote
    if ($remotes -contains 'origin') {
        & $git remote set-url origin $RepoUrl
        Write-Host "已更新 origin → $RepoUrl" -ForegroundColor Yellow
    } else {
        & $git remote add origin $RepoUrl
        Write-Host "已添加 origin → $RepoUrl" -ForegroundColor Yellow
    }

    Write-Host ''
    Write-Host '本地最近提交：' -ForegroundColor Cyan
    & $git -c core.quotepath=false log --oneline -10
    Write-Host ''
    Write-Host '开始推送（首次约 130 MB，视网速需 1~10 分钟；可能会弹出 GitHub 登录窗口）...' -ForegroundColor Cyan
    & $git push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host ''
        Write-Host '推送成功！打开仓库主页看看：' -ForegroundColor Green
        $web = $RepoUrl -replace '\.git$', ''
        Write-Host "  $web"
        try { Start-Process $web } catch { }
    } else {
        Write-Host ''
        Write-Host '推送失败。常见原因：' -ForegroundColor Red
        Write-Host '  1) 没登录：再跑一次，等 GitHub 登录窗口（或用 Personal Access Token 当密码）'
        Write-Host '  2) 远程仓库不是空的（建库时勾了 README/.gitignore）→ 先建一个空仓库，或执行：'
        Write-Host '     git push -u origin main --force'
        Write-Host "  3) 地址写错：当前 origin = $RepoUrl"
    }
} finally {
    Pop-Location
}
Write-Host ''
Read-Host '按回车键关闭窗口'

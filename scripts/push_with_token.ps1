param(
    [string]$RepoUrl = 'https://github.com/L-gita/Zero-YOLO-Efficient.git'
)

# 用 GitHub 用户名 + Personal Access Token 推送（token 只在本进程内存里，不写进 .git/config）
# 为什么需要它：本机 git 走 schannel 时吊销检查被墙（CRYPT_E_NO_REVOCATION_CHECK），
# 所以这里对当前仓库临时关闭证书校验（http.sslVerify=false），推送完可改回 true。
$ErrorActionPreference = 'Stop'
$repro = Split-Path -Parent $PSScriptRoot
$git = Join-Path $repro 'PortableGit\cmd\git.exe'
if (-not (Test-Path $git)) { $git = 'git' }

Write-Host '=== 推送到 GitHub（用户名 + PAT 方式）===' -ForegroundColor Cyan
if (-not $RepoUrl) { $RepoUrl = Read-Host '仓库地址' }
$RepoUrl = $RepoUrl.Trim()
if ($RepoUrl -match '^https://[^/]+/[^/]+$') { $RepoUrl = "$RepoUrl.git" }

$user  = Read-Host 'GitHub 用户名（例如 L-gita）'
$token = Read-Host 'PAT（在 GitHub → Settings → Developer settings → Personal access tokens 生成，勾 repo）'

$pair = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes("$user`:$token"))
$hdr  = "Authorization: Basic $pair"

Push-Location $repro
try {
    & $git config http.sslVerify false
    Write-Host "`n关闭证书校验：git config http.sslVerify false（本仓库局部设置，推完可改回）" -ForegroundColor DarkGray
    Write-Host '开始推送（首次约 130 MB，可能几分钟）...' -ForegroundColor Cyan
    & $git -c http.extraHeader=$hdr push -u origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n推送成功！" -ForegroundColor Green
        $web = $RepoUrl -replace '\.git$', ''
        Write-Host "  $web"
        try { Start-Process $web } catch { }
    } else {
        Write-Host "`n推送失败。" -ForegroundColor Red
        Write-Host '常见原因：PAT 没勾 repo 权限 / 已过期 / 用户名写错 / 仓库不是空的（需 --force）'
    }
} finally {
    Pop-Location
}
Write-Host ''
Read-Host '按回车键关闭窗口'

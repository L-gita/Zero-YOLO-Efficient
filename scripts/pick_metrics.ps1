param([Parameter(Position=0, ValueFromRemainingArguments=$true)][string[]]$Inputs)

$ErrorActionPreference = 'Stop'
$repro = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repro 'enhance.py'))) { throw "未找到 enhance.py，repro 路径不正确：$repro" }
$py = Join-Path $repro '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = 'python' }
$script = Join-Path $repro 'scripts\image_metrics.py'

$files = @()
if ($Inputs -and $Inputs.Count -gt 0) {
    foreach ($p in $Inputs) {
        if (Test-Path -LiteralPath $p -PathType Container) {
            $files += Get-ChildItem -LiteralPath $p -File -Recurse | Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } | ForEach-Object { $_.FullName }
        } elseif (Test-Path -LiteralPath $p) {
            $files += (Resolve-Path -LiteralPath $p).Path
        }
    }
} else {
    Add-Type -AssemblyName System.Windows.Forms | Out-Null
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = '选择要计算指标的图片（可多选）'
    $dlg.Multiselect = $true
    $dlg.Filter = '图片|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件|*.*'
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { Write-Host '已取消'; Read-Host '按回车退出'; exit 0 }
    $files = $dlg.FileNames
}

if ($files.Count -eq 0) { Write-Host '没有选中图片'; Read-Host '按回车退出'; exit 1 }

$first = @($files)[0]
$outDir = Join-Path (Split-Path -Parent $first) '指标结果'
Write-Host ''
Write-Host "共 $($files.Count) 张图，输出目录：$outDir" -ForegroundColor Cyan
Write-Host '正在计算输入图指标 + 最终版（去马赛克）增强后的指标 …' -ForegroundColor Cyan
Write-Host ''

& $py $script @files -o $outDir --with-base
$code = $LASTEXITCODE

Write-Host ''
if ($code -eq 0) {
    Write-Host '完成：全部成功' -ForegroundColor Green
} else {
    Write-Host "结束，但有失败项（退出码 $code），请看 report.md 里的失败清单" -ForegroundColor Yellow
}
if (Test-Path (Join-Path $outDir 'report.md')) { Start-Process notepad.exe (Join-Path $outDir 'report.md') }
Start-Process explorer.exe $outDir
Read-Host '按回车退出'

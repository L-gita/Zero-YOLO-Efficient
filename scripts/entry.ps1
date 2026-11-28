param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Inputs
)

# 三个版本入口的统一实现（由同目录 .bat 用环境变量 ZERODCE_ENTRY 指定）：
#   original → Zero 原版（不加任何改进）
#   edge     → 仅边缘增强版（论文 4.9~4.12 的 Sobel 边缘损失，无马赛克预处理）
#   mosaic   → 去马赛克最终版（auto v3 场景自适应预处理，不含边缘损失）
$Entry = "$($env:ZERODCE_ENTRY)".Trim().ToLower()
if ($Entry -notin @('original', 'edge', 'mosaic', 'region')) { $Entry = 'mosaic' }

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null

$repro = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repro 'enhance.py'))) {
    $repro = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
}
$py  = Join-Path $repro '.venv\Scripts\python.exe'
$enh = Join-Path $repro 'enhance.py'

$cfg = switch ($Entry) {
    'original' { @{ Title = 'Zero 原版（不加任何改进）'; DirName = 'Zero原版结果'; Args = @('-m', 'sice') } }
    'edge'     { @{ Title = '仅边缘增强版（Sobel 边缘损失）'; DirName = '边缘增强结果'; Args = @('-m', 'edge') } }
    'mosaic'   { @{ Title = '去马赛克最终版（场景自适应预处理）'; DirName = '去马赛克最终版结果'; Args = @('--preprocess', 'auto') } }
    'region'   { @{ Title = '单图分区增强（暗区单独处理，亮区保持原版）'; DirName = '单图分区增强结果'; Args = @('--preprocess', 'region', '--region-mode', 'gain') } }
}

function Collect-Files($paths) {
    $out = @()
    foreach ($p in $paths) {
        if (Test-Path -LiteralPath $p -PathType Container) {
            $out += Get-ChildItem -LiteralPath $p -Recurse -File |
                    Where-Object { $_.Extension -match '^\.(jpg|jpeg|png|bmp|tif|tiff|webp)$' } |
                    ForEach-Object { $_.FullName }
        } elseif (Test-Path -LiteralPath $p) {
            $out += (Resolve-Path -LiteralPath $p).Path
        }
    }
    return $out
}

if ($Inputs -and $Inputs.Count -gt 0) {
    $files = Collect-Files $Inputs
} else {
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Title = "选择照片：$($cfg.Title)（按住 Ctrl 可多选）"
    $dlg.Filter = '图片文件|*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.webp|所有文件|*.*'
    $dlg.Multiselect = $true
    if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit }
    $files = @($dlg.FileNames)
}

$files = @($files | Where-Object { $_ })
if ($files.Count -eq 0) {
    [System.Windows.Forms.MessageBox]::Show('没有找到任何图片。', 'Zero-DCE') | Out-Null
    exit
}

$parent = Split-Path -Parent $files[0]
$outDir = Join-Path $parent $cfg.DirName

Write-Host ""
Write-Host "版本：$($cfg.Title)" -ForegroundColor Cyan
Write-Host "共 $($files.Count) 张，输出目录：$outDir" -ForegroundColor DarkGray
if ($Entry -eq 'edge') {
    Write-Host "  使用权重 snapshots_sice_edge\Epoch49.pth（50 轮，λ=0.22）" -ForegroundColor DarkGray
} elseif ($Entry -eq 'region') {
    Write-Host "  分区增强：按局部亮度分位找暗区，暗区用 gamma 提亮（保纹理）、亮区保持原版；并做暗区色度平滑" -ForegroundColor DarkGray
} elseif ($Entry -eq 'mosaic') {
    Write-Host "  最终版：按噪声/块效应自适应——只对亮度降噪、去块按需、暗部补偿；不碰色度（不造色块）" -ForegroundColor DarkGray
}
Write-Host ""

Push-Location $repro
try {
    & $py $enh @files @($cfg.Args) -c -o $outDir
    $code = $LASTEXITCODE
} finally {
    Pop-Location
}

if ($code -eq 0) {
    Write-Host "`n完成！输出目录：" -ForegroundColor Green
    Write-Host "  $outDir"
    Write-Host '  每张：xxx_enhanced.png（成片）+ xxx_compare.png（左右对比）'
    try { Start-Process explorer.exe $outDir } catch { }
} else {
    Write-Host '处理过程中出错了，把上面的报错发我。' -ForegroundColor Red
}
Write-Host ''
Read-Host '按回车键关闭窗口'

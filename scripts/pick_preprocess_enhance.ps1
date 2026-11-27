param(
    [Parameter(Position = 0, ValueFromRemainingArguments = $true)][string[]]$Inputs
)

# 拖拽/选图入口：去马赛克增强（E10 场景自适应）
#   环境变量 ZERODCE_PRE = auto（默认）→ 场景自适应增强，输出 ...\增强结果_去马赛克
#                        = mosaic        → 旧版 E9 固定管线
#                        = compare       → 一次出「输入 | 基线 | 去马赛克」三列对比 + 指标
$Mode = "$($env:ZERODCE_PRE)".Trim().ToLower()
if ($Mode -notin @('auto', 'mosaic', 'compare')) { $Mode = 'auto' }
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms | Out-Null

# 本脚本放在 scripts\ 下，repro 根目录是它的上一级
$repro = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $repro 'enhance.py'))) { $repro = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }
$py    = Join-Path $repro '.venv\Scripts\python.exe'
$enh   = Join-Path $repro 'enhance.py'
$cmp   = Join-Path $repro 'scripts\compare_preprocess.py'

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
    $dlg.Title = '选择要去马赛克的照片（按住 Ctrl 可多选）'
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

if ($Mode -eq 'compare') {
    $outDir = Join-Path $parent '对比_基线vs去马赛克'
    Write-Host ''
    Write-Host "共 $($files.Count) 张：每张输出「输入 | 基线 | 去马赛克」三列对比 + 3× 局部放大" -ForegroundColor Cyan
    Write-Host "输出目录：$outDir" -ForegroundColor DarkGray
    Write-Host ''
    & $py $cmp @files -o $outDir --mode auto
} else {
    $outDir = Join-Path $parent '增强结果_去马赛克'
    $preset = if ($Mode -eq 'mosaic') { 'mosaic' } else { 'auto' }
    Write-Host ''
    Write-Host "共 $($files.Count) 张，使用 --preprocess $preset ..." -ForegroundColor Cyan
    Write-Host "  $preset = 场景自适应：噪声重才降噪、块效应才去块、只对暗图做近黑限增益" -ForegroundColor DarkGray
    Write-Host "输出目录：$outDir" -ForegroundColor DarkGray
    Write-Host ''
    Push-Location $repro
    & $py $enh @files --preprocess $preset -c -o $outDir
    Pop-Location
}

if ($LASTEXITCODE -eq 0) {
    Write-Host ''
    Write-Host '完成！输出目录：' -ForegroundColor Green
    Write-Host "  $outDir"
    if ($Mode -eq 'compare') {
        Write-Host '  每张：xxx_对比.png（上排三列、下排放大）；另有 metrics.csv / summary.md'
    } else {
        Write-Host '  每张：xxx_enhanced.png（成片）+ xxx_compare.png（左右对比）'
    }
    try { Start-Process explorer.exe $outDir } catch { }
} else {
    Write-Host '处理过程中出错了，把上面的报错发我。' -ForegroundColor Red
}
Write-Host ''
Read-Host '按回车键关闭窗口'